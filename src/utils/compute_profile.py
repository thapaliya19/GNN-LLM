# src/utils/compute_profile.py
from __future__ import annotations
import json, os, time, platform, socket, threading, subprocess
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, List

def _now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())

def _safe_run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode("utf-8", errors="ignore").strip()
    except Exception:
        return ""

def _get_git_rev() -> str:
    return _safe_run(["git", "rev-parse", "--short", "HEAD"]) or "unknown"

def _try_import_psutil():
    try:
        import psutil  # type: ignore
        return psutil
    except Exception:
        return None

def _try_import_pynvml():
    try:
        import pynvml  # type: ignore
        return pynvml
    except Exception:
        return None

@dataclass
class PhaseRec:
    seconds: float = 0.0
    cpu_rss_peak_mb: float = 0.0
    gpu_mem_peak_mb: float = 0.0

class ComputeProfile:
    """
    Minimal overhead profiler: phase timing + CPU/GPU peak memory + optional GPU power/util sampling.
    """
    def __init__(
        self,
        enabled: bool,
        out_path: Optional[str] = None,
        sample_gpu: bool = True,
        sample_interval_s: float = 0.5,
        params: Optional[Dict[str, Any]] = None,
    ):
        self.enabled = bool(enabled)
        self.out_path = out_path
        self.params = params or {}

        self.meta: Dict[str, Any] = {
            "created_at": _now_iso(),
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "git_rev": _get_git_rev(),
        }
        self.phases: Dict[str, PhaseRec] = {}
        self.counters: Dict[str, Any] = {}

        self._psutil = _try_import_psutil()
        self._proc = self._psutil.Process(os.getpid()) if self._psutil else None

        # GPU monitoring
        self._pynvml = _try_import_pynvml()
        self._gpu_sample_enabled = sample_gpu and (self._pynvml is not None)
        self._gpu_samples: List[Dict[str, float]] = []
        self._stop_evt = threading.Event()
        self._thr: Optional[threading.Thread] = None
        self._sample_interval_s = float(sample_interval_s)

        # torch peak mem stats (if CUDA)
        self._has_torch = False
        self._torch = None
        try:
            import torch  # type: ignore
            self._torch = torch
            self._has_torch = True
        except Exception:
            self._has_torch = False

        if self.enabled and self._gpu_sample_enabled:
            try:
                self._pynvml.nvmlInit()
            except Exception:
                self._gpu_sample_enabled = False

    def add(self, key: str, val: Any, mode: str = "sum"):
        if not self.enabled:
            return
        if mode == "sum":
            self.counters[key] = self.counters.get(key, 0) + val
        elif mode == "set":
            self.counters[key] = val
        else:
            self.counters[key] = val

    def _cpu_rss_mb(self) -> float:
        if not self._proc:
            return 0.0
        try:
            return float(self._proc.memory_info().rss) / (1024.0 * 1024.0)
        except Exception:
            return 0.0

    def _gpu_mem_mb_torch_peak(self) -> float:
        if not (self._has_torch and self._torch and self._torch.cuda.is_available()):
            return 0.0
        try:
            return float(self._torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
        except Exception:
            return 0.0

    def _reset_torch_gpu_peak(self):
        if not (self._has_torch and self._torch and self._torch.cuda.is_available()):
            return
        try:
            self._torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    def _gpu_poll_loop(self):
        # samples: util%, mem_used_mb, power_w
        h = None
        try:
            h = self._pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            return

        while not self._stop_evt.is_set():
            try:
                util = self._pynvml.nvmlDeviceGetUtilizationRates(h).gpu
                mem = self._pynvml.nvmlDeviceGetMemoryInfo(h).used / (1024.0 * 1024.0)
                # power may fail on some systems
                try:
                    pwr = self._pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
                except Exception:
                    pwr = 0.0
                self._gpu_samples.append({"util": float(util), "mem_mb": float(mem), "power_w": float(pwr)})
            except Exception:
                pass
            time.sleep(self._sample_interval_s)

    def start_gpu_sampling(self):
        if not (self.enabled and self._gpu_sample_enabled):
            return
        if self._thr is not None:
            return
        self._stop_evt.clear()
        self._thr = threading.Thread(target=self._gpu_poll_loop, daemon=True)
        self._thr.start()

    def stop_gpu_sampling(self):
        if not (self.enabled and self._gpu_sample_enabled and self._thr is not None):
            return
        self._stop_evt.set()
        self._thr.join(timeout=2.0)
        self._thr = None

    @contextmanager
    def phase(self, name: str):
        if not self.enabled:
            yield
            return

        t0 = time.perf_counter()
        cpu0 = self._cpu_rss_mb()

        # reset torch peak for this phase (so phase peak is meaningful)
        self._reset_torch_gpu_peak()

        yield

        dt = time.perf_counter() - t0
        cpu_peak = max(cpu0, self._cpu_rss_mb())
        gpu_peak = self._gpu_mem_mb_torch_peak()

        rec = self.phases.get(name, PhaseRec())
        rec.seconds += float(dt)
        rec.cpu_rss_peak_mb = max(rec.cpu_rss_peak_mb, float(cpu_peak))
        rec.gpu_mem_peak_mb = max(rec.gpu_mem_peak_mb, float(gpu_peak))
        self.phases[name] = rec

    def finalize(self):
        if not self.enabled:
            return
        # aggregate GPU samples
        if self._gpu_samples:
            util = [s["util"] for s in self._gpu_samples]
            mem  = [s["mem_mb"] for s in self._gpu_samples]
            pwr  = [s["power_w"] for s in self._gpu_samples if s["power_w"] > 0]
            self.counters["gpu_util_avg"] = sum(util)/len(util)
            self.counters["gpu_util_max"] = max(util)
            self.counters["gpu_mem_avg_mb"] = sum(mem)/len(mem)
            self.counters["gpu_mem_max_mb_nvml"] = max(mem)
            if pwr:
                self.counters["gpu_power_avg_w"] = sum(pwr)/len(pwr)
                self.counters["gpu_power_max_w"] = max(pwr)

    def dump(self, path: Optional[str] = None):
        if not self.enabled:
            return
        self.finalize()
        out = {
            "meta": self.meta,
            "params": self.params,
            "phases": {k: asdict(v) for k, v in self.phases.items()},
            "counters": self.counters,
            "finished_at": _now_iso(),
        }
        p = path or self.out_path
        if not p:
            return
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            json.dump(out, f, indent=2)
