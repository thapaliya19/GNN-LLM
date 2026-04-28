# src/utils/exp_naming.py
from __future__ import annotations
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

_SAFE = re.compile(r"[^a-zA-Z0-9._:=,+-]+")

def _safe(s: str) -> str:
    s = s.strip().replace(" ", "")
    s = _SAFE.sub("_", s)
    return s[:180]  # keep path components sane

def _stable_json(d: Dict[str, Any]) -> str:
    # stringify non-JSON types conservatively
    def _default(x):
        return str(x)
    return json.dumps(d, sort_keys=True, default=_default, separators=(",", ":"))

def short_hash(d: Dict[str, Any], n: int = 8) -> str:
    h = hashlib.sha1(_stable_json(d).encode("utf-8")).hexdigest()
    return h[:n]

def pick(d: Dict[str, Any], keys: Iterable[str]) -> Dict[str, Any]:
    return {k: d.get(k, None) for k in keys}

def build_run_slug(params: Dict[str, Any]) -> str:
    """
    Human-readable prefix + short hash over all "identity" params.
    Change ID_KEYS/HASH_KEYS if you want more/less verbosity.
    """
    ID_KEYS = [
        "data", "llm", "emb_method", "backbone",
        "sim_method", "buffer_k", "buffer_alpha", "embed_alpha",
        "ex_block_template", "prompt_template",
        "desc_mode", "node_text",
        "label_setting", "train_num", "val_num", "train_ratio", "val_ratio",
        "tta_max_iters", "buffer_size", "tta_topk", "entropy_thr",
        "struct_epochs", "seed",
    ]

    # Hash can include more stuff (e.g., file basenames) without making slug huge
    HASH_KEYS = list(set(ID_KEYS + [
        "desc_csv", "topo_csv", "tgt_desc_csv", "tgt_topo_csv",
        "rewrite_all", "rewrite_iters", "rewrite_batch_size",
    ]))

    # Normalize a couple fields so they don’t explode the path
    p = dict(params)

    # node_text can be list; make it stable
    if isinstance(p.get("node_text"), (list, tuple)):
        p["node_text"] = "_".join(map(str, p["node_text"]))

    # only keep basenames for csv paths (full paths still affect the hash if you want)
    for k in ["desc_csv", "topo_csv", "tgt_desc_csv", "tgt_topo_csv"]:
        if p.get(k):
            p[k] = Path(str(p[k])).name

    prefix = []
    for k in ID_KEYS:
        v = p.get(k, None)
        if v is None:
            continue
        prefix.append(f"{k}={v}")
    prefix_str = _safe(",".join(prefix))

    h = short_hash(pick(p, HASH_KEYS), n=10)
    tag = p.get("run_tag", None)
    if tag:
        return f"{prefix_str},tag={_safe(str(tag))},h={h}"
    return f"{prefix_str},h={h}"

def ensure_run_dir(run_dir: str) -> str:
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def dump_config(run_dir: str, params: Dict[str, Any]) -> None:
    Path(run_dir, "config.json").write_text(_stable_json(params), encoding="utf-8")
