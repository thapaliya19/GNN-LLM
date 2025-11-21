# pip install wandb pandas numpy scipy
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Optional
import time
import random
import argparse
import concurrent.futures as cf
from tqdm import tqdm


import wandb
import pandas as pd
import numpy as np
from scipy.stats import t

# ---------------- Argument Parser ----------------
parser = argparse.ArgumentParser(description="Aggregate W&B sweep results with seed coverage checking (parallel fetch).")
parser.add_argument("--entity", type=str, default="thapaliya19", help="W&B entity (user or org name)")
parser.add_argument("--project", type=str, default="autoprompt", help="W&B project name")
parser.add_argument("--sweep_id", type=str, required=True, help="W&B sweep ID (e.g., 'abc12345')")
parser.add_argument("--max_workers", type=int, default=16, help="Max parallel fetch workers")
parser.add_argument("--max_retries", type=int, default=5, help="Max retries per run fetch")
parser.add_argument("--timeout", type=int, default=60, help="W&B API timeout (seconds) per request")
args = parser.parse_args()

ENTITY = args.entity
PROJECT = args.project
SWEEP_ID = args.sweep_id

print(f"Finding sweep: {ENTITY}/{PROJECT}/{SWEEP_ID}")

# Seeds you intended to run (used to compute coverage)
REQUIRED_SEEDS = set(range(1, 31))
MIN_COVERAGE = 6  # require ≥10 seeds per config

# Metric keys as logged in your runs
VAL_KEY = "best/val"
TEST_KEY = "best/test"

HP_KEYS = ["hidden_dim", "decay", "dropout", "lr", "normalize", "num_layers"]  # exclude seed

def canonicalize_config(cfg: Dict[str, Any]) -> Tuple:
    """Return a hashable key for config excluding the seed and non-HP fields."""
    row = []
    for k in HP_KEYS:
        v = cfg.get(k, None)
        if isinstance(v, str) and v.lower() == "none":
            v = "none"
        row.append((k, v))
    return tuple(row)

@dataclass
class RunRec:
    config_key: Tuple
    seed: int
    val: float
    test: float
    run_id: str
    name: str

def dedupe_by_seed(recs: List[RunRec]) -> List[RunRec]:
    """
    For each (config_key, seed) keep the run with the best (val, then test).
    """
    best: Dict[Tuple[Tuple, int], RunRec] = {}
    for r in recs:
        k = (r.config_key, r.seed)
        if k not in best:
            best[k] = r
            continue
        # Prefer higher val; break ties by higher test
        prev = best[k]
        if (r.val, r.test) > (prev.val, prev.test):
            best[k] = r
    return list(best.values())



def fetch_runs_fast(entity: str, project: str, sweep_id: str,
                    timeout: int = 60,
                    per_page: int = 1000,
                    state: str = "finished") -> List[RunRec]:
    """
    Fetch runs for a sweep in bulk via api.runs(filters=...),
    avoiding per-run api.run() calls.
    """
    api = wandb.Api(timeout=timeout)
    # Filter by sweep ID and (optionally) state
    filters = {"$and": [{"sweep": sweep_id}]}
    if state:
        filters["$and"].append({"state": state})

    # Run listing pages already contain summary & config; no extra calls needed.
    runs = api.runs(f"{entity}/{project}", filters=filters, per_page=per_page)

    recs: List[RunRec] = []
    # tqdm without a total (unknown upfront) streams nicely; if you want a total,
    # you can pre-count by materializing to a list (but that doubles memory/time).
    for r in tqdm(runs, desc=f"Listing runs for {entity}/{project} (sweep {sweep_id})", unit="run"):
        try:
            cfg = r.config or {}
            seed = cfg.get("seed", None)
            if seed is None:
                continue

            summ = r.summary or {}
            val = summ.get(VAL_KEY, None)
            test = summ.get(TEST_KEY, None)
            if val is None or test is None:
                continue

            key = canonicalize_config(cfg)
            recs.append(RunRec(
                config_key=key,
                seed=int(seed),
                val=float(val),
                test=float(test),
                run_id=r.id,
                name=r.name or r.id
            ))
        except Exception as e:
            # Keep streaming; this only skips a problematic run
            tqdm.write(f"[WARN] Skipping run {getattr(r, 'id', 'unknown')}: {e}")
            continue
    return recs

# ---------------- Parallel run fetching ----------------
def _fetch_one_run(run_id: str) -> Optional[RunRec]:
    """
    Fetch a single run (with retries), returning RunRec or None if unusable.
    Uses its own Api() (safer for threads).
    """
    delay = 0.5
    for attempt in range(1, args.max_retries + 1):
        try:
            api = wandb.Api(timeout=args.timeout)
            r = api.run(f"{ENTITY}/{PROJECT}/{run_id}")

            # Extract fields
            seed = r.config.get("seed", None)
            if seed is None:
                return None

            # Pull metrics
            val = r.summary.get(VAL_KEY, None)
            test = r.summary.get(TEST_KEY, None)
            if val is None or test is None:
                return None

            key = canonicalize_config(r.config)
            name = r.name or r.id
            return RunRec(key, int(seed), float(val), float(test), r.id, name)

        except Exception as e:
            if attempt == args.max_retries:
                print(f"[WARN] Giving up run {run_id} after {attempt} attempts: {e}")
                return None
            # Exponential backoff with jitter
            sleep_s = delay * (2 ** (attempt - 1)) + random.uniform(0, 0.25)
            time.sleep(sleep_s)

def fetch_runs_parallel(entity: str, project: str, sweep_id: str) -> List[RunRec]:
    """List run IDs from the sweep, then fetch runs concurrently."""
    api = wandb.Api(timeout=args.timeout)
    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
    print("Sweep name:", sweep.name)

    # Force materialization of run ids first (cheap metadata pull)
    run_ids = [r.id for r in sweep.runs]
    total = len(run_ids)
    if total == 0:
        return []

    print(f"Found {total} runs. Fetching in parallel with {args.max_workers} workers...")

    recs: List[RunRec] = []
    completed = 0
    # ThreadPool is appropriate for network I/O
    with cf.ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = {ex.submit(_fetch_one_run, rid): rid for rid in run_ids}
        with tqdm(total=total, desc=f"Sweep {sweep_id}", ncols=90, colour="cyan", unit="run") as pbar:
            for fut in cf.as_completed(futures):
                rid = futures[fut]
                try:
                    rec = fut.result()
                    if rec is not None:
                        recs.append(rec)
                except Exception as e:
                    tqdm.write(f"[WARN] Exception fetching {rid}: {e}")
                finally:
                    pbar.update(1)

    return recs

# ---------------- Aggregation (unchanged) ----------------
def agg_by_config(recs: List[RunRec]) -> pd.DataFrame:
    buckets = defaultdict(list)
    for rr in recs:
        buckets[rr.config_key].append(rr)

    rows = []
    for key, rs in buckets.items():
        seeds = sorted({r.seed for r in rs})
        vals = np.array([r.val for r in rs], dtype=float)
        tests = np.array([r.test for r in rs], dtype=float)
        n = len(vals)
        # means / std / stderr
        mean_val, mean_test = float(vals.mean()), float(tests.mean())
        std_val, std_test = (float(vals.std(ddof=1)) if n > 1 else 0.0,
                             float(tests.std(ddof=1)) if n > 1 else 0.0)
        sem_val, sem_test = (std_val/np.sqrt(n) if n > 1 else 0.0,
                             std_test/np.sqrt(n) if n > 1 else 0.0)
        # 95% t CI (two-sided)
        if n > 1:
            tcrit = t.ppf(0.975, df=n-1)
            ci_val = (mean_val - tcrit*sem_val, mean_val + tcrit*sem_val)
            ci_test = (mean_test - tcrit*sem_test, mean_test + tcrit*sem_test)
        else:
            ci_val = (mean_val, mean_val)
            ci_test = (mean_test, mean_test)

        cfg = dict(key)
        rows.append({
            **cfg,
            "seeds_n": n,
            "seeds_covered": seeds,
            "coverage": n / len(REQUIRED_SEEDS) if len(REQUIRED_SEEDS) else np.nan,
            "mean_val": mean_val,
            "std_val": std_val,
            "sem_val": sem_val,
            "ci95_val_low": ci_val[0],
            "ci95_val_high": ci_val[1],
            "mean_test": mean_test,
            "std_test": std_test,
            "sem_test": sem_test,
            "ci95_test_low": ci_test[0],
            "ci95_test_high": ci_test[1],
            "mean_both": 0.5*(mean_val + mean_test),
            "min_both": min(mean_val, mean_test),
            "lcb_both": min(ci_val[0], ci_test[0]),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["mean_val", "mean_test"], ascending=False)
    return df

def pareto_front(df: pd.DataFrame, x="mean_val", y="mean_test") -> pd.DataFrame:
    """Return non-dominated configs maximizing both x and y."""
    d = df.sort_values([x, y], ascending=[False, False]).reset_index(drop=True)
    pareto_idx = []
    best_y = -np.inf
    for i, row in d.iterrows():
        if row[y] >= best_y - 1e-12:
            pareto_idx.append(i)
            best_y = row[y]
    return d.loc[pareto_idx]

def main():
    recs = fetch_runs_fast(ENTITY, PROJECT, SWEEP_ID, timeout=120, per_page=1000, state="finished")
    if not recs:
        print("No runs with both metrics found.")
        return
    
    recs = dedupe_by_seed(recs)


    df = agg_by_config(recs)

    # Filter by coverage
    df_cov = df[df["seeds_n"] >= MIN_COVERAGE].copy()
    if df_cov.empty:
        # fallback: keep the top coverage tier
        top_n = df["seeds_n"].max()
        df_cov = df[df["seeds_n"] == top_n].copy()

    # Three ways to surface "good" configs:
    top_by_val = df_cov.sort_values(["mean_val", "mean_test"], ascending=False).head(30)
    top_by_lcb = df_cov.sort_values("lcb_both", ascending=False).head(30)
    pareto = pareto_front(df_cov)

    # Save artifacts
    df.to_csv("all_configs_aggregated.csv", index=False)
    top_by_val.to_csv("top_by_val.csv", index=False)
    top_by_lcb.to_csv("top_by_lcb.csv", index=False)
    pareto.to_csv("pareto_front.csv", index=False)

    # Pretty print a short summary
    def brief(tbl, title):
        cols = HP_KEYS + ["seeds_n", "mean_val", "mean_test", "lcb_both"]
        print(f"\n== {title} ==")
        print(tbl[cols].to_string(index=False))

    brief(top_by_val, "Top-10 by mean validation (tiebreaker: mean test)")
    brief(top_by_lcb, "Top-10 by conservative lower CI across val/test")
    brief(pareto.head(10), "Pareto front (first 10) on mean_val vs mean_test")

if __name__ == "__main__":
    main()
