# pip install wandb pandas numpy scipy
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List
import wandb
import pandas as pd
import numpy as np
from scipy.stats import t
import argparse

# ---------------- Argument Parser ----------------
parser = argparse.ArgumentParser(description="Aggregate W&B sweep results with seed coverage checking.")
parser.add_argument("--entity", type=str, default="thapaliya19", help="W&B entity (user or org name)")
parser.add_argument("--project", type=str, default="autoprompt", help="W&B project name")
parser.add_argument("--sweep_id", type=str, required=True, help="W&B sweep ID (e.g., 'abc12345')")
args = parser.parse_args()

ENTITY = args.entity
PROJECT = args.project
SWEEP_ID = args.sweep_id

print(f"Finding sweep: {ENTITY}/{PROJECT}/{SWEEP_ID}")

# Seeds you intended to run (used to compute coverage)
REQUIRED_SEEDS = set(range(1, 31))
MIN_COVERAGE = 15      # e.g., require ≥10 seeds per config; set to len(REQUIRED_SEEDS) for full coverage

# Metric keys as logged in your runs
VAL_KEY = "best/val"
TEST_KEY = "best/test"

HP_KEYS = ["hidden_dim", "decay", "dropout", "lr", "normalize", "num_layers"]  # exclude seed

def canonicalize_config(cfg: Dict[str, Any]) -> Tuple:
    """Return a hashable key for config excluding the seed and non-HP fields."""
    # Pull only declared HP keys; normalize strings like 'none'/'None'
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

def fetch_runs(entity: str, project: str, sweep_id: str) -> List[RunRec]:
    api = wandb.Api(timeout=60)
    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
    print("Sweep name:", sweep.name)
    recs = []
    for r in sweep.runs:
        # extract seed
        seed = r.config.get("seed", None)
        if seed is None:
            continue
        # pull metrics from summary
        val = r.summary.get(VAL_KEY, None)
        test = r.summary.get(TEST_KEY, None)
        if val is None or test is None:
            continue
        key = canonicalize_config(r.config)
        recs.append(RunRec(key, int(seed), float(val), float(test), r.id, r.name or r.id))
    return recs

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
        std_val, std_test = float(vals.std(ddof=1)) if n > 1 else 0.0, float(tests.std(ddof=1)) if n > 1 else 0.0
        sem_val, sem_test = (std_val/np.sqrt(n) if n > 1 else 0.0, std_test/np.sqrt(n) if n > 1 else 0.0)
        # 95% t CI (two-sided)
        if n > 1:
            tcrit = t.ppf(0.975, df=n-1)
            ci_val = (mean_val - tcrit*sem_val, mean_val + tcrit*sem_val)
            ci_test = (mean_test - tcrit*sem_test, mean_test + tcrit*sem_test)
        else:
            ci_val = (mean_val, mean_val)
            ci_test = (mean_test, mean_test)

        # unpack config tuple back to dict columns
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
            # composite scores (optional)
            "mean_both": 0.5*(mean_val + mean_test),
            "min_both": min(mean_val, mean_test),        # conservative
            "lcb_both": min(ci_val[0], ci_test[0]),      # pick by lower CI bound across both
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
    recs = fetch_runs(ENTITY, PROJECT, SWEEP_ID)
    if not recs:
        print("No runs with both metrics found.")
        return
    df = agg_by_config(recs)

    # Filter by coverage
    df_cov = df[df["seeds_n"] >= MIN_COVERAGE].copy()
    if df_cov.empty:
        # fallback: keep the top coverage tier
        top_n = df["seeds_n"].max()
        df_cov = df[df["seeds_n"] == top_n].copy()

    # Three ways to surface "good" configs:
    top_by_val = df_cov.sort_values(["mean_val", "mean_test"], ascending=False).head(10)
    top_by_lcb = df_cov.sort_values("lcb_both", ascending=False).head(10)
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
