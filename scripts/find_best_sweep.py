import argparse
import wandb
import pandas as pd
import numpy as np
from scipy.stats import t

# ---------------- Argument Parser ----------------
parser = argparse.ArgumentParser(description="Aggregate W&B sweep results with seed coverage checking.")
parser.add_argument("--entity", type=str, default="thapaliya19", help="W&B entity (user or org name)")
parser.add_argument("--project", type=str, default="autoprompt", help="W&B project name")
parser.add_argument("--sweep_id", type=str, required=True, help="W&B sweep ID (e.g., 'abc12345')")
args = parser.parse_args()

ENTITY = args.entity
PROJECT = args.project
SWEEP_ID = args.sweep_id
# --------------------------------------------------

print(f"Finding sweep: {ENTITY}/{PROJECT}/{SWEEP_ID}")

# Required seed set & coverage threshold
REQUIRED_SEEDS = set(range(1, 31))
MIN_COVERAGE = len(REQUIRED_SEEDS)

# Candidate metric keys to tolerate common logging variants
METRIC_CANDIDATES = [
    "best/test",
    "best.test",
    "test",
    "metrics/test",
]

# Hyperparameters that define a "configuration" (exclude `seed`)
GROUP_COLS = ["lr", "hidden_dim", "dropout", "decay", "normalize", "num_layers"]

api = wandb.Api()
sweep = api.sweep(f"{ENTITY}/{PROJECT}/{SWEEP_ID}")

print("Sweep name:", sweep.name)
runs = sweep.runs

def get_metric(r):
    """Return (value, key_used) for the metric from summary or history."""
    smry = dict(r.summary or {})
    for k in METRIC_CANDIDATES:
        if k in smry:
            return smry[k], k
    # fallback: scan history (last non-NaN)
    try:
        hist = r.history(keys=METRIC_CANDIDATES, pandas=True)
        if hist is not None and len(hist):
            for k in METRIC_CANDIDATES:
                if k in hist.columns:
                    col = hist[k].dropna()
                    if len(col):
                        return col.iloc[-1], k
    except Exception:
        pass
    return None, None

rows = []
for r in runs:
    if r.state != "finished":
        continue
    metric, mkey = get_metric(r)
    cfg = r.config or {}
    seed_raw = cfg.get("seed")
    # normalize seed -> int when possible
    try:
        seed = int(seed_raw)
    except Exception:
        seed = seed_raw

    rows.append({
        "run_id": r.id,
        "metric": metric,
        "metric_key_used": mkey,
        "seed": seed,
        **{k: cfg.get(k) for k in GROUP_COLS},
    })

df = pd.DataFrame(rows)
if df.empty:
    raise RuntimeError("No finished runs found. Check ENTITY/PROJECT/SWEEP_ID.")

# Keep only runs with a valid metric and seed present
df = df.dropna(subset=["metric", "seed"])

# If seeds are strings, try to coerce to int where possible
def coerce_int(x):
    try: return int(x)
    except: return x
df["seed"] = df["seed"].map(coerce_int)

# --- Optional: if you suspect duplicate runs per (config, seed), keep the best ---
df = df.sort_values("metric").drop_duplicates(subset=GROUP_COLS + ["seed"], keep="last")

# Coverage + aggregation per config
def t_ci_halfwidth(n, std):
    if (n is None) or (n < 2) or (std is None) or np.isnan(std):
        return np.nan
    return t.ppf(0.975, df=n-1) * std / np.sqrt(n)

agg = (df
    .groupby(GROUP_COLS)
    .agg(mean=("metric","mean"),
         std=("metric","std"),
         n=("metric","count"),
         seeds=("seed", lambda s: sorted(set(s))))
    .reset_index())

# Compute coverage & missing seeds
agg["coverage"] = agg["seeds"].map(lambda s: len(set(s) & REQUIRED_SEEDS))
agg["missing_seeds"] = agg["seeds"].map(lambda s: sorted(REQUIRED_SEEDS - set(s)))

def select_with_coverage(min_cov):
    eligible = agg[agg["coverage"] >= min_cov].copy()
    return eligible.sort_values(by=["mean","std","n"], ascending=[False, True, False]).reset_index(drop=True)

ranked = select_with_coverage(MIN_COVERAGE)
current_min_cov = MIN_COVERAGE
while ranked.empty and current_min_cov > 3:
    current_min_cov -= 1
    print("Falling back to lower coverage threshold:", current_min_cov)
    ranked = select_with_coverage(current_min_cov)


if ranked.empty:
    raise RuntimeError("No configurations meet the coverage requirement. Consider lowering MIN_COVERAGE or checking filters.")

# Show top configs
TOP_K = 10
print("\nTOP CONFIGS (coverage-aware):")
print(ranked.head(TOP_K)[GROUP_COLS + ["mean","std","n","coverage","missing_seeds"]])

# Pick the best row
best = ranked.iloc[0]
print("\nBEST CONFIG (seed-robust selection):")
print(best[GROUP_COLS + ["mean","std","n","coverage","missing_seeds",]])

# List member runs (by seed) for the best config
mask = (df[GROUP_COLS] == best[GROUP_COLS]).all(axis=1)
members = df[mask].sort_values("seed")[["run_id","seed","metric","metric_key_used"]]
print("\nMember runs for the selected config:")
print(members.to_string(index=False))

# (Optional) Auto-generate a 30-seed sweep YAML for the winner:
WINNER = {k: best[k] for k in GROUP_COLS}

print(WINNER)
