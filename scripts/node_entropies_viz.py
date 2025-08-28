#!/usr/bin/env python3
"""
Bar plots of predictive entropy per node (sorted descending), colored by correctness.

Usage:
  python entropy_barplot_by_iter.py \
      --csv path/to/file.csv \
      --out entropy_bars.png \
      --dataset cora \
      --topk 100          # optional: only top-K highest-entropy nodes per iteration
  # Optional:
  #   --iteration 0       # plot only a single iteration
  #   --sep auto          # auto-detect delimiter
  #   --label_every 5     # write every N-th node_idx label on the x-axis (default: auto)
"""

import argparse
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read_csv_safely(path: str, sep_opt: str | None) -> pd.DataFrame:
    if sep_opt is None:
        return pd.read_csv(path)
    if sep_opt == "auto":
        try:
            return pd.read_csv(path, sep=None, engine="python")
        except Exception:
            return pd.read_csv(path)
    return pd.read_csv(path, sep=sep_opt)

def ensure_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to CSV with columns incl. iteration,node_idx,true_label,pred_label,entropy")
    ap.add_argument("--out", default="entropy_bars.png", help="Output image path")
    ap.add_argument("--dataset", default="(dataset)", help="Printed in the figure title")
    ap.add_argument("--sep", default=None, help='Delimiter (",", "\\t", or "auto")')
    ap.add_argument("--iteration", type=int, default=None, help="If set, plot only this iteration")
    ap.add_argument("--topk", type=int, default=None, help="Show only top-K highest-entropy nodes per iteration")
    ap.add_argument("--label_every", type=int, default=None, help="Write every N-th node_idx as an xtick (auto if omitted)")
    args = ap.parse_args()

    df = read_csv_safely(args.csv, args.sep)

    required = ["iteration", "node_idx", "true_label", "pred_label", "entropy"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    ensure_numeric(df, ["iteration", "node_idx", "true_label", "pred_label", "entropy"])
    df = df.dropna(subset=["iteration", "node_idx", "true_label", "pred_label", "entropy"]).copy()
    df["is_wrong"] = (df["true_label"] != df["pred_label"]).astype(int)

    if args.iteration is not None:
        df = df[df["iteration"] == args.iteration]
        if df.empty:
            raise ValueError(f"No rows for iteration={args.iteration}")

    iters = sorted(df["iteration"].unique().tolist())
    n = len(iters)
    ncols = min(1, n)
    nrows = n

    # Shared y-limit for comparability
    ymax = float(df["entropy"].max())
    ypad = 0.05 * (ymax if np.isfinite(ymax) else 1.0)
    ymax = ymax + ypad

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.6 * nrows), squeeze=False, sharey=True)

    # Legend proxies
    import matplotlib.lines as mlines
    correct_proxy = mlines.Line2D([], [], color="C0", marker="s", linestyle="None", label="Correct")
    wrong_proxy   = mlines.Line2D([], [], color="C1", marker="s", linestyle="None", label="Wrong")

    for i, it in enumerate(iters):
        ax = axes[i // ncols][i % ncols]
        g = df[df["iteration"] == it].copy().sort_values("entropy", ascending=False)
        if args.topk:
            g = g.head(args.topk)
        

    
        ax.set_ylim(0, ymax)
        ax.set_title(f"iter={it}  n={len(g)}")
        ax.set_xlabel("nodes (sorted by entropy)")
        if i % ncols == 0:
            ax.set_ylabel("predictive entropy")

        # Nicely sparse x tick labels (node_idx)
        if args.label_every is None:
            step = max(1, len(g) // 30)  # ~30 labels max
        else:
            step = max(1, args.label_every)
        # inside the loop where you build each subplot
        gap = 0.25                        # 0 = no extra gap; 0.25 = 25% extra spacing
        x = np.arange(len(g)) * (1 + gap) # spread bars apart
        bar_w = 0.7                       # narrower bars
        colors = np.where(g["is_wrong"].values == 1, "C1", "C0")

        ax.bar(x, g["entropy"].values, width=bar_w,
            color=colors, edgecolor="black", linewidth=0.4)
        ax.xaxis.set_visible(False)
        ax.spines["bottom"].set_visible(False)

        # make xticks line up with the spaced bars
        # ax.set_xticks(x[::step], [str(int(v)) for v in g["node_idx"].values[::step]], rotation=90)
        # idxs = np.arange(0, len(g), step)
        # ax.set_xticks(idxs, [str(int(v)) for v in g["node_idx"].values[idxs]], rotation=90)

    # Hide unused panels
    total = nrows * ncols
    for j in range(n, total):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle(f"Predictive Entropy per Node (sorted desc) — {args.dataset}", y=0.995)
    fig.legend([correct_proxy, wrong_proxy], ["Correct", "Wrong"],
               loc="upper right", bbox_to_anchor=(0.98, 0.98), frameon=False, ncol=1)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.out, dpi=160)
    print(f"Saved: {args.out}")

if __name__ == "__main__":
    main()
