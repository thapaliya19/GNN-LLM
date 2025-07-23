#!/usr/bin/env python3
import os
import glob
import json
import re
import argparse
import itertools

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def load_results(directory):
    """Load all TTA result JSONs into a DataFrame."""
    pattern = os.path.join(directory, "tta_results_seed_*_iter_*.json")
    recs = []
    for fp in glob.glob(pattern):
        try:
            with open(fp) as f:
                recs.append(json.load(f))
        except Exception as e:
            print(f"⚠️ Skipping {fp}: {e}")
    return pd.DataFrame(recs)

def find_missing(df, seeds, iters):
    exp = set(itertools.product(seeds, iters))
    found = set(zip(df['seed'], df['iteration']))
    miss = exp - found
    out = {}
    for s, grp in itertools.groupby(sorted(miss), key=lambda x: x[0]):
        out[s] = [i for (_,i) in grp]
    return out

def print_missing(miss):
    if not miss:
        print("✅ All expected JSON files found")
    else:
        print("⚠️ Missing JSONs:")
        for s, its in sorted(miss.items()):
            print(f"  seed {s}: missing iterations {', '.join(map(str,its))}")

def load_baseline(logs_dir, seeds):
    """
    Returns dict: seed -> {train, val, test}.
    Parses 'After Train Result/Before TTA - Train: X, Val: Y, Test: Z'.
    """
    pat = re.compile(
      r"After Train Result/Before TTA\s*-\s*Train:\s*([\d.]+),\s*Val:\s*([\d.]+),\s*Test:\s*([\d.]+)"
    )
    out = {}
    for s in seeds:
        p = os.path.join(logs_dir, f"tta_seed_{s}.out")
        if not os.path.isfile(p): continue
        with open(p) as f:
            for line in f:
                m = pat.search(line)
                if m:
                    out[s] = {
                      "train": float(m.group(1)),
                      "val":   float(m.group(2)),
                      "test":  float(m.group(3))
                    }
                    break
    return out

def load_entropy(logs_dir, seeds, iters):
    """
    Returns a list of (seed, iteration, nodes).
    Parses lines:
      'Test-time adaptation iteration i/N'  and 
      'Found X high-entropy nodes.'
    """
    iter_pat  = re.compile(r"Test-time adaptation iteration\s*(\d+)/\d+")
    found_pat = re.compile(r"Found\s*(\d+)\s*high-entropy nodes\.")
    records = []
    for s in seeds:
        p = os.path.join(logs_dir, f"tta_seed_{s}.out")
        if not os.path.isfile(p): continue
        cur_iter = None
        with open(p) as f:
            for line in f:
                im = iter_pat.search(line)
                if im:
                    cur_iter = int(im.group(1)) - 1  # zero‑based iteration
                fm = found_pat.search(line)
                if fm and cur_iter is not None:
                    records.append({
                        "seed":      s,
                        "iteration": cur_iter,
                        "nodes":     int(fm.group(1))
                    })
                    cur_iter = None
    return pd.DataFrame(records)

def print_table(df, name="Data"):
    print(f"\n📋 {name}:")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

def main():
    p = argparse.ArgumentParser(
        description="Combine TTA + baseline + entropy; tables & iteration plots."
    )
    p.add_argument("--base_dir", default="./outputs",
                   help="Base dir with <dataset>/<llm>/ and <dataset>_logs/")
    p.add_argument("--dataset", help="Name of dataset folder under base_dir")
    p.add_argument("--llm",     help="LLM subfolder under <base_dir>/<dataset>/")
    p.add_argument("--plot", action="store_true",
                   help="Save iteration plots (accuracy & entropy vs iteration)")
    args = p.parse_args()

    results_dir = os.path.join(args.base_dir, args.dataset, args.llm)
    logs_dir    = os.path.join(f"{args.dataset}_logs")

    # Load TTA JSONs
    df_tta = load_results(results_dir)
    if df_tta.empty:
        print(f"No TTA JSONs in {results_dir}"); return

    # Check missing
    seeds = list(range(1,31))
    iters = list(range(0,5))
    print_missing(find_missing(df_tta, seeds, iters))

    # Load baseline
    bl = load_baseline(logs_dir, seeds)
    if bl:
        bdf = (pd.DataFrame.from_dict(bl, orient='index')
                .reset_index().rename(columns={
                  'index':'seed','train':'baseline_train',
                  'val':'baseline_val','test':'baseline_test'
                }))
        bdf.sort_values('seed', inplace=True)
        baseline_mean = bdf['baseline_test'].mean()
        baseline_std  = bdf['baseline_test'].std()
        print_table(bdf, "Pre‑TTA baseline by seed")
        print(f"\nOverall baseline test: mean={baseline_mean:.4f}, std={baseline_std:.4f}\n")
    else:
        baseline_mean = None
        print(f"\n⚠️  No baseline logs in {logs_dir}\n")

    # Combined TTA table
    df_tta_sorted = df_tta.sort_values(['seed','iteration']).reset_index(drop=True)
    print_table(df_tta_sorted, "Combined TTA results (seed, iter, train, val, test)")

    # Iteration summary (accuracy)
    iter_acc = df_tta.groupby('iteration')['test'].agg(['mean','std']).reset_index()
    iter_acc.columns = ['iteration','mean_test','std_test']
    print_table(iter_acc, "TTA Test by iteration (mean ± std)")

    # Baseline summary (just once, above) and now entropy
    df_ent = load_entropy(logs_dir, seeds, iters)
    if not df_ent.empty:
        ent_stats = df_ent.groupby('iteration')['nodes'].agg(['mean','std']).reset_index()
        ent_stats.columns = ['iteration','mean_nodes','std_nodes']
        print_table(ent_stats, "High‑entropy nodes by iteration (mean ± std)")
    else:
        ent_stats = pd.DataFrame(columns=['iteration','mean_nodes','std_nodes'])
        print(f"\n⚠️  No high‑entropy data parsed in {logs_dir}\n")

    # Plotting
    if args.plot:
        os.makedirs(results_dir, exist_ok=True)

        # Plot 1: accuracy vs iteration + baseline
        fig1, ax1 = plt.subplots()
        ax1.plot(iter_acc['iteration'], iter_acc['mean_test'],
                 marker='o', label='TTA Mean Test')
        if baseline_mean is not None:
            ax1.hlines(baseline_mean,
                       xmin=iter_acc['iteration'].min(),
                       xmax=iter_acc['iteration'].max(),
                       colors='gray', linestyles='--',
                       label='Baseline Mean Test')
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Test Accuracy')
        ax1.set_title('Mean TTA Test Accuracy by Iteration')
        ax1.grid(True)
        ax1.legend()
        f1 = os.path.join(results_dir, 'tta_accuracy_by_iteration.png')
        fig1.tight_layout()
        fig1.savefig(f1)
        print(f"📈 Saved plot: {f1}")
        plt.close(fig1)

        # Plot 2: accuracy & entropy vs iteration (dual axis)
        fig2, ax2 = plt.subplots()
        ax2.plot(iter_acc['iteration'], iter_acc['mean_test'],
                 marker='o', label='TTA Mean Test')
        if baseline_mean is not None:
            ax2.hlines(baseline_mean,
                       xmin=iter_acc['iteration'].min(),
                       xmax=iter_acc['iteration'].max(),
                       colors='red', linestyles='--',
                       label='Baseline Mean Test')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Test Accuracy')
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax2.grid(True)

        ax3 = ax2.twinx()
        ax3.plot(ent_stats['iteration'], ent_stats['mean_nodes'],
                 marker='s', color='tab:orange', label='Mean High‑Entropy Nodes')
        ax3.set_ylabel('Mean # High‑Entropy Nodes')

        # combine legends
        lines, labels = [], []
        for ax in (ax2, ax3):
            l, lab = ax.get_legend_handles_labels()
            lines += l; labels += lab
        ax2.legend(lines, labels, loc='best')

        ax2.set_title('TTA Test Accuracy & High‑Entropy Nodes by Iteration')
        f2 = os.path.join(results_dir, 'accuracy_vs_entropy_by_iteration.png')
        fig2.tight_layout()
        fig2.savefig(f2)
        print(f"📈 Saved plot: {f2}")
        plt.close(fig2)

if __name__ == "__main__":
    main()
