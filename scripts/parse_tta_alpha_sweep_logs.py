#!/usr/bin/env python3
import re
import argparse
import glob
from pathlib import Path

import pandas as pd
import numpy as np

# -------- Regex patterns --------
RE_BASELINE = re.compile(
    r'After\s+Train\s+Result/Before\s+TTA\s*-\s*Train:\s*([0-9.]+),\s*Val:\s*([0-9.]+),\s*Test:\s*([0-9.]+)',
    re.IGNORECASE
)
RE_ITER_HDR = re.compile(r'Test-time\s+adaptation\s+iteration\s+(\d+)\s*/\s*(\d+)', re.IGNORECASE)
RE_HI_ENT_COUNT_1 = re.compile(r'(\d+)\s+high-entropy\s+nodes', re.IGNORECASE)
RE_HI_ENT_COUNT_2 = re.compile(r'Found\s+(\d+)\s+high-entropy\s+nodes', re.IGNORECASE)
RE_TTA_RESULT = re.compile(
    r'TTA\s+Result\s*-\s*Train:\s*([0-9.]+),\s*Val:\s*([0-9.]+),\s*Test:\s*([0-9.]+)',
    re.IGNORECASE
)
RE_ALPHA_DONE = re.compile(
    r'->\s*alpha\s*=\s*([0-9.]+)\s*yields\s*val\s*accuracy\s*([0-9.]+)',
    re.IGNORECASE
)

RE_EMBED_ALPHA_DONE = re.compile(
    r'->\s*embed_alpha\s*=\s*([0-9.]+)\s*yields\s*val\s*accuracy\s*([0-9.]+)',
    re.IGNORECASE
)

RE_BUFFER_ALPHA_DONE = re.compile(
    r'->\s*buffer_alpha\s*=\s*([0-9.]+)\s*yields\s*val\s*accuracy\s*([0-9.]+)',
    re.IGNORECASE
)

RE_ENTROPY_THR_DONE = re.compile(
    r'->\s*entropy_thr\s*=\s*([0-9.]+)\s*yields\s*val\s*accuracy\s*([0-9.]+)',
    re.IGNORECASE
)

def parse_file(path: Path, param: str):
    """
    Returns:
      - detailed rows (baseline + each TTA iteration with entropy count)
      - alpha_finals rows with final reported val (from '-> alpha=... yields ...')
    """
    rows_buffer = []
    all_rows = []
    alpha_finals = []

    current_iter = None
    current_iter_total = None
    last_entropy_count = None

    with path.open('r', encoding='utf-8', errors='ignore') as f:
        for raw in f:
            line = raw.strip()

            m = RE_BASELINE.search(line)
            if m:
                train, val, test = map(float, m.groups())
                rows_buffer.append({
                    'file': path.name,
                    'alpha': None,
                    'iteration': 0,  # baseline
                    'iterations_total': None,
                    'train': train,
                    'val': val,
                    'test': test,
                    'high_entropy_count': None
                })
                continue

            m = RE_ITER_HDR.search(line)
            if m:
                current_iter = int(m.group(1))
                current_iter_total = int(m.group(2))
                last_entropy_count = None
                continue

            m = RE_HI_ENT_COUNT_1.search(line) or RE_HI_ENT_COUNT_2.search(line)
            if m:
                last_entropy_count = int(m.group(1))
                continue

            m = RE_TTA_RESULT.search(line)
            if m:
                train, val, test = map(float, m.groups())
                rows_buffer.append({
                    'file': path.name,
                    'alpha': None,
                    'iteration': current_iter,
                    'iterations_total': current_iter_total,
                    'train': train,
                    'val': val,
                    'test': test,
                    'high_entropy_count': last_entropy_count
                })
                continue
            if param == "embed_alpha":
                m = RE_EMBED_ALPHA_DONE.search(line)
            elif param == "entropy_thr":
                m = RE_ENTROPY_THR_DONE.search(line)
            elif param == "alpha":
                m = RE_ALPHA_DONE.search(line)
            elif param == "buffer_alpha":
                m = RE_BUFFER_ALPHA_DONE.search(line)
            if m:
                alpha_val = float(m.group(1))
                final_val = float(m.group(2))
                for r in rows_buffer:
                    r['alpha'] = alpha_val
                all_rows.extend(rows_buffer)
                rows_buffer = []
                alpha_finals.append({
                    'file': path.name,
                    'alpha': alpha_val,
                    'final_reported_val': final_val
                })
                continue

    # flush any remaining rows (alpha=None if tail incomplete)
    all_rows.extend(rows_buffer)
    return all_rows, alpha_finals

def summarize_alpha_per_seed(detailed_df, finals_df):
    """One row per file×alpha with best Val/Test across iterations and mean entropy."""
    iter_df = detailed_df[detailed_df['iteration'] > 0].copy()
    if iter_df.empty:
        return pd.DataFrame(columns=[
            'file','alpha','best_val_across_iterations','best_test_across_iterations','mean_high_entropy_count','final_reported_val'
        ])
    summary = iter_df.groupby(['file', 'alpha']).agg(
        best_val_across_iterations=('val', 'max'),
        best_test_across_iterations=('test', 'max'),
        mean_high_entropy_count=('high_entropy_count', 'mean')
    ).reset_index()
    summary = summary.merge(finals_df, on=['file', 'alpha'], how='left')
    return summary

def aggregate_across_seeds(summary_df):
    """Aggregate file×alpha summary across seeds (files)."""
    if summary_df.empty:
        return pd.DataFrame(columns=['alpha','seeds',
                                     'final_val_mean±std','best_val_mean±std','best_test_mean±std','entropy_mean±std'])
    grouped = summary_df.groupby('alpha').agg(
        seeds=('file', 'nunique'),
        mean_final_val=('final_reported_val', 'mean'),
        std_final_val=('final_reported_val', 'std'),
        mean_best_val=('best_val_across_iterations', 'mean'),
        std_best_val=('best_val_across_iterations', 'std'),
        mean_best_test=('best_test_across_iterations', 'mean'),
        std_best_test=('best_test_across_iterations', 'std'),
        mean_entropy=('mean_high_entropy_count', 'mean'),
        std_entropy=('mean_high_entropy_count', 'std'),
    ).reset_index()

    # format mean ± std columns
    def fmt(m, s):
        if pd.isna(m): return ''
        if pd.isna(s): s = 0.0
        return f"{m:.4f} ± {s:.4f}"

    grouped['final_val_mean±std'] = grouped.apply(lambda r: fmt(r['mean_final_val'], r['std_final_val']), axis=1)
    grouped['best_val_mean±std']  = grouped.apply(lambda r: fmt(r['mean_best_val'], r['std_best_val']), axis=1)
    grouped['best_test_mean±std'] = grouped.apply(lambda r: fmt(r['mean_best_test'], r['std_best_test']), axis=1)
    grouped['entropy_mean±std']   = grouped.apply(lambda r: fmt(r['mean_entropy'], r['std_entropy']), axis=1)

    cols = ['alpha','seeds','final_val_mean±std','best_val_mean±std','best_test_mean±std','entropy_mean±std']
    return grouped[cols].sort_values('alpha')

def entropy_by_iteration_across_seeds(detailed_df):
    """
    For each alpha and iteration (>0), compute mean ± std of high_entropy_count across seeds.
    Returns a wide table with columns: alpha, iter1, iter2, ...
    """
    iter_df = detailed_df[detailed_df['iteration'] > 0].copy()
    if iter_df.empty:
        return pd.DataFrame(columns=['alpha'])

    # aggregate per alpha×iteration
    agg = iter_df.groupby(['alpha','iteration']).agg(
        mean_entropy=('high_entropy_count','mean'),
        std_entropy=('high_entropy_count','std'),
        n=('file','nunique')
    ).reset_index()

    # format "mean ± std (n=..)" strings
    def fmt(m, s, n):
        if pd.isna(m): return ''
        if pd.isna(s): s = 0.0
        return f"{m:.2f} ± {s:.2f} (n={int(n)})"

    agg['mean±std'] = agg.apply(lambda r: fmt(r['mean_entropy'], r['std_entropy'], r['n']), axis=1)

    # pivot wide with iteration-specific columns
    wide = agg.pivot(index='alpha', columns='iteration', values='mean±std').reset_index()
    # rename iteration columns to Iter# for clarity
    wide.columns = ['alpha'] + [f'Iter{int(c)}' for c in wide.columns[1:]]
    # sort columns by iteration order
    iter_cols = sorted([c for c in wide.columns if c.startswith('Iter')], key=lambda x: int(x.replace('Iter','')))
    return wide[['alpha'] + iter_cols].sort_values('alpha')

def main():
    ap = argparse.ArgumentParser(description="Parse TTA logs and show CLI tables with seed-wise and across-seed stats.")
    ap.add_argument('--paths', nargs='+', required=True, help='Glob patterns for log files, e.g., logs/*.txt')
    ap.add_argument('--sort-by', default='best_val', choices=['best_val','final_val','best_test'],
                    help='Sort final aggregated table by this metric (descending).')
    ap.add_argument('--param', type=str, default=None,)
    args = ap.parse_args()

    files = []
    for pat in args.paths:
        files.extend(glob.glob(pat))
    files = sorted(set(files))
    if not files:
        print("No files found.")
        return

    all_rows, all_alpha_finals = [], []
    for fp in files:
        rows, finals = parse_file(Path(fp), args.param)
        all_rows.extend(rows)
        all_alpha_finals.extend(finals)

    detailed_df = pd.DataFrame(all_rows)
    finals_df = pd.DataFrame(all_alpha_finals)

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 160)

    print("\n=== Detailed Per-Iteration Data (per file) ===")
    if detailed_df.empty:
        print("No data found.")
        return
    # order for readability
    detailed_df = detailed_df.sort_values(['file','alpha','iteration'])
    print(detailed_df[['file','alpha','iteration','train','val','test','high_entropy_count']].to_string(index=False))

    print("\n=== Alpha Summary per Seed (file×alpha) ===")
    per_seed = summarize_alpha_per_seed(detailed_df, finals_df)
    if per_seed.empty:
        print("No alpha summary available.")
    else:
        per_seed = per_seed.sort_values(['file','alpha'])
        print(per_seed[['file','alpha','best_val_across_iterations',
                        'best_test_across_iterations','mean_high_entropy_count',
                        'final_reported_val']].to_string(index=False))

    print("\n=== Alpha Mean ± Std Across Seeds ===")
    if not per_seed.empty:
        agg = aggregate_across_seeds(per_seed)

        # optional sorting by chosen metric
        sort_map = {
            'best_val': 'best_val_mean±std',
            'final_val': 'final_val_mean±std',
            'best_test': 'best_test_mean±std'
        }
        # For sorting numerically, extract the mean part
        metric_col = sort_map.get(args.sort_by, 'best_val_mean±std')
        # Create a hidden numeric column for sorting
        def pull_mean(cell):
            try:
                return float(cell.split('±')[0])
            except Exception:
                return -np.inf
        agg = agg.assign(_sort=agg[metric_col].map(pull_mean)).sort_values('_sort', ascending=False).drop(columns=['_sort'])
        print(agg.to_string(index=False))
    else:
        print("No aggregated alpha results available.")

    print("\n=== High-Entropy Nodes by Iteration Across Seeds (mean ± std per alpha) ===")
    ent_iter = entropy_by_iteration_across_seeds(detailed_df)
    if ent_iter.empty:
        print("No iteration-wise entropy stats available.")
    else:
        print(ent_iter.to_string(index=False))

if __name__ == "__main__":
    main()
