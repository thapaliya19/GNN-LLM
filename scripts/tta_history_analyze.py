#!/usr/bin/env python3
"""
Analyze TTA history CSVs named like: tta_history_seed_<seed_value>.csv

Each CSV must contain columns:
  iteration, node_idx, true_label, pred_label, confidence, old_desc, new_desc

This script aggregates across ALL seed files found in a folder and reports:
  1) For each (iteration, node_idx): how many times across seeds it was
     correctly vs. wrongly classified (plus accuracy and seed list).
  2) For each (seed, iteration): counts of correct vs. wrong nodes.
  3) For each iteration (aggregated across seeds): counts of correct vs. wrong nodes.
  4) For each seed, the **average number** of correct vs. wrong nodes across all iterations.
  5) Overall average number of correct vs. wrong nodes across all seeds and iterations.
  6) For iteration 0, display the old and new descriptions of the most wrong nodes with high confidence.

Mapping Cora classes to their respective label values:
    'Case Based': 0,
    'Genetic Algorithms': 1,
    'Neural Networks': 2,
    'Probabilistic Methods': 3,
    'Reinforcement Learning': 4,
    'Rule Learning': 5,
    'Theory': 6
"""

import argparse
import re
from pathlib import Path
import pandas as pd

# Cora class label mapping
cora_class_mapping = {
    'Case Based': 0,
    'Genetic Algorithms': 1,
    'Neural Networks': 2,
    'Probabilistic Methods': 3,
    'Reinforcement Learning': 4,
    'Rule Learning': 5,
    'Theory': 6
}

# Reverse mapping from numeric to class names for display
reverse_cora_class_mapping = {v: k for k, v in cora_class_mapping.items()}

DEFAULT_PATTERN = "tta_history_seed_*.csv"

SEED_REGEX = re.compile(r"tta_history_seed_(-?\d+)\.csv$", re.IGNORECASE)

def extract_seed(path: Path):
    m = SEED_REGEX.search(path.name)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return m.group(1)  # fallback to raw string
    return None

def read_history_file(path: Path) -> pd.DataFrame:
    # Robust delimiter detection (comma or tab) and safe dtype handling
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        # Fallback to comma
        df = pd.read_csv(path)

    required = {"iteration", "node_idx", "true_label", "pred_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    df["seed"] = extract_seed(path)

    # Normalize types to avoid mismatches in equality comparison
    df["iteration"] = pd.to_numeric(df["iteration"], errors="coerce").astype("Int64")
    df["node_idx"] = pd.to_numeric(df["node_idx"], errors="coerce").astype("Int64")
    
    # Correctly map numeric labels to their class names using reverse mapping
    df["true_label_numeric"] = pd.to_numeric(df["true_label"], errors="coerce")
    df["pred_label_numeric"] = pd.to_numeric(df["pred_label"], errors="coerce")
    
    # Ensure to retrieve class names based on numeric labels
    df["true_label_name"] = df["true_label_numeric"].map(reverse_cora_class_mapping)
    df["pred_label_name"] = df["pred_label_numeric"].map(reverse_cora_class_mapping)

    # Compare labels for correctness
    df["is_correct"] = df["true_label_numeric"] == df["pred_label_numeric"]

    return df

def format_markdown(df: pd.DataFrame) -> str:
    # Convert dataframe to markdown table
    return df.to_markdown(index=False)

def analyze(folder: Path, pattern: str):
    files = sorted(folder.glob(pattern))
    if not files:
        raise SystemExit(f"No files matched pattern '{pattern}' in {folder}")

    dfs = []
    for f in files:
        df = read_history_file(f)
        df["source_file"] = f.name
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)

    # Filter data for iteration 0
    iteration_0_df = all_df[all_df["iteration"] == 0]

    # Identify most wrong nodes with high confidence
    wrong_nodes_df = iteration_0_df[iteration_0_df["is_correct"] == False]
    wrong_nodes_df_sorted = wrong_nodes_df.sort_values(by="confidence", ascending=False)

    # Select top N most wrong nodes with high confidence
    most_wrong_nodes = wrong_nodes_df_sorted.head(10)  # Top 10 most wrong nodes for example

    # Print the most wrong nodes with their old and new descriptions
    print("\n### Most Wrong Nodes with High Confidence (Iteration 0)\n")
    for index, row in most_wrong_nodes.iterrows():
        # Print the information with both numeric and class name
        print(f"**Node Index:** {row['node_idx']}")
        print(f"**True Label (Numeric):** {row['true_label_numeric']} ({row['true_label_name']})")
        print(f"**Predicted Label (Numeric):** {row['pred_label_numeric']} ({row['pred_label_name']})")
        print(f"**Confidence:** {row['confidence']}")
        print(f"**Old Description:** {row['old_desc']}")
        print(f"**New Description:** {row['new_desc']}\n")

def main():
    ap = argparse.ArgumentParser(description="Analyze TTA history across seeds")
    ap.add_argument("folder", type=str, help="Folder containing tta_history_seed_*.csv files")
    ap.add_argument("--pattern", type=str, default=DEFAULT_PATTERN,
                    help=f"Glob pattern for files (default: {DEFAULT_PATTERN})")
    args = ap.parse_args()

    folder = Path(args.folder).expanduser().resolve()
    analyze(folder, args.pattern)

if __name__ == "__main__":
    main()
