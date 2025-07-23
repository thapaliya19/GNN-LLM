#!/usr/bin/env python3
"""
process_csvs.py

Load all CSV files from an input directory, add a `node_idx` column (row index),
reorder the columns as [node_idx, answer], and overwrite each file in place,
using multiprocessing for speed.
"""
import os
import argparse
import pandas as pd
from multiprocessing import Pool

def process_file(file_path):
    # Read CSV
    df = pd.read_csv(file_path)
    # Add node_idx column
    df['node_idx'] = df.index
    # Reorder columns
    df = df[['node_idx', 'answer']]
    # Overwrite same file
    df.to_csv(file_path, index=False)
    return os.path.basename(file_path)

def main():
    parser = argparse.ArgumentParser(
        description='Add node_idx and reorder columns in CSV files (in place)'
    )
    parser.add_argument(
        '--input_dir', required=True,
        help='Directory containing CSV files to update'
    )
    parser.add_argument(
        '--workers', type=int, default=os.cpu_count(),
        help='Number of parallel worker processes'
    )
    args = parser.parse_args()

    files = [
        os.path.join(args.input_dir, f)
        for f in os.listdir(args.input_dir)
        if f.lower().endswith('.csv')
    ]

    with Pool(processes=args.workers) as pool:
        for fname in pool.imap_unordered(process_file, files):
            print(f"Processed: {fname}")

if __name__ == '__main__':
    main()
