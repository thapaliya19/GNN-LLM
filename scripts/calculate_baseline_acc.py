#!/usr/bin/env python3
import re
import argparse
from pathlib import Path
from statistics import mean, stdev, pstdev

def parse_out_files(folder: Path):
    """Parse .out files in the given folder and extract test values."""
    pattern = re.compile(
        r"After\s+Train\s+Result/Before\s+TTA\s*-\s*Train:\s*([0-9.+-eE]+)\s*,\s*Val:\s*([0-9.+-eE]+)\s*,\s*Test:\s*([0-9.+-eE]+)"
    )

    test_values = []
    files_scanned = 0
    files_matched = 0

    for p in folder.glob("*.out"):
        files_scanned += 1
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"[WARN] Could not read {p}: {e}")
            continue

        match = pattern.search(text)
        if match:
            files_matched += 1
            test_val = float(match.group(3))
            test_values.append(test_val)

    return test_values, files_scanned, files_matched


def main():
    parser = argparse.ArgumentParser(
        description="Parse .out files in a folder and compute mean/std of Test values."
    )
    parser.add_argument(
        "folder",
        type=str,
        help="Path to the folder containing .out files"
    )
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"Error: '{folder}' is not a valid directory.")
        return

    test_values, files_scanned, files_matched = parse_out_files(folder)

    if not test_values:
        print(f"No matching lines found in {files_scanned} .out files under: {folder.resolve()}")
        return

    m = mean(test_values)
    s_sample = stdev(test_values) if len(test_values) > 1 else 0.0
    s_pop = pstdev(test_values) if len(test_values) > 1 else 0.0

    print(f"\nScanned files: {files_scanned}")
    print(f"Files matched: {files_matched}")
    print(f"Count (n): {len(test_values)}")
    print(f"Test values: {test_values}")
    print(f"Mean(Test): {m:.6f}")
    print(f"Std(Test) [sample, ddof=1]: {s_sample:.6f}")
    print(f"Std(Test) [population, ddof=0]: {s_pop:.6f}")
    print(f"Mean ± Std: {m:.2f} ± {s_pop:.2f}")



if __name__ == "__main__":
    main()
