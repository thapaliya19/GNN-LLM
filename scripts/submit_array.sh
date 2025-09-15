#!/bin/bash
set -euo pipefail

MANIFEST="scripts/manifest.tsv"
if [[ ! -s "$MANIFEST" ]]; then
  echo "Manifest not found or empty: $MANIFEST" >&2
  exit 1
fi

# count data rows (minus header)
N=$(awk 'END{print NR-1}' "$MANIFEST")
if (( N <= 0 )); then
  echo "No data rows in manifest (has header only?): $MANIFEST" >&2
  exit 1
fi

mkdir -p grid_logs
sed "s/{{N}}/${N}/g" scripts/array_runner.sbatch | sbatch
echo "Submitted array of ${N} tasks."
