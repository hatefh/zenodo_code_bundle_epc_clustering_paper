#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$(pwd)"

python -m code.steps.00_features_from_vs_csv
python -m code.steps.01_cluster_with_custom_impl
python -m code.steps.02_validation_metrics
python -m code.steps.03_pairwise_agreement

echo
echo "---- outputs ----"
ls -lh outputs/
