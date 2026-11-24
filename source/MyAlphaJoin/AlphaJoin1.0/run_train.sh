#!/usr/bin/env bash
# Run AlphaJoin training and write logs to learning_logs.out.
# Run this in one terminal; use training_curves.ipynb in another to watch loss/accuracy.
set -e
cd "$(dirname "$0")"
# Activate the Python virtualenv at the repository root (../../../venv).
# Adjust the path if your venv lives elsewhere.
source ../../../venv/bin/activate
# Optional: --train-test-split by_query --test-query-ratio 0.2  (test on unseen query IDs)
python3 -u 6.train_network.py \
  --data-file ./t6.sql \
  --train-steps 1600000 \
  --lr 0.01 \
  --save-every 200000 \
  --test-every 5000 \
  > learning_logs.out 2>&1
