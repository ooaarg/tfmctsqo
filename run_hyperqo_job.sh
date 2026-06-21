#!/usr/bin/env bash
# Reproduce HyperQO on the JOB workload (113 IMDb queries).
#
# Drives the generic runner in the HyperQO submodule, pointing it at the JOB
# queries owned by this repo (source/resource). Outputs land at the repo root
# where analysis.ipynb reads them:
#   per_query-job113.csv  stats-job113.json  log_job113.txt
# Run from anywhere; requires a live PostgreSQL with the IMDb DB loaded.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The runner imports its sibling modules and resolves model/ relative to cwd,
# so it must run from inside the submodule.
cd "$ROOT/source/HyperQO"

python hyperqo_run_dir.py \
    --sql-dir "$ROOT/source/resource/imdb_original_queries" \
    --mcts-model model/log_c3_h64_s4_t3.pth \
    --log-file "$ROOT/log_job113.txt" \
    --stats-json "$ROOT/stats-job113.json" \
    --stats-csv "$ROOT/per_query-job113.csv"
