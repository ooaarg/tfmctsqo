#!/usr/bin/env bash
# Reproduce HyperQO on the JOB-Complex workload.
#
# Drives the generic runner in the HyperQO submodule, pointing it at the
# JOB-Complex queries owned by this repo (source/resource). Outputs land at the
# repo root where analysis.ipynb reads them:
#   per_query-job-complex.csv  stats-job-complex.json  log_job-complex.txt
# Run from anywhere; requires a live PostgreSQL with the IMDb DB loaded.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The runner imports its sibling modules and resolves model/ relative to cwd,
# so it must run from inside the submodule.
cd "$ROOT/source/HyperQO"

python hyperqo_run_dir.py \
    --sql-dir "$ROOT/source/resource/job_complex" \
    --mcts-model model/log_c3_h64_s4_t3.pth \
    --log-file "$ROOT/log_job-complex.txt" \
    --stats-json "$ROOT/stats-job-complex.json" \
    --stats-csv "$ROOT/per_query-job-complex.csv"
