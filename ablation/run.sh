#!/usr/bin/env bash
# Self-rexec to bash so `sh run.sh` doesn't silently die on `local` / pipefail.
[ -z "${BASH_VERSION-}" ] && exec bash "$0" "$@"
#
# Ablation runner for mcts_extreme.reward_map and uct_aggregation across the
# 12+ rel subset of JOB.  Raw JOB SQL files live in $JOB_DIR; this script
# prepends EXPLAIN (ANALYZE, ...) to each query at runtime via a temp file,
# so the ablation/ directory holds only configuration (GUC SQL files + this
# runner), not per-query SQL.
#
# Sweeps the reward map (cost->reward mapping) in {neg_log, neg_cost,
# norm_neg_log} crossed with uct_aggregation (UCT Q-value
# aggregation) in {best, average}.  Each combination is repeated N times, and
# run k uses mcts_extreme.random_seed = k so the whole experiment is bit-exact
# reproducible.  Luby's "independent samples" assumption is still satisfied:
# mcts_root_restart() derives each phase's seed via splitmix64(master, phase),
# so phases inside a run are decorrelated even with a fixed master seed.
#
# Every run is preceded by `sync + drop_caches + cluster restart`, matching
# bench/run.sh, so cold pagecache and cold shared_buffers are honored.
#
# Usage:
#   ablation/run.sh                 # default queries=12+ rel JOB, n=20
#   ablation/run.sh -n 10           # 10 runs per cell
#   ablation/run.sh -v              # verbose: also print per-run lines
#   ablation/run.sh -m "neg_log norm_neg_log"   # restrict reward maps
#   ablation/run.sh -a "best"                 # restrict aggregations
#   ablation/run.sh -q "29c 28a"              # restrict queries
#
# Sudoers setup for full OS-cache flushing (one time):
#   sudo tee /etc/sudoers.d/mcts_extreme_drop_caches <<EOF
#   $USER ALL=(root) NOPASSWD: /usr/bin/tee /proc/sys/vm/drop_caches
#   EOF
#   sudo chmod 0440 /etc/sudoers.d/mcts_extreme_drop_caches
# Without it the script falls back to cluster-restart-only.

set -euo pipefail

PGBIN=${PGBIN:-$(dirname "$(command -v pg_ctl 2>/dev/null)" 2>/dev/null)}
PGDATA=${PGDATA:?set PGDATA to your Postgres data dir (e.g. PGDATA=/srv/pg/data)}
PGLOG=${PGLOG:-/tmp/mcts_extreme_pg.log}
DB=${DB:-imdb}
ABL_DIR="$(cd "$(dirname "$0")" && pwd)"
JOB_DIR=${JOB_DIR:-"$(cd "$ABL_DIR/.." && pwd)/source/resource/imdb_original_queries"}

# Default queries: 12+ relations on JOB (17 rels: 29*, 14 rels: 28*/33*,
# 12 rels: 24a/24b/26*/27*/30*).  Use -q to restrict.
default_queries="29a 29b 29c 28a 28b 28c 33a 33b 33c 26a 26b 26c 27a 27b 27c 30a 30b 30c 24a 24b"

runs=20
verbose=0
modes="neg_log norm_neg_log"
aggregations="best average"
queries="$default_queries"
while getopts "n:m:a:q:v" opt; do
  case $opt in
    n) runs=$OPTARG ;;
    m) modes=$OPTARG ;;
    a) aggregations=$OPTARG ;;
    q) queries=$OPTARG ;;
    v) verbose=1 ;;
    *) exit 2 ;;
  esac
done

# Validate JOB_DIR and every requested query up front so the script doesn't
# burn 10 minutes of sweep before discovering a typo.
[ -d "$JOB_DIR" ] || { echo "JOB_DIR not a directory: $JOB_DIR" >&2; exit 2; }
for q in $queries; do
  [ -f "$JOB_DIR/$q.sql" ] || { echo "missing query: $JOB_DIR/$q.sql" >&2; exit 3; }
done

# Per-query temp file holding "EXPLAIN (ANALYZE, ...) <raw query>".  Cleaned
# up on exit so re-runs always regenerate from $JOB_DIR.
TMPDIR=$(mktemp -d -t mcts_ablation.XXXXXX)
trap 'rm -rf "$TMPDIR"' EXIT

build_analyze_sql() {
  local q=$1
  local raw="$JOB_DIR/$q.sql"
  local out="$TMPDIR/$q.analyze.sql"
  {
    echo "EXPLAIN (ANALYZE, BUFFERS, COSTS on, SUMMARY on, TIMING on)"
    cat "$raw"
  } > "$out"
  echo "$out"
}

# Sweep size for progress reporting.
num_queries=$(echo "$queries" | wc -w)
num_aggs=$(echo "$aggregations" | wc -w)
num_modes=$(echo "$modes" | wc -w)
total_cells=$((num_queries * num_aggs * num_modes * 2))   # *2: luby + single
total_runs=$((total_cells * runs))
cell_idx=0
sweep_start_ts=$SECONDS

flush_caches() {
  sync
  # `|| true` so `set -e` doesn't kill the script if sudo isn't granted;
  # stdout is silenced because tee echoes "3" into the log on success, and
  # stderr is silenced so the absence of the sudoers rule isn't spammed.
  echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || true
}

PG_CTL="${PGBIN:+$PGBIN/}pg_ctl"
PSQL="${PGBIN:+$PGBIN/}psql"

restart_cluster() {
  "$PG_CTL" -D "$PGDATA" -l "$PGLOG" restart -m fast -w > /dev/null
}

# Pull the numeric field at $2 from the first line of stdin matching $1.
extract_num() {
  awk -v pat="$1" -v field="$2" '
    $0 ~ pat {
      v = $field
      if (v ~ /^[+-]?[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?$/) print v
      exit
    }'
}

# Total cost of the top plan node from EXPLAIN output.
extract_plan_cost() {
  grep -m1 '(cost=' | sed -E 's/.*\(cost=[0-9.]+\.\.([0-9.]+).*/\1/'
}

# MCTS best cost reported by EXPLAIN (extension stats line).  Falls back
# to empty if the run failed over to the chained planner.
extract_mcts_best() {
  awk '/MCTS Best Cost:/ {
    v = $4
    if (v ~ /^[+-]?[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?$/) print v
    exit
  }'
}

# Actual number of MCTS iterations performed (vs. the configured budget,
# which is an upper bound — tree exhaustion / patience can cut it short).
extract_mcts_iters() {
  awk '/MCTS Iterations:/ {
    v = $3
    if (v ~ /^[0-9]+$/) print v
    exit
  }'
}

# Summarize floats from stdin as: best / mean / median / worst.
summarize() {
  awk '
    NF == 0 { next }
    $1 == "" { next }
    { v[++n]=$1; s+=$1 }
    END {
      if (n == 0) { printf "%10s / %10s / %10s / %10s", "no data", "no data", "no data", "no data"; exit }
      for (i=1;i<=n;i++)
        for (j=i+1;j<=n;j++)
          if (v[i] > v[j]) { t=v[i]; v[i]=v[j]; v[j]=t }
      mid = int((n+1)/2)
      printf "%10.2f / %10.2f / %10.2f / %10.2f", v[1], s/n, v[mid], v[n]
    }
  '
}

# $1 = reward_mode, $2 = uct_aggregation, $3 = config label, $4 = GUC sql file,
# $5 = query short name (e.g. 29c), $6 = path to generated analyze sql
run_mode() {
  local reward=$1 agg=$2 cfg=$3 sqlfile=$4 query=$5 analyze_sql=$6
  local costs=() mcts_bests=() plans=() xtimes=() iters=()
  local out cost mcts_best plan xtime iter

  cell_idx=$((cell_idx + 1))
  local elapsed=$((SECONDS - sweep_start_ts))
  echo "===== [cell $cell_idx/$total_cells  elapsed ${elapsed}s] query=$query  reward_mode=$reward  uct_aggregation=$agg  config=$cfg  n=$runs runs ====="

  for run in $(seq 1 $runs); do
    flush_caches
    restart_cluster
    sleep 0.3

    out=$("$PSQL" -h /tmp -d "$DB" -X \
          -v reward="$reward" \
          -v agg="$agg" \
          -v seed="$run" \
          -v analyze_sql="$analyze_sql" \
          -f "$sqlfile" 2>&1)

    plan=$(echo "$out"  | extract_num "^ Planning Time:"  3)
    xtime=$(echo "$out" | extract_num "^ Execution Time:" 3)
    cost=$(echo "$out"  | extract_plan_cost)
    mcts_best=$(echo "$out" | extract_mcts_best)
    iter=$(echo "$out" | extract_mcts_iters)

    plans+=("$plan")
    xtimes+=("$xtime")
    costs+=("$cost")
    mcts_bests+=("$mcts_best")
    iters+=("$iter")

    if [ $verbose -eq 1 ]; then
      printf "  run %2d  iters=%-5s mcts_best=%-10s plan_cost=%-10s plan_ms=%-8s exec_ms=%s\n" \
             "$run" "$iter" "$mcts_best" "$cost" "$plan" "$xtime"
    fi
  done

  echo "  summary  (best /       mean /     median /      worst)"
  printf '    MCTS Iters     :%s\n' "$(printf '%s\n' "${iters[@]}"      | summarize)"
  printf '    MCTS Best Cost :%s\n' "$(printf '%s\n' "${mcts_bests[@]}" | summarize)"
  printf '    Plan Cost      :%s\n' "$(printf '%s\n' "${costs[@]}"      | summarize)"
  printf '    Plan Time ms   :%s\n' "$(printf '%s\n' "${plans[@]}"      | summarize)"
  printf '    Exec Time ms   :%s\n' "$(printf '%s\n' "${xtimes[@]}"     | summarize)"
  echo
}

echo "Sweep plan: $num_queries queries x $num_aggs aggs x $num_modes modes x 2 configs x $runs runs"
echo "           = $total_cells cells / $total_runs total runs"
echo

# Two configurations are swept across the same reward modes so that the
# effect of reward shaping can be read separately from the effect of Luby
# restarts:
#
#   luby   : phases=8, start_budget=20, patience=0, depth=4 (mcts_luby.sql)
#   single : phases=1, start_budget=260, patience=0, depth=4 (mcts_single.sql)
# Both cap at 20 * sum(luby(1..8)) = 260 iters and share depth=4 (tree cap
# ~top_k^4 = 625 leaves).  The ONLY difference is whether those iterations
# are spent on eight restarted trees or one persistent tree, so any quality
# gap is attributable to Luby restarts rather than to depth or budget.
# patience=0 forces all 8 phases to run so Luby's restart schedule is
# fully exercised.  The `MCTS Iters` row reports cumulative iters across
# phases (post-iter-counter fix).
for q in $queries; do
  analyze_sql=$(build_analyze_sql "$q")
  for a in $aggregations; do
    for m in $modes; do
      run_mode "$m" "$a" "luby"   "$ABL_DIR/mcts_luby.sql"   "$q" "$analyze_sql"
    done
    for m in $modes; do
      run_mode "$m" "$a" "single" "$ABL_DIR/mcts_single.sql" "$q" "$analyze_sql"
    done
  done
done

echo "Done."
echo "Notes:"
echo "  - queries: $queries"
echo "  - random_seed = run index (1..$runs): each repetition is bit-exact"
echo "    reproducible.  Luby phases stay decorrelated because"
echo "    mcts_root_restart() derives phase j's seed via splitmix64(seed, j)."
echo "  - All other GUCs are held fixed across cells within a config (see"
echo "    mcts_luby.sql and mcts_single.sql) so only reward shaping,"
echo "    UCT Q-aggregation, and the Luby-vs-single-phase axis differ."
echo "  - uct_aggregation = 'best'    : Q-value = best reward observed below"
echo "                                  (UCT-Extreme classic)."
echo "  - uct_aggregation = 'average' : Q-value = mean rollout reward"
echo "                                  (standard UCT)."
echo "  - config=luby   : phases=8, start_budget=20, patience=0, depth=4"
echo "                    (cap 260 iters across 8 restarted trees,"
echo "                    Luby weights 1,1,2,1,1,2,4,1)."
echo "  - config=single : phases=1, start_budget=260, patience=0, depth=4"
echo "                    (cap 260 iters in 1 persistent tree)."
echo "  - Depth and total budget are matched so the only axis of difference"
echo "    is Luby restarts.  Compare MCTS Iters / MCTS Best Cost across"
echo "    configs to read the restart effect directly."
echo "  - MCTS Best Cost is the extension's reported best (post-search);"
echo "    Plan Cost is the top-of-EXPLAIN cost of the plan actually chosen."
