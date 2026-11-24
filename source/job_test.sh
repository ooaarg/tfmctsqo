# Run from inside the source/ directory, e.g.:
#   cd source && bash ./job_test.sh
#
# Configure the variables below for your environment before running.

INSTDIR=$(dirname "$(command -v psql)")   # PostgreSQL bin directory (override if using a custom build)
QUERIES_DIR=$(pwd)/resource/imdb_original_queries
PLANS_DIR=$(pwd)/../plans                 # written to repo root (sibling of source/)
TIMING_CSV=$(pwd)/../original_time.csv

disabled_iters=10

# Set PG environment variables for correct access to the DBMS
export PGDATA=$(pwd)/../pgdata            # PostgreSQL data directory (sibling of source/)
export PGPORT=5496

mkdir -p "$PLANS_DIR"

#rm explains.txt

# ##############################################################################
#
# Test conditions No.1: Quick pass in 'disabled' mode with statistics and
# forced usage of a bunch of parallel workers.
#
# - Disabled mode with a stat gathering and AQO details in explain
# - Force usage of parallel workers aggressively
# - Enable pg_stat_statements statistics
#
# ##############################################################################

for query_f in "$QUERIES_DIR"/*.sql
  do
    query=`cat "$query_f"`
    times_file=$(mktemp)

    echo "$query_f";
    for i in $(seq 1 $disabled_iters); do
        echo "EXPLAIN " > test.sql
        echo $query >> test.sql
        $INSTDIR/psql postgres -f test.sql > "$PLANS_DIR/$(basename "$query_f")_${i}"
        sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
        $INSTDIR/psql postgres -t \
  -c '\timing on' \
  -f "$query_f" \
| awk -v t="$times_file" '/Time:/ { print $2 >> t }'

    done
    awk -v q="$query_f" -v iters="$disabled_iters" '{ sum += $1 } END { if (iters) printf "%s,%.2f\n", q, sum/iters }' "$times_file" \
| tee -a "$TIMING_CSV"
    rm -f "$times_file"
done
