#!/usr/bin/env bash
#
# setup_tpch.sh — standalone TPC-H setup from scratch on any machine.
#
# On a clean machine this single file does EVERYTHING for TPC-H:
#   1. git clone pg_tpch        (schema, query templates, SQL harness)
#   2. git clone + build dbgen  (official TPC-H data/query generator)
#   3. initdb + start cluster in PGDATA (if not already running)
#   4. createdb tpch
#   5. generate data (dbgen) and the 22 queries (qgen)
#   6. load: schema + data + primary/foreign keys + indexes + ANALYZE
#
# It does NOT run the benchmark — this is preparation only. Queries are run
# afterwards by pg_tpch/run_tpch.sh -L.
#
# Inputs (exactly two positional args):
#   $1 = BIN     PostgreSQL bin dir (psql/pg_ctl/initdb)
#   $2 = PGDATA  cluster dir (created if empty)
#
# Optional (env): PGPORT (default 5432), SCALE (default 1),
#                 BENCH_HOME (where to clone repos, default $HOME).
#
# Examples:
#   ./setup_tpch.sh /usr/lib/postgresql/17/bin ~/pgdata
#   ./setup_tpch.sh ~/pg/inst/bin ~/pgdata
#   PGPORT=5499 SCALE=10 ./setup_tpch.sh ~/pg/inst/bin ~/pgdata
#
set -euo pipefail

# ── input ────────────────────────────────────────────────────────────────────
BIN="${1:-}"
PGDATA_DIR="${2:-}"
[[ -n "$BIN" && -n "$PGDATA_DIR" ]] || {
    sed -n '2,26p' "$0"; exit 1;
}

PGPORT="${PGPORT:-5432}"
SCALE="${SCALE:-1}"
BENCH_HOME="${BENCH_HOME:-$HOME}"
DATA_DIR="${DATA_DIR:-/tmp/dss-data}"
DBNAME="${DBNAME:-tpch}"

TPCH_REPO="git@github.com:Alena0704/pg_tpch.git"
TPCH_REPO_HTTPS="https://github.com/Alena0704/pg_tpch.git"
DBGEN_REPO="https://github.com/electrum/tpch-dbgen.git"

TPCH_DIR="$BENCH_HOME/pg_tpch"
DBGEN_DIR="$TPCH_DIR/tpch-dbgen"
DBUSER="$(whoami)"

export PATH="$BIN:$PATH"
export PGPORT
PSQL="psql -X -v ON_ERROR_STOP=1 -p $PGPORT"
TPCH_TABLES="region nation part supplier partsupp customer orders lineitem"

log()  { printf '\n[%s] === %s ===\n' "$(date +%H:%M:%S)" "$*"; }
die()  { printf '\n[ERROR] %s\n' "$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "command '$1' not found${2:+ — $2}"; }

[[ "$(id -u)" -ne 0 ]] || die "do not run as root/sudo — PostgreSQL refuses to run as root."
[[ -x "$BIN/psql" ]]   || die "no psql in BIN ($BIN) — point to the correct PostgreSQL bin dir."
need git

# ── 1. pg_tpch ───────────────────────────────────────────────────────────────
if [[ ! -d "$TPCH_DIR/.git" ]]; then
    log "git clone pg_tpch → $TPCH_DIR"
    git clone "$TPCH_REPO" "$TPCH_DIR" 2>/dev/null \
        || git clone "$TPCH_REPO_HTTPS" "$TPCH_DIR" \
        || die "failed to clone pg_tpch (neither SSH nor HTTPS)."
else
    log "pg_tpch already present: $TPCH_DIR"
fi
[[ -f "$TPCH_DIR/dss/tpch-load.sql" ]] || die "missing dss/tpch-load.sql in $TPCH_DIR — incomplete repo."

# ── 2. dbgen: clone + build ──────────────────────────────────────────────────
if [[ ! -x "$DBGEN_DIR/dbgen" || ! -x "$DBGEN_DIR/qgen" ]]; then
    if [[ ! -e "$DBGEN_DIR/makefile.suite" && ! -e "$DBGEN_DIR/dbgen.c" ]]; then
        log "git clone tpch-dbgen → $DBGEN_DIR"
        git clone --depth 1 "$DBGEN_REPO" "$DBGEN_DIR"
    fi
    need make; need gcc
    case "$(uname -s)" in
        Darwin) machine=MACOS ;;   # if the build fails, switch to MAC
        *)      machine=LINUX ;;
    esac
    [[ -f "$DBGEN_DIR/Makefile" || -f "$DBGEN_DIR/makefile" ]] \
        || cp "$DBGEN_DIR/makefile.suite" "$DBGEN_DIR/Makefile"
    log "building dbgen/qgen (MACHINE=$machine)"
    rm -f "$DBGEN_DIR"/*.o          # stray .o files from the repo break linking
    make -C "$DBGEN_DIR" CC=gcc DATABASE=ORACLE MACHINE="$machine" WORKLOAD=TPCH \
        || die "dbgen build failed (gcc and make required)."
    [[ -x "$DBGEN_DIR/dbgen" && -x "$DBGEN_DIR/qgen" ]] || die "dbgen/qgen were not produced."
else
    log "dbgen already built: $DBGEN_DIR"
fi

# ── 3. cluster ───────────────────────────────────────────────────────────────
need pg_isready
if pg_isready -p "$PGPORT" -q 2>/dev/null; then
    log "PostgreSQL already answering on port $PGPORT"
else
    need initdb; need pg_ctl
    if [[ ! -d "$PGDATA_DIR" || -z "$(ls -A "$PGDATA_DIR" 2>/dev/null)" ]]; then
        log "initdb → $PGDATA_DIR"
        mkdir -p "$PGDATA_DIR"
        initdb -D "$PGDATA_DIR" -U "$DBUSER" --encoding=UTF8 --locale=C
        echo "port = $PGPORT" >> "$PGDATA_DIR/postgresql.conf"
    fi
    log "starting cluster (PGDATA=$PGDATA_DIR, port $PGPORT)"
    pg_ctl -D "$PGDATA_DIR" -o "-p $PGPORT" -l "$PGDATA_DIR/server.log" -w start
fi

# ── 4. database and role ─────────────────────────────────────────────────────
log "database $DBNAME and role $DBUSER (creating if needed)"
psql -X -p "$PGPORT" -d postgres -tAc \
    "SELECT 1 FROM pg_roles WHERE rolname='$DBUSER'" | grep -q 1 \
    || createuser -p "$PGPORT" -s "$DBUSER"
psql -X -p "$PGPORT" -d postgres -tAc \
    "SELECT 1 FROM pg_database WHERE datname='$DBNAME'" | grep -q 1 \
    || createdb -p "$PGPORT" -O "$DBUSER" "$DBNAME"

# ── 5. data and query generation ─────────────────────────────────────────────
have_all=1
for t in $TPCH_TABLES; do [[ -f "$DATA_DIR/$t.csv" ]] || have_all=0; done
if [[ $have_all -eq 0 ]]; then
    log "generating data with dbgen (SF=$SCALE) → $DATA_DIR"
    mkdir -p "$DATA_DIR"
    ( cd "$DATA_DIR" && "$DBGEN_DIR/dbgen" -b "$DBGEN_DIR/dists.dss" -f -s "$SCALE" )
    chmod u+rw "$DATA_DIR"/*.tbl          # dbgen sometimes creates .tbl without the read bit
    for f in "$DATA_DIR"/*.tbl; do
        base=$(basename "${f%.tbl}")
        sed 's/|$//' "$f" > "$DATA_DIR/$base.csv"
        rm -f "$f"
    done
    for t in $TPCH_TABLES; do
        [[ -s "$DATA_DIR/$t.csv" ]] || die "table $t.csv is empty — generation failed (check permissions in $DATA_DIR)."
    done
else
    log "data already present in $DATA_DIR (all 8 tables) — skipping dbgen"
fi

QDIR="$TPCH_DIR/dss/queries"
TPL="$TPCH_DIR/dss/templates"
if [[ -z "$(ls -A "$QDIR" 2>/dev/null)" ]]; then
    log "generating 22 queries with qgen → $QDIR"
    mkdir -p "$QDIR"
    for q in $(seq 1 22); do
        ( cd "$DBGEN_DIR" && DSS_QUERY="$TPL" ./qgen "$q" ) > "$QDIR/$q.sql"
    done
else
    log "queries already generated in $QDIR"
fi

# ── 6. load into database ────────────────────────────────────────────────────
log "loading data (COPY FROM $DATA_DIR)"
sed "s#/tmp/dss-data#$DATA_DIR#g" "$TPCH_DIR/dss/tpch-load.sql" | $PSQL -d "$DBNAME"
log "primary keys";  $PSQL -d "$DBNAME" < "$TPCH_DIR/dss/tpch-pkeys.sql"
log "foreign keys";  $PSQL -d "$DBNAME" < "$TPCH_DIR/dss/tpch-alter.sql"
log "indexes";       $PSQL -d "$DBNAME" < "$TPCH_DIR/dss/tpch-index.sql"
log "ANALYZE";       $PSQL -d "$DBNAME" -c "ANALYZE;"

log "DONE. TPC-H (SF=$SCALE) loaded into database '$DBNAME' on port $PGPORT."
printf '\nRun the 22 queries:\n  BIN=%q PGPORT=%s bash %q -L -d %s\n\n' \
    "$BIN" "$PGPORT" "$TPCH_DIR/run_tpch.sh" "$DBNAME"
