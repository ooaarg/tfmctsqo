# `ablation/`

Sweep runner and Streamlit results interface for PostgreSQL and DuckDB MCTS experiments.
For the included IMDb, Stack, and DuckDB presets, start with the
[reproduction guide](../REPRODUCING.md). The PostgreSQL setup below describes
the `mcts_extreme` runner; DuckDB uses its own runtime paths and adapter.

The archived runs in [`../runs/`](../runs/README.md) are the measurement records
of the paper. Each archive's `config.json` is authoritative for its query list,
seeds, cells, and settings. The presets under `configs/` launch new sweeps; some
default to a different seed count than the archive. Two presets,
`ceb_budget_topk_uct.toml` and `ceb_rel12_budget_topk_uct.toml`, are exploratory
large-budget/top-k sweeps that are not reported in the paper.

## Setup for a new machine

You need: Python 3.12+, [`uv`](https://docs.astral.sh/uv/), Postgres built with
the `mcts_extreme` extension, the IMDB dataset loaded, and a few env vars
pointing the sweep at your cluster.

For the archived PostgreSQL experiments, follow the
[Stack setup](../REPRODUCING.md#postgresql-stack) and set `DB=stack`.
The generic IMDb examples below require a separately loaded JOB database.

### 1. Build and install the `mcts_extreme` extension

```bash
cd ../mcts_extreme
make PG_CONFIG=/path/to/your/pg/bin/pg_config install
```

Add it to `shared_preload_libraries` in `postgresql.conf` (the sweep also
`LOAD 'mcts_extreme'` per session, but the GUCs are registered at server
start either way), then restart the cluster.

### 2. Install the Python deps

If you don't have [`uv`](https://docs.astral.sh/uv/) yet, install it first:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env    # or restart your shell so `uv` is on PATH
```

Then:

```bash
cd ablation
uv sync --group dev
```

`uv` creates a local `.venv/`; you don't need to activate it manually — `uv
run …` picks it up.

### 3. Point the sweep at your cluster

`sweep.py` reads these from the environment:

```
PGDATA=/path/to/postgres/data    # required; fails fast if unset
PGBIN=/path/to/postgres/bin      # optional; falls back to pg_ctl on PATH
PGLOG=/tmp/mcts_extreme_pg.log   # optional
DB=imdb                          # optional; defaults to 'imdb'
DOCKER_CONTAINER=name            # optional; use docker restart/exec instead of pg_ctl/psql
```

Add them to your shell rc so you don't have to re-export each time.

Browsing existing runs in the Streamlit UI doesn't need any of these.

### 4. Launch the UI

```bash
uv run streamlit run app.py
```

Open the URL it prints, click **+ New experiment**, pick a base config from
`configs/`, adjust the knobs, and **Launch sweep**.

The top row of the dialog includes search variants:

- **Budget mode**: `flex budget` stops when expansion is exhausted; `full budget`
  spends the remaining budget with classic UCB1.
- **Expand scenario**: `cost`, `row`, `mixed_025`, `mixed_050`, or
  `selectivity` controls top-k expansion ranking.
- **Measurement**: `full protocol` runs `EXPLAIN ANALYZE`; `cost only` runs
  `EXPLAIN` without executing the query and forces one seed.

## `runs/<id>/` layout

```
runs/2026-05-17T18-30-00-my-grid/
├── status                                # running | done | failed | stopped
├── pid                                   # sweep.py PID (gone on exit)
├── README.md                             # auto-generated summary
├── config.json                           # resolved config + grid axes
├── results.jsonl                         # one row per (cell × query × seed)
├── luby.sql, single.sql                  # rendered MCTS templates
│   …or luby__<combo>.sql when grid varies…
├── dp.sql, geqo.sql                      # baseline templates
├── <cell>/<query>/seed-N.plan            # raw EXPLAIN ANALYZE captures
└── errors.log                            # only on failure
```

One `results.jsonl` row (fields omitted when absent, e.g. on timeout):

```json
{"ts": "...", "query": "1a", "source": "job", "config": "luby",
 "reward": "norm_neg_log", "agg": "best", "combo_id": "d2_L_sb20",
 "gucs": {...}, "seed": 1, "plan_cost": 12345.6, "plan_time_ms": 8.4,
 "exec_time_ms": 132.7, "mcts_best_cost": 12345.6, "iters": 260,
 "rels": 8, "phases": 8, "timed_out": true, "plan_path": "..."}
```

## CLI

```bash
# Launch a sweep. `--gucs` accepts scalar OR list values per leaf
# (lists become a Cartesian-product grid over the varying axes).
uv run sweep.py -c configs/job_jobcomplex_full.toml --queries "27a 27b" --seeds 5

uv run sweep.py -c configs/bench_ceb_rel9_seed42_200.toml \
    --gucs '{"depth": [2, 3, 4], "luby": {"start_budget": [20, 40, 80]}}'

# Print the plan + run-dir preview without touching Postgres.
uv run sweep.py -c configs/bench_ceb_rel9_seed42_200.toml --dry-run

# Docker-backed smoke run for the OG_project side-by-side container.
uv run sweep.py -c configs/job_geqo.toml \
    --queries "1a" --seeds 1 --configs geqo \
    --docker-container practical_mcts_qo_pg_main_20260521

# Parallelism + warmup knobs.
uv run sweep.py -c configs/bench_ceb_rel9_seed42_200.toml -j 4        # 4 cells per batch (default; see below)
uv run sweep.py -c configs/bench_ceb_rel9_seed42_200.toml -j 1        # sequential
uv run sweep.py -c configs/bench_ceb_rel9_seed42_200.toml --prewarm   # warm catalog once per batch (rarely useful)

# Workbook converter.
uv run sharing.py export ../runs/<id> -o report.xlsx
uv run sharing.py import workbook.xlsx -o ../runs/imported-foo/
```

### Parallelism (`-j`)

Each `(query, seed)` batch runs all cells concurrently against the same
freshly-restarted cluster, so `plan_cost` is identical to a sequential run
but wall-clock drops by roughly `len(cells)`.

`plan_time` accuracy depends on CPU contention. On the dev machine
(12th-gen i7-12700, 8 P + 4 E cores):

| `-j` | speedup | plan_time accuracy |
|---|---|---|
| 4 (default) | ~2.3× | within seed noise — paper grade |
| 8 | ~3.6× | ±25% per-cell drift — fine for search/exploration |
| ≥10 | ~6× | systematic bias — don't use for headline numbers |

Pick a value ≤ your number of P-cores. The UI's New-experiment dialog has
the same control.

## Verifying reproducibility

Two runs with the same config should produce identical `plan_cost` and
`mcts_best_cost` rows. Pair the rows by
`(source, query, config, reward, agg, combo_id, seed)` and compare those two
fields; a plain file diff is not enough because timestamps and plan paths
differ. The consistency criteria are:

- Any `plan_cost` / `mcts_best_cost` mismatch (should be zero — flag a bug
  if not).
- `plan_time` delta distribution + per-mode 2σ tolerance check.
- Side-by-side per-mode aggregates and MCTS-vs-baseline speedups.

Use this whenever you change the harness, the extension, or the parallelism
level, to confirm the numbers still agree.

## Streamlit UI tabs

- **Summary** — bar grid of cost / e2e / exec / plan / iters across the
  workload. Metrics with no data anywhere are dropped (baselines don't
  contribute to the iters panel, etc.). Toggle to relative-to-baseline view.
- **Winners** — best-cell-per-query tables (which mode wins on cost, on
  plan-time, on exec-time, …).
- **Trajectory** — per-cell line plot vs seed for one query.
- **Image** — matplotlib snapshot for paper figures.
- **Timeouts** — table of `(cell, query, seed)` runs that hit
  `statement_timeout`.
- **Plans** — raw EXPLAIN output for one `(cell, query, seed)` triple.

## Configs

`configs/*.toml` ship the defaults the New-experiment dialog pre-fills.
Source names (`"job"`, `"jobcomplex"`) resolve through `KNOWN_SOURCES` in
`sweep.py` to `../source/resource/imdb_original_queries/` and
`../source/resource/job_complex/`. Absolute paths work too.

The selected release experiments and their launch commands are described in
[REPRODUCING.md](../REPRODUCING.md).

Every rendered SQL template sets `statement_timeout = 5min`; rows that hit
it get `timed_out: true` and are skipped from the bar plots.

## Dev

```bash
uv run ruff format .
uv run ruff check --fix .
uv run flake8 .
```

All three should report zero. Disable rationale lives next to each disabled
rule in `pyproject.toml` (ruff) and `setup.cfg` (wemake).

## Stack GEQO-balanced 200

Choose `stack_geqo_balanced_200.toml` in **+ New experiment**. The bundled
`stack-geqo-balanced-200` source contains the same query IDs and SQL bytes as
the reference corpus: 100 q3 queries (12 relations), 50 q2 (11), and 50 q8 (9), sampled
with seed 42. The UI offers relation-count presets for this source.

The preset targets database `stack` and runs single-phase MCTS (depth 10,
top-k 10, budget 480, neg_log/best, no Luby), DP, and GEQO, with three seeds.
That is 1,800 measurements. The GEQO baseline uses `geqo_threshold=2`, forcing
GEQO for all 200 queries; it does not measure PostgreSQL's default threshold-12
switch. DP is forced separately. Existing five-minute timeouts apply.

Preview without connecting to PostgreSQL:

```bash
cd ablation
uv run sweep.py -c configs/stack_geqo_balanced_200.toml --dry-run
```

Launch against your restored Stack cluster (set `PGBIN` to the matching build):

```bash
export PGDATA=/path/to/postgres/data
export PGHOST=/path/to/postgres/socket
export PGPORT=55432
export PGBIN=/path/to/matching/postgres/bin
export DB=stack
uv run sweep.py -c configs/stack_geqo_balanced_200.toml -j 1
```

Local readiness checks, prewarming, and measurements respect `PGHOST` (default
`/tmp`) and libpq's `PGPORT`. The runner restarts the specified cluster under its
normal execution protocol. For cloud runs, clone this repository, restore Stack,
build/install the matching PostgreSQL and extension, and set those variables to
the cloud paths. Query files are bundled; no sibling the reference corpus checkout is required.
`DB` and `JOB_DIR` environment variables override the preset's database/source.

### MCTS versus default PostgreSQL planner choice

Choose `stack_mcts_vs_default_pg.toml` in the UI, or run:

```bash
uv run sweep.py -c configs/stack_mcts_vs_default_pg.toml --dry-run
# Remove --dry-run to launch against the configured Stack cluster.
```

This selects `single` and `pg_default`: 200 queries × 2 cells × 3 seeds =
1,200 measurements. The **Default PG (DP <12 / GEQO ≥12)** option is also
available in the Configs picker for other workloads. CLI overrides can use
`--configs "single pg_default"`.

`pg_default` disables MCTS and enables GEQO at threshold 12, with its default
seed 0. Thus Stack q2/q8 use DP and q3 uses GEQO. Collapse limits remain 64,
matched to the other benchmark cells; this baseline tests PostgreSQL's default
planner cutover, not all stock server settings. In general the threshold applies
to each join search problem. Forced `dp` and `geqo` remain separate options.

## Shared-core adapters

`implementation` selects `mcts_extreme` (default), `pg_mctsopt`, or
`duckdb_mctsopt`. Existing configs keep their original SQL templates and behavior.
The UI exposes the selector under **New experiment**.

```sh
uv run python sweep.py -c configs/duckdb_shared_job.toml --dry-run
uv run python sweep.py -c configs/pg_shared_stack.toml --dry-run
```

Set paths in `[runtime]`, or override them with `--runtime '{"binary":"..."}'`.
Use `--gucs '{"start_budget":[125,250,500],"top_k":[10,20]}'` for a grid.
`seed_values = [101, 709, 2027]` fixes explicit seeds; `--seeds N` overrides them
with 1 through N. The example configs select small query subsets; use `--queries`
to select a full workload.

New adapter runs are sequential and rotate cell order across queries and seeds.
Cold cache is the default. DuckDB evicts its closed database file and verifies zero
resident pages. PostgreSQL restarts the configured cluster and uses the existing
cache-drop helper. Set `runtime.cache = "warm"` to skip eviction and restarts.
The default statement timeout is 30 seconds. The subprocess watchdog allows setup
and result collection beyond that limit.

DuckDB executes the query and records its JSON profile. E2E is the profile's query
time; planning includes parser, planner, optimizer, and physical planner. PostgreSQL
uses EXPLAIN ANALYZE planning plus execution time, then executes an untimed copy to
collect the answer. Setup, cache eviction, and answer checking are excluded.
`cost_only` saves a plan without executing or checking answers.

Answers are compared as bags against successful baseline cells for the same query
and seed. Without a successful baseline, correctness is marked `no_baseline`.
Errors, timeouts, and wrong answers remain in totals with a timeout-sized penalty;
`penalized` distinguishes these values from measured latency. Fallback is recorded
separately. Raw output, SQL, plans, and DuckDB diagnostics remain beside each row.

Baselines: `pg_default`, `dp`, `geqo` for PostgreSQL; `duckdb_default` for DuckDB.
`pg_default` uses the native DP/GEQO cutoff with matched collapse limits (64 by
default), rather than every stock session setting. DuckDB leaves native optimizer
settings unchanged. Do not preload either PG MCTS extension in the benchmark server.

New result config IDs include the implementation. Workbook export/import preserves
raw rows and resolved config in additional sheets. Existing workbooks and plot
inputs retain their previous format. SQL and plan files travel with the run folder,
not the workbook.
