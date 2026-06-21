# `ablation/`

Sweep runner + Streamlit UI for the `mcts_extreme` PostgreSQL planner extension.

## Setup for a new machine

You need: Python 3.12+, [`uv`](https://docs.astral.sh/uv/), Postgres built with
the `mcts_extreme` extension, the IMDB dataset loaded, and a few env vars
pointing the sweep at your cluster.

If you're starting from scratch on the Postgres + IMDB side, follow the
[repo-root README](../README.md) first — those steps are common across all
optimizers in this repo. Once you have a working `psql -d imdb` against the
JOB schema, come back here.

### 1. Build and install the `mcts_extreme` extension

```bash
cd ../mcts_extreme
make PG_CONFIG=/path/to/your/pg/bin/pg_config install
```

Add it to `shared_preload_libraries` in `postgresql.conf` (the sweep also
`LOAD 'mcts_extreme'` per session, but the GUCs are registered at server
start either way), then restart the cluster.

### 2. Install the Python deps

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
uv run sweep.py -c configs/bench_final.toml --queries "27a 27b" --seeds 5

uv run sweep.py -c configs/paper_search_structure.toml \
    --gucs '{"depth": [2, 3, 4], "luby": {"start_budget": [20, 40, 80]}}'

# Print the plan + run-dir preview without touching Postgres.
uv run sweep.py -c configs/bench_final.toml --dry-run

# Docker-backed smoke run for the OG_project side-by-side container.
uv run sweep.py -c configs/job_geqo.toml \
    --queries "1a" --seeds 1 --configs geqo \
    --docker-container practical_mcts_qo_pg_main_20260521

# Parallelism + warmup knobs.
uv run sweep.py -c configs/bench_final.toml -j 4        # 4 cells per batch (default; see below)
uv run sweep.py -c configs/bench_final.toml -j 1        # sequential
uv run sweep.py -c configs/bench_final.toml --prewarm   # warm catalog once per batch (rarely useful)

# Workbook converter.
uv run sharing.py export ../runs/<id> -o report.xlsx
uv run sharing.py import workbook.xlsx -o ../runs/imported-foo/

# Compare two runs (e.g. parallel vs sequential cross-check).
uv run verify.py ../runs/<run_A> ../runs/<run_B>
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
`mcts_best_cost` rows. Cross-check with:

```bash
uv run verify.py ../runs/<run_A> ../runs/<run_B>
```

It pairs rows by `(source, query, config, reward, agg, combo_id, seed)` and
reports:

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

`configs/PAPER_RUNS.md` and `configs/STAGED_SEARCH.md` describe the staged
search → benchmark → ablation plan for the paper runs.

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
