# Reproducing the experiments

This guide covers environment setup and the launch procedure for the three
experiment families in the paper: PostgreSQL on the IMDb workloads, PostgreSQL
on Stack, and DuckDB on the IMDb workloads. The archived runs are the reference
output; each archive's `config.json` records the exact queries, seeds, cells, and
settings that produced it.

## Python tooling

The harness is in `ablation/`. From the repository root:

```sh
cd ablation
uv sync --frozen
```

The CLI and the Streamlit browser are documented in `ablation/README.md`. The
browser starts with `uv run streamlit run app.py`.

## PostgreSQL: IMDb workloads

The JOB, JOB-Complex, and IMDb-CEB runs use `mcts_extreme` (Section 5.1). Build
and install it against the PostgreSQL installation used for the experiments:

```sh
make -C mcts_extreme PG_CONFIG=/path/to/pg_config
make -C mcts_extreme PG_CONFIG=/path/to/pg_config install
```

PostgreSQL was built from `master` (`19devel`, commit `73d60ac`, February
2026). Restore the IMDb database published with JOB under the name expected by
the preset (`imdb` by default). The query corpora are in
`source/resource/imdb_original_queries/` (JOB), `source/resource/job_complex/`
(JOB-Complex), and
`source/resource/imdb-ceb/queries/rel9_seed42_200_flat/` (the 200-query
IMDb-CEB subset).

The archived configuration records are
`runs/JOB-JOBComplex/config.json` and `runs/CEB/config.json`. The second is a
merged archive of five source runs; its `merged_from` field names them. The
main comparison used one MCTS cell, `neg_log` reward, best aggregation,
depth 10, top-k 10, 480 iterations, and three seeds; DPSize and GEQO ran on the
same queries.

The presets in `ablation/configs/` are launchers, not byte-for-byte copies of
the archived launches. For example, `job_jobcomplex_full.toml` defaults to five
seeds and an extra Luby cell, while the archive used three seeds and one cell.
Use the archive `config.json` as the authority, or pass `--seeds 3` and the
matching cell when rerunning.

From `ablation/`, preview a run without a database:

```sh
DB=imdb uv run python sweep.py -c configs/job_jobcomplex_full.toml --dry-run
DB=imdb uv run python sweep.py -c configs/bench_ceb_rel9_seed42_200.toml --dry-run
DB=imdb uv run python sweep.py -c configs/ceb_saiovsii.toml --dry-run
```

Expected archived counts: 1287 rows for JOB + JOB-Complex, 1200 rows for the
SAIO/iterative-improvement control, and 7800 rows in the merged IMDb-CEB
archive. To execute, configure `PGDATA`, `PGBIN`, and optionally
`PGHOST`/`PGPORT`, then remove `--dry-run`. Use a dedicated benchmark cluster:
the runner restarts PostgreSQL and drops caches before every query.

## PostgreSQL: Stack

Build and install `mcts_extreme` as above, then restore the external Stack
database to `stack`. The bundled 200-query selection is documented in
`source/resource/stack-geqo-balanced-200/README.md`; the database dump is not
included. The archives do not record a complete PostgreSQL server version and
settings record, so do not assume the timings are portable to another server
build or machine.

Preview the three selected experiments without a database:

```sh
DB=stack uv run python sweep.py -c configs/stack_mcts_vs_geqo_200.toml --dry-run
DB=stack uv run python sweep.py -c configs/dp_stack_12plus.toml --dry-run
DB=stack uv run python sweep.py -c configs/dp_stack_12minus.toml --dry-run
```

Expected measurement counts are 1200, 300, and 300. The presets match the
archived runs: `stack_mcts_vs_geqo_200` has 200 queries, three seeds, MCTS and
GEQO; the two DPSize presets have 100 queries, three seeds, and forced DPSize.

## DuckDB

Build the patched DuckDB host and the extension together using
`mctsopt/duckdb_mctsopt/README.md`. The DuckDB commit is
`ead2df641ce60cfce32f4903a98a7f59c4491e1f`; apply the three patches in
`mctsopt/duckdb_mctsopt/patches/` in the documented order. The extension and the
shared core are included; the upstream DuckDB source must be obtained
separately.

The archived timings require the original physical IMDb database layout,
identified by SHA-256
`d75e81bb2d03c5e4d3a9506eba2aaf2bc6ce66e5976ff7ae9ff4e59510a2a9f8`. That file is
not in this repository and no public download location is supplied. Loading the
same logical rows into a fresh DuckDB database is not equivalent: row-group
layout affects retention and execution performance. Obtain the matching file
separately for a faithful rerun; otherwise label the results as a different
layout.

From `ablation/`:

```sh
uv run python sweep.py -c configs/duckdb_job_12plus.toml --dry-run
uv run python sweep.py -c configs/duckdb_jobcomplex_12plus.toml --dry-run
uv run python sweep.py -c configs/duckdb_ceb_12plus.toml --dry-run
```

Expected counts are 500, 650, and 1425. Each preset retains all five cells and
the archived seed list. The checked retention cell uses threshold 0.5; the
unchecked cell uses 1.0. Both disabled-retention cells are retained as separate
measurements.

Set the three `runtime` paths in a copy of the preset, or override them at
launch:

```sh
taskset -c 4 uv run python sweep.py -c configs/duckdb_job_12plus.toml \
  --runtime '{"binary":"/path/to/duckdb","database":"/path/to/imdb-original-layout.duckdb","extension":"/path/to/duckdb_mctsopt.duckdb_extension"}'
```

CPU affinity is supplied by `taskset`, not the harness. Use an available
benchmark CPU consistently. The database must be checkpointed and closed (no
WAL). The runner opens it read-only, evicts file pages for cold-cache
measurements, and checks residency. One thread, 8192 MB, and 30-second timeouts
are specified in the presets.

## Numbers and figures

The paper's own generators are in `paper/`. They read `runs/` and `results/` and
reproduce every table body and both data figures:

```sh
cd paper
python3 gen_numbers.py
python3 gen_figures.py
```

See `paper/README.md` for the item-by-item mapping and the output files.
