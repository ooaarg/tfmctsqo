# Archived benchmark runs

This directory holds the measurement archives behind the tables and figures of
the paper. Every run directory contains `config.json` (queries, seeds, settings,
cell definitions), `results.jsonl` (one JSON object per seed-run), the SQL
templates used for the run, and the generated plan files where the archive kept
them. `status` is `done` for every complete run.

All archives use the same result schema for the PostgreSQL runs: one row per
query, seed, and configuration, with planning time, execution time, estimated
plan cost, and the chosen plan path. The DuckDB rows carry the profile-derived
end-to-end time and the correctness check instead.

| Run | Engine | Workload | Configurations | Queries | Seeds | Rows | Dates |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| `CEB` | PostgreSQL | IMDb-CEB, 200-query subset | MCTS-Extreme variants, DPSize, GEQO | 200 | 3 | 7800 | 2026-05-25 to 28 |
| `JOB-JOBComplex` | PostgreSQL | JOB (113) and JOB-Complex (30) | MCTS-Extreme, DPSize, GEQO | 143 | 3 | 1287 | 2026-05-25 to 26 |
| `CEB-SAIOvsII` | PostgreSQL | IMDb-CEB, 200-query subset | MCTS-Extreme, SAIO, iterative improvement | 200 | 3 | 1200 | 2026-06-04 to 05 |
| `stack-mcts-vs-geqo-200` | PostgreSQL | Stack, 200-query subset | MCTS-Extreme, GEQO | 200 | 3 | 1200 | 2026-09-14 to 16 |
| `dp-stack-12plus` | PostgreSQL | Stack, 12 relations | DPSize | 100 | 3 | 300 | 2026-09-16 |
| `dp-stack-12minus` | PostgreSQL | Stack, 9 or 11 relations | DPSize | 100 | 3 | 300 | 2026-09-16 |
| `duckdb-job-12plus` | DuckDB | JOB, at least 12 relations | Native, retention on/off | 20 | 5 | 500 | 2026-09-17 |
| `duckdb-jobcomplex-12plus` | DuckDB | JOB-Complex, at least 12 relations | Native, retention on/off | 13 | 10 | 650 | 2026-09-17 |
| `duckdb-ceb-12plus` | DuckDB | IMDb-CEB, at least 12 relations | Native, retention on/off | 57 | 5 | 1425 | 2026-09-17 |

`runs-manifest.json` at the repository root records the counts and the SHA-256 of
every `results.jsonl`. For the DuckDB runs it also
stores the hash of the file before machine-local paths were replaced with
`/path/to/` placeholders.

`CEB` is a merged archive assembled from five source runs named in its
`config.json` (`merged_from`, `merged_at`): the main DPSize/GEQO comparison, an
aggregation ablation, a reward ablation, a depth series, and a top-k series. The
`combo_id`, `reward`, `agg`, and `gucs` fields identify the variant of each row.
The ablation in Table 7 selects the rows by these fields.

## What the archives do not contain

The archives are the measurements cited in the paper; debug runs, failed
launches, and intermediate experiment snapshots are not included. Database
files, prebuilt binaries, and raw query output are not included either. In
particular, the Stack database dump and the DuckDB database layout are
external. `../REPRODUCING.md` states the reproduction limits.

## DuckDB configuration

Each DuckDB run retains all five measured cells: native DuckDB; retention enabled
with `retention_max_group_fraction` 1.0 or 0.5; retention disabled with either
threshold. 1.0 disables the row-group check. When retention is disabled, the
threshold is inactive; those cells remain separate measurements and must not be
deduplicated. The primary selected MCTS configuration is `retTrue_ret0.5`.

Shared search settings: 250 iterations, one phase, depth 10, top-k 20,
rank-biased rollouts, `neg_log` reward, best aggregation, exploration constant
0.05, gamma 0.5, greedy initialization disabled. The actual seeds are in
`seed_values` and in each result row; the seed in `resolved_cells` is a template
value.

Recorded protocol: cold cache, one thread, 8192 MB memory, 30-second timeout, a
fresh CLI per measurement, CPU 4 affinity, and rotating cell order. All selected
DuckDB results report matched answers, no timeouts, and no fallbacks. Hardware
details are not recorded in the copied configurations.

The archived DuckDB timings require the original physical database layout: row
group layout affects retention and execution performance. Loading the same
logical rows into a fresh database is not equivalent. `../REPRODUCING.md`
records the layout identifier.

For setup and launch commands, see `../REPRODUCING.md` and
`../ablation/README.md`.
