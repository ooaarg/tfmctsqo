# Practical Training-Free MCTS Query Optimization

This repository contains code, scripts, and collected benchmark results for query optimization experiments on the JOB, JOB-Complex, and IMDb-CEB workloads.

It combines three optimizer tracks:
- `MCTSExtreme` (MCTS-based join-order search)
- `HyperQO` (learning + search hybrid optimizer)
- `AlphaJoin` (AlphaJoin pipeline and variants)

It also includes some precomputed outputs in `results/` and figures in `plots/`.

## Cloning

This repo uses **git submodules** (HyperQO, AlphaJoin) and **Git LFS** (the
`runs/` sweep outputs). Install Git LFS first, then clone recursively:

```bash
git lfs install
git clone --recurse-submodules <repo-url>
# or, if already cloned:
git submodule update --init --recursive
git lfs pull
```

> `runs/` is quite large in LFS.** It is stored as **~10,000 LFS objects
> (~140 MB)**, so the LFS fetch is slow and bandwidth-heavy. If you only need the
> code and figures (not the raw per-query run artifacts), skip the LFS download
> and pull a subset on demand:
>
> ```bash
> GIT_LFS_SKIP_SMUDGE=1 git clone --recurse-submodules <repo-url>
> # later, fetch just what you need, e.g.:
> git lfs pull --include="runs/CEB/results.jsonl"
> ```

## Repository layout

- `source/`: implementation code and runnable scripts.
- `source/HyperQO/`: HyperQO optimizer — **git submodule** (`ooaarg/HyperQO`).
- `source/MyAlphaJoin/`: AlphaJoin pipeline + variants — **git submodule** (`ooaarg/AlphaJoin`).
- `source/resource/imdb_original_queries/`: original JOB queries.
- `source/resource/job_complex/`: JOB-Complex queries.
- `mcts_extreme/`: the MCTS-Extreme PostgreSQL extension (C).
- `ablation/`: the MCTS sweep harness (`sweep.py`) and run configs (`ablation/configs/`).
- `runs/`: MCTS-Extreme sweep results (`CEB`, `JOB-JOBComplex`, `CEB-SAIOvsII`).
- `results/`: collected benchmark runs (organized by workload / DB version / method).
- `plots/`: figure generators (`gen_results.py`, `gen_stats.py`); output in `plots/figures/`.

## Environment assumptions

There is no single lockfile (`requirements.txt`/`pyproject.toml`) at repository root. Setup is component-specific.

Common requirements across scripts:
- Python 3
- PostgreSQL instance with IMDB/JOB data loaded
- `psycopg2`
- `torch` (for HyperQO/AlphaJoin)

Some scripts also assume:
- `pg_hint_plan`
- Java + JDBC driver (`source/*/tools/postgresql-42.2.6.jar`)

Important: many shell scripts and configs currently may use hardcoded local paths. Please update them before running in a different environment.

## PostgreSQL installation: making `psql`, `pg_ctl`, etc. discoverable

The bash scripts in this repo (`source/job_test.sh`,
`source/job_complex_test.sh`) are designed to be **run from inside `source/`**;
they derive other paths from `$(pwd)` (e.g. `$(pwd)/resource`,
`$(pwd)/../pgdata`). They resolve PostgreSQL binaries through your shell
`PATH`, e.g.:

```bash
INSTDIR=$(dirname "$(command -v psql)")
```

This means **`psql`, `pg_ctl`, `pg_config`, etc. must be on `PATH`** for the
scripts to find them. Typical invocation:

```bash
cd source
bash ./job_test.sh
bash ./job_complex_test.sh
```

## Quick start

### 1 Review and configure scripts

Start with:
- `source/README.md`
- `source/HyperQO/README.md`
- `source/MyAlphaJoin/README.md`

Configure the database connection via the standard libpq environment variables
(`PGHOST`, `PGPORT`, `PGDATABASE`, `PGUSER`, `PGPASSWORD`) — nothing is hardcoded
(see `source/HyperQO/ImportantConfig.py` and `source/MyAlphaJoin/tools/JDBCUtil.java`).

### 2 Run one of the methods

HyperQO (directory benchmark) — driver scripts at the repo root run the HyperQO
submodule against the JOB / JOB-Complex queries:

```bash
bash run_hyperqo_job.sh           # JOB (113 queries)
bash run_hyperqo_job_complex.sh   # JOB-Complex
```

Or call the runner directly:

```bash
cd source/HyperQO
python hyperqo_run_dir.py \
  --sql-dir ../resource/imdb_original_queries \
  --log-file run-log.txt \
  --stats-json stats.json \
  --stats-csv perquery.csv
```

AlphaJoin pipeline:
- Follow `source/MyAlphaJoin/README.md` for full training + inference + evaluation steps.

### Caveats

This is research software, use cold caching in your own risk.

- The `drop_caches` step requires passwordless `sudo` (or remove that line
  to skip cold-cache enforcement — results will then include cache effects).

## Existing results in this repo

Stored outputs are already included, for example:
- `results/job/pg19/default/perquery.csv`
- `results/job/pg19/HyperQO/perquery.csv`
- `results/job/pg19/MCTSExtreme/perquery.csv`
- `results/job/pg19/AlphaJoin/perquery.csv`
- `results/jobcomplex/pg19/HyperQO/perquery.csv`
- `results/jobcomplex/pg19/MCTSExtreme/perquery.csv`

Figures are produced by `plots/gen_results.py` and `plots/gen_stats.py` (they read
`runs/`) into `plots/figures/`, e.g.:
- `plots/figures/job_kind_stats.png`
- `plots/figures/job_complex_stats.png`
- `plots/figures/ceb_by_rels.png`
