# OLAP DBMS Benchmark Results repository

Repository for storing DBMS benchmark data and results. See other branches for branch-specific results. We use PostreSQL19 (Feb 2026)

## AlphaJoin

- **`MyAlphaJoin/`** — Alpha Join with additional fixes. For the full pipeline, training recommendations, experiments, and observations see **[MyAlphaJoin/README.md](MyAlphaJoin/README.md)**.

- **`AlphaJoin2/`** — original AlphaJoin (unmodified).

## HyperQO

- **`HyperQO/`** — With additional fixes with no GPU requirement. Simply enter DBMS credentials in ImportantConfig to use, or use run_*.sh to launch certain benchmark

## Query workloads

- **`resource/imdb_original_queries/`** — original JOB queries.
- **`resource/job_complex/`** — JOB-Complex queries.
