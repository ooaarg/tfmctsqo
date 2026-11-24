# Learned optimizers and query corpora

This directory contains the learned MCTS optimizers used for the comparison in
Section 6.4 and Table 6, and the query corpora used by every experiment.

See `../README.md` for the repository overview and the
per-table mapping.

## AlphaJoin

`MyAlphaJoin/` is the AlphaJoin pipeline with the fixes used for the
reproduction. Its model files are under `MyAlphaJoin/models/` and its query
copies under `MyAlphaJoin/resource/jobquery/`. The per-query outputs are in
`../results/job/pg19/AlphaJoin/`. See `MyAlphaJoin/README.md` for the pipeline
and the environment variables.

## HyperQO

`HyperQO/` is HyperQO with the fixes used for the reproduction; it runs without
a GPU. The trained model is `HyperQO/model/log_c3_h64_s4_t3.pth`. Configure the
database through the standard `PG*` environment variables (see
`HyperQO/ImportantConfig.py`), or use `../run_hyperqo_job.sh` and
`../run_hyperqo_job_complex.sh`. The per-query outputs are in
`../results/job/pg19/HyperQO/` and `../results/jobcomplex/pg19/HyperQO/`.

## Query corpora

| Path | Workload |
| --- | --- |
| `resource/imdb_original_queries/` | JOB, 113 queries |
| `resource/job_complex/` | JOB-Complex, 30 queries |
| `resource/imdb-ceb/queries/rel9_seed42_200_flat/` | IMDb-CEB, 200-query subset (seed 42, at least 9 relations) |
| `resource/stack-geqo-balanced-200/queries/` | Stack, 200-query subset |

The IMDb-CEB subset is reproducible with
`resource/imdb-ceb/script/select_rel_queries.py`; `manifest.csv` records the
selection. The Stack selection is documented in
`resource/stack-geqo-balanced-200/README.md` and its
`selection-manifest.json`.

`job_test.sh` and `job_complex_test.sh` are the standalone timing scripts for
JOB and JOB-Complex. Run them from inside `source/` and set the connection
variables at the top of each script first.
