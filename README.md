# Practical Training-Free MCTS Query Optimization

This repository contains the source code of MCTS-Extreme (a PostgreSQL join-order search extension), the
portable search core with its DuckDB adapter and host patches, the experiment
harness with the configuration presets used in the paper, the query corpora, and
the measurement archives behind every table and figure of the paper.

MCTS-Extreme searches join orders inside the host planner. It uses the host cost
model and cardinality estimates unchanged and needs no model training.

## Directory map

| Path | Contents |
| --- | --- |
| `mcts_extreme/` | PostgreSQL extension used for all IMDb and Stack experiments (Section 5.1). |
| `mctsopt/` | Portable C11 search core, PostgreSQL adapter, and DuckDB adapter with the three DuckDB host patches (Section 5.2, Appendix A). |
| `ablation/` | Sweep runner, adapters, configuration presets, Streamlit browser, and tests. |
| `source/HyperQO/`, `source/MyAlphaJoin/` | Learned MCTS optimizers used for the comparison in Section 6.4 and Table 6. |
| `source/resource/` | Query corpora: JOB, JOB-Complex, the 200-query IMDb-CEB subset, and the 200-query Stack subset. |
| `results/` | Per-query outputs of the learned optimizers and of the default PostgreSQL policy. |
| `runs/` | Measurement archives: three IMDb runs, three Stack runs, three DuckDB runs. |
| `paper/` | Scripts that regenerate every number, table body, figure data file, and both figures from `runs/` and `results/`. |
| `runs-manifest.json` | Query, seed, and measurement counts plus SHA-256 of every archived result file. |

## Where each paper item comes from

Table and figure numbers follow the paper. The scripts are run from the
repository root with Python 3.12 (standard library only for numbers; `numpy`,
`pandas`, and `matplotlib` for figures). Outputs land in `paper/build/generated/`
and `paper/figures/`.

| Paper item | Data | Script and output |
| --- | --- | --- |
| Table 1, speedup significance | `runs/CEB`, `runs/JOB-JOBComplex` | `paper/gen_numbers.py` → `tab_imdb_e2e.tex` |
| Table 2, full-workload aggregates | `runs/CEB`, `runs/JOB-JOBComplex` | `paper/gen_numbers.py` → `tab_imdb_e2e.tex`, `tab_imdb_plan_cost.tex` |
| Table 3, large-join aggregates | `runs/CEB`, `runs/JOB-JOBComplex` | `paper/gen_numbers.py` → `tab_imdb_e2e.tex`, `tab_imdb_plan_cost.tex` |
| Table 4, co-winner counts | `runs/CEB`, `runs/JOB-JOBComplex` | `paper/gen_numbers.py` → `tab_winners.tex` |
| Figure 5, per-query outcomes | `runs/CEB`, `runs/JOB-JOBComplex` | `paper/gen_figures.py` → `figures/outcomes_all.pdf` (data in `fig_outcomes.dat`) |
| Figure 6, per-query speedups | `runs/CEB`, `runs/JOB-JOBComplex` | `paper/gen_figures.py` → `figures/results_detail.pdf` (data in `fig_perquery_job.dat`, `fig_perquery_jobc.dat`, `fig_ceb_buckets.dat`) |
| Table 5, Stack subset | `runs/stack-mcts-vs-geqo-200`, `runs/dp-stack-12plus`, `runs/dp-stack-12minus` | `paper/gen_numbers.py` → `tab_stack.tex` |
| Table 6, learned optimizers (§6.4) | `results/job/`, `results/jobcomplex/`, `runs/CEB`, `runs/JOB-JOBComplex` | `paper/gen_numbers.py` → `tab_learned.tex` |
| Table 7, ablation | `runs/CEB` | `paper/gen_numbers.py` → `tab_ablation.tex` |
| Table 8, local-search controls | `runs/CEB-SAIOvsII`, `runs/CEB` | `paper/gen_numbers.py` → `tab_controls.tex` |
| Table 9, DuckDB adapter (Appendix A) | `runs/duckdb-job-12plus`, `runs/duckdb-jobcomplex-12plus`, `runs/duckdb-ceb-12plus` | `paper/gen_numbers.py` → `tab_duckdb.tex` |
| Prose numbers (Section 6) | all of the above | `paper/gen_numbers.py` → `numbers.tex` |

`paper/gen_numbers.py` also writes `paper/tfmctsqo_numbers.json`, an audit file
with the intermediate aggregates. Section 6.5 and Appendix A use Tables 7–9.

## Reproduce

Environment: Ubuntu 24.04, PostgreSQL `19devel` (commit `73d60ac`, February
2026), and the AMD EPYC 9654 node described in Section 6.1. Build the extension
with PGXS against the PostgreSQL installation used for the experiments:

```sh
make -C mcts_extreme PG_CONFIG=/path/to/pg_config
make -C mcts_extreme PG_CONFIG=/path/to/pg_config install
```

The sweep harness lives in `ablation/`. It needs Python 3.12+ and
[uv](https://docs.astral.sh/uv/):

```sh
cd ablation
uv sync --frozen
uv run streamlit run app.py          # browse an archived run without a database
uv run python sweep.py -c configs/bench_ceb_rel9_seed42_200.toml --dry-run
```

[`REPRODUCING.md`](REPRODUCING.md) gives the full procedure for the IMDb, Stack,
and DuckDB experiments, including the database prerequisites and the DuckDB host
patches. The archived `config.json` files in
`runs/` are the authoritative record of each launch; the presets under
`ablation/configs/` are generic launchers and in a few cases default to a
different seed count than the archive.

The learned optimizers in Section 6.4 are reproduced from `source/HyperQO/` and
`source/MyAlphaJoin/`; their recorded per-query outputs are in `results/`.

## Data integrity

`runs-manifest.json` lists every archived run with its query, seed, and
measurement counts and the SHA-256 of `runs/<run>/results.jsonl`. To check the
archives:

```sh
python3 - <<'PY'
import hashlib, json
for e in json.load(open("runs-manifest.json")):
    h = hashlib.sha256(open(f"runs/{e['run']}/results.jsonl", "rb").read()).hexdigest()
    print(e["run"], h == e["published_results_sha256"])
PY
```

## Licenses

This repository is released under the MIT license (`LICENSE`). Vendored components
keep their own licenses: `mctsopt/LICENSE`, `mctsopt/duckdb_mctsopt/LICENSE.duckdb`,
and the notices in `source/HyperQO/` and `source/MyAlphaJoin/`.
