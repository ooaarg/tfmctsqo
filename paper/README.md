# Paper numbers and figures

This directory holds the two scripts that produce every number, table body,
figure data file, and both data figures of the paper from the archived runs.

## Requirements

* `gen_numbers.py`: Python 3.12, standard library only.
* `gen_figures.py`: Python 3.12 with `numpy`, `pandas`, and `matplotlib`.
  The paper's figures use Linux Libertine, which is not shipped because the
  fonts carry their own licenses. To match the paper's typography, download
  Linux Libertine from <https://www.linuxlibertine.org/>, convert the regular,
  bold, and italic faces to TrueType if needed, and place them in `paper/fonts/`
  as `LinLibertine_R.ttf`, `LinLibertine_RB.ttf`, and `LinLibertine_I.ttf`.
  Without them the script uses the default matplotlib font.

## Run

From this directory:

```sh
python3 gen_numbers.py
python3 gen_figures.py
```

`gen_numbers.py` writes to `build/generated/`:

| File | Paper item |
| --- | --- |
| `tab_imdb_e2e.tex` | Tables 1–3 (totals, speedups, confidence intervals, p-values) |
| `tab_imdb_plan_cost.tex` | Tables 2–3 (planning time, execution time, estimated cost) |
| `tab_winners.tex` | Table 4 |
| `tab_stack.tex` | Table 5 |
| `tab_learned.tex` | Table 6 |
| `tab_ablation.tex` | Table 7 |
| `tab_controls.tex` | Table 8 |
| `tab_duckdb.tex` | Table 9 |
| `fig_outcomes.dat` | Figure 5 data |
| `fig_perquery_job.dat`, `fig_perquery_jobc.dat`, `fig_ceb_buckets.dat` | Figure 6 data |
| `numbers.tex` | Macros used in the prose of Section 6 |

It also writes `tfmctsqo_numbers.json`, an audit file with the intermediate
aggregates.

`gen_figures.py` writes `figures/outcomes_all.pdf` (Figure 5) and
`figures/results_detail.pdf` (Figure 6), plus PNG previews.

## Inputs

| Input | Used for |
| --- | --- |
| `../runs/CEB/results.jsonl` | IMDb-CEB main run, Figure 6c, Table 7 ablation, MCTS rows of Table 8 |
| `../runs/JOB-JOBComplex/results.jsonl` | JOB and JOB-Complex main runs, Figures 5–6 |
| `../runs/CEB-SAIOvsII/results.jsonl` | SAIO and iterative-improvement controls (Table 8) |
| `../runs/stack-mcts-vs-geqo-200/`, `../runs/dp-stack-12plus/`, `../runs/dp-stack-12minus/` | Table 5 |
| `../runs/duckdb-*-12plus/` | Table 9 |
| `../results/job/pg19/`, `../results/jobcomplex/pg19/` | Learned optimizers in Table 6 |
| `../source/resource/job_complex/12.sql` | Relation count of JOB-Complex query 12 |

The generator treats a seed-run as complete when it did not time out and has
both planning and execution time. Per-query values are means over seeds; totals
cover the queries that every compared method completes on every seed; the 95%
confidence interval is a paired bootstrap over queries with 10^4 resamples
(seed 42); the p-value is a two-sided Wilcoxon signed-rank test on paired
per-query values; the tie band is ±5%.
