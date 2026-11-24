# CEB — IMDb-CEB benchmark (rel ≥ 9, seed 42, 200-query flat)

MCTS-Extreme on the IMDb-CEB workload: the main comparison against PostgreSQL's
DP and GEQO optimizers, plus the MCTS ablations (reward, aggregation, top-k, and
plan-shape kernels). Run group: **MCTS-paper**.

- **Workload:** IMDb-CEB, rel ≥ 9, seed 42 — 200 flattened queries
- **Seeds:** 3
- **Baselines:** PostgreSQL DP (`dp`), GEQO (`geqo`)
- **MCTS default:** single-kernel, reward `neg_log`, aggregation `best`, depth 10, top_k 10, start_budget 480
- **Rows:** 7800 in `results.jsonl` (13 series × 200 queries × 3 seeds)

## Series (13 × 600 rows)

| config | reward | agg | combo | what it is |
|---|---|---|---|---|
| `dp` | — | — | — | PostgreSQL DP baseline |
| `geqo` | — | — | — | PostgreSQL GEQO baseline |
| `single` | `neg_log` | best | — | base replicate |
| `single` | `neg_cost` | best | — | reward ablation: −cost |
| `single` | `norm_neg_log` | best | — | reward ablation: normalized −log |
| `single` | `neg_log` | average | `d10_tk10_S_sb480` | aggregation ablation: average vs best |
| `single` | `neg_log` | best | `tk8`…`tk12` | top-k sweep; |
| `single` | `neg_log` | best | `k2`, `k3` | plan-shape kernel ablation (K = 2, 3) |

## Layout
- `results.jsonl` — all 7800 measurement rows
- `config.json` — run configuration (queries, GUCs, MCTS cells)
- `dp/`, `geqo/`, `single__*/` — per-series chosen plans and `.sql`
- `status` — run status

## Provenance

This is a **merge of four `ablation/sweep.py` sweeps** around the base
MCTS-Extreme config (`single`, `neg_log`, `best`, depth 10, top_k 10,
start_budget 480) — each varies one ablation axis — plus the PostgreSQL
baselines. It is not a single sweep: one part is a grid, the other three are
one-at-a-time cell sweeps. To regenerate, run each and concatenate the
`results.jsonl` files (dedupe identical baselines):

| # | sweep | mode | varies |
|---|---|---|---|
| 1 | reward + baselines | grid | `reward_modes = [neg_log, neg_cost, norm_neg_log]`, `configs = [single, dp, geqo]` |
| 2 | aggregation | cell | `agg = average` (vs base `best`) |
| 3 | top-k | cell | `top_k = 8, 9, 10, 11, 12` |
| 4 | plan-shape | cell | `plan_shape = 2, 3` (vs base K = 1) |

Note: the stored rows record this knob as `gucs.kernels`; current `sweep.py`
calls it `plan_shape` (same parameter, renamed).
