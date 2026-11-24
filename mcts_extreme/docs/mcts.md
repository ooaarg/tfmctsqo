# MCTS backend (`mcts/mcts.c`)

Monte Carlo Tree Search for join ordering — the MCTS-Extreme search
described in the paper, run natively as a PostgreSQL planner extension.

## Algorithm

**UCT-Extreme selection**: rather than tracking the average rollout cost
per node, each node stores the *best* cost seen in its subtree — its
*subtree incumbent*.  The reward map φ (`mcts_extreme.reward_map`) turns
that incumbent into the per-node reward used by the selection score, so
selection targets the best plan in each subtree rather than the average.

**Reward map φ** (`mcts_extreme.reward_map`): how a subtree-incumbent
cost becomes a scalar reward.  All maps are monotone in plan quality:
- `neg_log` (default): `-log(cost)`; compares multiplicative cost ratios.
- `neg_cost`: `-cost`; preserves absolute cost differences.
- `norm_neg_log`: log-cost rescaled into `[0, 1]` over the observed cost
  envelope `[cost_min, cost_max]`.

**Restart phases**: the search is divided into `phases` restart cycles.
Each phase rebuilds the tree from scratch with a fresh PRNG seed; with
`mcts_extreme.luby` on (default), phase `j` runs for
`start_budget * luby(j+1)` iterations (Luby sequence 1, 1, 2, 1, 1, 2,
4, ...), otherwise every phase gets a flat `start_budget`.

**Phase-aware seeding**: each phase's seed is derived via splitmix64 of
`(base_seed, phase)` so all three `erand48` words differ between phases,
giving each restart an independent rollout stream.

**Plan-shape parameter K** (`mcts_extreme.plan_shape`):
- `0` (bushy): any pair of clumps may be joined.
- `1` (linear / zig-zag, default): once a chain (size > 1) exists, only
  joins that extend it with a base relation are considered.  The chain
  may appear on either input side, so this is the zig-zag class, not a
  strict left-deep restriction.
- `≥2` (K-component bushy): form up to K independent chains, then merge
  them.  `shape_allows_pair()` gates expansion + rollout candidates.

**Selectable rollout strategy** (`mcts_extreme.rollout`):
- `random` (default): bushy or linear random rollouts, chosen by
  `mcts_extreme.plan_shape`.  PRNG seed is fixed for the phase.
- `luby`: identical rollout body, but the PRNG is re-seeded before
  each rollout by XORing the phase seed with `luby(rollout_idx)`.
  Decorrelates consecutive rollouts within a phase.

## Cache

A simple two-field key canonicalizes `(A join B)` and `(B join A)` to the
same entry, caching the join's cost and resulting joinrel:

```c
typedef struct JoinCostCacheKey {
    uint64 left_bits;     /* relids->words[0] from the smaller side */
    uint64 right_bits;    /* relids->words[0] from the larger side */
} JoinCostCacheKey;
```

Collision-prone above 64 base relations but cache correctness does not
depend on key uniqueness — MCTS replays the chosen plan from scratch in
`mcts_replay_best_order`.

## EXPLAIN integration

`mcts_explain_per_plan` appends a detailed stats block to every
`EXPLAIN ANALYZE` output:

```
MCTS Relations:        17
MCTS Phases:           5
MCTS Iterations:       400        (last phase's count — counter resets per restart)
MCTS Depth-Limit Rollouts: 1024
MCTS Total Random Rollouts: 1024
MCTS Rollout Failures: 0
MCTS Exhausted:        128
MCTS Best Plan:
  MCTS Best Cost:      4022.43    or "no plan from MCTS (fallback used)"
  Found at Phase:      3
  Found at Iteration:  178
MCTS Cache Effectiveness:
  Cost Evals:          15032
  Cache Hits:          11244
  Cache Size:          1801
  Cache Max Size:      50000
  Cache Hit Ratio:     42.8%
MCTS Total Planning Time: 1748.927 ms
MCTS Search Time Breakdown:
  MCTS Search Time:    1747.514 ms
    Selection Time:    23.012 ms
    Expansion Time:    154.881 ms
    Rollout Time:      1569.215 ms
    Backprop Time:     0.398 ms
  MCTS Replay Time:    1.413 ms
```

`Iterations` is **per-phase**, not cumulative — `mcts_root_restart`
resets the counter.  Likewise `Cache Hit Ratio` is per-phase because
the cache is destroyed between restarts.

## GUC parameters

All `PGC_USERSET`.

### Algorithm

| GUC | Type | Default | Range | Description |
|---|---|---|---|---|
| `mcts_extreme.enabled` | bool | `true` | — | Master kill switch |
| `mcts_extreme.min_relations` | int | 3 | 3..BITS_PER_BITMAPWORD | Skip MCTS below this many initial rels |
| `mcts_extreme.exploration_constant` | real | 1.4 | 0..1000 | Exploration constant c |
| `mcts_extreme.gamma` | real | 0.5 | 0.01..2.0 | Exploration exponent γ |
| `mcts_extreme.random_seed` | int | 0 (time-based) | 0..INT_MAX | Base seed; varied per phase via splitmix64 |
| `mcts_extreme.depth` | int | 2 | 2..100 | Tree depth before forcing rollout |
| `mcts_extreme.start_budget` | int | 100 | 1..100M | Phase-1 iterations; Luby scales later phases |
| `mcts_extreme.phases` | int | 5 | 1..1000 | Number of restart phases |
| `mcts_extreme.patience` | int | 0 (off) | 0..1000 | Halt after N phases without improvement |
| `mcts_extreme.top_k` | int | 5 | 0..10000 | Limit per-state actions to top-k cheapest (0 = all) |
| `mcts_extreme.expand_strategy` | enum | `cost` | cost / row / mixed_025 / mixed_050 / selectivity | Top-k expansion ranking |
| `mcts_extreme.rollouts_per_leaf` | int | 1 | 1..10000 | Random rollouts per depth-limited leaf |
| `mcts_extreme.full_budget` | bool | `false` | — | Continue after expansion is exhausted and spend remaining budget with classic UCB1 |
| `mcts_extreme.plan_shape` | int | 1 | 0..100 | Plan-shape K: 0=bushy, 1=linear (zig-zag), ≥2=K-component bushy |
| `mcts_extreme.reward_map` | enum | `neg_log` | neg_log / neg_cost / norm_neg_log | Reward map φ: cost → UCT-Extreme reward |
| `mcts_extreme.uct_aggregation` | enum | `best` | best / average | `best` = UCT-Extreme (subtree incumbent), `average` = mean UCT (ablation) |
| `mcts_extreme.rollout` | enum | `random` | random / luby | Random rollout; `luby` re-seeds the PRNG per rollout for decorrelation |

### Cache

| GUC | Type | Default | Range | Description |
|---|---|---|---|---|
| `mcts_extreme.cache_enabled` | bool | `true` | — | Enable join cost cache |
| `mcts_extreme.cache_size` | int | 256 | 256..1M | Initial hash table size |
| `mcts_extreme.cache_max_size` | int | 50000 | 0..INT_MAX | 0 = unlimited |

### Logging

| GUC | Type | Default | Description |
|---|---|---|---|
| `mcts_extreme.log_debug` | bool | `false` | Emit verbose WARNING traces |
| `mcts_extreme.log_steps` | bool | `false` | Log the best merge order with relation aliases |

## Tuning hints

| Use case | Recommended config |
|---|---|
| **Fast default** (production) | `rollout=random`, `start_budget=200`, `phases=5`, `top_k=5`, `plan_shape=1` |
| **Decorrelated rollouts** | `rollout=luby`, `start_budget=200`, `phases=5`, `top_k=5`, `plan_shape=1` |

`patience=3` is a safe early-stopping value; set to `0` to disable.

## Known caveats

- The `EXPLAIN` block's `Iterations` shows the *last phase's* count
  because `mcts_root_restart` resets it.  Total iterations = sum across
  phases.  Same for cache hit ratio.
- A rollout that cannot complete returns `DBL_MAX`; `mcts_backpropagate`
  bumps `visits` along the path but skips the reward update, so such
  attempts widen exploration without corrupting plan-quality statistics.
