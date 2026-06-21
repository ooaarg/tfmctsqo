# mcts_extreme — MCTS join-order search for PostgreSQL

A PostgreSQL extension that replaces the standard join-order selection
with **Monte-Carlo Tree Search** (UCT-Extreme variant + Luby restart
phases).  Installed via PostgreSQL's `join_search_hook`, so loading
the extension is enough — no SQL DDL changes needed in your queries.

```
                join_search_hook
                       │
              core/mcts_extreme.c    <-- entry point: installs the hook
                       │
                  mcts/mcts.c         <-- UCT-Extreme search + Luby phases
```

## Rollouts

The MCTS rollout (the random playout used to estimate the value of a
partially-built join state) has two flavours, picked via
`mcts_extreme.rollout`:

| Value | Behaviour |
|---|---|
| `random` *(default)* | Pure uniform random rollout. PRNG seed is fixed for the search and varied across **phases** by splitmix64. |
| `luby` | Same uniform random rollout, but the PRNG is re-seeded **per rollout** with `seed XOR luby(rollout_idx)`, decorrelating consecutive rollouts within a single phase. |

`dp` / `geqo` rollouts (delegating to PostgreSQL's DP or GEQO) are
intentionally **not** included — they would pull in the full GEQO and
DP source trees as build dependencies of this extension.

## Build & install

`mcts_extreme` uses PGXS, so you only need a working `pg_config` on
`PATH`.  If you followed the top-level
[README — PostgreSQL installation](../README.md#postgresql-installation-making-psql-pg_ctl-etc-discoverable)
section and installed PostgreSQL into `source/tmp_install/`, that
already gives you the right `pg_config`.

From the repository root:

```bash
cd mcts_extreme
make
make install
```

This installs:

- `$(pg_config --pkglibdir)/mcts_extreme.so`
- `$(pg_config --sharedir)/extension/mcts_extreme.control`
- `$(pg_config --sharedir)/extension/mcts_extreme--1.0.sql`

To target a different PostgreSQL build, pass `PG_CONFIG`:

```bash
make PG_CONFIG=/path/to/other/pg_config install
```

## Loading the extension

`mcts_extreme` needs to be loaded into every session that should use
MCTS — either via `shared_preload_libraries` (cluster-wide) or
`session_preload_libraries` / `LOAD` (per-session).

### Cluster-wide (recommended)

Edit your `postgresql.conf`:

```
shared_preload_libraries = 'mcts_extreme'
```

…then restart the cluster:

```bash
pg_ctl -D ./pgdata restart
```

Register the extension once per database:

```sql
CREATE EXTENSION mcts_extreme;
```

### Per-session (quick try-out)

```sql
LOAD 'mcts_extreme';
CREATE EXTENSION mcts_extreme;       -- only the first time, per database
```

After this, every join-search call in the session goes through MCTS.

## Quick smoke test

```sql
LOAD 'mcts_extreme';
SET mcts_extreme.log_debug = on;     -- emit per-step WARNINGs

EXPLAIN ANALYZE
SELECT *
FROM t1 JOIN t2 USING (a)
        JOIN t3 USING (b)
        JOIN t4 USING (c)
        JOIN t5 USING (d);
```

You should see `WARNING: mcts_extreme:` lines in `psql` output, and
the final plan's join order is the one chosen by MCTS.

## GUC reference (high-impact)

| GUC | Type | Default | Notes |
|---|---|---|---|
| `mcts_extreme.enabled` | bool | `true` | Master kill switch. When `off`, the hook delegates immediately. |
| `mcts_extreme.min_relations` | int | `3` | Skip MCTS for queries with fewer base rels (DP is fine there). |
| `mcts_extreme.rollout` | enum | `random` | `random` or `luby` — see the rollout table above. |
| `mcts_extreme.exploration_constant` | real | `1.4` | Exploration constant `c`. |
| `mcts_extreme.gamma` | real | `0.5` | Exploration exponent `γ`. |
| `mcts_extreme.start_budget` | int | `100` | Iterations in phase 0; later phases multiply by `luby(j)`. |
| `mcts_extreme.phases` | int | `5` | Number of Luby restart phases. |
| `mcts_extreme.top_k` | int | `5` | Best-plan replay candidates per phase. |
| `mcts_extreme.expand_strategy` | enum | `cost` | Top-k expansion ranking: `cost`, `row`, `mixed_025`, `mixed_050`, or `selectivity`; `mix` is accepted as a legacy alias for `mixed_050`. |
| `mcts_extreme.rollouts_per_leaf` | int | `1` | Independent rollouts per expansion. |
| `mcts_extreme.full_budget` | bool | `false` | When `on`, continue after expansion is exhausted and spend the remaining iteration budget with classic UCB1. |
| `mcts_extreme.cache_enabled` | bool | `true` | Memoise rollout costs by clump bitmap. |
| `mcts_extreme.log_debug` | bool | `false` | Per-step `WARNING` logs (very chatty). |
| `mcts_extreme.log_steps` | bool | `false` | Per-iteration progress line. |
| `mcts_extreme.plan_shape` | int | `1` | Plan-shape K: `0` = bushy, `1` = linear (zig-zag), `>=2` = K-component bushy. |
| `mcts_extreme.reward_map` | enum | `neg_log` | Reward map φ: `neg_log`, `neg_cost`, or `norm_neg_log`. |
| `mcts_extreme.random_seed` | int | `0` (time-based) | Base PRNG seed; phases use splitmix64 over this. |

See [docs/mcts.md](docs/mcts.md) for the full GUC list and the
algorithm walk-through (selection / expansion / rollout / backprop).

## Disabling without uninstalling

If something goes wrong mid-benchmark, the safest off-switch is the
master GUC:

```sql
SET mcts_extreme.enabled = off;
```

The hook is still installed but every call falls straight through to
`standard_join_search` (or the previous hook if anything else is
chained).  To remove the hook entirely, drop the extension from
`shared_preload_libraries` and restart the cluster.
