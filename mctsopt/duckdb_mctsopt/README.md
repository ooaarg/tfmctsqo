# duckdb_mctsopt

DuckDB adapter for `mctsopt_core`, version 0.6.0.

## Build

Base DuckDB commit: `ead2df641ce60cfce32f4903a98a7f59c4491e1f`.
Apply `patches/duckdb-join-search-hook.patch`, then
`patches/duckdb-explicit-initialization.patch`, followed by
`patches/duckdb-candidate-cost-access.patch`. The last patch exposes cached local costs
and candidate diagnostics already used by the research host. It routes ComputeCost and the diagnostic recosts
through one cost assembly so they cannot drift apart; native costs are bit-identical and the memo is never rewritten.
Use the module with the host from the same build.

Run from `mctsopt/`:

```sh
MCTSOPT_SOURCE=$(pwd -P)
BUILD_ROOT=/absolute/path/to/a-new-mctsopt-build
mkdir "$BUILD_ROOT"
git clone https://github.com/duckdb/duckdb.git "$BUILD_ROOT/duckdb-src"
git -C "$BUILD_ROOT/duckdb-src" checkout --detach ead2df641ce60cfce32f4903a98a7f59c4491e1f
git -C "$BUILD_ROOT/duckdb-src" apply --check "$MCTSOPT_SOURCE/duckdb_mctsopt/patches/duckdb-join-search-hook.patch"
git -C "$BUILD_ROOT/duckdb-src" apply "$MCTSOPT_SOURCE/duckdb_mctsopt/patches/duckdb-join-search-hook.patch"
git -C "$BUILD_ROOT/duckdb-src" apply --check "$MCTSOPT_SOURCE/duckdb_mctsopt/patches/duckdb-explicit-initialization.patch"
git -C "$BUILD_ROOT/duckdb-src" apply "$MCTSOPT_SOURCE/duckdb_mctsopt/patches/duckdb-explicit-initialization.patch"
git -C "$BUILD_ROOT/duckdb-src" apply --check "$MCTSOPT_SOURCE/duckdb_mctsopt/patches/duckdb-candidate-cost-access.patch"
git -C "$BUILD_ROOT/duckdb-src" apply "$MCTSOPT_SOURCE/duckdb_mctsopt/patches/duckdb-candidate-cost-access.patch"
cmake -S "$BUILD_ROOT/duckdb-src" -B "$BUILD_ROOT/duckdb-build" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE='-O2 -DNDEBUG' \
  -DBUILD_UNITTESTS=OFF \
  -DENABLE_EXTENSION_AUTOINSTALL=OFF -DENABLE_EXTENSION_AUTOLOADING=OFF \
  -DGIT_COMMIT_HASH=6bf2ab7454 -DDUCKDB_EXPLICIT_VERSION=v2.0.0-dev84604 \
  -DDUCKDB_EXTENSION_CONFIGS="$MCTSOPT_SOURCE/duckdb_mctsopt/extension_config.cmake"
cmake --build "$BUILD_ROOT/duckdb-build" \
  --target duckdb duckdb_mctsopt_loadable_extension shell -j2
```

## Settings

All settings use the `duckdb_mctsopt_` prefix. Loading leaves MCTS disabled.
Search defaults use the selected pure JOB/JOB-Complex configuration: 250 iterations,
K=1, top-k 20, depth 10, exploration 0.05, one phase, rank-biased rollout, retained
candidate selection, and no Luby or greedy initialization. This is not a universal
winner: final CEB 14+ latency regressed by 32.93%. PostgreSQL defaults are unchanged.

```sql
LOAD '/absolute/path/to/duckdb_mctsopt.duckdb_extension';
SET duckdb_mctsopt_enabled=true;
SET duckdb_mctsopt_min_relations=12;
SET duckdb_mctsopt_seed=42;
SET duckdb_mctsopt_iterations=250;
SET duckdb_mctsopt_greedy_init=false;
```

| Suffix | Default | Meaning |
| --- | --- | --- |
| `enabled` | false | Enable the adapter |
| `scope` | all | Hook stage: before native search, or only approximate fallback; independent of the size threshold |
| `min_relations` | 12 | Apply independently to each native join region at or above this count; smaller regions use native |
| `greedy_init` | false | Explicitly initialize once before Monte Carlo phases |
| `greedy_init_mode` | native | `native` or deterministic `shape`; inactive when greedy_init=false |
| `seed` | 42 | Shared-core seed; **zero uses entropy** |
| `iterations` | 250 | Positive per-phase base budget; zero is rejected |
| `phases` | 1 | Restart phases |
| `luby_enabled` | false | Weight phase budgets using Luby |
| `patience` | 0 | Stop after phases without strict best-cost improvement; zero disables |
| `top_k` | 20 | Expansion candidate cap; zero keeps all |
| `depth` | 10 | Tree depth limit; zero is unlimited |
| `plan_shape` | 1 | Search shape: 0 bushy, 1 linear, >=2 K-component |
| `force_left_tree` | false | Keep the growing kernel left in linear search; no effect for K>=2 |
| `rollouts_per_leaf` | 1 | Rollouts per depth-limited leaf |
| `rollout` | rank_biased | `random`, `luby` reseeding, or stochastic `rank_biased` |
| `rollout_epsilon` | 0.25 | Uniform probability mass in rank-biased rollout; [0, 1] |
| `expand_strategy` | cost | `cost`, `row`, `mixed_025`, `mixed_050`, `selectivity` |
| `reward_map` | neg_log | `neg_cost`, `neg_log`, `norm_neg_log` |
| `uct_aggregation` | best | `best` or `average` |
| `exploration_constant` | 0.05 | UCT exploration constant |
| `gamma` | 0.5 | UCT Extreme exploration exponent |
| `full_budget` | false | Continue after frontier exhaustion using classic replay selection |
| `legacy_selection` | false | Restore historical premature-exhaustion behavior for attribution only |
| `legacy_normalization` | false | Restore historical cached normalized rewards for attribution only |
| `rank_all` | false | Rank unpruned actions to populate memo, keeping action order |
| `cache_connections` | true | Cache ordered graph lookups during one MCTS search |
| `pair_cache` | false | Skip unchanged native pair emissions; experimental, pure MCTS only |
| `retention_enabled` | true | Preserve and select alternative complete trees |
| `retention_max_group_fraction` | 1.0 | Maximum expected row-group fraction; 1.0 disables the check, archived checked cells use 0.5 |
| `retain_attempted_roots` | true | Include eligible roots rejected by memo cost |
| `retention_leaf_cap` | 4 | Maximum estimated filtered leaf cardinality |
| `retention_max_leaf_fraction` | 0.05 | Maximum filtered/base leaf estimate ratio |
| `retention_scan_evidence` | true | Require a local filter and known scan reduction |
| `retention_path_risk` | true | Include estimated build barriers in target paths |
| `retention_cost_guard` | 1.1 | Candidate/final MCTS memo cost ceiling; 0 disables |
| `retention_log_score` | true | Score as cost times (1 + log1p(target rows)) |
| `retention_lambda` | 0.01 | Additive penalty weight when log scoring is disabled |

Rank-biased rollout assigns 25% of probability uniformly and 75% by inverse cost
midrank. It evaluates every admitted action, so the same iteration budget can take
more planning time than random rollout. Tied costs have equal weight. Candidate
retention uses estimated filter-path risk, not measured execution time; its cost
guard cannot guarantee a runtime bound. The chosen frozen tree is replayed and checked.

To reproduce the previous default algorithm, set `rollout=random`,
`retention_enabled=false`, `plan_shape=2`, and `cache_connections=false`.
`random` and `luby` retain their original sampling semantics; retention is independent.
Rebuild the core and adapters together after updating public C structures.

`min_relations` counts native join-order regions after logical rewrites. Each
qualifying region gets its own budget. Smaller regions use the native optimizer.
`patience` counts phases without cost improvement. Cost is not execution latency.

Pure mode clears non-leaf memo entries and supplies no initial-plan callback.
Greedy initialization runs once before search. `native` uses DuckDB's greedy search;
`shape` uses deterministic, shape-aware minimum-cost merges. A failed search or
initializer records fallback and invokes native search.

`plan_shape` constrains exploration. DuckDB memo reconstruction can combine explored
subplans into a different final shape. Native search is selected by disabling the
extension; the native approximate-search threshold remains unchanged.

## Search caches

Both caches live in the adapter and use the existing host contract. No additional
DuckDB patch is required. Enable them independently for comparisons:

```sql
SET duckdb_mctsopt_cache_connections=true;
SET duckdb_mctsopt_pair_cache=true;
```

Connection entries retain graph edges for an ordered pair of relation sets. Pair
entries retain stable relation-set handles, not replaceable memo-node pointers.
A pair hit requires unchanged input costs/cardinalities, output cost and split,
the cardinality used by the native tie-breaker, and connection identities.
Costs are still read from the current native memo.

Each cache holds at most 50,000 entries and is released after one search, including
replay. Both are cleared before an empty-connection emission, which may add graph
edges. Failed emissions and generated cross-product pairs are not memoized by
the pair cache. Pair caching rejects greedy initialization; connection caching
applies to the Monte Carlo search, not the initializer.

Diagnostics report connection and pair hits/misses, pair entries remaining at
search end, and conservative graph invalidations. These are planner caches,
independent of database buffer or OS page-cache policy. Connection caching defaults
to true; pair caching remains experimental and defaults to false.

## Diagnostics and checks

Save the target profile before querying diagnostics:

```sql
PRAGMA disable_profiling;
SET duckdb_mctsopt_enabled=false;
SELECT duckdb_mctsopt_diagnostics(), duckdb_mctsopt_last_cost(),
       duckdb_mctsopt_core_iterations(), duckdb_mctsopt_fallback_count();
```

Diagnostics include selected/skipped regions, seed, iterations, initialization,
search time, stop reason, cost, and fallback. Counters accumulate per connection.
Use a fresh connection for each measurement.

SQLLogicTest suites:

- `rollout.test`: policy defaults, seeded replay, retention verification, invalid epsilon,
  and recovery.
- `joins.test`: seven join cases with native planning and three initialization modes,
  across three tree shapes.
- `initialization.test`: initial incumbent, search activity, and same-seed replay.
- `threshold.test`: region threshold and disabled search.
- `budgets.test`: phase budgets with and without Luby, reward and aggregation modes.
- `fallback.test`: failed search/initialization, native fallback, and warning records.
- `errors.test`: invalid budget, impossible joins, cancellation, and recovery.
- `cache.test`: cache combinations across seeds, budgets and tree shapes; result
  and non-timing diagnostic equivalence for inner, left, cross and hyperedge joins.
Run from `mctsopt/`, using the matching host build above:

```sh
cmake -S "$BUILD_ROOT/duckdb-src" -B "$BUILD_ROOT/duckdb-build" \
  -DBUILD_UNITTESTS=ON -DENABLE_UNITTEST_CPP_TESTS=OFF \
  -DDUCKDB_EXTENSION_CONFIGS="$MCTSOPT_SOURCE/duckdb_mctsopt/test/extension_config.cmake"
cmake --build "$BUILD_ROOT/duckdb-build" \
  --target unittest duckdb_local_extension_repo -j2
"$BUILD_ROOT/duckdb-build/test/unittest" 'test/sql/*' \
  --test-dir "$MCTSOPT_SOURCE/duckdb_mctsopt" \
  --temp-dir-root "$BUILD_ROOT/sql-tests" \
  --require duckdb_mctsopt --require json
```

The test configuration adds JSON support for reading diagnostics. Required
extensions must be available: missing extensions fail instead of skipping tests.

See [the host contract](../docs/host-contract.md) and [DuckDB attribution](LICENSE.duckdb).
