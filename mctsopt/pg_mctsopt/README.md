# pg_mctsopt

PostgreSQL adapter for `mctsopt_core`.
Target: PostgreSQL 19devel, commit `73d60ac385a93684f68297ae0ccb8f75bc6f23e1`.

```sh
make -C mctsopt/pg_mctsopt PG_CONFIG=/path/to/pg_config
```

Load the built module in a fresh connection:

```sql
LOAD '/absolute/path/to/pg_mctsopt.so';
SET pg_mctsopt.min_relations = 12;
SET pg_mctsopt.random_seed = 42;
SET pg_mctsopt.start_budget = 250;
SET pg_mctsopt.phases = 1;
SET pg_mctsopt.luby_enabled = false;
EXPLAIN (ANALYZE, BUFFERS, SUMMARY) SELECT ...;
```

Settings are registered in `controls.inc`. EXPLAIN includes search statistics.
The adapter uses native DP/GEQO when disabled or below `min_relations`.
Do not preload or load `mcts_extreme` in the same backend: the modules share C symbols
and install competing hooks. Use separate connections and an instance without either
module in `shared_preload_libraries`.

Greedy initialization is currently exposed by the DuckDB adapter only.

Install into a dedicated PostgreSQL test installation, then run `pg_regress`:

```sh
make -C mctsopt/pg_mctsopt install PG_CONFIG=/path/to/test/pg_config
make -C mctsopt/pg_mctsopt installcheck PG_CONFIG=/path/to/test/pg_config \
  PGHOST=/path/to/test/socket PGPORT=5432
```

`installcheck` creates and replaces `contrib_regression` on that test server.
For a temporary server, use
`EXTRA_REGRESS_OPTS="--temp-instance=/absolute/path/to/new-test-instance"`.

- `basic.sql`: extension loading, answers, public search/phase traces, Luby budgets,
  tracing controls, and invalid settings.
- `threshold.sql`: search at 14 relations, skipping at 8 with `min_relations=12`,
  stale statistics, and disabled search.
