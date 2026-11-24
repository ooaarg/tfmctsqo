-- Postgres dynamic-programming baseline (no MCTS, no GEQO).
--
-- Same EXPLAIN-include shape as the MCTS configs.  `:reward`, `:agg`,
-- and `:seed` are still injected by run.sh / sweep.py but ignored here
-- since DP is deterministic.  geqo_threshold=64 forces DP even on the
-- biggest JOB queries (default cutover is 12).
--
-- from_collapse_limit / join_collapse_limit default to 8 in PostgreSQL,
-- which makes the planner slice the join list into ≤8-rel sub-groups
-- joined in syntactic order.  DP then only ever runs on ≤8 rels — never
-- the full query — masking the exponential blow-up DP is supposed to
-- exhibit on big joins.  Raise both so DP sees the whole join.
LOAD 'mcts_extreme';
SET statement_timeout                    = '5min';
SET pg_mcts_extreme.enabled              = off;
SET mcts_extreme.enabled                 = off;
SET geqo                                 = off;
SET geqo_threshold                       = 64;
SET from_collapse_limit                  = 64;
SET join_collapse_limit                  = 64;

\timing on
\i :analyze_sql
