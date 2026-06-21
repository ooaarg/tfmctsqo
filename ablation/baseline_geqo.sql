LOAD 'mcts_extreme';
SET statement_timeout                    = '5min';
SET pg_mcts_extreme.enabled              = off;
SET mcts_extreme.enabled                 = off;
SET geqo                                 = on;
SET geqo_threshold                       = 2;
SET geqo_seed                            = :geqo_seed;
-- Without these, the planner slices the join list at 8 rels and GEQO
-- only ever runs on ≤8-rel sub-groups — the opposite of what GEQO is
-- designed for.  Raise both so GEQO actually does its genetic search
-- over the full join.
SET from_collapse_limit                  = 64;
SET join_collapse_limit                  = 64;

\timing on
\i :analyze_sql
