-- PostgreSQL's default DP/GEQO cutover, with collapse limits matched to MCTS.
-- Threshold applies to each join search problem, not a textual table count.
-- Stack's selected flat joins expose all 9/11/12 relations with these limits.
LOAD 'mcts_extreme';
SET statement_timeout                    = '5min';
SET pg_mcts_extreme.enabled              = off;
SET mcts_extreme.enabled                 = off;
SET geqo                                 = on;
SET geqo_threshold                       = 12;
-- PostgreSQL's default seed; repetitions measure the same baseline plan.
SET geqo_seed                            = 0;
SET from_collapse_limit                  = 64;
SET join_collapse_limit                  = 64;

\timing on
\i :analyze_sql
