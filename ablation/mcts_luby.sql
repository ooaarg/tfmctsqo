LOAD 'mcts_extreme';
SET statement_timeout                    = '5min';
-- Raise collapse limits so MCTS sees the full join list, not ≤8-rel
-- sub-groups (PG default).  Matched in baseline_dp.sql / baseline_geqo.sql.
SET from_collapse_limit                  = 64;
SET join_collapse_limit                  = 64;
SET pg_mcts_extreme.enabled              = off;
SET mcts_extreme.enabled                 = on;
SET mcts_extreme.search_algorithm        = 'mcts';
SET mcts_extreme.log_debug               = off;
SET mcts_extreme.log_steps               = off;
SET mcts_extreme.random_seed             = :seed;
SET mcts_extreme.rollout                 = 'random';
SET mcts_extreme.plan_shape              = 1;
SET mcts_extreme.force_left_tree         = off;
SET mcts_extreme.depth                   = 4;
SET mcts_extreme.start_budget            = 20;
SET mcts_extreme.phases                  = 8;
SET mcts_extreme.patience                = 0;
SET mcts_extreme.rollouts_per_leaf       = 1;
SET mcts_extreme.full_budget             = off;
SET mcts_extreme.top_k                   = 5;
SET mcts_extreme.expand_strategy         = 'cost';
SET mcts_extreme.exploration_constant    = 1.4;
SET mcts_extreme.saio_max_iterations     = 100000;
SET mcts_extreme.reward_map              = :'reward';
SET mcts_extreme.uct_aggregation         = :'agg';
SET geqo                                 = off;

\timing on
\i :analyze_sql
