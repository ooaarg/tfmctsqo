CREATE EXTENSION pg_mctsopt;
LOAD 'pg_mctsopt';
SHOW pg_mctsopt.trace_search;
SELECT count(*) AS trace_before FROM mcts_search_trace();

CREATE TABLE t1 AS SELECT generate_series(1, 10) AS id;
CREATE TABLE t2 AS SELECT * FROM t1;
CREATE TABLE t3 AS SELECT * FROM t1;
CREATE TABLE t4 AS SELECT * FROM t1;
ANALYZE t1, t2, t3, t4;

SET statement_timeout = '30s';
SET pg_mctsopt.min_relations = 3;
SET pg_mctsopt.random_seed = 42;
SET pg_mctsopt.start_budget = 25;
SET pg_mctsopt.phases = 3;
SET pg_mctsopt.luby_enabled = true;
SET pg_mctsopt.full_budget = true;
SET pg_mctsopt.trace_search = on;

SELECT count(*) FROM t1, t2, t3, t4
WHERE t1.id=t2.id AND t2.id=t3.id AND t3.id=t4.id;

SELECT count(*) AS roots FROM mcts_search_trace() WHERE parent_id IS NULL;
SELECT count(*) > 0 AS chosen FROM mcts_search_trace() WHERE chosen;
SELECT count(*) AS invalid_dropped FROM mcts_search_trace() WHERE dropped AND visits <> 0;
SELECT phase, luby, budget FROM mcts_phase_trace() ORDER BY phase;
SELECT bool_and(iterations > 0 AND iterations <= budget) AS within_budget FROM mcts_phase_trace();
SELECT count(*) > 0 AS iterations_recorded FROM mcts_iter_trace();

SET pg_mctsopt.trace_search = off;
SELECT count(*) > 0 AS trace_retained FROM mcts_search_trace();
SET pg_mctsopt.trace_search = on;
SET pg_mctsopt.luby_enabled = false;
SET pg_mctsopt.reward_map = 'norm_neg_log';
SET pg_mctsopt.uct_aggregation = 'average';
SET pg_mctsopt.plan_shape = 0;
SELECT count(*) FROM t1, t2, t3, t4
WHERE t1.id=t2.id AND t2.id=t3.id AND t3.id=t4.id;
SELECT phase, luby, budget FROM mcts_phase_trace() ORDER BY phase;

-- A mixed join must finish and preserve the answer.
SELECT count(*) FROM t1
JOIN t2 ON t1.id=t2.id
LEFT JOIN t3 ON t2.id=t3.id
JOIN t4 ON t1.id=t4.id;

SET pg_mctsopt.start_budget = 0;
SELECT count(*) FROM t1, t2, t3, t4
WHERE t1.id=t2.id AND t2.id=t3.id AND t3.id=t4.id;
DROP TABLE t1, t2, t3, t4;
DROP EXTENSION pg_mctsopt;
