-- Basic regression tests for mcts_extreme.
--
-- Covers: extension load + GUC registration, the search-decision traces
-- (mcts_search_trace / mcts_phase_trace), and that outer joins are
-- disabled (the planner transparently falls back for them).
--
-- The tests avoid asserting anything stochastic (costs, search counts);
-- they check structural invariants that hold regardless of the MCTS seed.

CREATE EXTENSION mcts_extreme;

-- Loading the module installs join_search_hook and registers the GUCs.
LOAD 'mcts_extreme';

-- GUC defaults.
SHOW mcts_extreme.trace_search;

-- Before any traced query, the search trace is empty.
SELECT count(*) AS rows_before FROM mcts_search_trace();

-- Fixtures: four base relations to force a 4-way join search.
CREATE TABLE t1 (id int, a int);
CREATE TABLE t2 (id int, b int);
CREATE TABLE t3 (id int, c int);
CREATE TABLE t4 (id int, d int);
INSERT INTO t1 SELECT g, g % 50 FROM generate_series(1, 500) g;
INSERT INTO t2 SELECT g % 500 + 1, g % 50 FROM generate_series(1, 500) g;
INSERT INTO t3 SELECT g % 500 + 1, g % 50 FROM generate_series(1, 500) g;
INSERT INTO t4 SELECT g % 500 + 1, g % 50 FROM generate_series(1, 500) g;
ANALYZE t1, t2, t3, t4;

-- MCTS parameters needed for a deterministic run that populates the traces:
-- enable MCTS, allow >=3 base rels, fix the seed, quiet the debug log, keep
-- top-k and Luby at their defaults (start_budget=100, top_k=5), and turn on
-- the search trace before planning.
SET mcts_extreme.enabled = on;
SET mcts_extreme.min_relations = 3;
SET mcts_extreme.random_seed = 42;
SET mcts_extreme.start_budget = 100;
SET mcts_extreme.top_k = 5;
SET mcts_extreme.log_debug = off;
SET mcts_extreme.trace_search = on;

-- Plan a 4-way inner join through MCTS (result is deterministic; the plan
-- shape is not, so we only check the answer).
SELECT count(*) FROM t1, t2, t3, t4
 WHERE t1.id = t2.id AND t2.id = t3.id AND t3.id = t4.id;

-- Search-decision trace (why this join order): one start node, a non-empty
-- chosen spine, and dropped rows (if any) carry no UCT visits.
SELECT count(*) > 0 AS search_has_rows FROM mcts_search_trace();
SELECT count(*) AS search_roots FROM mcts_search_trace() WHERE parent_id IS NULL;
SELECT count(*) > 0 AS has_chosen FROM mcts_search_trace() WHERE chosen;
SELECT bool_and(visits = 0) AS dropped_have_no_visits
  FROM mcts_search_trace() WHERE dropped;

-- Per-phase Luby trace: budget = start_budget (100) * luby multiplier, and the
-- first phase always has luby = 1.
SELECT count(*) > 0 AS phase_has_rows FROM mcts_phase_trace();
SELECT bool_and(budget = 100 * luby) AS budget_matches_luby FROM mcts_phase_trace();
SELECT luby AS phase1_luby FROM mcts_phase_trace() WHERE phase = 1;

-- Turning tracing off leaves the buffer untouched (still queryable).
SET mcts_extreme.trace_search = off;
SELECT count(*) > 0 AS still_has_rows FROM mcts_search_trace();

-- Search-algorithm controls (the non-tree local-search baselines): SAIO
-- and iterative improvement are selected via mcts_extreme.search_algorithm
-- and run on a single linear kernel instead of the MCTS tree.  Their chosen
-- order is stochastic but the query result is not, so we only assert the
-- answer.  Defaults: search_algorithm = mcts, force_left_tree = off.
SHOW mcts_extreme.search_algorithm;
SHOW mcts_extreme.force_left_tree;
SET mcts_extreme.search_algorithm = 'saio';
SELECT count(*) FROM t1, t2, t3, t4
 WHERE t1.id = t2.id AND t2.id = t3.id AND t3.id = t4.id;
SET mcts_extreme.search_algorithm = 'iterative_improvement';
SELECT count(*) FROM t1, t2, t3, t4
 WHERE t1.id = t2.id AND t2.id = t3.id AND t3.id = t4.id;
RESET mcts_extreme.search_algorithm;

-- Outer joins are disabled: MCTS declines and the planner falls back, but
-- the query still plans and returns the correct answer.
SET mcts_extreme.trace_search = on;
SELECT count(*) FROM t1
  LEFT JOIN t2 ON t1.id = t2.id
  LEFT JOIN t3 ON t2.id = t3.id
  LEFT JOIN t4 ON t3.id = t4.id;

-- Regression guard: a query that MIXES inner and outer joins is also
-- declined up front.  This used to send a rollout into an infinite loop
-- once only outer-join pairs remained to merge (make_join_rel returns
-- NULL for them).  MCTS now falls back and the answer is still correct.
SELECT count(*) FROM t1
  JOIN t2 ON t1.id = t2.id
  LEFT JOIN t3 ON t2.id = t3.id
  JOIN t4 ON t1.id = t4.id;

DROP EXTENSION mcts_extreme;
