\echo Use "CREATE EXTENSION mcts_extreme" to load this file. \quit

-- Search-decision trace: why MCTS chose this join order.
--
-- The decision tree along the winning order: for each step, the join action
-- MCTS picked and its sibling alternatives, with their UCT statistics.  The
-- chosen action (chosen = true) continues down the spine; alternatives are
-- leaves showing why they lost (lower best_reward / higher best_cost).
--
-- Reflects the most recently planned query for which
-- mcts_extreme.trace_search was on.
CREATE FUNCTION mcts_search_trace(
    OUT id          int,     -- node id (also row order)
    OUT parent_id   int,     -- parent decision node, NULL at the start state
    OUT depth       int,     -- join step number (0 = initial state)
    OUT "left"      text,    -- left side of the join action
    OUT "right"     text,    -- right side of the join action
    OUT visits      int,     -- UCT visit count of this action's subtree
    OUT best_reward float8,  -- best reward backpropagated under it
    OUT sum_reward  float8,  -- sum of rollout rewards (AVERAGE mode)
    OUT best_cost   float8,  -- best plan cost found in this subtree (kept nodes)
    OUT topk_cost   float8,  -- immediate join cost top-k ranked the action by
    OUT est_rows    float8,  -- predicted cardinality of this join (estimate)
    OUT chosen      bool,    -- on the winning order's spine? (Selection)
    OUT dropped     bool     -- evaluated then dropped by top-k (Expansion)?
)
RETURNS SETOF record
AS 'MODULE_PATHNAME', 'mcts_search_trace'
LANGUAGE C VOLATILE;

-- Per-phase (Luby restart) trace: one row per MCTS phase.  The iteration
-- budget changes per restart (start_budget * luby_value(phase)); this shows
-- the Luby multiplier, the budget, how many iterations actually ran, the best
-- cost found in that phase, and whether it improved the global best.
--
-- Reflects the most recently planned query for which
-- mcts_extreme.trace_search was on.
CREATE FUNCTION mcts_phase_trace(
    OUT phase           int,     -- 1-based phase number
    OUT luby            int,     -- Luby multiplier for this phase
    OUT budget          int,     -- iteration budget = start_budget * luby
    OUT iterations      int,     -- iterations actually run (< budget if exhausted)
    OUT phase_best_cost float8,  -- best plan cost found in this phase
    OUT improved        bool     -- did this phase improve the global best?
)
RETURNS SETOF record
AS 'MODULE_PATHNAME', 'mcts_phase_trace'
LANGUAGE C VOLATILE;

-- Per-iteration trace: one row per MCTS iteration (within a phase).  Shows when
-- an improvement was found (best cost so far) and how deep selection reached.
CREATE FUNCTION mcts_iter_trace(
    OUT phase            int,     -- 1-based phase number
    OUT iteration        int,     -- 1-based iteration within the phase
    OUT phase_best_cost  float8,  -- best cost so far within this phase
    OUT global_best_cost float8,  -- best cost so far across all phases
    OUT depth            int      -- join depth the selection reached this iter
)
RETURNS SETOF record
AS 'MODULE_PATHNAME', 'mcts_iter_trace'
LANGUAGE C VOLATILE;

-- Per-phase subplan trace: the joinrels (building blocks) of each phase's best
-- plan.  Compare relids across phases to see whether phases reuse the same
-- early clumps or each reaches the plan a different way.
CREATE FUNCTION mcts_phasesub_trace(
    OUT phase   int,      -- 1-based phase number
    OUT relids  text,     -- a joinrel on that phase's best plan (relation set)
    OUT size    int       -- number of base relations in the joinrel
)
RETURNS SETOF record
AS 'MODULE_PATHNAME', 'mcts_phasesub_trace'
LANGUAGE C VOLATILE;
