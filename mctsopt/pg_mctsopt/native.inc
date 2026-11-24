/* Native PG helpers and settings derived from reference/mcts_extreme/mcts/mcts.c. */
 #include "postgres.h"

 #include <float.h>
 #include <limits.h>
 #include <math.h>
 #include <stdlib.h>

 #include "fmgr.h"
 #include "commands/explain.h"
 #include "commands/explain_format.h"
 #include "commands/explain_state.h"
 #include "lib/stringinfo.h"
 #include "miscadmin.h"
 #include "nodes/bitmapset.h"
 #include "nodes/pg_list.h"
 #include "optimizer/geqo.h"
 #include "optimizer/joininfo.h"
 #include "optimizer/pathnode.h"
 #include "optimizer/paths.h"
 #include "optimizer/planmain.h"
 #include "optimizer/planner.h"
 #include "portability/instr_time.h"
 #include "utils/builtins.h"
 #include "utils/guc.h"
 #include "utils/hsearch.h"
 #include "utils/memutils.h"
 #include "utils/timestamp.h"

 #include "mcts_trace.h"
 #include "mcts_internal.h"		/* Native adapter structs and helpers. */

 /* Conditional debug logging; emits WARNING only when pg_mctsopt.log_debug is on */
 #define mcts_debug_log(...) \
     do { if (mcts_log_debug) elog(WARNING, __VA_ARGS__); } while (0)


 /* ----------
  *  GUC variables
  * ----------
  */
 static bool pg_mctsopt_enabled = true;
 static int  pg_mctsopt_min_relations = 3;
 static double pg_mctsopt_exploration_constant = 1.4;
 static double pg_mctsopt_gamma = 0.5;
 static int  pg_mctsopt_random_seed = 0;
 int         mcts_start_budget = 480;
 static int  mcts_phases = 1;
 static int  mcts_top_k = 10;
 bool        mcts_cache_enabled = true;
 int         mcts_cache_size = 256;
int         mcts_cache_max_size = 50000;
static bool mcts_log_debug = false;
static bool mcts_log_steps = false;
static bool mcts_full_budget = false;
static int  mcts_plan_shape = 1;	/* plan-shape K: 0=bushy, 1=linear (zig-zag), >=2=K-component bushy */
 int         mcts_cost_eval_count = 0;
 static int  depth = 10;
static int  mcts_rollouts_per_leaf = 1;
int         mcts_patience = 0;

static bool mcts_force_left_tree = false;

typedef enum
{
    MCTS_EXTREME_EXPAND_COST,
    MCTS_EXTREME_EXPAND_ROW,
    MCTS_EXTREME_EXPAND_MIXED_025,
    MCTS_EXTREME_EXPAND_MIXED_050,
    MCTS_EXTREME_EXPAND_SELECTIVITY
} MctsExtremeExpandStrategy;

static const struct config_enum_entry pg_mctsopt_expand_strategy_options[] = {
    {"cost",        MCTS_EXTREME_EXPAND_COST,        false},
    {"row",         MCTS_EXTREME_EXPAND_ROW,         false},
    {"rows",        MCTS_EXTREME_EXPAND_ROW,         true},
    {"mixed_025",   MCTS_EXTREME_EXPAND_MIXED_025,   false},
    {"mix025",      MCTS_EXTREME_EXPAND_MIXED_025,   true},
    {"mixed_050",   MCTS_EXTREME_EXPAND_MIXED_050,   false},
    {"mix050",      MCTS_EXTREME_EXPAND_MIXED_050,   true},
    {"mix",         MCTS_EXTREME_EXPAND_MIXED_050,   true},
    {"selectivity", MCTS_EXTREME_EXPAND_SELECTIVITY, false},
    {NULL, 0, false}
};

static int  mcts_expand_strategy = MCTS_EXTREME_EXPAND_COST;

static const char *
mcts_expand_strategy_name(void)
{
    switch (mcts_expand_strategy)
    {
        case MCTS_EXTREME_EXPAND_ROW:
            return "row";
        case MCTS_EXTREME_EXPAND_MIXED_025:
            return "mixed_025";
        case MCTS_EXTREME_EXPAND_MIXED_050:
            return "mixed_050";
        case MCTS_EXTREME_EXPAND_SELECTIVITY:
            return "selectivity";
        case MCTS_EXTREME_EXPAND_COST:
        default:
            return "cost";
    }
}

 typedef enum
 {
     MCTS_EXTREME_ROLLOUT_RANDOM,        /* pure-uniform random rollout */
     MCTS_EXTREME_ROLLOUT_LUBY            /* random rollout, but the PRNG is re-seeded
                                           * per-rollout with seed XOR luby(rollout_idx)
                                           * to decorrelate consecutive rollouts.    */
 } MctsExtremeRollout;

 static const struct config_enum_entry pg_mctsopt_rollout_options[] = {
     {"random", MCTS_EXTREME_ROLLOUT_RANDOM, false},
     {"luby",   MCTS_EXTREME_ROLLOUT_LUBY,   false},
     {NULL, 0, false}
 };

 static int  mcts_rollout_mode = MCTS_EXTREME_ROLLOUT_RANDOM;

 /*
  * UCT Q-value aggregation mode (controls uct_extreme_score()).
  *   BEST    -- classic UCT-Extreme: each node's Q-value is the best reward
  *              ever observed below it.
  *   AVERAGE -- standard UCT: Q-value is the running mean of rollout rewards.
  *
  * Selected at runtime via pg_mctsopt.uct_aggregation.
  */
 typedef enum
 {
     MCTS_EXTREME_UCT_AGG_BEST,
     MCTS_EXTREME_UCT_AGG_AVERAGE
 } MctsExtremeUctAggregation;

 static const struct config_enum_entry pg_mctsopt_uct_aggregation_options[] = {
     {"best",    MCTS_EXTREME_UCT_AGG_BEST,    false},
     {"average", MCTS_EXTREME_UCT_AGG_AVERAGE, false},
     {"avg",     MCTS_EXTREME_UCT_AGG_AVERAGE, true},
     {NULL, 0, false}
 };

 static int  mcts_uct_aggregation = MCTS_EXTREME_UCT_AGG_BEST;

 /*
  * Reward map phi: converts a subtree-incumbent cost into the scalar
  * reward used by the UCT-Extreme selection score.  All three maps are
  * monotone in plan quality and differ only in how they normalize.
  *
  *  - neg_log (default):  -log(cost)
  *        Compares multiplicative cost differences; no envelope.
  *  - neg_cost:           -cost
  *        Raw negated cost; preserves the absolute scale of differences.
  *  - norm_neg_log:       (log(cost_max) - log(cost)) /
  *                        (log(cost_max) - log(cost_min))
  *        Min/max-normalized log reward over the observed cost envelope;
  *        result in [0, 1], clamped outside the envelope.
  */
 typedef enum
 {
     MCTS_EXTREME_REWARD_NEG_COST,
     MCTS_EXTREME_REWARD_NEG_LOG,
     MCTS_EXTREME_REWARD_NORM_NEG_LOG
 } MctsExtremeReward;

 static const struct config_enum_entry pg_mctsopt_reward_options[] = {
     {"neg_cost",     MCTS_EXTREME_REWARD_NEG_COST,     false},
     {"neg_log",      MCTS_EXTREME_REWARD_NEG_LOG,      false},
     {"norm_neg_log", MCTS_EXTREME_REWARD_NORM_NEG_LOG, false},
     {NULL, 0, false}
 };

 static int  mcts_reward_map = MCTS_EXTREME_REWARD_NEG_LOG;

 /*
  * Luby-pattern restart budgeting.  When true (default), phase j has budget
  * (mcts_luby_enabled ? mcts_start_budget * luby_value(j + 1) : mcts_start_budget) -- the classic Luby restart strategy.
  * When false, every phase gets a flat mcts_start_budget so we can isolate
  * the contribution of Luby restarts in approbation experiments.
  */
 static bool mcts_luby_enabled = true;

 /*
  * Statistics from the most recent run, reported via EXPLAIN and the debug
  * log. The type is declared in mcts_internal.h.
  */
 MctsLastStats mcts_last_stats = {false};


#include "mctsopt.h"
 typedef struct JoinCostCacheKey
 {
     uint64      left_bits;
     uint64      right_bits;
 } JoinCostCacheKey;

 typedef struct JoinCostCacheEntry
 {
     JoinCostCacheKey key;
     Cost        cost;
     RelOptInfo *joinrel;        /* cached joinrel pointer (NULL if from eval_context) */
 } JoinCostCacheEntry;

 static uint64
 relids_to_bits(Relids relids)
 {
     if (relids == NULL)
         return 0;
     return (uint64) relids->words[0];
 }

 /*
  * make_join_cache_key
  *	  Build a canonical (smaller-bits, larger-bits) key so (A,B) and
  *	  (B,A) hash to the same entry.
  */
 static JoinCostCacheKey
 make_join_cache_key(Relids left, Relids right)
 {
     JoinCostCacheKey k;
     uint64      lb = relids_to_bits(left);
     uint64      rb = relids_to_bits(right);

     if (lb <= rb)
     {
         k.left_bits = lb;
         k.right_bits = rb;
     }
     else
     {
         k.left_bits = rb;
         k.right_bits = lb;
     }
     return k;
 }

 /*
  * create_join_cost_cache
  *	  Allocate a new hash table for caching join costs in the given context.
  */
 HTAB *
 create_join_cost_cache(MemoryContext cxt, int nentries)
 {
     HASHCTL     ctl;

     memset(&ctl, 0, sizeof(ctl));
     ctl.keysize = sizeof(JoinCostCacheKey);
     ctl.entrysize = sizeof(JoinCostCacheEntry);
     ctl.hcxt = cxt;
     return hash_create("MCTS JoinCost Cache", nentries,
                        &ctl, HASH_ELEM | HASH_BLOBS | HASH_CONTEXT);
 }

 /*
  * join_cost_cache_lookup
  *	  Look up a cached join cost.  Returns true if found, filling in
  *	  the cost and optionally the cached RelOptInfo pointer.
  */
 static bool
 join_cost_cache_lookup(HTAB *cache, Relids left, Relids right,
                        Cost *cost, RelOptInfo **joinrel_out)
 {
     JoinCostCacheKey k = make_join_cache_key(left, right);
     JoinCostCacheEntry *entry;

     entry = (JoinCostCacheEntry *) hash_search(cache, &k, HASH_FIND, NULL);
     if (entry)
     {
         *cost = entry->cost;
         if (joinrel_out)
             *joinrel_out = entry->joinrel;
         return true;
     }
     return false;
 }

 /*
  * join_cost_cache_insert
  *	  Insert or update a join cost cache entry.  Keep the lower cost
  *	  if the entry already exists.  When the cache is at max capacity,
  *	  only update existing entries (no eviction policy).
  */
 static void
 join_cost_cache_insert(HTAB *cache, Relids left, Relids right,
                        Cost cost, RelOptInfo *joinrel)
 {
     JoinCostCacheKey k = make_join_cache_key(left, right);
     JoinCostCacheEntry *entry;
     bool        found;

     if (mcts_cache_max_size > 0 &&
         hash_get_num_entries(cache) >= (long) mcts_cache_max_size)
     {
         entry = (JoinCostCacheEntry *) hash_search(cache, &k, HASH_FIND, NULL);
         if (entry && cost < entry->cost)
         {
             entry->cost = cost;
             entry->joinrel = joinrel;
         }
         return;
     }

     entry = (JoinCostCacheEntry *) hash_search(cache, &k, HASH_ENTER, &found);
     if (!found || cost < entry->cost)
     {
         entry->cost = cost;
         entry->joinrel = joinrel;
     }
 }

 /*
  * mcts_finalize_joinrel_paths
  *    set_cheapest(jr) plus eager-aggregation grouped_rel maintenance.
  *
  * When the query has aggregates AND a GROUP BY (root->agg_clause_list and
  * root->group_expr_list both non-NIL, e.g. CEB "SELECT ... COUNT(*) ...
  * GROUP BY ..." queries), make_join_rel() internally builds a sibling
  * jr->grouped_rel via make_grouped_join_rel().  standard_join_search
  * materialises its paths in the per-level loop right after set_cheapest(rel);
  * MCTS owns its own search loop and must do the same, otherwise the next
  * make_join_rel() that consumes jr->grouped_rel as an input (via
  * make_grouped_join_rel -> populate_joinrel_with_paths -> add_paths_to_joinrel)
  * dereferences a NULL cheapest_total_path and segfaults.  JOB/JOBComplex
  * never hit this because they have no aggregates, so jr->grouped_rel stays
  * NULL throughout.
  *
  * Skip the topmost rel to match standard PG: grouping_planner generates the
  * final group/agg paths separately and reads jr->grouped_rel only when its
  * pathlist is non-empty, so leaving the top untouched is safe.  Vanilla PG18
  * does not expose grouped_rel/generate_grouped_paths in RelOptInfo, so this
  * maintenance is compiled only when the grouped-rel API is available.
  */
static void
mcts_finalize_joinrel_paths(PlannerInfo *root, RelOptInfo *jr)
{
     set_cheapest(jr);

#if defined(IS_GROUPED_REL)
     if (jr->grouped_rel != NULL && IS_GROUPED_REL(jr->grouped_rel) &&
         !bms_equal(jr->relids, root->all_query_rels))
     {
         RelOptInfo *grouped_rel = jr->grouped_rel;

         generate_grouped_paths(root, grouped_rel, jr);
         if (grouped_rel->pathlist != NIL)
             set_cheapest(grouped_rel);
     }
#else
     (void) root;
#endif
}

 /*
  * mcts_get_or_build_join
  *    Cache-aware joinrel construction.  Returns a usable joinrel for
  *    (outer, inner): a cached RelOptInfo on hit, otherwise a fresh
  *    make_join_rel result that gets inserted into the cache.
  *
  *    cache_hits_out: when non-NULL, bumped on every cache hit.
  *    store_joinrel:  when true, store the joinrel pointer (so future
  *                    hits can reuse it directly); when false, cache
  *                    only the cost.
  *
  *    Returns NULL when make_join_rel produced no joinrel (illegal pair).
  */
 RelOptInfo *
 mcts_get_or_build_join(PlannerInfo *root, HTAB *cache,
                        RelOptInfo *outer, RelOptInfo *inner,
                        int *cache_hits_out, bool store_joinrel)
 {
     Cost        cached_cost;
     RelOptInfo *cached_jr = NULL;
     RelOptInfo *jr;

     if (cache != NULL &&
         join_cost_cache_lookup(cache, outer->relids, inner->relids,
                                &cached_cost, &cached_jr) &&
         cached_jr != NULL && REL_HAS_CHEAPEST_PATH(cached_jr))
     {
         if (cache_hits_out)
             (*cache_hits_out)++;
         return cached_jr;
     }

     jr = make_join_rel(root, outer, inner);
     if (!jr)
         return NULL;

     mcts_cost_eval_count++;
     mcts_finalize_joinrel_paths(root, jr);

     if (cache && REL_HAS_CHEAPEST_PATH(jr))
         join_cost_cache_insert(cache, outer->relids, inner->relids,
                                REL_CHEAPEST_PATH(jr)->total_cost,
                                store_joinrel ? jr : NULL);
     return jr;
 }

 static bool
 desirable_join(PlannerInfo *root, RelOptInfo *outer_rel, RelOptInfo *inner_rel)
 {
     if (have_relevant_joinclause(root, outer_rel, inner_rel))
         return true;
     if (have_join_order_restriction(root, outer_rel, inner_rel))
         return true;
     return false;
 }

 /*
  * find_clump_by_relids
  *	  Find the clump in the list whose joinrel matches the given relids.
  *	  Returns NULL if no match, which should not happen during replay.
  */
 static MctsClump *
 find_clump_by_relids(List *clumps, Relids relids)
 {
     ListCell *lc;

     foreach(lc, clumps)
     {
         MctsClump *c = (MctsClump *) lfirst(lc);
         if (bms_equal(c->joinrel->relids, relids))
             return c;
     }
     return NULL;
 }

 /*
  * mcts_replay_best_order
  *	  Re-execute the best join order found during search.
  *
  * We replay the recorded merge steps using make_join_rel to build
  * proper join RelOptInfos that the rest of the planner can use.
  * This is necessary because the joins created during MCTS search
  * may have been discarded when we truncated join_rel_list between
  * phases.
  *
  * TODO: look up already-computed joins from the cost cache to
  * avoid redundant make_join_rel calls during replay.
  */
 RelOptInfo *
 mcts_replay_best_order(PlannerInfo *root, List *initial_rels, List *merge_order)
 {
     List     *clumps = NIL;
     ListCell *lc;

     /* Build initial clump list from base relations */
     foreach(lc, initial_rels)
     {
         MctsClump *c = (MctsClump *) palloc(sizeof(MctsClump));
         c->joinrel = (RelOptInfo *) lfirst(lc);
         c->size = 1;
         clumps = lappend(clumps, c);
     }

     foreach(lc, merge_order)
     {
         MctsMergeStep *step = (MctsMergeStep *) lfirst(lc);
         MctsClump *left = find_clump_by_relids(clumps, step->left);
         MctsClump *right = find_clump_by_relids(clumps, step->right);
         RelOptInfo *jr;

         jr = make_join_rel(root, left->joinrel, right->joinrel);

         mcts_cost_eval_count++;
         mcts_finalize_joinrel_paths(root, jr);

         left->joinrel = jr;
         left->size += right->size;
         clumps = list_delete_ptr(clumps, right);
         pfree(right);
     }

     return ((MctsClump *) linitial(clumps))->joinrel;
 }

