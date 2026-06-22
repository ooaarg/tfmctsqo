/*-------------------------------------------------------------------------
 *
 * mcts.c
 *	  UCT-Extreme Monte Carlo Tree Search for join-order optimization.
 *
 *	  Implements the MCTS-Extreme search: a training-free join optimizer
 *	  that reuses PostgreSQL's own cost model.  Each iteration runs the
 *	  four MCTS steps (selection, expansion, rollout, backpropagation);
 *	  selection uses the UCT-Extreme rule, which tracks the best
 *	  (incumbent) cost in each subtree rather than the average.  The
 *	  chosen join tree is replayed through make_join_rel so the rest of
 *	  the planner can use it.  EXPLAIN output is provided via
 *	  explain_per_plan_hook.
 *
 *-------------------------------------------------------------------------
 */
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
 #include "mcts_internal.h"		/* shared structs/helpers + local_search.c */

 /* Conditional debug logging; emits WARNING only when mcts_extreme.log_debug is on */
 #define mcts_debug_log(...) \
     do { if (mcts_log_debug) elog(WARNING, __VA_ARGS__); } while (0)


 /* ----------
  *  GUC variables
  * ----------
  */
 static bool mcts_extreme_enabled = true;
 static int  mcts_extreme_min_relations = 3;
 static double mcts_extreme_exploration_constant = 1.4;
 static double mcts_extreme_gamma = 0.5;
 static int  mcts_extreme_random_seed = 0;
 int         mcts_start_budget = 480;	/* shared with local_search.c */
 static int  mcts_phases = 1;
 static int  mcts_top_k = 10;
 bool        mcts_cache_enabled = true;	/* shared with local_search.c */
 int         mcts_cache_size = 256;		/* shared with local_search.c */
int         mcts_cache_max_size = 50000;	/* shared with local_search.c */
static bool mcts_log_debug = false;
static bool mcts_log_steps = false;
static bool mcts_full_budget = false;
static int  mcts_plan_shape = 1;	/* plan-shape K: 0=bushy, 1=linear (zig-zag), >=2=K-component bushy */
 int         mcts_cost_eval_count = 0;	/* shared with local_search.c */
 static int  depth = 10;
static int  mcts_rollouts_per_leaf = 1;
int         mcts_patience = 0;			/* shared with local_search.c */

/*
 * search_algorithm / force_left_tree / saio_* and the local-search baselines
 * (SAIO, iterative improvement) live in local_search.c; see mcts_internal.h.
 */

typedef enum
{
    MCTS_EXTREME_EXPAND_COST,
    MCTS_EXTREME_EXPAND_ROW,
    MCTS_EXTREME_EXPAND_MIXED_025,
    MCTS_EXTREME_EXPAND_MIXED_050,
    MCTS_EXTREME_EXPAND_SELECTIVITY
} MctsExtremeExpandStrategy;

static const struct config_enum_entry mcts_extreme_expand_strategy_options[] = {
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

 static const struct config_enum_entry mcts_extreme_rollout_options[] = {
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
  * Selected at runtime via mcts_extreme.uct_aggregation.
  */
 typedef enum
 {
     MCTS_EXTREME_UCT_AGG_BEST,
     MCTS_EXTREME_UCT_AGG_AVERAGE
 } MctsExtremeUctAggregation;

 static const struct config_enum_entry mcts_extreme_uct_aggregation_options[] = {
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

 static const struct config_enum_entry mcts_extreme_reward_options[] = {
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
  * log.  The type lives in mcts_internal.h so local_search.c can fill it too.
  */
 MctsLastStats mcts_last_stats = {false};

 /* ----------
  *  Data structures
  * ----------
  */

 /* MctsClump, MctsAction, MctsMergeStep and MctsContext are in mcts_internal.h. */

 /*
  * MctsDroppedAction -- a candidate join that top-k filtering evaluated (built
  * its joinrel to rank it) but then dropped because it was not among the k
  * cheapest.  Recorded for the search trace only (mcts_extreme.trace_search),
  * so the "why this order" view can show that these joins were considered, not
  * pruned outright.
  */
 typedef struct MctsDroppedAction
 {
     Relids      left;
     Relids      right;
     Cost        cost;			/* the top-k ranking cost */
     double      rows;			/* estimated joinrel cardinality */
 } MctsDroppedAction;

 /*
  * MctsNode -- one node in the MCTS search tree.  A node represents a
  * partial join state (a list of clumps); a root-to-leaf path is a join
  * order and a terminal node is a complete plan.
  *
  * Each node tracks the UCT-Extreme statistics used by selection.  Unlike
  * classical UCT, which averages rollout outcomes, we keep the *best* cost
  * seen anywhere in the node's subtree -- its subtree incumbent -- because
  * the objective is the single best plan, not the average plan.
  */
 typedef struct MctsNode
 {
     struct MctsNode *parent;
     List       *children;			/* List of MctsNode* */
     List       *clumps;				/* List of MctsClump*, current state */
     List       *untried_actions;	/* List of MctsAction* */
     List       *topk_dropped;		/* List of MctsDroppedAction* (trace only) */
     Cost        immediate_cost;		/* top-k rank cost of the creating join (trace) */
     double      est_rows;			/* estimated cardinality of the creating join */
     int         visits;				/* n_j: times this node was on a backprop path */
     double      best_reward;		/* reward map applied to the subtree incumbent */
     double      sum_reward;			/* sum of rollout rewards (for mean-UCT ablation) */
     Cost        best_cost_in_subtree;	/* subtree incumbent: min rollout cost below here */
     bool        terminal;
     Bitmapset  *all_query_relids;
     Relids      creation_left;		/* relids of left side that created this node */
     Relids      creation_right;		/* relids of right side that created this node */
     int         node_depth;
     bool        exhausted;			/* all descendants fully explored */
 } MctsNode;

 /*
  * (MctsMergeStep is defined in mcts_internal.h.)
  */

 /*
  * JoinCostCacheKey / JoinCostCacheEntry -- hash table entries that cache
  * the cost (and resulting joinrel) of joining two clumps, so the same
  * partial plan is not re-evaluated across rollouts.
  *
  * Keys are canonicalized so (A join B) and (B join A) map to the same
  * entry.  The key uses only the first bitmap word of each side's relids,
  * which is collision-prone for queries with >64 baserels but works for
  * everything in practice; correctness does not depend on key uniqueness
  * since the cache stores the actual RelOptInfo and we re-use it directly.
  */
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

 /* MctsContext is defined in mcts_internal.h (shared with local_search.c). */

/*
 * Derive ctx->seedbuf from (mcts_extreme_random_seed, c0, c1, c2) via splitmix64.
 * Used so that every PRNG reseed in the MCTS loop is reproducible given the
 * user-supplied seed.  When mcts_extreme_random_seed == 0, falls back to
 * wall-clock so unseeded runs still get a per-call source of entropy.
 *
 * Calling with c1=c2=0 reproduces the historic per-phase mix bit-for-bit,
 * so existing fixed-seed runs reproduce exactly.
 */
void
mcts_mix_seedbuf(MctsContext *ctx, uint64 c0, uint64 c1, uint64 c2)
{
    uint64 base;
    uint64 mix;

    if (mcts_extreme_random_seed != 0)
        base = (uint64) mcts_extreme_random_seed;
    else
        base = (uint64) GetCurrentTimestamp();

    mix = base * UINT64CONST(0x9e3779b97f4a7c15)
        + c0 * UINT64CONST(0xbf58476d1ce4e5b9)
        + c1 * UINT64CONST(0x94d049bb133111eb)
        + c2 * UINT64CONST(0xc6bc279692b5c323);
    mix ^= mix >> 30;
    mix *= UINT64CONST(0xbf58476d1ce4e5b9);
    mix ^= mix >> 27;
    mix *= UINT64CONST(0x94d049bb133111eb);
    mix ^= mix >> 31;

    ctx->seedbuf[0] = (unsigned short) (mix & 0xFFFF);
    ctx->seedbuf[1] = (unsigned short) ((mix >> 16) & 0xFFFF);
    ctx->seedbuf[2] = (unsigned short) ((mix >> 32) & 0xFFFF);
}

 /* ----------
  *  Forward declarations
  * ----------
  */
static double cost_to_reward(MctsContext *ctx, Cost total_cost);
 static bool desirable_join(PlannerInfo *root, RelOptInfo *outer_rel, RelOptInfo *inner_rel);
 static Cost mcts_eval_candidate(PlannerInfo *root, List *clumps,
                                 RelOptInfo **best_rel_out, HTAB *join_cost_cache);
 static List *mcts_enumerate_legal_actions(MctsContext *ctx, List *clumps);
static MctsNode *mcts_select_uct(MctsContext *ctx, MctsNode *node);
static MctsNode *mcts_select_classic_ucb(MctsContext *ctx, MctsNode *node);
 static MctsNode *mcts_expand(MctsContext *ctx, MctsNode *leaf);
 static Cost mcts_rollout_bushy(MctsContext *ctx, MctsNode *node, RelOptInfo **out_rel,
                                List **out_merge_order);
 static Cost mcts_rollout_linear(MctsContext *ctx, MctsNode *node, RelOptInfo **out_rel,
                                   List **out_merge_order);
static void mcts_backpropagate(MctsContext *ctx, MctsNode *node, Cost rollout_cost);
 static MctsNode *mcts_build_root(MctsContext *ctx);
 static List *get_node_merge_path(MctsNode *node, MemoryContext cxt);

 /* ----------
  *  Join cost cache helpers
  * ----------
  */

 /*
  * relids_to_bits
  *	  Fold a Relids to a 64-bit fingerprint.
  *
  * We use only the first bitmap word.  This is collision-prone above 64
  * baserels but cache correctness does not depend on key uniqueness (we
  * store the actual RelOptInfo and reuse it directly), and JOB-scale
  * workloads never exceed 64.
  */
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

 /* ----------
  *  Utility helpers
  * ----------
  */

 /*
  * mcts_clone_clumps
  *    Shallow-clone a list of MctsClump entries.  The joinrel pointers are
  *    shared with the source list; only the wrapper structs are duplicated
  *    so the caller can mutate the list independently.
  */
 static List *
 mcts_clone_clumps(List *clumps)
 {
     List     *out = NIL;
     ListCell *lc;

     foreach(lc, clumps)
     {
         MctsClump *src = (MctsClump *) lfirst(lc);
         MctsClump *dup = (MctsClump *) palloc(sizeof(MctsClump));

         dup->joinrel = src->joinrel;
         dup->size = src->size;
         out = lappend(out, dup);
     }
     return out;
 }

 /*
  * luby_value
  *	  Compute the i-th element of the Luby sequence (1, 1, 2, 1, 1, 2, 4, ...).
  *
  * We use this to set the iteration budget for each restart phase,
  * providing a universal restart strategy with provably optimal
  * expected cost for Las Vegas algorithms.
  */
 static int
 luby_value(int i)
 {
     int size = 1;
     int seq = 1;

     while (size < i)
     {
         size = 2 * size + 1;
         seq *= 2;
     }
     while (size > i)
     {
         seq /= 2;
         size /= 2;
         if (i > size)
             i -= size;
     }
     return seq;
 }

 /* ----------
  *  Cost / reward conversion
  * ----------
  */

 /*
  * cost_to_reward
  *	  Apply the reward map phi: turn a plan cost into the scalar reward
  *	  that UCT-Extreme maximizes during selection.  A lower cost must
  *	  always map to a higher reward, so every map is monotone decreasing
  *	  in cost.  The active map is chosen by mcts_extreme.reward_map:
  *
  *   neg_log (default):  -log(cost)
  *       Unbounded; equal multiplicative cost ratios map to equal reward
  *       gaps, which suits plan costs that span many orders of magnitude.
  *
  *   neg_cost:           -cost
  *       Raw negated cost; preserves absolute cost differences.
  *
  *   norm_neg_log:       (log(cost_max) - log(cost)) /
  *                       (log(cost_max) - log(cost_min))
  *       Same shape as neg_log but rescaled into [0, 1] using the
  *       running cost envelope [cost_min, cost_max] (maintained by
  *       mcts_backpropagate); costs outside the envelope clamp to 0/1.
  *
  * Degenerate inputs (non-positive or DBL_MAX cost) are mapped to the
  * map's best/worst value.  mcts_backpropagate filters DBL_MAX rollouts
  * before reaching this function on the hot path.
  */
 static double
cost_to_reward(MctsContext *ctx, Cost total_cost)
 {
    switch (mcts_reward_map)
    {
        case MCTS_EXTREME_REWARD_NEG_COST:
            if (total_cost >= (Cost) DBL_MAX)
                return -DBL_MAX;
            return -(double) total_cost;

        case MCTS_EXTREME_REWARD_NORM_NEG_LOG:
        {
            double  c;
            double  lo = ctx->cost_min;
            double  hi = ctx->cost_max;
            double  log_lo,
                    log_hi,
                    log_c;

            if (total_cost <= 0)
                return 1.0;
            if (total_cost >= (Cost) DBL_MAX)
                return 0.0;

            /*
             * Before we have any finite observation lo/hi are still at
             * their sentinels (lo=+inf, hi<=0).  Treat the first plan
             * we score as the (currently sole) best one.
             */
            if (!isfinite(lo) || lo <= 0 || !isfinite(hi) || hi <= 0)
                return 1.0;

            c = (double) total_cost;
            if (c <= lo)
                return 1.0;
            if (c >= hi)
                return 0.0;

            log_lo = log(lo);
            log_hi = log(hi);
            log_c = log(c);
            if (log_hi - log_lo <= 0)
                return 1.0;
            return (log_hi - log_c) / (log_hi - log_lo);
        }

        case MCTS_EXTREME_REWARD_NEG_LOG:
        default:
            if (total_cost <= 0)
                return -log(DBL_MIN);
            if (total_cost >= (Cost) DBL_MAX)
                return -log(DBL_MAX);
            return -log((double) total_cost);
    }
 }

 /*
  * desirable_join
  *	  Return true if a join between these two rels is "interesting" --
  *	  either there is a relevant join clause or a join order restriction.
  *
  * Non-desirable joins are used only as a fallback when no desirable
  * joins exist, preventing Cartesian products when possible.
  */
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

 /*
  * mcts_eval_candidate
  *	  Greedily join a list of clumps and return the total cost.
  *
  * This performs a simple greedy merge: repeatedly join the first
  * joinable pair found.  We use the join cost cache when available
  * to avoid calling make_join_rel for previously seen pairs.
  *
  * We save and restore root->join_rel_list and join_rel_hash so
  * that speculative joins created here do not pollute the planner's
  * permanent state.
  */
 static Cost
 mcts_eval_candidate(PlannerInfo *root, List *clumps, RelOptInfo **best_rel_out,
                     HTAB *join_cost_cache)
 {
     MemoryContext oldcxt;
     int         saved_len;
     struct HTAB *save_hash;
     RelOptInfo *joinrel;
     Cost        total_cost;

     *best_rel_out = NULL;
     mcts_debug_log("mcts_extreme.mcts: mcts_eval_candidate start clumps=%d",
                    list_length(clumps));
     if (list_length(clumps) == 0)
     {
         mcts_debug_log("mcts_extreme.mcts: mcts_eval_candidate early return (0 clumps)");
         return (Cost) DBL_MAX;
     }

     if (list_length(clumps) == 1)
     {
         MctsClump *c = (MctsClump *) linitial(clumps);
         *best_rel_out = c->joinrel;
         if (c->joinrel && REL_HAS_CHEAPEST_PATH(c->joinrel))
         {
             mcts_debug_log("mcts_extreme.mcts: mcts_eval_candidate early return (1 clump) cost=%.2f",
                            (double) REL_CHEAPEST_PATH(c->joinrel)->total_cost);
             return REL_CHEAPEST_PATH(c->joinrel)->total_cost;
         }
         return (Cost) DBL_MAX;
     }

     oldcxt = MemoryContextSwitchTo(root->planner_cxt);
     saved_len = list_length(root->join_rel_list);
     save_hash = root->join_rel_hash;
     root->join_rel_hash = NULL;

     while (list_length(clumps) > 1)
     {
         int na = list_length(clumps);
         int ia, ib;
         bool merged = false;

         for (ia = 0; ia < na && !merged; ia++)
         {
             MctsClump *ca = (MctsClump *) list_nth(clumps, ia);

             for (ib = ia + 1; ib < na && !merged; ib++)
             {
                 MctsClump  *cb = (MctsClump *) list_nth(clumps, ib);
                 RelOptInfo *jr;

                 jr = mcts_get_or_build_join(root, join_cost_cache,
                                             ca->joinrel, cb->joinrel,
                                             NULL, true);
                 if (jr == NULL)
                     continue;

                 ca->joinrel = jr;
                 ca->size += cb->size;
                 clumps = list_delete_nth_cell(clumps, ib);
                 pfree(cb);
                 merged = true;
             }
         }
         if (!merged)
             break;
     }

     if (list_length(clumps) != 1)
     {
         mcts_debug_log("mcts_extreme.mcts: mcts_eval_candidate failed merge (clumps=%d)",
                        list_length(clumps));
         root->join_rel_list = list_truncate(root->join_rel_list, saved_len);
         root->join_rel_hash = save_hash;
         MemoryContextSwitchTo(oldcxt);
         return (Cost) DBL_MAX;
     }

     joinrel = ((MctsClump *) linitial(clumps))->joinrel;
     *best_rel_out = joinrel;
     total_cost = REL_HAS_CHEAPEST_PATH(joinrel)
         ? REL_CHEAPEST_PATH(joinrel)->total_cost : (Cost) DBL_MAX;

     mcts_debug_log("mcts_extreme.mcts: mcts_eval_candidate done cost=%.2f", (double) total_cost);

     root->join_rel_list = list_truncate(root->join_rel_list, saved_len);
     root->join_rel_hash = save_hash;
     MemoryContextSwitchTo(oldcxt);
     return total_cost;
 }



 /* ----------
  *  Top-k filtering for actions
  * ----------
  */
typedef struct ActionCost
{
    MctsAction *action;
    double      score;
    double      rows;			/* estimated joinrel cardinality (trace) */
} ActionCost;

 /*
  * filter_actions_top_k
  *	  Prune the action list to only the k cheapest joins.
  *
  * We evaluate every candidate action's cost (using the cache where
  * possible), maintain a sorted array of the top-k cheapest, and
  * discard the rest.  This reduces branching factor at the expense
  * of potentially missing good joins that have high intermediate cost.
  */
/*
 * record_topk_dropped
 *    For the search trace: remember a candidate join that top-k evaluated
 *    (costed) and then discarded.  No-op unless mcts_extreme.trace_search.
 */
static void
record_topk_dropped(MctsContext *ctx, List *clumps, MctsAction *a, Cost cost,
                    double rows)
{
    MctsClump  *cl;
    MctsClump  *cr;
    MctsDroppedAction *d;
    MemoryContext oldcxt;

    if (!mcts_trace_search)
        return;

    cl = (MctsClump *) list_nth(clumps, a->left_idx);
    cr = (MctsClump *) list_nth(clumps, a->right_idx);
    oldcxt = MemoryContextSwitchTo(ctx->run_context);
    d = (MctsDroppedAction *) palloc(sizeof(MctsDroppedAction));
    d->left = bms_copy(cl->joinrel->relids);
    d->right = bms_copy(cr->joinrel->relids);
    d->cost = cost;
    d->rows = rows;
    ctx->last_topk_dropped = lappend(ctx->last_topk_dropped, d);
    MemoryContextSwitchTo(oldcxt);
}

static List *
filter_actions_top_k(MctsContext *ctx, List *clumps, List *actions, int k)
{
     int         n = list_length(actions);
     ActionCost *top;
     int         top_len = 0;
     ListCell   *lc;
     int         i;
     List       *result = NIL;
     int         saved_len;
     struct HTAB *save_hash;

     if (k <= 0 || n <= k)
         return actions;

     saved_len = list_length(ctx->root->join_rel_list);
     save_hash = ctx->root->join_rel_hash;
     ctx->root->join_rel_hash = NULL;

     top = (ActionCost *) palloc(sizeof(ActionCost) * k);

     foreach(lc, actions)
     {
         MctsAction *a = (MctsAction *) lfirst(lc);
        MctsClump  *cl = (MctsClump *) list_nth(clumps, a->left_idx);
        MctsClump  *cr = (MctsClump *) list_nth(clumps, a->right_idx);
        Cost        c = (Cost) DBL_MAX;
        double      rows = DBL_MAX;
        double      score = DBL_MAX;
        double      left_rows = Max(cl->joinrel->rows, 1.0);
        double      right_rows = Max(cr->joinrel->rows, 1.0);
        RelOptInfo *jr = NULL;

        /*
         * Top-k filtering consumes a cheap ranking score.  In cost mode the
         * cached cost is enough.  In row/mix modes we also need joinrel->rows;
         * cached entries normally carry the joinrel pointer, but if they do
         * not, rebuild this candidate just like a cache miss.
         */
        if (ctx->join_cost_cache &&
             join_cost_cache_lookup(ctx->join_cost_cache,
                                    cl->joinrel->relids,
                                    cr->joinrel->relids,
                                    &c, &jr) &&
            (mcts_expand_strategy == MCTS_EXTREME_EXPAND_COST || jr != NULL))
        {
             ctx->cache_hits++;
        }
        else
        {
             jr = make_join_rel(ctx->root, cl->joinrel, cr->joinrel);
             if (jr)
             {
                 mcts_cost_eval_count++;
                 mcts_finalize_joinrel_paths(ctx->root, jr);
                 if (REL_HAS_CHEAPEST_PATH(jr))
                 {
                     c = REL_CHEAPEST_PATH(jr)->total_cost;
                     if (ctx->join_cost_cache)
                         join_cost_cache_insert(ctx->join_cost_cache,
                                                cl->joinrel->relids,
                                                cr->joinrel->relids,
                                                c, jr);
                 }
             }
         }

         if (jr && REL_HAS_CHEAPEST_PATH(jr))
         {
             c = REL_CHEAPEST_PATH(jr)->total_cost;
             rows = Max(jr->rows, 1.0);
         }

         switch (mcts_expand_strategy)
         {
             case MCTS_EXTREME_EXPAND_ROW:
                 score = rows;
                 break;
             case MCTS_EXTREME_EXPAND_MIXED_025:
                 score = log(rows + 1.0) + 0.25 * log(((double) c) + 1.0);
                 break;
             case MCTS_EXTREME_EXPAND_MIXED_050:
                 score = log(rows + 1.0) + 0.50 * log(((double) c) + 1.0);
                 break;
             case MCTS_EXTREME_EXPAND_SELECTIVITY:
                 score = rows / Max(left_rows * right_rows, 1.0);
                 break;
             case MCTS_EXTREME_EXPAND_COST:
             default:
                 score = (double) c;
                 break;
         }

         /* Maintain top-k via insertion sort */
         if (top_len < k)
         {
             int pos = top_len;
             while (pos > 0 && top[pos - 1].score > score)
             {
                 top[pos] = top[pos - 1];
                 pos--;
             }
             top[pos].action = a;
             top[pos].score = score;
             top[pos].rows = rows;
             top_len++;
         }
         else if (score < top[top_len - 1].score)
         {
             record_topk_dropped(ctx, clumps, top[top_len - 1].action,
                                 (Cost) top[top_len - 1].score,
                                 top[top_len - 1].rows);
             pfree(top[top_len - 1].action);
             int pos = top_len - 1;
             while (pos > 0 && top[pos - 1].score > score)
             {
                 top[pos] = top[pos - 1];
                 pos--;
             }
             top[pos].action = a;
             top[pos].score = score;
             top[pos].rows = rows;
         }
         else
         {
             record_topk_dropped(ctx, clumps, a, (Cost) score, rows);
             pfree(a);
         }
     }

     ctx->root->join_rel_list = list_truncate(ctx->root->join_rel_list, saved_len);
     ctx->root->join_rel_hash = save_hash;

     for (i = 0; i < top_len; i++)
     {
         /* remember the immediate cost top-k ranked each kept action by */
         top[i].action->rank_cost = (Cost) top[i].score;
         result = lappend(result, top[i].action);
     }

     pfree(top);
     list_free(actions);
     return result;
 }

 /*
  * shape_allows_pair
  *    Plan-shape gate (the paper's plan-shape parameter K).  Shared by
  *    mcts_enumerate_legal_actions and the linear rollout's primary scan
  *    pass; returns false if joining clumps ci and cj would violate the
  *    shape selected by mcts_plan_shape.  A "chain" is a clump of size > 1
  *    (an accumulated intermediate); a base relation has size 1.
  *
  *    K=1 (linear/zig-zag): forbid joining two chains, and once a chain
  *    exists forbid starting a second one (base+base) -- so every join has
  *    at least one base-relation input, on either side.
  *    K>=2 (K-component bushy): allow up to K independent chains to form,
  *    then merge them.
  *
  *    relax_chain_only requests the loosened predicate used by the rollout
  *    safety-valve pass: keep the no-chain+chain-merge rule while base rels
  *    remain, but drop the base+base prohibition so a rollout can always
  *    make progress.
  */
 static inline bool
 shape_allows_pair(MctsClump *ci, MctsClump *cj,
                   int num_chains, int num_base_rels,
                   bool relax_chain_only)
 {
     if (mcts_plan_shape == 1)
     {
         if (ci->size > 1 && cj->size > 1)
             return false;
         if (!relax_chain_only && num_chains > 0 &&
             ci->size == 1 && cj->size == 1)
             return false;
     }
     else if (mcts_plan_shape >= 2)
     {
         bool both_base  = (ci->size == 1 && cj->size == 1);
         bool both_chain = (ci->size > 1 && cj->size > 1);
         bool one_chain  = (!both_base && !both_chain);

         if (relax_chain_only)
         {
             if (both_chain && num_base_rels > 0)
                 return false;
         }
         else
         {
             if (both_chain && num_base_rels > 0)
                 return false;
             if (both_base && num_chains >= mcts_plan_shape)
                 return false;
             if (one_chain && num_chains < mcts_plan_shape && num_base_rels >= 2)
                 return false;
         }
     }
     return true;
 }

 /*
  * force_left_tree gate for kernels=1: once a chain exists, it must remain
  * the left input and every later step must add a base rel on the right.
  * Base+base is allowed only before the first kernel is formed.
  */
 static inline bool
 force_left_tree_orient_action(MctsClump *ci, MctsClump *cj,
                               int i, int j, int num_chains,
                               int *left_idx, int *right_idx)
 {
     bool ci_chain = ci->size > 1;
     bool cj_chain = cj->size > 1;

     if (!mcts_force_left_tree || mcts_plan_shape != 1)
     {
         *left_idx = i;
         *right_idx = j;
         return true;
     }

     if (ci_chain && cj_chain)
         return false;
     if (!ci_chain && !cj_chain)
     {
         if (num_chains > 0)
             return false;
         *left_idx = i;
         *right_idx = j;
         return true;
     }

     if (ci_chain)
     {
         *left_idx = i;
         *right_idx = j;
     }
     else
     {
         *left_idx = j;
         *right_idx = i;
     }
     return true;
 }

 /*
  * mcts_enumerate_legal_actions
  *	  Build a list of legal join actions for the given clump state.
  *
  * We enumerate all pairs of non-overlapping clumps subject to the
  * plan-shape parameter K (mcts_plan_shape):
  *   0   = bushy (any pair allowed)
  *   1   = linear / zig-zag (extend a single chain with base rels)
  *   >=2 = K-component bushy (form up to K chains, then merge them)
  *
  * Desirable joins (those with join clauses) are preferred; Cartesian
  * products are used only when no desirable joins exist.  If top-k
  * filtering is enabled, we further prune via filter_actions_top_k.
  */
 static List *
 mcts_enumerate_legal_actions(MctsContext *ctx, List *clumps)
 {
     List       *actions = NIL;
     List       *fallback_actions = NIL;
     int         n = list_length(clumps);
     int         i, j;
     MemoryContext oldcxt = MemoryContextSwitchTo(ctx->run_context);

     /* Reset per-call top-k drop record (filter_actions_top_k repopulates it). */
     ctx->last_topk_dropped = NIL;

     /* Count existing chains (size > 1) vs base rels for shape enforcement */
     int num_chains = 0;
     int num_base = 0;
     if (mcts_plan_shape >= 1)
     {
         for (i = 0; i < n; i++)
         {
             MctsClump *ci = (MctsClump *) list_nth(clumps, i);
             if (ci->size > 1) num_chains++;
             else               num_base++;
         }
     }

     for (i = 0; i < n; i++)
     {
         MctsClump *ci = (MctsClump *) list_nth(clumps, i);

         for (j = i + 1; j < n; j++)
         {
             MctsClump  *cj = (MctsClump *) list_nth(clumps, j);
             MctsAction *a;
             int         left_idx;
             int         right_idx;

             if (!shape_allows_pair(ci, cj, num_chains, num_base, false))
                 continue;
             if (!force_left_tree_orient_action(ci, cj, i, j, num_chains,
                                                &left_idx, &right_idx))
                 continue;

             a = (MctsAction *) palloc(sizeof(MctsAction));
             a->left_idx = left_idx;
             a->right_idx = right_idx;
             a->rank_cost = (Cost) DBL_MAX;

             if (desirable_join(ctx->root, ci->joinrel, cj->joinrel))
                 actions = lappend(actions, a);
             else
                 fallback_actions = lappend(fallback_actions, a);
         }
     }

     if (actions == NIL)
         actions = fallback_actions;
     else
         list_free(fallback_actions);

     if (mcts_top_k > 0)
         actions = filter_actions_top_k(ctx, clumps, actions, mcts_top_k);

     MemoryContextSwitchTo(oldcxt);
     return actions;
 }

 /* ----------
  *  UCT-Extreme scoring and selection
  * ----------
  */
 /*
  * uct_extreme_score
  *	  Compute the UCT-Extreme selection score for a child node.
  *
  * Score = r_hat + (c * ln(parent_visits) / child_visits)^gamma
  *
  * where r_hat is the child's reward statistic and the second term is the
  * exploration bonus: c is the exploration constant and gamma the
  * exploration exponent (both paper parameters).  Unlike classical UCB1,
  * the exploration term is raised to the power gamma rather than square
  * rooted, and r_hat defaults to the child's *best* reward in its subtree
  * (UCT-Extreme) rather than its mean.
  *
  *   - UCT-Extreme (default): r_hat = best reward seen below the child,
  *     derived from its subtree-incumbent cost.  This targets the best
  *     plan in each subtree, which is the right objective for join search.
  *   - mean UCT (mcts_extreme.uct_aggregation = average): r_hat = mean
  *     rollout reward, included for ablation.
  *
  * Unvisited children get a large sentinel score so they are tried first.
  */
static double
uct_extreme_score(MctsContext *ctx, MctsNode *child, int parent_visits)
{
     double q_value;
     double explore;

     if (child->visits == 0)
         return 1e30;

     if (mcts_uct_aggregation == MCTS_EXTREME_UCT_AGG_AVERAGE)
         q_value = child->sum_reward / (double) child->visits;
     else
         q_value = child->best_reward;

     if (parent_visits <= 0)
         return q_value;
     explore = (ctx->exploration_c * log((double)(parent_visits)) / (double)(child->visits));
     explore = pow(explore, ctx->gamma);
    return q_value + explore;
}

/*
 * classic_ucb_score
 *     Standard UCB1 score used only by mcts_extreme.full_budget after the
 *     expandable frontier has been exhausted.  At that point no new nodes can
 *     be added, so we keep spending the configured budget by selecting already
 *     expanded leaves and rolling out from them.
 */
static double
classic_ucb_score(MctsContext *ctx, MctsNode *child, int parent_visits)
{
    double q_value;
    double explore;

    if (child->visits == 0)
        return 1e30;

    q_value = child->sum_reward / (double) child->visits;
    if (parent_visits <= 1)
        return q_value;

    explore = ctx->exploration_c *
        sqrt(log((double) parent_visits) / (double) child->visits);
    return q_value + explore;
}

/*
 * mcts_select_uct
 *	  Walk from the given node to a leaf using UCT Extreme scores.
  *
  * Returns the selected leaf node, or NULL if the entire subtree is
  * exhausted (all children fully explored or dead ends).
  */
 static MctsNode *
 mcts_select_uct(MctsContext *ctx, MctsNode *node)
 {
     MctsNode   *best = NULL;
     double      best_score = -1e30;
     ListCell   *lc;

     if (node->exhausted)
         return NULL;

     if (node->terminal || list_length(node->children) == 0)
     {
         if (!node->terminal && list_length(node->children) == 0 &&
             (!node->untried_actions || list_length(node->untried_actions) == 0))
         {
             mcts_debug_log("mcts_extreme.mcts: select_uct dead-end leaf (clumps=%d), return NULL",
                            list_length(node->clumps));
             return NULL;
         }
         mcts_debug_log("mcts_extreme.mcts: select_uct leaf reached (terminal=%d children=%d clumps=%d)",
                        node->terminal, list_length(node->children), list_length(node->clumps));
         return node;
     }

     if (node->untried_actions && list_length(node->untried_actions) > 0)
     {
         mcts_debug_log("mcts_extreme.mcts: select_uct returning node with untried actions=%d clumps=%d",
                        list_length(node->untried_actions), list_length(node->clumps));
         return node;
     }

     mcts_debug_log("mcts_extreme.mcts: select_uct at node clumps=%d children=%d visits=%d",
                    list_length(node->clumps), list_length(node->children), node->visits);

     foreach(lc, node->children)
     {
         MctsNode *ch = (MctsNode *) lfirst(lc);

         if (ch->exhausted)
             continue;
         if (list_length(ch->children) == 0 &&
             (!ch->untried_actions || list_length(ch->untried_actions) == 0))
             continue;

         double s = uct_extreme_score(ctx, ch, node->visits);
         if (s > best_score)
         {
             best_score = s;
             best = ch;
         }
     }

     if (best)
     {
         mcts_debug_log("mcts_extreme.mcts: select_uct best child ucb=%.6f (clumps=%d visits=%d best_reward=%.4f)",
                        best_score, list_length(best->clumps), best->visits, best->best_reward);
         return mcts_select_uct(ctx, best);
     }

    mcts_debug_log("mcts_extreme.mcts: select_uct all children dead ends, cannot find expandable leaf");
    return NULL;
}

/*
 * mcts_select_classic_ucb
 *     Select an already-expanded rollout/terminal leaf with classic UCB1.
 *     Unlike mcts_select_uct(), this deliberately ignores exhausted flags and
 *     untried_actions because it is entered only after expansion has no
 *     usable leaf left.  It lets a full-budget run continue sampling the tree
 *     instead of treating the iteration budget as a mere upper bound.
 */
static MctsNode *
mcts_select_classic_ucb(MctsContext *ctx, MctsNode *node)
{
    MctsNode   *best = NULL;
    double      best_score = -1e30;
    ListCell   *lc;

    if (node == NULL)
        return NULL;
    if (node->terminal || list_length(node->children) == 0)
        return node;

    foreach(lc, node->children)
    {
        MctsNode *ch = (MctsNode *) lfirst(lc);
        double s = classic_ucb_score(ctx, ch, Max(node->visits, 1));

        if (s > best_score)
        {
            best_score = s;
            best = ch;
        }
    }

    if (best == NULL)
        return node;

    mcts_debug_log("mcts_extreme.mcts: classic UCB picked child score=%.6f (clumps=%d visits=%d)",
                   best_score, list_length(best->clumps), best->visits);
    return mcts_select_classic_ucb(ctx, best);
}

/* ----------
 *  Tree expansion
 * ----------
  */

 /*
  * mcts_expand
  *	  Expand a leaf node by applying one untried action.
  *
  * We pop the first untried action, create the join via the cache-aware
  * mcts_get_or_build_join, build a new child node with the resulting
  * clump state, and enumerate its legal actions.  Plan-shape constraints
  * (bushy / linear / K-component) are enforced upstream when
  * mcts_enumerate_legal_actions filters the untried action list, so this
  * function does not need to care about which shape is active.
  */
 static MctsNode *
 mcts_expand(MctsContext *ctx, MctsNode *leaf)
 {
     MctsAction *a;
     MctsNode   *child;
     List       *new_clumps;
     int         i;
     RelOptInfo *dummy_rel;
     MemoryContext oldcxt;
     Relids      creation_left_relids = NULL;
     Relids      creation_right_relids = NULL;
     double      child_est_rows = -1.0;	/* predicted cardinality of the join */

     if (list_length(leaf->untried_actions) == 0)
     {
         mcts_debug_log("mcts_extreme.mcts: expand skip, no untried_actions (leaf clumps=%d)",
                        list_length(leaf->clumps));
         return leaf;
     }

     mcts_debug_log("mcts_extreme.mcts: expand start, leaf clumps=%d untried_actions=%d",
                    list_length(leaf->clumps), list_length(leaf->untried_actions));

     a = (MctsAction *) linitial(leaf->untried_actions);
     leaf->untried_actions = list_delete_first(leaf->untried_actions);

     mcts_debug_log("mcts_extreme.mcts: expand picked action left_idx=%d right_idx=%d, untried left=%d",
                    a->left_idx, a->right_idx, list_length(leaf->untried_actions));

     oldcxt = MemoryContextSwitchTo(ctx->run_context);
     {
         int         saved_len = list_length(ctx->root->join_rel_list);
         struct HTAB *save_hash = ctx->root->join_rel_hash;
         int         n_clumps = list_length(leaf->clumps);
         MctsClump  *cl = (MctsClump *) list_nth(leaf->clumps, a->left_idx);
         MctsClump  *cr = (MctsClump *) list_nth(leaf->clumps, a->right_idx);
         RelOptInfo *jr;

         ctx->root->join_rel_hash = NULL;

         if (bms_overlap(cl->joinrel->relids, cr->joinrel->relids))
         {
             mcts_debug_log("mcts_extreme.mcts: expand overlap, action discarded");
             ctx->root->join_rel_hash = save_hash;
             pfree(a);
             MemoryContextSwitchTo(oldcxt);
             return leaf;
         }

         jr = mcts_get_or_build_join(ctx->root, ctx->join_cost_cache,
                                     cl->joinrel, cr->joinrel,
                                     &ctx->cache_hits, true);
         if (jr == NULL)
         {
             mcts_debug_log("mcts_extreme.mcts: expand make_join_rel failed, action discarded");
             ctx->root->join_rel_list = list_truncate(ctx->root->join_rel_list, saved_len);
             ctx->root->join_rel_hash = save_hash;
             pfree(a);
             MemoryContextSwitchTo(oldcxt);
             return leaf;
         }

         creation_left_relids = bms_copy(cl->joinrel->relids);
         creation_right_relids = bms_copy(cr->joinrel->relids);
         child_est_rows = jr->rows;

         new_clumps = NIL;
         for (i = 0; i < n_clumps; i++)
         {
             MctsClump *c = (MctsClump *) list_nth(leaf->clumps, i);
             MctsClump *copy;

             if (i == a->right_idx)
                 continue;

             copy = (MctsClump *) palloc(sizeof(MctsClump));
             copy->joinrel = c->joinrel;
             copy->size = c->size;
             if (i == a->left_idx)
             {
                 copy->joinrel = jr;
                 copy->size = cl->size + cr->size;
             }
             new_clumps = lappend(new_clumps, copy);
         }

         mcts_debug_log("mcts_extreme.mcts: expand built new_clumps=%d", list_length(new_clumps));

         ctx->root->join_rel_list = list_truncate(ctx->root->join_rel_list, saved_len);
         ctx->root->join_rel_hash = save_hash;
     }

     child = (MctsNode *) MemoryContextAlloc(ctx->run_context, sizeof(MctsNode));
     child->parent = leaf;
     child->children = NIL;
     child->clumps = new_clumps;
     child->visits = 0;
     child->best_reward = -DBL_MAX;
     child->sum_reward = 0.0;
     child->best_cost_in_subtree = (Cost) DBL_MAX;
     child->all_query_relids = ctx->all_query_relids;
     child->terminal = (list_length(new_clumps) == 1 &&
                        bms_equal(((MctsClump *) linitial(new_clumps))->joinrel->relids,
                                  ctx->all_query_relids));
     child->untried_actions = child->terminal ? NIL : mcts_enumerate_legal_actions(ctx, new_clumps);
     child->topk_dropped = child->terminal ? NIL : ctx->last_topk_dropped;
     child->immediate_cost = a->rank_cost;
     child->est_rows = child_est_rows;
     child->creation_left = creation_left_relids;
     child->creation_right = creation_right_relids;
     child->node_depth = leaf->node_depth + 1;
     child->exhausted = false;
     leaf->children = lappend(leaf->children, child);

     mcts_debug_log("========== MCTS depth %d / %d ==========", child->node_depth, depth);
     mcts_debug_log("mcts_extreme.mcts: expand child created terminal=%d untried_actions=%d",
                    child->terminal, child->untried_actions ? list_length(child->untried_actions) : 0);

     if (child->terminal)
     {
         Cost cost = mcts_eval_candidate(ctx->root, new_clumps, &dummy_rel,
                                         ctx->join_cost_cache);
        double r = cost_to_reward(ctx, cost);
         child->best_reward = r;
         child->sum_reward = r;
         child->best_cost_in_subtree = cost;
         child->visits = 1;
         mcts_debug_log("mcts_extreme.mcts: expand terminal node eval cost=%g reward=%.4f",
                        (double) cost, r);
     }

     mcts_debug_log("mcts_extreme.mcts: expand done, returning child");
     MemoryContextSwitchTo(oldcxt);
     return child;
 }
 /* ----------
  *  Rollout strategies
  * ----------
  */

 /*
  * mcts_rollout_bushy
  *	  Perform a random rollout from the given node using bushy joins.
  *
  * We copy the node's clump list, then repeatedly pick a random
  * desirable join (falling back to any join if none exists) until
  * a single clump remains.  Returns the final cost and optionally
  * the merge order for later replay.
  */
 static Cost
 mcts_rollout_bushy(MctsContext *ctx, MctsNode *node, RelOptInfo **out_rel,
                    List **out_merge_order)
 {
     List       *clumps;
     List       *merge_order = NIL;
     MemoryContext oldcxt;
     int         saved_len;
     struct HTAB *save_hash;
     Cost        cost;

     *out_rel = NULL;
     oldcxt = MemoryContextSwitchTo(ctx->eval_context);

     clumps = mcts_clone_clumps(node->clumps);

     saved_len = list_length(ctx->root->join_rel_list);
     save_hash = ctx->root->join_rel_hash;
     ctx->root->join_rel_hash = NULL;

     while (list_length(clumps) > 1)
     {
         int na = list_length(clumps);
         int ia, ib;
         List       *candidates = NIL;
         List       *fallback = NIL;
         MctsAction *chosen;
         MctsClump  *ca, *cb;
         RelOptInfo *jr;
         int         pick;

         for (ia = 0; ia < na; ia++)
         {
             MctsClump *ci = (MctsClump *) list_nth(clumps, ia);
             for (ib = ia + 1; ib < na; ib++)
             {
                 MctsClump *cj = (MctsClump *) list_nth(clumps, ib);
                 MctsAction *a;

                 if (bms_overlap(ci->joinrel->relids, cj->joinrel->relids))
                     continue;

                 a = (MctsAction *) palloc(sizeof(MctsAction));
                 a->left_idx = ia;
                 a->right_idx = ib;

                 if (desirable_join(ctx->root, ci->joinrel, cj->joinrel))
                     candidates = lappend(candidates, a);
                 else
                     fallback = lappend(fallback, a);
             }
         }

         if (candidates == NIL)
         {
             candidates = fallback;
             fallback = NIL;
         }
         else
             list_free_deep(fallback);

         if (candidates == NIL)
         {
             ctx->rollout_failures++;
             ctx->root->join_rel_list = list_truncate(ctx->root->join_rel_list, saved_len);
             ctx->root->join_rel_hash = save_hash;
             if (out_merge_order)
                 *out_merge_order = NIL;
             MemoryContextSwitchTo(oldcxt);
             return (Cost) DBL_MAX;
         }

         /*
          * Pick a random candidate and try to build it; if it cannot be
          * built (e.g. it would form an outer join, which this extension
          * does not construct), drop it and try another.  If no candidate in
          * the set can be built, the rollout cannot complete -- report
          * failure instead of `continue`, which would rebuild the same
          * unbuildable set and spin forever.
          */
         jr = NULL;
         while (candidates != NIL)
         {
             pick = (int) (erand48(ctx->seedbuf) * list_length(candidates));
             chosen = (MctsAction *) list_nth(candidates, pick);
             ca = (MctsClump *) list_nth(clumps, chosen->left_idx);
             cb = (MctsClump *) list_nth(clumps, chosen->right_idx);

             jr = mcts_get_or_build_join(ctx->root, ctx->join_cost_cache,
                                         ca->joinrel, cb->joinrel,
                                         &ctx->cache_hits, false);
             if (jr != NULL)
                 break;
             candidates = list_delete_nth_cell(candidates, pick);
         }
         if (jr == NULL)
         {
             ctx->rollout_failures++;
             ctx->root->join_rel_list = list_truncate(ctx->root->join_rel_list, saved_len);
             ctx->root->join_rel_hash = save_hash;
             if (out_merge_order)
                 *out_merge_order = NIL;
             MemoryContextSwitchTo(oldcxt);
             return (Cost) DBL_MAX;
         }

         {
             MctsMergeStep *step;
             MemoryContext savecxt = MemoryContextSwitchTo(ctx->run_context);
             step = (MctsMergeStep *) palloc(sizeof(MctsMergeStep));
             step->left = bms_copy(ca->joinrel->relids);
             step->right = bms_copy(cb->joinrel->relids);
             merge_order = lappend(merge_order, step);
             MemoryContextSwitchTo(savecxt);
         }

         ca->joinrel = jr;
         ca->size += cb->size;
         clumps = list_delete_nth_cell(clumps, chosen->right_idx);
         pfree(cb);
         list_free_deep(candidates);
     }

     cost = REL_HAS_CHEAPEST_PATH(((MctsClump *) linitial(clumps))->joinrel)
         ? REL_CHEAPEST_PATH(((MctsClump *) linitial(clumps))->joinrel)->total_cost
         : (Cost) DBL_MAX;
     *out_rel = ((MctsClump *) linitial(clumps))->joinrel;
     if (out_merge_order)
         *out_merge_order = merge_order;

     ctx->root->join_rel_list = list_truncate(ctx->root->join_rel_list, saved_len);
     ctx->root->join_rel_hash = save_hash;
     MemoryContextSwitchTo(oldcxt);
     return cost;
 }

 /*
  * mcts_rollout_linear
  *	  Perform a random rollout enforcing the K=1 linear (zig-zag) shape.
  *
  * Once a chain (clump with size > 1) exists, only actions that extend it
  * with a base relation are considered.  Because the accumulated chain may
  * appear on either input side of the next join, this is the zig-zag class
  * (left-deep and right-deep combined), not a strict left-deep restriction.
  *
  * Three safety valves ensure the rollout always completes on any join
  * graph:
  * 1. If no desirable joins exist, allow any shape-compatible pair.
  * 2. If that also fails, relax the shape constraint but keep structure.
  * 3. As a last resort, allow any non-overlapping pair (bushy fallback).
  */
 static Cost
 mcts_rollout_linear(MctsContext *ctx, MctsNode *node, RelOptInfo **out_rel,
                       List **out_merge_order)
 {
     List       *clumps;
     List       *merge_order = NIL;
     MemoryContext oldcxt;
     int         saved_len;
     struct HTAB *save_hash;
     Cost        cost;

     *out_rel = NULL;
     oldcxt = MemoryContextSwitchTo(ctx->eval_context);

     clumps = mcts_clone_clumps(node->clumps);

     saved_len = list_length(ctx->root->join_rel_list);
     save_hash = ctx->root->join_rel_hash;
     ctx->root->join_rel_hash = NULL;

     while (list_length(clumps) > 1)
     {
         int na = list_length(clumps);
         int ia, ib;
         List       *candidates = NIL;
         List       *fallback = NIL;
         MctsAction *chosen;
         MctsClump  *ca, *cb;
         RelOptInfo *jr;
         int         pick;

         /* Count chains vs base rels for shape enforcement */
         int num_chains = 0;
         int num_base_rels = 0;
         for (ia = 0; ia < na; ia++)
         {
             MctsClump *ci = (MctsClump *) list_nth(clumps, ia);
             if (ci->size > 1) num_chains++;
             else               num_base_rels++;
         }

         /*
          * Pass 0: full shape constraint, prefer desirable joins.
          * Pass 1: relax shape (single chain still enforced for K==1).
          * Pass 2: any non-overlapping pair is acceptable.
          */
         for (int pass = 0; pass < 3 && candidates == NIL; pass++)
         {
             for (ia = 0; ia < na; ia++)
             {
                 MctsClump *ci = (MctsClump *) list_nth(clumps, ia);
                 for (ib = ia + 1; ib < na; ib++)
                 {
                     MctsClump *cj = (MctsClump *) list_nth(clumps, ib);
                     MctsAction *a;
                     int         left_idx;
                     int         right_idx;

                     if (bms_overlap(ci->joinrel->relids, cj->joinrel->relids))
                         continue;

                     if (pass < 2 &&
                         !shape_allows_pair(ci, cj, num_chains, num_base_rels,
                                            pass == 1))
                         continue;
                     if (!force_left_tree_orient_action(ci, cj, ia, ib,
                                                        num_chains,
                                                        &left_idx, &right_idx))
                         continue;

                     a = (MctsAction *) palloc(sizeof(MctsAction));
                     a->left_idx = left_idx;
                     a->right_idx = right_idx;

                     if (pass < 2 &&
                         desirable_join(ctx->root, ci->joinrel, cj->joinrel))
                         candidates = lappend(candidates, a);
                     else
                         fallback = lappend(fallback, a);
                 }
             }

             if (candidates == NIL)
             {
                 candidates = fallback;
                 fallback = NIL;
             }
             else
                 list_free_deep(fallback);
         }

         if (candidates == NIL)
         {
             ctx->rollout_failures++;
             ctx->root->join_rel_list = list_truncate(ctx->root->join_rel_list, saved_len);
             ctx->root->join_rel_hash = save_hash;
             if (out_merge_order)
                 *out_merge_order = NIL;
             MemoryContextSwitchTo(oldcxt);
             return (Cost) DBL_MAX;
         }

         /*
          * Pick a random candidate and try to build it; if it cannot be
          * built (e.g. it would form an outer join, which this extension
          * does not construct), drop it and try another.  If no candidate in
          * the set can be built, the rollout cannot complete -- report
          * failure instead of `continue`, which would rebuild the same
          * unbuildable set and spin forever.
          */
         jr = NULL;
         while (candidates != NIL)
         {
             pick = (int) (erand48(ctx->seedbuf) * list_length(candidates));
             chosen = (MctsAction *) list_nth(candidates, pick);
             ca = (MctsClump *) list_nth(clumps, chosen->left_idx);
             cb = (MctsClump *) list_nth(clumps, chosen->right_idx);

             jr = mcts_get_or_build_join(ctx->root, ctx->join_cost_cache,
                                         ca->joinrel, cb->joinrel,
                                         &ctx->cache_hits, false);
             if (jr != NULL)
                 break;
             candidates = list_delete_nth_cell(candidates, pick);
         }
         if (jr == NULL)
         {
             ctx->rollout_failures++;
             ctx->root->join_rel_list = list_truncate(ctx->root->join_rel_list, saved_len);
             ctx->root->join_rel_hash = save_hash;
             if (out_merge_order)
                 *out_merge_order = NIL;
             MemoryContextSwitchTo(oldcxt);
             return (Cost) DBL_MAX;
         }

         {
             MctsMergeStep *step;
             MemoryContext savecxt = MemoryContextSwitchTo(ctx->run_context);
             step = (MctsMergeStep *) palloc(sizeof(MctsMergeStep));
             step->left = bms_copy(ca->joinrel->relids);
             step->right = bms_copy(cb->joinrel->relids);
             merge_order = lappend(merge_order, step);
             MemoryContextSwitchTo(savecxt);
         }

         ca->joinrel = jr;
         ca->size += cb->size;
         clumps = list_delete_nth_cell(clumps, chosen->right_idx);
         pfree(cb);
         list_free_deep(candidates);
     }

     cost = REL_HAS_CHEAPEST_PATH(((MctsClump *) linitial(clumps))->joinrel)
         ? REL_CHEAPEST_PATH(((MctsClump *) linitial(clumps))->joinrel)->total_cost
         : (Cost) DBL_MAX;
     *out_rel = ((MctsClump *) linitial(clumps))->joinrel;
     if (out_merge_order)
         *out_merge_order = merge_order;

     ctx->root->join_rel_list = list_truncate(ctx->root->join_rel_list, saved_len);
     ctx->root->join_rel_hash = save_hash;
     MemoryContextSwitchTo(oldcxt);
     return cost;
 }

 /* ----------
  *  Backpropagation
  * ----------
  */

 /*
  * mcts_backpropagate
  *	  Walk from the rolled-out node up to the root, updating each node's
  *	  visit count and subtree incumbent (the best cost seen below it).
  *
  * This is the Backpropagation step.  UCT-Extreme keeps the best (not the
  * average) outcome in each subtree, so for every node on the path we take
  * the min of its current subtree incumbent and this rollout's cost, then
  * recompute its reward via the active reward map.
  */
 static void
mcts_backpropagate(MctsContext *ctx, MctsNode *node, Cost rollout_cost)
 {
     mcts_debug_log("mcts_extreme.mcts: mcts_backpropagate start node=%p cost=%.2f",
                    (void *) node, (double) rollout_cost);

     /*
      * Failed rollouts return DBL_MAX (the bushy/linear paths when no legal
      * pair exists).  We still walk up to bump `visits` — a failed attempt
      * costs an iteration and selection should keep exploring around it —
      * but skip the reward update.  Feeding DBL_MAX through the reward map
      * would otherwise make failed leaves look maximally attractive.
      */
     if (rollout_cost >= (Cost) DBL_MAX)
     {
         for (; node != NULL; node = node->parent)
             node->visits++;
         return;
     }

     /*
      * Maintain the running min/max envelope used by norm_neg_log.  Done
      * here (rather than at the rollout-cost-discovered sites in
      * mcts_extreme()) so every successful backprop contributes — including
      * those that don't improve the global/phase best.
      */
     if ((double) rollout_cost > 0 && (double) rollout_cost < ctx->cost_min)
         ctx->cost_min = (double) rollout_cost;
     if ((double) rollout_cost > ctx->cost_max)
         ctx->cost_max = (double) rollout_cost;

     {
         double rollout_reward = cost_to_reward(ctx, rollout_cost);

         for (; node != NULL; node = node->parent)
         {
             node->visits++;
             node->sum_reward += rollout_reward;
             if (rollout_cost < node->best_cost_in_subtree)
                 node->best_cost_in_subtree = rollout_cost;
             node->best_reward = cost_to_reward(ctx, node->best_cost_in_subtree);
         }
     }
 }

 /*
  * mcts_build_root
  *	  Create the root MctsNode for a new MCTS phase.
  *
  * The root state has one clump per base relation and enumerates
  * all legal first-join actions.
  */
 static MctsNode *
 mcts_build_root(MctsContext *ctx)
 {
     MctsNode   *root_node;
     List       *clumps = NIL;
     ListCell   *lc;
     MemoryContext oldcxt = MemoryContextSwitchTo(ctx->run_context);

     mcts_debug_log("mcts_extreme.mcts: mcts_build_root start n_rels=%d",
                    list_length(ctx->initial_rels));

     foreach(lc, ctx->initial_rels)
     {
         RelOptInfo *rel = (RelOptInfo *) lfirst(lc);
         MctsClump  *c = (MctsClump *) palloc(sizeof(MctsClump));
         c->joinrel = rel;
         c->size = 1;
         clumps = lappend(clumps, c);
     }

     root_node = (MctsNode *) palloc(sizeof(MctsNode));
     root_node->parent = NULL;
     root_node->children = NIL;
     root_node->clumps = clumps;
     root_node->visits = 0;
     root_node->best_reward = -DBL_MAX;
     root_node->sum_reward = 0.0;
     root_node->best_cost_in_subtree = (Cost) DBL_MAX;
     root_node->all_query_relids = ctx->all_query_relids;
     root_node->terminal = (list_length(clumps) == 1);
     root_node->untried_actions = root_node->terminal ? NIL : mcts_enumerate_legal_actions(ctx, clumps);
     root_node->topk_dropped = root_node->terminal ? NIL : ctx->last_topk_dropped;
     root_node->immediate_cost = (Cost) DBL_MAX;
     root_node->est_rows = -1.0;		/* root is not a single join */
     root_node->creation_left = NULL;
     root_node->creation_right = NULL;
     root_node->node_depth = 0;
     root_node->exhausted = false;

     mcts_debug_log("mcts_extreme.mcts: mcts_build_root done clumps=%d terminal=%d untried=%d",
                    list_length(clumps), root_node->terminal,
                    root_node->untried_actions ? list_length(root_node->untried_actions) : 0);

     MemoryContextSwitchTo(oldcxt);
     return root_node;
 }

 /*
  * get_node_merge_path
  *	  Reconstruct the sequence of merge steps from the root to this node
  *	  by walking the parent chain.  Used to record the tree-expansion
  *	  prefix of the best join order.
  */
 static List *
 get_node_merge_path(MctsNode *node, MemoryContext cxt)
 {
     List       *path = NIL;
     MemoryContext oldcxt = MemoryContextSwitchTo(cxt);

     for (; node != NULL && node->creation_left != NULL; node = node->parent)
     {
         MctsMergeStep *step = (MctsMergeStep *) palloc(sizeof(MctsMergeStep));
         step->left = node->creation_left;
         step->right = node->creation_right;
         path = lcons(step, path);
     }

     MemoryContextSwitchTo(oldcxt);
     return path;
 }

 /*
  * mcts_capture_order
  *   Build a complete join order (tree-expansion prefix + rollout steps) for a
  *   node, copied into cxt so it survives to phase end for per-phase subplan
  *   tracing.  steps may be NIL (terminal leaf).
  */
 static List *
 mcts_capture_order(MctsNode *node, List *steps, MemoryContext cxt)
 {
     List       *order = get_node_merge_path(node, cxt);

     if (steps != NIL)
     {
         MemoryContext old = MemoryContextSwitchTo(cxt);

         order = list_concat(order, list_copy(steps));
         MemoryContextSwitchTo(old);
     }
     return order;
 }

 /* Cap on emitted search-trace nodes, so big searches don't blow up the buffer. */
 #define MCTS_SEARCHTRACE_MAX 2000

 /*
  * mcts_emit_subtree
  *	  Recursively emit a node's children (the expanded tree) and its
  *	  top-k-dropped actions into the search trace.  Nodes on the winning
  *	  order's path (the spine) are flagged chosen.  Returns the running count
  *	  of emitted nodes (capped at MCTS_SEARCHTRACE_MAX).
  */
 static int
 mcts_emit_subtree(PlannerInfo *root, MctsNode *node, int snap, int depth,
                   MctsNode **spine, int spine_len, int emitted)
 {
     ListCell   *lc;

     if (depth >= 64)
         return emitted;

     /* expanded children (the actual MCTS tree) */
     foreach(lc, node->children)
     {
         MctsNode   *ch = (MctsNode *) lfirst(lc);
         bool        is_chosen = false;
         int         i;
         int         sid;

         if (emitted >= MCTS_SEARCHTRACE_MAX)
             break;
         for (i = 0; i < spine_len; i++)
             if (spine[i] == ch)
             {
                 is_chosen = true;
                 break;
             }

         sid = mcts_searchtrace_add(snap, depth + 1,
                                    mcts_trace_relids_text(root, ch->creation_left),
                                    mcts_trace_relids_text(root, ch->creation_right),
                                    ch->visits, ch->best_reward, ch->sum_reward,
                                    (double) ch->best_cost_in_subtree,
                                    (double) ch->immediate_cost, ch->est_rows,
                                    is_chosen, false);
         emitted++;
         emitted = mcts_emit_subtree(root, ch, sid, depth + 1, spine,
                                     spine_len, emitted);
     }

     /*
      * Top-k-dropped actions at this state: candidate joins top-k evaluated
      * (built + costed during Expansion) but discarded for not being among the
      * k cheapest.  They never became tree nodes (no UCT stats).
      */
     foreach(lc, node->topk_dropped)
     {
         MctsDroppedAction *d = (MctsDroppedAction *) lfirst(lc);

         if (emitted >= MCTS_SEARCHTRACE_MAX)
             break;
         mcts_searchtrace_add(snap, depth + 1,
                              mcts_trace_relids_text(root, d->left),
                              mcts_trace_relids_text(root, d->right),
                              0, 0.0, 0.0, (double) DBL_MAX,
                              (double) d->cost, d->rows, false, true);
         emitted++;
     }
     return emitted;
 }

 /*
  * mcts_capture_search_tree
  *	  Snapshot the WHOLE MCTS search tree of a phase for the search trace
  *	  (why MCTS chose this join order).  Called at the end of a phase that
  *	  improved the global best, with that phase's accumulated root node.
  *
  * Emits every expanded node (so non-best branches and how far they were
  * explored are visible too), plus each node's top-k-dropped candidates.  The
  * nodes lying on the actual winning order (best_order) are flagged chosen, so
  * the spine is the real best order, not a cost-tie-break of our own.
  * Overwrites any prior snapshot, so the last improving phase wins.
  */
 static void
 mcts_capture_search_tree(PlannerInfo *root, MctsNode *rootnode, List *best_order)
 {
     MctsNode   *spine[128];
     int         spine_len = 0;
     MctsNode   *cur;
     ListCell   *stepcell;
     int         s0;

     if (!mcts_trace_search || rootnode == NULL)
         return;

     /* Resolve the winning order's path through this phase's tree (the spine). */
     cur = rootnode;
     stepcell = list_head(best_order);
     while (cur != NULL && stepcell != NULL && spine_len < 128)
     {
         MctsMergeStep *step = (MctsMergeStep *) lfirst(stepcell);
         MctsNode   *next = NULL;
         ListCell   *lc;

         foreach(lc, cur->children)
         {
             MctsNode   *ch = (MctsNode *) lfirst(lc);

             if ((bms_equal(ch->creation_left, step->left) &&
                  bms_equal(ch->creation_right, step->right)) ||
                 (bms_equal(ch->creation_left, step->right) &&
                  bms_equal(ch->creation_right, step->left)))
             {
                 next = ch;
                 break;
             }
         }
         if (next == NULL)
             break;
         spine[spine_len++] = next;
         cur = next;
         stepcell = lnext(best_order, stepcell);
     }

     mcts_searchtrace_begin();

     /* the initial state (all relations separate) */
     s0 = mcts_searchtrace_add(-1, 0, NULL, NULL,
                               rootnode->visits, rootnode->best_reward,
                               rootnode->sum_reward,
                               (double) rootnode->best_cost_in_subtree,
                               (double) DBL_MAX, rootnode->est_rows,
                               true, false);

     mcts_emit_subtree(root, rootnode, s0, 0, spine, spine_len, 0);
 }

 /*
  * mcts_root_restart
  *	  Reset the MCTS context for a new restart phase.
  *
  * We destroy the old join cost cache (so stale entries from a
  * different tree topology do not mislead the search), re-seed the
  * PRNG, and build a fresh root node.
  */
 static MctsNode *
 mcts_root_restart(MctsContext *ctx, int phase)
 {
     MctsNode *root_node;

     ctx->exploration_c = mcts_extreme_exploration_constant;
     ctx->gamma = mcts_extreme_gamma;
     /*
      * iterations_done is *cumulative across phases* (initialized in
      * mcts_extreme() before the phase loop), so don't zero it here.
      */
     ctx->rollout_failures = 0;
     ctx->best_cost = (Cost) DBL_MAX;
     ctx->best_merge_order = NIL;

     /* Destroy and recreate the cache so each phase starts fresh */
     if (ctx->join_cost_cache)
     {
         hash_destroy(ctx->join_cost_cache);
         ctx->join_cost_cache = create_join_cost_cache(ctx->run_context, mcts_cache_size);
     }
     ctx->cache_hits = 0;

    /*
     * Per-phase seed: splitmix64(base_seed, phase) mixes the phase index
     * through all three erand48 state words, so successive restart phases
     * draw independent rollout streams rather than correlated ones.
     */
    mcts_mix_seedbuf(ctx, (uint64) phase, 0, 0);

     root_node = mcts_build_root(ctx);
     return root_node;
 }

 /* ----------
  *  Main entry point
  * ----------
  */

 /*
  * mcts_extreme
  *	  Run the UCT-Extreme MCTS join-order search and return the chosen
  *	  join relation.  This is the top-level entry installed on
  *	  join_search_hook (see core/mcts_extreme.c).
  *
  * The search runs one or more restart phases, each building a fresh MCTS
  * tree (phase budgets follow the Luby sequence when mcts_extreme.luby is
  * on).  Each iteration performs the four MCTS steps: select a leaf via
  * the UCT-Extreme score, expand it with one untried join, roll out to a
  * complete join order under the active plan shape, and backpropagate the
  * cost.  Depth-limited leaves are rolled out repeatedly instead of being
  * expanded further.
  *
  * After every phase completes, the best merge order found is replayed
  * through make_join_rel to build join RelOptInfos the planner can use.
  */
 RelOptInfo *
 mcts_extreme(PlannerInfo *root, List *initial_rels)
 {
     MctsContext ctx;
     MctsNode   *root_node;
     RelOptInfo *best_rel = NULL;
     Cost        best_cost = (Cost) DBL_MAX;
     int         i;
     int         len_phases = mcts_phases;

     if (list_length(initial_rels) < 2)
     {
         mcts_debug_log("mcts_extreme.mcts: mcts_extreme skip (< 2 relations)");
         return NULL;
     }

     /*
      * MCTS does not build outer joins: make_join_rel rejects
      * JOIN_LEFT/RIGHT/FULL and returns NULL, so a join problem that contains
      * one has no complete plan in this search.  Decline such queries up front
      * and let the planner fall back to its standard join search -- otherwise
      * a rollout could spin forever trying to merge a pair that can never be
      * built.
      */
     {
         ListCell *lc_sji;

         foreach(lc_sji, root->join_info_list)
         {
             SpecialJoinInfo *sji = (SpecialJoinInfo *) lfirst(lc_sji);

             if (sji->jointype == JOIN_LEFT ||
                 sji->jointype == JOIN_RIGHT ||
                 sji->jointype == JOIN_FULL)
             {
                 mcts_debug_log("mcts_extreme.mcts: declining query with outer join");
                 return NULL;
             }
         }
     }

     if (mcts_search_algorithm == MCTS_EXTREME_SEARCH_SAIO)
         return saio_one_kernel(root, initial_rels);
     if (mcts_search_algorithm == MCTS_EXTREME_SEARCH_ITERATIVE_IMPROVEMENT)
         return mcts_iterative_improvement(root, initial_rels);

     mcts_debug_log("mcts_extreme.mcts: mcts_extreme start n_rels=%d",
                    list_length(initial_rels));
     mcts_debug_log("mcts_extreme.mcts: mcts_extreme.depth = %d", depth);

     ctx.root = root;
     ctx.initial_rels = initial_rels;
     ctx.num_rels = list_length(initial_rels);
     /* Build relids from the actual initial_rels, not root->all_baserels,
      * because we may be planning a subset (group) of the full query */
     {
         ListCell *lc_tmp;
         ctx.all_query_relids = NULL;
         foreach(lc_tmp, initial_rels)
         {
             RelOptInfo *r = (RelOptInfo *) lfirst(lc_tmp);
             ctx.all_query_relids = bms_union(ctx.all_query_relids, r->relids);
         }
     }
     ctx.run_context = AllocSetContextCreate(CurrentMemoryContext,
                                             "MCTS run",
                                             ALLOCSET_DEFAULT_MINSIZE,
                                             ALLOCSET_DEFAULT_INITSIZE,
                                             ALLOCSET_DEFAULT_MAXSIZE);
     ctx.eval_context = AllocSetContextCreate(ctx.run_context,
                                              "MCTS eval",
                                              ALLOCSET_DEFAULT_MINSIZE,
                                              ALLOCSET_DEFAULT_INITSIZE,
                                              ALLOCSET_DEFAULT_MAXSIZE);
     ctx.join_cost_cache = mcts_cache_enabled
         ? create_join_cost_cache(ctx.run_context, mcts_cache_size) : NULL;
     ctx.cache_hits = 0;
     ctx.iterations_done = 0;
    ctx.phase_best_cost = (Cost) DBL_MAX;
    ctx.cost_min = DBL_MAX;
    ctx.cost_max = 0.0;

     /* Remember pre-search state so we can truncate before replay */
     int saved_join_rel_len = list_length(root->join_rel_list);

     double best_cost_global = DBL_MAX;
     RelOptInfo *best_rel_global = NULL;
     List *best_merge_order_global = NIL;

     /* Clear any prior search-decision trace; snapshot the best phase below. */
     Cost searchtrace_best = (Cost) DBL_MAX;

     if (mcts_trace_search)
     {
         mcts_searchtrace_begin();
         mcts_phasetrace_begin();
         mcts_itertrace_begin();
         mcts_phasesubtrace_begin();
     }

     instr_time phase_start, phase_end;
     double time_selection_ms = 0.0;
     double time_expansion_ms = 0.0;
     double time_rollout_ms = 0.0;
     double time_backprop_ms = 0.0;
     instr_time total_start, total_end;
     int total_rollouts = 0;
     int total_exhausted = 0;
     int best_cost_phase = 0;
     int best_cost_iteration = 0;

     INSTR_TIME_SET_CURRENT(total_start);

     int phases_since_improvement = 0;
     for (int j = 0; j < len_phases; j++)
     {
         List       *phase_best_order = NIL;	/* best plan's join order this phase */

         root_node = mcts_root_restart(&ctx, j);
         ctx.best_cost = best_cost_global;
        /*
         * phase_best_cost tracks the best cost found within the current
         * phase; it feeds the per-iteration trace and logging.  Reset it
         * at every restart so it reflects this phase only.
         */
        ctx.phase_best_cost = (Cost) DBL_MAX;

         mcts_debug_log("mcts_extreme.mcts: root built, clumps=%d untried_actions=%d",
                        list_length(root_node->clumps),
                        root_node->untried_actions ? list_length(root_node->untried_actions) : 0);
         /* Log initial clumps with relation aliases for debugging */
         if (mcts_log_debug)
         {
             StringInfoData buf;
             ListCell *clc;
             initStringInfo(&buf);
             appendStringInfoChar(&buf, '(');
             foreach(clc, root_node->clumps)
             {
                 MctsClump *c = (MctsClump *) lfirst(clc);
                 int nmembers = bms_num_members(c->joinrel->relids);
                 if (clc != list_head(root_node->clumps))
                     appendStringInfoChar(&buf, ' ');
                 if (nmembers > 1)
                     appendStringInfoChar(&buf, '{');
                 int x = -1;
                 bool first = true;
                 while ((x = bms_next_member(c->joinrel->relids, x)) >= 0)
                 {
                     RangeTblEntry *rte = root->simple_rte_array[x];
                     if (!first)
                         appendStringInfoChar(&buf, ',');
                     appendStringInfoString(&buf, rte->eref->aliasname);
                     first = false;
                 }
                 if (nmembers > 1)
                     appendStringInfoChar(&buf, '}');
             }
             appendStringInfoChar(&buf, ')');
             if (mcts_log_debug)
                 elog(WARNING, "mcts_extreme.mcts: clumps = %s", buf.data);
             pfree(buf.data);
         }

         mcts_debug_log("mcts_extreme.mcts: phase %d budget=%d best_cost_global=%.2f",
                        j + 1, (mcts_luby_enabled ? mcts_start_budget * luby_value(j + 1) : mcts_start_budget), best_cost_global);

         for (i = 0; i < (mcts_luby_enabled ? mcts_start_budget * luby_value(j + 1) : mcts_start_budget); i++)
         {
            MctsNode   *leaf;
            MctsNode   *child;
            Cost        rollout_cost;
            RelOptInfo *rollout_rel = NULL;
            bool        full_budget_replay = false;

             INSTR_TIME_SET_CURRENT(phase_start);
             leaf = mcts_select_uct(&ctx, root_node);
             INSTR_TIME_SET_CURRENT(phase_end);
             INSTR_TIME_SUBTRACT(phase_end, phase_start);
             time_selection_ms += INSTR_TIME_GET_MILLISEC(phase_end);

            if (leaf == NULL)
            {
                mcts_debug_log("mcts_extreme.mcts: no expandable leaf found, tree exhausted");
                if (!mcts_full_budget)
                    break;

                leaf = mcts_select_classic_ucb(&ctx, root_node);
                if (leaf == NULL)
                    break;
                full_budget_replay = true;
                mcts_debug_log("mcts_extreme.mcts: continuing after exhaustion via classic UCB");
            }

            mcts_debug_log("mcts_extreme.mcts: iter %d selection done, leaf clumps=%d depth=%d",
                            i + 1, list_length(leaf->clumps), leaf->node_depth);

            /*
             * Per-iteration trace: best-so-far (phase and global) and the depth
             * selection reached.  Recorded at iteration start, so an improvement
             * found during iteration i shows from iteration i+1.
             */
            if (mcts_trace_search)
                mcts_itertrace_add(j + 1, i + 1, ctx.phase_best_cost,
                                   best_cost_global, leaf->node_depth);

            /*
             * Full-budget replay: expansion has been exhausted, so spend the
             * remaining budget by re-sampling existing leaves selected by
             * classic UCB1.  Terminal leaves are re-scored directly; partial
             * leaves run another rollout without marking them exhausted.
             */
            if (full_budget_replay)
            {
                if (leaf->terminal)
                {
                    rollout_cost = leaf->best_cost_in_subtree;
                    if (rollout_cost < (Cost) DBL_MAX &&
                        rollout_cost < ctx.phase_best_cost)
                    {
                        ctx.phase_best_cost = rollout_cost;
                        if (mcts_trace_search)
                            phase_best_order = mcts_capture_order(leaf, NIL, ctx.run_context);
                    }

                    if (leaf->clumps && list_length(leaf->clumps) == 1)
                    {
                        MctsClump *c = (MctsClump *) linitial(leaf->clumps);
                        if (REL_HAS_CHEAPEST_PATH(c->joinrel) &&
                            REL_CHEAPEST_PATH(c->joinrel)->total_cost < best_cost)
                        {
                            List *full_order = get_node_merge_path(leaf, ctx.run_context);

                            best_cost = REL_CHEAPEST_PATH(c->joinrel)->total_cost;
                            best_rel = c->joinrel;
                            best_rel_global = best_rel;
                            ctx.best_merge_order = full_order;
                            ctx.best_cost = best_cost;
                            best_cost_global = best_cost;
                            best_merge_order_global = full_order;
                            best_cost_phase = j + 1;
                            best_cost_iteration = i + 1;
                        }
                    }
                }
                else
                {
                    List       *rollout_steps = NIL;
                    Cost        best_rollout_cost = (Cost) DBL_MAX;
                    RelOptInfo *best_rollout_rel = NULL;
                    List       *best_rollout_steps = NIL;

                    INSTR_TIME_SET_CURRENT(phase_start);
                    mcts_mix_seedbuf(&ctx, (uint64) j, (uint64) i, 0);

                    for (int rr = 0; rr < mcts_rollouts_per_leaf; rr++)
                    {
                        if (mcts_rollout_mode == MCTS_EXTREME_ROLLOUT_LUBY)
                        {
                            unsigned long long lubymix =
                                (unsigned long long) luby_value(rr + 1);
                            ctx.seedbuf[0] ^= (unsigned short) (lubymix & 0xFFFF);
                            ctx.seedbuf[1] ^= (unsigned short) ((lubymix >> 16) & 0xFFFF);
                            ctx.seedbuf[2] ^= (unsigned short) ((lubymix >> 32) & 0xFFFF);
                        }
                        if (mcts_plan_shape == 0)
                            rollout_cost = mcts_rollout_bushy(&ctx, leaf, &rollout_rel, &rollout_steps);
                        else
                            rollout_cost = mcts_rollout_linear(&ctx, leaf, &rollout_rel, &rollout_steps);

                        if (rollout_cost < best_rollout_cost)
                        {
                            best_rollout_cost = rollout_cost;
                            best_rollout_rel = rollout_rel;
                            best_rollout_steps = rollout_steps;
                        }

                        mcts_mix_seedbuf(&ctx, (uint64) j, (uint64) i, (uint64) (rr + 1));
                    }

                    rollout_cost = best_rollout_cost;
                    rollout_rel = best_rollout_rel;

                    if (rollout_cost < (Cost) DBL_MAX &&
                        rollout_cost < ctx.phase_best_cost)
                    {
                        ctx.phase_best_cost = rollout_cost;
                        if (mcts_trace_search)
                            phase_best_order = mcts_capture_order(leaf, best_rollout_steps, ctx.run_context);
                    }

                    if (rollout_rel && REL_HAS_CHEAPEST_PATH(rollout_rel) &&
                        rollout_cost < best_cost)
                    {
                        List *tree_path = get_node_merge_path(leaf, ctx.run_context);
                        List *full_order = list_concat(tree_path, best_rollout_steps);

                        best_cost = rollout_cost;
                        best_rel = rollout_rel;
                        ctx.best_merge_order = full_order;
                        ctx.best_cost = best_cost;
                        best_rel_global = best_rel;
                        best_cost_global = best_cost;
                        best_merge_order_global = full_order;
                        best_cost_phase = j + 1;
                        best_cost_iteration = i + 1;
                    }

                    INSTR_TIME_SET_CURRENT(phase_end);
                    INSTR_TIME_SUBTRACT(phase_end, phase_start);
                    time_rollout_ms += INSTR_TIME_GET_MILLISEC(phase_end);
                    total_rollouts++;
                }

                INSTR_TIME_SET_CURRENT(phase_start);
                mcts_backpropagate(&ctx, leaf, rollout_cost);
                INSTR_TIME_SET_CURRENT(phase_end);
                INSTR_TIME_SUBTRACT(phase_end, phase_start);
                time_backprop_ms += INSTR_TIME_GET_MILLISEC(phase_end);
                ctx.iterations_done++;
                continue;
            }

            /* Depth-limited: rollout directly without further tree expansion */
            if (depth > 0 && leaf->node_depth >= depth && !leaf->terminal)
            {
                 List       *rollout_steps = NIL;
                 Cost        best_rollout_cost = (Cost) DBL_MAX;
                 RelOptInfo *best_rollout_rel = NULL;
                 List       *best_rollout_steps = NIL;

                 mcts_debug_log("mcts_extreme.mcts: iter %d depth limit reached (%d >= %d), rollout from leaf",
                                i + 1, leaf->node_depth, depth);

                 INSTR_TIME_SET_CURRENT(phase_start);

                 /*
                  * Re-seed PRNG per rollout for diversity.  Derived from
                  * (random_seed, phase j, iter i, rollout rr) so that a
                  * fixed random_seed reproduces the rollout sequence
                  * exactly even under parallel sweep load.  When
                  * random_seed = 0 the helper falls back to wall-clock.
                  */
                 mcts_mix_seedbuf(&ctx, (uint64) j, (uint64) i, 0);

                 for (int rr = 0; rr < mcts_rollouts_per_leaf; rr++)
                 {
                     /*
                      * rollout=luby: re-seed the PRNG per rollout (see the
                      * non-depth-limited dispatch site below for the rationale).
                      */
                     if (mcts_rollout_mode == MCTS_EXTREME_ROLLOUT_LUBY)
                     {
                         unsigned long long lubymix =
                             (unsigned long long) luby_value(rr + 1);
                         ctx.seedbuf[0] ^= (unsigned short) (lubymix & 0xFFFF);
                         ctx.seedbuf[1] ^= (unsigned short) ((lubymix >> 16) & 0xFFFF);
                         ctx.seedbuf[2] ^= (unsigned short) ((lubymix >> 32) & 0xFFFF);
                     }
                     if (mcts_plan_shape == 0)
                         rollout_cost = mcts_rollout_bushy(&ctx, leaf, &rollout_rel, &rollout_steps);
                     else
                         rollout_cost = mcts_rollout_linear(&ctx, leaf, &rollout_rel, &rollout_steps);
                     mcts_debug_log("exhausted rollout num:%d; cost found:%f mode=%d", rr, (double)rollout_cost, mcts_rollout_mode);
                     if (rollout_cost < best_rollout_cost)
                     {
                         best_rollout_cost = rollout_cost;
                         best_rollout_rel = rollout_rel;
                         best_rollout_steps = rollout_steps;
                     }

                     /* Prepare a distinct deterministic seed for rollout rr+1. */
                     mcts_mix_seedbuf(&ctx, (uint64) j, (uint64) i, (uint64) (rr + 1));
                 }

                 rollout_cost = best_rollout_cost;
                 rollout_rel = best_rollout_rel;

                if (rollout_cost < (Cost) DBL_MAX &&
                    rollout_cost < ctx.phase_best_cost)
                {
                    ctx.phase_best_cost = rollout_cost;
                    if (mcts_trace_search)
                        phase_best_order = mcts_capture_order(leaf, best_rollout_steps, ctx.run_context);
                }

                 mcts_debug_log("mcts_extreme.mcts: iter %d depth-limit rollout done, cost=%.2f reward=%.4f",
                               i + 1, (double) rollout_cost, cost_to_reward(&ctx, rollout_cost));

                 if (rollout_rel && REL_HAS_CHEAPEST_PATH(rollout_rel) &&
                     rollout_cost < best_cost)
                 {
                     List *tree_path = get_node_merge_path(leaf, ctx.run_context);
                     List *full_order = list_concat(tree_path, best_rollout_steps);

                     best_cost = rollout_cost;
                     best_rel = rollout_rel;
                     ctx.best_merge_order = full_order;
                     ctx.best_cost = best_cost;
                     best_rel_global = best_rel;
                     best_cost_global = best_cost;
                     best_merge_order_global = full_order;
                     best_cost_phase = j + 1;
                     best_cost_iteration = i + 1;
                 }

                 INSTR_TIME_SET_CURRENT(phase_end);
                 INSTR_TIME_SUBTRACT(phase_end, phase_start);
                 time_rollout_ms += INSTR_TIME_GET_MILLISEC(phase_end);
                 total_rollouts++;

                 leaf->exhausted = true;
                 total_exhausted++;

                 INSTR_TIME_SET_CURRENT(phase_start);
                mcts_backpropagate(&ctx, leaf, rollout_cost);
                 INSTR_TIME_SET_CURRENT(phase_end);
                 INSTR_TIME_SUBTRACT(phase_end, phase_start);
                 time_backprop_ms += INSTR_TIME_GET_MILLISEC(phase_end);
                 mcts_debug_log("mcts_extreme.mcts: iter %d backpropagation done best_reward=%.4f best_cost=%.2f",
                                i + 1, root_node->best_reward, (double) best_cost);
                 ctx.iterations_done++;
                 continue;
             }

             /* Within depth limit: expand the tree by one node */
             INSTR_TIME_SET_CURRENT(phase_start);
             child = mcts_expand(&ctx, leaf);
             INSTR_TIME_SET_CURRENT(phase_end);
             INSTR_TIME_SUBTRACT(phase_end, phase_start);
             time_expansion_ms += INSTR_TIME_GET_MILLISEC(phase_end);

             if (child == leaf)
             {
                 mcts_debug_log("mcts_extreme.mcts: iter %d expansion skipped", i + 1);
                 continue;
             }

             mcts_debug_log("mcts_extreme.mcts: iter %d expansion done, child terminal=%d clumps=%d",
                            i + 1, child->terminal, list_length(child->clumps));

             if (!child->terminal)
             {
                 List *rollout_steps = NIL;

                 INSTR_TIME_SET_CURRENT(phase_start);
                 /*
                  * For rollout=luby, re-seed the PRNG before each rollout by
                  * XORing the current phase seed with luby(rollout_idx),
                  * decorrelating consecutive rollouts within the phase.  For
                  * rollout=random we leave the PRNG alone — the phase seed
                  * is fixed for the whole phase, matching the historic
                  * behaviour.
                  */
                 if (mcts_rollout_mode == MCTS_EXTREME_ROLLOUT_LUBY)
                 {
                     unsigned long long lubymix =
                         (unsigned long long) luby_value(total_rollouts + 1);
                     ctx.seedbuf[0] ^= (unsigned short) (lubymix & 0xFFFF);
                     ctx.seedbuf[1] ^= (unsigned short) ((lubymix >> 16) & 0xFFFF);
                     ctx.seedbuf[2] ^= (unsigned short) ((lubymix >> 32) & 0xFFFF);
                 }
                 /* Dispatch by plan shape K: 0 = bushy, else linear (zig-zag). */
                 if (mcts_plan_shape == 0)
                     rollout_cost = mcts_rollout_bushy(&ctx, child, &rollout_rel, &rollout_steps);
                 else
                     rollout_cost = mcts_rollout_linear(&ctx, child, &rollout_rel, &rollout_steps);
                 INSTR_TIME_SET_CURRENT(phase_end);
                 INSTR_TIME_SUBTRACT(phase_end, phase_start);
                 time_rollout_ms += INSTR_TIME_GET_MILLISEC(phase_end);
                 total_rollouts++;

                if (rollout_cost < (Cost) DBL_MAX &&
                    rollout_cost < ctx.phase_best_cost)
                {
                    ctx.phase_best_cost = rollout_cost;
                    if (mcts_trace_search)
                        phase_best_order = mcts_capture_order(child, rollout_steps, ctx.run_context);
                }

                 mcts_debug_log("mcts_extreme.mcts: iter %d rollout done, cost=%.2f reward=%.4f mode=%d",
                               i + 1, (double) rollout_cost, cost_to_reward(&ctx, rollout_cost), mcts_rollout_mode);

                 if (rollout_rel && REL_HAS_CHEAPEST_PATH(rollout_rel) &&
                     rollout_cost < best_cost)
                 {
                     List *tree_path = get_node_merge_path(child, ctx.run_context);
                     List *full_order = list_concat(tree_path, rollout_steps);

                     best_cost = rollout_cost;
                     best_rel = rollout_rel;
                     best_rel_global = best_rel;
                     ctx.best_merge_order = full_order;
                     ctx.best_cost = best_cost;
                     best_cost_global = best_cost;
                     best_merge_order_global = full_order;
                     best_cost_phase = j + 1;
                     best_cost_iteration = i + 1;
                 }
             }
             else
             {
                 rollout_cost = child->best_cost_in_subtree;
                if (rollout_cost < (Cost) DBL_MAX &&
                    rollout_cost < ctx.phase_best_cost)
                {
                    ctx.phase_best_cost = rollout_cost;
                    if (mcts_trace_search)
                        phase_best_order = mcts_capture_order(child, NIL, ctx.run_context);
                }
                 mcts_debug_log("mcts_extreme.mcts: iter %d terminal node, cost=%.2f reward=%.4f",
                               i + 1, (double) rollout_cost, cost_to_reward(&ctx, rollout_cost));

                 if (child->clumps && list_length(child->clumps) == 1)
                 {
                     MctsClump *c = (MctsClump *) linitial(child->clumps);
                     if (REL_HAS_CHEAPEST_PATH(c->joinrel) &&
                         REL_CHEAPEST_PATH(c->joinrel)->total_cost < best_cost)
                     {
                         List *full_order = get_node_merge_path(child, ctx.run_context);

                         best_cost = REL_CHEAPEST_PATH(c->joinrel)->total_cost;
                         best_rel = c->joinrel;
                         best_rel_global = best_rel;
                         ctx.best_merge_order = full_order;
                         ctx.best_cost = best_cost;
                         best_cost_global = best_cost;
                         best_merge_order_global = full_order;
                         best_cost_phase = j + 1;
                         best_cost_iteration = i + 1;
                     }
                 }
             }

             INSTR_TIME_SET_CURRENT(phase_start);
            mcts_backpropagate(&ctx, child, rollout_cost);
             INSTR_TIME_SET_CURRENT(phase_end);
             INSTR_TIME_SUBTRACT(phase_end, phase_start);
             time_backprop_ms += INSTR_TIME_GET_MILLISEC(phase_end);
             mcts_debug_log("mcts_extreme.mcts: iter %d backpropagation done best_reward=%.4f best_cost=%.2f",
                            i + 1, root_node->best_reward, (double) best_cost);
             ctx.iterations_done++;
         }
         mcts_debug_log("mcts_extreme.mcts: after phase %d best_cost=%.2f", j + 1, best_cost);

         /*
          * Per-phase Luby trace: the iteration budget changes per restart
          * (start_budget * luby_value(phase)); record it for introspection.
          */
         if (mcts_trace_search)
         {
             int     luby = mcts_luby_enabled ? luby_value(j + 1) : 1;
             ListCell *stepcell_phasesub;

             mcts_phasetrace_add(j + 1, luby, mcts_start_budget * luby, i,
                                 (double) ctx.phase_best_cost,
                                 best_cost_phase == j + 1);

             /*
              * Per-phase subplan trace: record the joinrels (building blocks) of
              * this phase's best plan, so subplan reuse across phases can be
              * compared.  Each merge step's result is left UNION right.
              */
             foreach(stepcell_phasesub, phase_best_order)
             {
                 MctsMergeStep *step = (MctsMergeStep *) lfirst(stepcell_phasesub);
                 Bitmapset  *u = bms_union(step->left, step->right);

                 mcts_phasesubtrace_add(j + 1,
                                        mcts_trace_relids_text(root, u),
                                        bms_num_members(u));
                 bms_free(u);
             }
         }

         /*
          * Search-decision trace: snapshot this phase's accumulated tree if it
          * reaches a cost at least as good as any phase captured so far, so the
          * final buffer describes the tree of the phase that found the winner.
          */
         if (mcts_trace_search && root_node != NULL &&
             root_node->best_cost_in_subtree <= searchtrace_best)
         {
             searchtrace_best = root_node->best_cost_in_subtree;
             mcts_capture_search_tree(root, root_node, best_merge_order_global);
         }

         /* Early stopping via patience: halt if no improvement for N phases */
         if (best_cost_phase == j + 1)
             phases_since_improvement = 0;
         else
             phases_since_improvement++;

         if (mcts_patience > 0 && phases_since_improvement >= mcts_patience)
         {
             mcts_debug_log("mcts_extreme.mcts: early stopping after %d phases without improvement", phases_since_improvement);
             break;
         }

         /* Log best merge order as a nested join tree for diagnostics */
         if (best_merge_order_global != NIL)
         {
             int nrels = list_length(ctx.initial_rels);
             int max_relid = root->simple_rel_array_size;
             const char **rel_names = (const char **) palloc0(max_relid * sizeof(const char *));
             StringInfoData *labels = (StringInfoData *) palloc0(max_relid * sizeof(StringInfoData));
             for (int r = 0; r < nrels; r++)
             {
                 RelOptInfo *ri = (RelOptInfo *) list_nth(ctx.initial_rels, r);
                 int x = bms_next_member(ri->relids, -1);
                 initStringInfo(&labels[x]);
                 appendStringInfoString(&labels[x], root->simple_rte_array[x]->eref->aliasname);
             }

             ListCell *slc;
             int step_num = 0;
             foreach(slc, best_merge_order_global)
             {
                 MctsMergeStep *step = (MctsMergeStep *) lfirst(slc);
                 int lx = bms_next_member(step->left, -1);
                 int rx = bms_next_member(step->right, -1);
                 int left_size = bms_num_members(step->left);
                 int right_size = bms_num_members(step->right);
                 step_num++;
                 if (mcts_log_steps)
                     elog(WARNING, "mcts_extreme.mcts: step %d: (%d)%s + (%d)%s",
                          step_num,
                          left_size, labels[lx].data ? labels[lx].data : "?",
                          right_size, labels[rx].data ? labels[rx].data : "?");
                 StringInfoData merged;
                 initStringInfo(&merged);
                 appendStringInfo(&merged, "(%s %s)", labels[lx].data, labels[rx].data);
                 /* Propagate new label to all members of the merged relid set */
                 int m = -1;
                 Relids both = bms_union(step->left, step->right);
                 while ((m = bms_next_member(both, m)) >= 0)
                 {
                     if (labels[m].data)
                         pfree(labels[m].data);
                     initStringInfo(&labels[m]);
                     appendStringInfoString(&labels[m], merged.data);
                 }
                 pfree(merged.data);
                 bms_free(both);
             }
             /* Final label lives at any member of the full relid set */
             int final_idx = bms_next_member(ctx.all_query_relids, -1);
             if (mcts_log_debug)
                 elog(WARNING, "mcts_extreme.mcts: best order = %s", labels[final_idx].data);
             pfree(labels);
             pfree(rel_names);
         }
     }

     mcts_debug_log("mcts_extreme.mcts: best_cost_global=%.2f", best_cost_global);
     mcts_debug_log("mcts_extreme.mcts: done iterations=%d rollout_failures=%d best_cost=%g",
                    ctx.iterations_done, ctx.rollout_failures, (double) best_cost);
     mcts_debug_log("mcts_extreme.mcts: cost_evals=%d cache_hits=%d",
                    mcts_cost_eval_count, ctx.cache_hits);

     INSTR_TIME_SET_CURRENT(total_end);
     INSTR_TIME_SUBTRACT(total_end, total_start);

     /* Replay the best merge order to build final join RelOptInfos */
     {
         instr_time replay_start, replay_end;

         INSTR_TIME_SET_CURRENT(replay_start);
         if (best_merge_order_global != NIL)
         {
             /* Truncate to pre-search state so replay builds clean join rels */
             root->join_rel_list = list_truncate(root->join_rel_list, saved_join_rel_len);
             root->join_rel_hash = NULL;
             best_rel = mcts_replay_best_order(root, initial_rels, best_merge_order_global);
         }
         INSTR_TIME_SET_CURRENT(replay_end);
         INSTR_TIME_SUBTRACT(replay_end, replay_start);

         {
             int    total_lookups = mcts_cost_eval_count + ctx.cache_hits;
             double hit_ratio = total_lookups > 0
                 ? 100.0 * ctx.cache_hits / total_lookups : 0.0;
             double search_ms = INSTR_TIME_GET_MILLISEC(total_end);
             double replay_ms = INSTR_TIME_GET_MILLISEC(replay_end);

             mcts_last_stats.valid = true;
             mcts_last_stats.algorithm = mcts_search_algorithm_name();
             mcts_last_stats.mode = (mcts_plan_shape == 0) ? "bushy" :
                 (mcts_plan_shape == 1 ? "linear" : "K-component");
             mcts_last_stats.num_rels = ctx.num_rels;
             mcts_last_stats.phases = len_phases;
             mcts_last_stats.iterations = ctx.iterations_done;
             mcts_last_stats.accepted_moves = 0;
             mcts_last_stats.depth_limit_hits = total_rollouts;
             mcts_last_stats.total_random_rollouts = total_rollouts * mcts_rollouts_per_leaf;
             mcts_last_stats.rollout_failures = ctx.rollout_failures;
             mcts_last_stats.exhausted = total_exhausted;
             mcts_last_stats.best_cost = best_cost_global;
             mcts_last_stats.best_cost_phase = best_cost_phase;
             mcts_last_stats.best_cost_iteration = best_cost_iteration;
             mcts_last_stats.cost_evals = mcts_cost_eval_count;
             mcts_last_stats.cache_on = mcts_cache_enabled;
             mcts_last_stats.cache_hits = ctx.cache_hits;
             mcts_last_stats.cache_size = (mcts_cache_enabled && ctx.join_cost_cache)
                 ? (long long) hash_get_num_entries(ctx.join_cost_cache) : 0LL;
             mcts_last_stats.cache_max_size = mcts_cache_max_size;
             mcts_last_stats.cache_hit_ratio = hit_ratio;
             mcts_last_stats.selection_time_ms = time_selection_ms;
             mcts_last_stats.expansion_time_ms = time_expansion_ms;
             mcts_last_stats.rollout_time_ms = time_rollout_ms;
             mcts_last_stats.backprop_time_ms = time_backprop_ms;
             mcts_last_stats.search_time_ms = search_ms;
             mcts_last_stats.replay_time_ms = replay_ms;
             mcts_last_stats.total_planning_mcts_time_ms = search_ms + replay_ms;
             mcts_last_stats.final_temperature = 0.0;

             /* Debug summary; the same numbers also appear in EXPLAIN output. */
             if (mcts_log_debug)
             elog(WARNING, "mcts_extreme.mcts stats:\n"
                  "  plan_shape      = %s (K=%d)\n"
                  "  rollout_mode    = %s\n"
                  "  uct_aggregation = %s\n"
                  "  expand_strategy = %s\n"
                  "  full_budget     = %s\n"
                  "  luby_budget     = %s\n"
                  "  rels            = %d\n"
                  "  phases          = %d\n"
                  "  iterations      = %d\n"
                  "  best_cost       = %.2f (phase=%d iter=%d)\n"
                  "  cache_hits      = %d\n"
                  "  cache_size      = %lld\n"
                  "  hit_ratio       = %.1f%%\n"
                  "  search_time     = %.3f ms\n"
                  "  replay_time     = %.3f ms\n"
                  "  total_time      = %.3f ms\n"
                  "  selection_time  = %.3f ms\n"
                  "  expansion_time  = %.3f ms\n"
                  "  rollout_time    = %.3f ms\n"
                  "  backprop_time   = %.3f ms",
                  (mcts_plan_shape == 0) ? "bushy" :
                      (mcts_plan_shape == 1 ? "linear" : "K-component"),
                  mcts_plan_shape,
                  (mcts_rollout_mode == MCTS_EXTREME_ROLLOUT_LUBY) ? "luby" : "random",
                  (mcts_uct_aggregation == MCTS_EXTREME_UCT_AGG_AVERAGE) ? "average" : "best",
                  mcts_expand_strategy_name(),
                  mcts_full_budget ? "on" : "off",
                  mcts_luby_enabled ? "on" : "off",
                  ctx.num_rels,
                  len_phases,
                  ctx.iterations_done,
                  best_cost_global, best_cost_phase, best_cost_iteration,
                  ctx.cache_hits,
                  mcts_last_stats.cache_size,
                  hit_ratio,
                  search_ms,
                  replay_ms,
                  search_ms + replay_ms,
                  time_selection_ms,
                  time_expansion_ms,
                  time_rollout_ms,
                  time_backprop_ms);
         }
     }

     return best_rel_global;
 }


 /* ----------
  *  EXPLAIN integration
  *
  *	  Appends the most recent MCTS statistics block to EXPLAIN output.
  * ----------
  */
 static explain_per_plan_hook_type prev_explain_per_plan_hook = NULL;

 static void
 mcts_explain_per_plan(PlannedStmt *plannedstmt, IntoClause *into,
                       ExplainState *es, const char *queryString,
                       ParamListInfo params, QueryEnvironment *queryEnv)
 {
     if (prev_explain_per_plan_hook)
         prev_explain_per_plan_hook(plannedstmt, into, es, queryString,
                                    params, queryEnv);

     if (!mcts_last_stats.valid)
         return;

     ExplainOpenGroup("mcts_extreme.mcts", "mcts_extreme.mcts", true, es);
     ExplainPropertyText("MCTS Search Algorithm",
                         mcts_last_stats.algorithm ? mcts_last_stats.algorithm : "mcts", es);
     if (mcts_last_stats.mode)
         ExplainPropertyText("MCTS Mode", mcts_last_stats.mode, es);
     ExplainPropertyInteger("MCTS Relations", NULL, mcts_last_stats.num_rels, es);
    ExplainPropertyInteger("MCTS Phases", NULL, mcts_last_stats.phases, es);
    ExplainPropertyInteger("MCTS Iterations", NULL, mcts_last_stats.iterations, es);
    if (mcts_last_stats.accepted_moves > 0)
        ExplainPropertyInteger("MCTS Accepted Moves", NULL, mcts_last_stats.accepted_moves, es);
    if (mcts_last_stats.final_temperature > 0.0)
        ExplainPropertyFloat("MCTS Final Temperature", NULL, mcts_last_stats.final_temperature, 3, es);
    ExplainPropertyText("MCTS Expand Strategy",
                        mcts_expand_strategy_name(), es);
    ExplainPropertyText("MCTS Full Budget", mcts_full_budget ? "on" : "off", es);
    ExplainPropertyInteger("MCTS Depth-Limit Rollouts", NULL, mcts_last_stats.depth_limit_hits, es);
     ExplainPropertyInteger("MCTS Total Random Rollouts", NULL, mcts_last_stats.total_random_rollouts, es);
     ExplainPropertyInteger("MCTS Rollout Failures", NULL, mcts_last_stats.rollout_failures, es);
     ExplainPropertyInteger("MCTS Exhausted", NULL, mcts_last_stats.exhausted, es);
     ExplainOpenGroup("MCTS Best Plan", "MCTS Best Plan", true, es);
     if (mcts_last_stats.best_cost >= (Cost) DBL_MAX)
     {
         /*
          * MCTS never found a complete plan in this run -- for example the
          * depth limit was too low to reach a terminal node during tree
          * expansion.  The query still planned because the hook fell through
          * to the regular planner; reporting a DBL_MAX cost would be
          * misleading, so we say so plainly instead.
          */
         ExplainPropertyText("MCTS Best Cost", "no plan from MCTS (fallback used)", es);
     }
     else
     {
         ExplainPropertyFloat("MCTS Best Cost", NULL, mcts_last_stats.best_cost, 2, es);
         ExplainPropertyInteger("  Found at Phase", NULL, mcts_last_stats.best_cost_phase, es);
         ExplainPropertyInteger("  Found at Iteration", NULL, mcts_last_stats.best_cost_iteration, es);
     }
     ExplainCloseGroup("MCTS Best Plan", "MCTS Best Plan", true, es);
     ExplainOpenGroup("MCTS Cache Effectiveness", "MCTS Cache Effectiveness", true, es);
     ExplainPropertyText("MCTS Cache", mcts_last_stats.cache_on ? "on" : "off", es);
     ExplainPropertyInteger("  Cost Evals", NULL, mcts_last_stats.cost_evals, es);
     ExplainPropertyInteger("  Cache Hits", NULL, mcts_last_stats.cache_hits, es);
     ExplainPropertyInteger("  Cache Size", NULL, (int64) mcts_last_stats.cache_size, es);
     if (mcts_last_stats.cache_max_size > 0)
         ExplainPropertyInteger("  Cache Max Size", NULL, mcts_last_stats.cache_max_size, es);
     else
         ExplainPropertyText("  Cache Max Size", "unlimited", es);
     ExplainPropertyFloat("  Cache Hit Ratio", "%", mcts_last_stats.cache_hit_ratio, 1, es);
     ExplainCloseGroup("MCTS Cache Effectiveness", "MCTS Cache Effectiveness", true, es);
     ExplainPropertyFloat("MCTS Total Planning Time", "ms", mcts_last_stats.total_planning_mcts_time_ms, 3, es);
     ExplainOpenGroup("MCTS Search Time Breakdown", "MCTS Search Time Breakdown", true, es);
     ExplainPropertyFloat("MCTS Search Time", "ms", mcts_last_stats.search_time_ms, 3, es);
     ExplainPropertyFloat("  Selection Time", "ms", mcts_last_stats.selection_time_ms, 3, es);
     ExplainPropertyFloat("  Expansion Time", "ms", mcts_last_stats.expansion_time_ms, 3, es);
     ExplainPropertyFloat("  Rollout Time", "ms", mcts_last_stats.rollout_time_ms, 3, es);
     ExplainPropertyFloat("  Backprop Time", "ms", mcts_last_stats.backprop_time_ms, 3, es);
     ExplainPropertyFloat("MCTS Replay Time", "ms", mcts_last_stats.replay_time_ms, 3, es);
     ExplainCloseGroup("MCTS Search Time Breakdown", "MCTS Search Time Breakdown", true, es);
     ExplainCloseGroup("mcts_extreme.mcts", "mcts_extreme.mcts", true, es);

     mcts_last_stats.valid = false;
 }

 /* ----------
  *  GUC registration + hook install
  * ----------
  */

 /*
  * _PG_init_mcts_gucs
  *	  Register all mcts_extreme.* GUC variables and install the
  *	  EXPLAIN hook. Called from _PG_init in core/mcts_extreme.c.
  */
 bool
 mcts_is_enabled(void)
 {
     return mcts_extreme_enabled;
 }

 void
 _PG_init_mcts_gucs(void)
 {
     DefineCustomBoolVariable("mcts_extreme.enabled",
                              "Enable MCTS Extreme join ordering",
                              NULL,
                              &mcts_extreme_enabled,
                              true,
                              PGC_USERSET,
                              0, NULL, NULL, NULL);

     DefineCustomIntVariable("mcts_extreme.min_relations",
                             "Minimum relations to trigger MCTS",
                             NULL,
                             &mcts_extreme_min_relations,
                             3,
                             3, BITS_PER_BITMAPWORD,
                             PGC_USERSET,
                             0, NULL, NULL, NULL);

     DefineCustomRealVariable("mcts_extreme.exploration_constant",
                              "UCT exploration constant",
                              NULL,
                              &mcts_extreme_exploration_constant,
                              1.4,
                              0.0, 1000.0,
                              PGC_USERSET,
                              0, NULL, NULL, NULL);

     DefineCustomRealVariable("mcts_extreme.gamma",
                              "UCT Extreme exploration exponent",
                              NULL,
                              &mcts_extreme_gamma,
                              0.5,
                              0.01, 2.0,
                              PGC_USERSET,
                              0, NULL, NULL, NULL);

     mcts_trace_define_gucs();

     DefineCustomIntVariable("mcts_extreme.random_seed",
                             "Random seed (0 = time-based)",
                             NULL,
                             &mcts_extreme_random_seed,
                             0,
                             0, 2147483647,
                             PGC_USERSET,
                             0, NULL, NULL, NULL);

     DefineCustomIntVariable("mcts_extreme.depth",
                             "MCTS tree depth limit",
                             NULL,
                             &depth,
                             10,
                             2, 100,
                             PGC_USERSET,
                             0, NULL, NULL, NULL);

     DefineCustomIntVariable("mcts_extreme.start_budget",
                             "Start budget for MCTS (iterations in phase 1)",
                             NULL,
                             &mcts_start_budget,
                             480,
                             1, 100000000,
                             PGC_USERSET,
                             0, NULL, NULL, NULL);

     DefineCustomIntVariable("mcts_extreme.phases",
                             "Number of Luby-sequence restart phases",
                             NULL,
                             &mcts_phases,
                             1,
                             1, 1000,
                             PGC_USERSET,
                             0, NULL, NULL, NULL);

     DefineCustomIntVariable("mcts_extreme.patience",
                             "Stop after this many phases without improvement (0 = disabled)",
                             NULL,
                             &mcts_patience,
                             0,
                             0, 1000,
                             PGC_USERSET,
                             0, NULL, NULL, NULL);

     DefineCustomIntVariable("mcts_extreme.top_k",
                             "Limit actions to top-k cheapest (0 = all)",
                             NULL,
                             &mcts_top_k,
                             10,
                             0, 10000,
                             PGC_USERSET,
                             0, NULL, NULL, NULL);

     DefineCustomEnumVariable("mcts_extreme.expand_strategy",
                              "Top-k expansion ranking: cost, row, mixed_025, "
                              "mixed_050, or selectivity.",
                              NULL,
                              &mcts_expand_strategy,
                              MCTS_EXTREME_EXPAND_COST,
                              mcts_extreme_expand_strategy_options,
                              PGC_USERSET,
                              0, NULL, NULL, NULL);

     DefineCustomIntVariable("mcts_extreme.rollouts_per_leaf",
                             "Number of random rollouts per depth-limited leaf",
                             NULL,
                             &mcts_rollouts_per_leaf,
                             1,
                             1, 10000,
                             PGC_USERSET,
                             0, NULL, NULL, NULL);

     DefineCustomBoolVariable("mcts_extreme.full_budget",
                              "Continue after the expandable frontier is exhausted, "
                              "spending the whole iteration budget with classic UCB1.",
                              NULL,
                              &mcts_full_budget,
                              false,
                              PGC_USERSET,
                              0, NULL, NULL, NULL);

     DefineCustomBoolVariable("mcts_extreme.cache_enabled",
                              "Enable join cost caching",
                              NULL,
                              &mcts_cache_enabled,
                              true,
                              PGC_USERSET,
                              0, NULL, NULL, NULL);

     DefineCustomIntVariable("mcts_extreme.cache_size",
                             "Initial hash table entries for cost cache",
                             NULL,
                             &mcts_cache_size,
                             256,
                             256, 1000000,
                             PGC_USERSET,
                             0, NULL, NULL, NULL);

     DefineCustomIntVariable("mcts_extreme.cache_max_size",
                             "Max cache entries (0 = unlimited)",
                             NULL,
                             &mcts_cache_max_size,
                             50000,
                             0, INT_MAX,
                             PGC_USERSET,
                             0, NULL, NULL, NULL);

     DefineCustomBoolVariable("mcts_extreme.log_debug",
                              "Emit detailed WARNING messages during MCTS",
                              NULL,
                              &mcts_log_debug,
                              false,
                              PGC_USERSET,
                              0, NULL, NULL, NULL);
     DefineCustomBoolVariable("mcts_extreme.log_steps",
                              "Log each join step of the best merge order with sizes",
                              NULL,
                              &mcts_log_steps,
                              false,
                              PGC_USERSET,
                              0, NULL, NULL, NULL);
     DefineCustomIntVariable("mcts_extreme.plan_shape",
                             "Plan-shape parameter K: 0 = bushy, "
                             "1 = linear (zig-zag), >=2 = K-component bushy",
                             NULL,
                             &mcts_plan_shape,
                             1,
                             0, 100,
                             PGC_USERSET,
                             0, NULL, NULL, NULL);

     /* search_algorithm + force_left_tree + saio_* live in local_search.c */
     local_search_register_gucs();

     DefineCustomEnumVariable("mcts_extreme.rollout",
                              "Rollout strategy: 'random' (fixed PRNG seed per "
                              "phase) or 'luby' (re-seed the PRNG per rollout to "
                              "decorrelate consecutive rollouts).",
                              NULL,
                              &mcts_rollout_mode,
                              MCTS_EXTREME_ROLLOUT_RANDOM,
                              mcts_extreme_rollout_options,
                              PGC_USERSET,
                              0, NULL, NULL, NULL);

     DefineCustomEnumVariable("mcts_extreme.uct_aggregation",
                              "UCT Q-value aggregation: 'best' (UCT-Extreme, best "
                              "reward in subtree) or 'average' (standard UCT, "
                              "mean reward).",
                              NULL,
                              &mcts_uct_aggregation,
                              MCTS_EXTREME_UCT_AGG_BEST,
                              mcts_extreme_uct_aggregation_options,
                              PGC_USERSET,
                              0, NULL, NULL, NULL);

     DefineCustomEnumVariable("mcts_extreme.reward_map",
                              "Reward map phi (cost -> UCT-Extreme reward): "
                              "neg_log (-log(cost), default), "
                              "neg_cost (-cost), or "
                              "norm_neg_log (log-cost normalized into [0,1] "
                              "over the observed cost envelope).",
                              NULL,
                              &mcts_reward_map,
                              MCTS_EXTREME_REWARD_NEG_LOG,
                              mcts_extreme_reward_options,
                              PGC_USERSET,
                              0, NULL, NULL, NULL);

     DefineCustomBoolVariable("mcts_extreme.luby_enabled",
                              "Apply Luby-sequence weighting to per-phase "
                              "iteration budget (off = flat budget per phase).",
                              NULL,
                              &mcts_luby_enabled,
                              true,
                              PGC_USERSET,
                              0, NULL, NULL, NULL);

     prev_explain_per_plan_hook = explain_per_plan_hook;
     explain_per_plan_hook = mcts_explain_per_plan;
 }
