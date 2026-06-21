/*-------------------------------------------------------------------------
 *
 * local_search.c
 *	  Non-tree local-search baselines for join-order optimization: SAIO
 *	  (simulated annealing) and iterative improvement.
 *
 *	  These are the controls used to isolate the contribution of MCTS's
 *	  tree-guided search from the K=1 linear/zig-zag plan-shape restriction.
 *	  Both explore the same linear search space under the same iteration
 *	  budget as MCTS, reusing mcts.c's join-building, cost cache and replay
 *	  machinery (see mcts_internal.h).  Selected at run time via
 *	  mcts_extreme.search_algorithm.
 *
 *-------------------------------------------------------------------------
 */
#include "postgres.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>

#include "miscadmin.h"
#include "nodes/bitmapset.h"
#include "nodes/pg_list.h"
#include "optimizer/pathnode.h"
#include "optimizer/paths.h"
#include "portability/instr_time.h"
#include "utils/guc.h"
#include "utils/memutils.h"

#include "mcts_internal.h"

/* ----------
 *  GUC variables
 *
 *  search_algorithm and force_left_tree are shared with mcts.c (declared in
 *  mcts_internal.h); the saio_* knobs are private to this file.
 * ----------
 */
static const struct config_enum_entry mcts_extreme_search_algorithm_options[] = {
	{"mcts", MCTS_EXTREME_SEARCH_MCTS, false},
	{"saio", MCTS_EXTREME_SEARCH_SAIO, false},
	{"iterative_improvement", MCTS_EXTREME_SEARCH_ITERATIVE_IMPROVEMENT, false},
	{"ii", MCTS_EXTREME_SEARCH_ITERATIVE_IMPROVEMENT, true},
	{NULL, 0, false}
};

int		mcts_search_algorithm = MCTS_EXTREME_SEARCH_MCTS;
bool	mcts_force_left_tree = false;

static int    saio_equilibrium_factor = 16;
static double saio_initial_temperature_factor = 2.0;
static double saio_temperature_reduction_factor = 0.9;
static int    saio_moves_before_frozen = 4;
static int    saio_max_iterations = 100000;

/*
 * IterativeImprovementPlan -- a single linear (one-kernel) join order used
 * by the iterative-improvement search.
 */
typedef struct IterativeImprovementPlan
{
	int		   *order;			/* indexes into initial_rels */
	bool	   *base_left;		/* for positions >= 2: join(base, kernel) */
	int			nrels;
} IterativeImprovementPlan;

const char *
mcts_search_algorithm_name(void)
{
    switch (mcts_search_algorithm)
    {
        case MCTS_EXTREME_SEARCH_ITERATIVE_IMPROVEMENT:
            return "iterative_improvement";
        case MCTS_EXTREME_SEARCH_SAIO:
            return "saio";
        case MCTS_EXTREME_SEARCH_MCTS:
        default:
            return "mcts";
    }
}

static List *
saio_copy_merge_order(List *src, MemoryContext cxt)
{
    MemoryContext oldcxt = MemoryContextSwitchTo(cxt);
    List       *out = NIL;
    ListCell   *lc;

    foreach(lc, src)
    {
        MctsMergeStep *step = (MctsMergeStep *) lfirst(lc);
        MctsMergeStep *dup = (MctsMergeStep *) palloc(sizeof(MctsMergeStep));

        dup->left = bms_copy(step->left);
        dup->right = bms_copy(step->right);
        out = lappend(out, dup);
    }

    MemoryContextSwitchTo(oldcxt);
    return out;
}

static void
saio_append_merge_step(List **merge_order, MemoryContext step_cxt,
                       Relids left, Relids right)
{
    MemoryContext oldcxt = MemoryContextSwitchTo(step_cxt);
    MctsMergeStep *step = (MctsMergeStep *) palloc(sizeof(MctsMergeStep));

    step->left = bms_copy(left);
    step->right = bms_copy(right);
    *merge_order = lappend(*merge_order, step);
    MemoryContextSwitchTo(oldcxt);
}

static int
saio_random_index(MctsContext *ctx, int n)
{
    int idx;

    if (n <= 1)
        return 0;
    idx = (int) floor(erand48(ctx->seedbuf) * (double) n);
    if (idx < 0)
        idx = 0;
    if (idx >= n)
        idx = n - 1;
    return idx;
}

static List *
saio_swapped_order(List *order, int a, int b, MemoryContext cxt)
{
    MemoryContext oldcxt = MemoryContextSwitchTo(cxt);
    List       *out = list_copy(order);
    ListCell   *ca;
    ListCell   *cb;
    void       *tmp;

    ca = list_nth_cell(out, a);
    cb = list_nth_cell(out, b);
    tmp = lfirst(ca);
    lfirst(ca) = lfirst(cb);
    lfirst(cb) = tmp;

    MemoryContextSwitchTo(oldcxt);
    return out;
}

static Cost
saio_eval_leftdeep_order(MctsContext *ctx, List *order,
                         MemoryContext step_cxt, List **merge_order_out)
{
    PlannerInfo *root = ctx->root;
    MemoryContext oldcxt;
    int         saved_len;
    struct HTAB *save_hash;
    RelOptInfo *current;
    Cost        total_cost = (Cost) DBL_MAX;
    int         i;
    int         n = list_length(order);
    List       *merge_order = NIL;

    *merge_order_out = NIL;
    if (n == 0)
        return (Cost) DBL_MAX;
    if (n == 1)
    {
        current = (RelOptInfo *) linitial(order);
        if (current && REL_HAS_CHEAPEST_PATH(current))
            return REL_CHEAPEST_PATH(current)->total_cost;
        return (Cost) DBL_MAX;
    }

    oldcxt = MemoryContextSwitchTo(root->planner_cxt);
    saved_len = list_length(root->join_rel_list);
    save_hash = root->join_rel_hash;
    root->join_rel_hash = NULL;

    current = (RelOptInfo *) linitial(order);
    for (i = 1; i < n; i++)
    {
        RelOptInfo *next = (RelOptInfo *) list_nth(order, i);
        RelOptInfo *jr;

        jr = mcts_get_or_build_join(root, ctx->join_cost_cache,
                                    current, next, &ctx->cache_hits, true);
        if (jr == NULL || !REL_HAS_CHEAPEST_PATH(jr))
        {
            total_cost = (Cost) DBL_MAX;
            goto done;
        }

        saio_append_merge_step(&merge_order, step_cxt,
                               current->relids, next->relids);
        current = jr;
    }

    total_cost = REL_CHEAPEST_PATH(current)->total_cost;
    *merge_order_out = merge_order;

done:
    root->join_rel_list = list_truncate(root->join_rel_list, saved_len);
    root->join_rel_hash = save_hash;
    MemoryContextSwitchTo(oldcxt);
    return total_cost;
}

static bool
saio_accept_move(Cost old_cost, Cost new_cost, double temperature,
                 unsigned short seedbuf[3])
{
    double probability;

    if (new_cost >= (Cost) DBL_MAX || !isfinite((double) new_cost))
        return false;
    if (old_cost >= (Cost) DBL_MAX || !isfinite((double) old_cost))
        return true;
    if (new_cost < old_cost)
        return true;
    if (temperature < 1.0)
        return false;

    probability = exp(((double) old_cost - (double) new_cost) / temperature);
    return erand48(seedbuf) < probability;
}

RelOptInfo *
saio_one_kernel(PlannerInfo *root, List *initial_rels)
{
    MctsContext ctx;
    MemoryContext run_context;
    MemoryContext eval_context;
    instr_time total_start, total_end;
    instr_time replay_start, replay_end;
    List       *current_order;
    List       *best_merge_order = NIL;
    List       *initial_steps = NIL;
    Cost        current_cost;
    Cost        best_cost;
    double      temperature;
    int         equilibrium_loops;
    int         failed_moves = 0;
    int         elapsed_at_temperature = 0;
    int         iterations = 0;
    int         accepted_moves = 0;
    int         best_iteration = 0;
    int         saved_join_rel_len;
    RelOptInfo *best_rel = NULL;

    if (list_length(initial_rels) < 2)
        return NULL;

    memset(&ctx, 0, sizeof(ctx));
    ctx.root = root;
    ctx.initial_rels = initial_rels;
    ctx.num_rels = list_length(initial_rels);
    ctx.best_cost = (Cost) DBL_MAX;
    ctx.phase_best_cost = (Cost) DBL_MAX;
    ctx.cost_min = DBL_MAX;
    ctx.cost_max = 0.0;
    ctx.cache_hits = 0;

    run_context = AllocSetContextCreate(CurrentMemoryContext,
                                        "SAIO one-kernel run",
                                        ALLOCSET_DEFAULT_MINSIZE,
                                        ALLOCSET_DEFAULT_INITSIZE,
                                        ALLOCSET_DEFAULT_MAXSIZE);
    eval_context = AllocSetContextCreate(run_context,
                                         "SAIO one-kernel eval",
                                         ALLOCSET_DEFAULT_MINSIZE,
                                         ALLOCSET_DEFAULT_INITSIZE,
                                         ALLOCSET_DEFAULT_MAXSIZE);
    ctx.run_context = run_context;
    ctx.eval_context = eval_context;
    ctx.join_cost_cache = mcts_cache_enabled
        ? create_join_cost_cache(run_context, mcts_cache_size) : NULL;

    mcts_cost_eval_count = 0;
    mcts_mix_seedbuf(&ctx, 0, 0, 0);
    saved_join_rel_len = list_length(root->join_rel_list);

    INSTR_TIME_SET_CURRENT(total_start);

    current_order = list_copy(initial_rels);
    MemoryContextReset(eval_context);
    current_cost = saio_eval_leftdeep_order(&ctx, current_order,
                                            eval_context, &initial_steps);

    if (current_cost >= (Cost) DBL_MAX)
    {
        int attempts = ctx.num_rels * ctx.num_rels;

        for (int a = 0; a < attempts; a++)
        {
            int i = saio_random_index(&ctx, ctx.num_rels);
            int j = saio_random_index(&ctx, ctx.num_rels);
            List *candidate_order;
            List *candidate_steps = NIL;

            if (i == j)
                j = (j + 1) % ctx.num_rels;
            candidate_order = saio_swapped_order(current_order, i, j, run_context);
            MemoryContextReset(eval_context);
            current_cost = saio_eval_leftdeep_order(&ctx, candidate_order,
                                                    eval_context,
                                                    &candidate_steps);
            if (current_cost < (Cost) DBL_MAX)
            {
                current_order = candidate_order;
                initial_steps = candidate_steps;
                break;
            }
        }
    }

    if (current_cost >= (Cost) DBL_MAX)
    {
        MemoryContextDelete(run_context);
        return NULL;
    }

    best_cost = current_cost;
    best_merge_order = saio_copy_merge_order(initial_steps, run_context);
    temperature = (double) current_cost * saio_initial_temperature_factor;
    equilibrium_loops = Max(1, ctx.num_rels * saio_equilibrium_factor);

    while (!(temperature <= 1.0 && failed_moves >= saio_moves_before_frozen))
    {
        int i;
        int j;
        List *candidate_order;
        List *candidate_steps = NIL;
        Cost candidate_cost;
        bool accepted;

        if (saio_max_iterations > 0 && iterations >= saio_max_iterations)
            break;

        i = saio_random_index(&ctx, ctx.num_rels);
        j = saio_random_index(&ctx, ctx.num_rels);
        if (i == j)
            j = (j + 1) % ctx.num_rels;

        candidate_order = saio_swapped_order(current_order, i, j, run_context);
        MemoryContextReset(eval_context);
        candidate_cost = saio_eval_leftdeep_order(&ctx, candidate_order,
                                                  eval_context,
                                                  &candidate_steps);
        accepted = saio_accept_move(current_cost, candidate_cost,
                                    temperature, ctx.seedbuf);

        iterations++;
        elapsed_at_temperature++;

        if (accepted)
        {
            current_order = candidate_order;
            current_cost = candidate_cost;
            accepted_moves++;
            failed_moves = 0;

            if (candidate_cost < best_cost)
            {
                best_cost = candidate_cost;
                best_merge_order = saio_copy_merge_order(candidate_steps,
                                                         run_context);
                best_iteration = iterations;
            }
        }
        else
        {
            failed_moves++;
        }

        if (elapsed_at_temperature >= equilibrium_loops)
        {
            elapsed_at_temperature = 0;
            temperature *= saio_temperature_reduction_factor;
        }
    }

    INSTR_TIME_SET_CURRENT(total_end);
    INSTR_TIME_SUBTRACT(total_end, total_start);

    INSTR_TIME_SET_CURRENT(replay_start);
    if (best_merge_order != NIL)
    {
        root->join_rel_list = list_truncate(root->join_rel_list, saved_join_rel_len);
        root->join_rel_hash = NULL;
        best_rel = mcts_replay_best_order(root, initial_rels, best_merge_order);
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
        mcts_last_stats.algorithm = "saio";
        mcts_last_stats.mode = "one-kernel";
        mcts_last_stats.num_rels = ctx.num_rels;
        mcts_last_stats.phases = 1;
        mcts_last_stats.iterations = iterations;
        mcts_last_stats.accepted_moves = accepted_moves;
        mcts_last_stats.depth_limit_hits = 0;
        mcts_last_stats.total_random_rollouts = 0;
        mcts_last_stats.rollout_failures = 0;
        mcts_last_stats.exhausted = failed_moves;
        mcts_last_stats.best_cost = best_cost;
        mcts_last_stats.best_cost_phase = 1;
        mcts_last_stats.best_cost_iteration = best_iteration;
        mcts_last_stats.cost_evals = mcts_cost_eval_count;
        mcts_last_stats.cache_on = mcts_cache_enabled;
        mcts_last_stats.cache_hits = ctx.cache_hits;
        mcts_last_stats.cache_size = (mcts_cache_enabled && ctx.join_cost_cache)
            ? (long long) hash_get_num_entries(ctx.join_cost_cache) : 0LL;
        mcts_last_stats.cache_max_size = mcts_cache_max_size;
        mcts_last_stats.cache_hit_ratio = hit_ratio;
        mcts_last_stats.selection_time_ms = 0.0;
        mcts_last_stats.expansion_time_ms = 0.0;
        mcts_last_stats.rollout_time_ms = 0.0;
        mcts_last_stats.backprop_time_ms = 0.0;
        mcts_last_stats.search_time_ms = search_ms;
        mcts_last_stats.replay_time_ms = replay_ms;
        mcts_last_stats.total_planning_mcts_time_ms = search_ms + replay_ms;
        mcts_last_stats.final_temperature = temperature;
    }

    MemoryContextDelete(run_context);
    return best_rel;
}

static IterativeImprovementPlan *
ii_alloc_plan(int nrels, MemoryContext cxt)
{
    MemoryContext oldcxt = MemoryContextSwitchTo(cxt);
    IterativeImprovementPlan *plan;

    plan = (IterativeImprovementPlan *) palloc(sizeof(IterativeImprovementPlan));
    plan->nrels = nrels;
    plan->order = (int *) palloc(sizeof(int) * nrels);
    plan->base_left = (bool *) palloc0(sizeof(bool) * nrels);
    MemoryContextSwitchTo(oldcxt);
    return plan;
}

static IterativeImprovementPlan *
ii_copy_plan(const IterativeImprovementPlan *src, MemoryContext cxt)
{
    IterativeImprovementPlan *dst = ii_alloc_plan(src->nrels, cxt);

    memcpy(dst->order, src->order, sizeof(int) * src->nrels);
    memcpy(dst->base_left, src->base_left, sizeof(bool) * src->nrels);
    return dst;
}

static IterativeImprovementPlan *
ii_random_single_kernel_plan(MctsContext *ctx, MemoryContext cxt)
{
    IterativeImprovementPlan *plan;
    int         i;

    plan = ii_alloc_plan(ctx->num_rels, cxt);
    for (i = 0; i < ctx->num_rels; i++)
        plan->order[i] = i;

    for (i = ctx->num_rels - 1; i > 0; i--)
    {
        int         j = saio_random_index(ctx, i + 1);
        int         tmp = plan->order[i];

        plan->order[i] = plan->order[j];
        plan->order[j] = tmp;
    }

    if (!mcts_force_left_tree)
    {
        for (i = 2; i < ctx->num_rels; i++)
            plan->base_left[i] = (erand48(ctx->seedbuf) < 0.5);
    }

    return plan;
}

static void
ii_mutate_single_kernel_plan(MctsContext *ctx, IterativeImprovementPlan *plan)
{
    int         n = plan->nrels;
    int         op;

    if (n <= 1)
        return;

    op = (n <= 2) ? 0 : saio_random_index(ctx, mcts_force_left_tree ? 2 : 3);

    if (op == 0)
    {
        int         i = saio_random_index(ctx, n);
        int         j = saio_random_index(ctx, n);
        int         order_tmp;
        bool        left_tmp;

        if (i == j)
            j = (j + 1) % n;

        order_tmp = plan->order[i];
        plan->order[i] = plan->order[j];
        plan->order[j] = order_tmp;

        left_tmp = plan->base_left[i];
        plan->base_left[i] = plan->base_left[j];
        plan->base_left[j] = left_tmp;
    }
    else if (op == 1)
    {
        int         from = saio_random_index(ctx, n);
        int         to = saio_random_index(ctx, n);
        int         moved_order;
        bool        moved_left;
        int         i;

        if (from == to)
            to = (to + 1) % n;

        moved_order = plan->order[from];
        moved_left = plan->base_left[from];

        if (from < to)
        {
            for (i = from; i < to; i++)
            {
                plan->order[i] = plan->order[i + 1];
                plan->base_left[i] = plan->base_left[i + 1];
            }
        }
        else
        {
            for (i = from; i > to; i--)
            {
                plan->order[i] = plan->order[i - 1];
                plan->base_left[i] = plan->base_left[i - 1];
            }
        }

        plan->order[to] = moved_order;
        plan->base_left[to] = moved_left;
    }
    else
    {
        int         pos = 2 + saio_random_index(ctx, n - 2);

        plan->base_left[pos] = !plan->base_left[pos];
    }

    if (n > 0)
        plan->base_left[0] = false;
    if (n > 1)
        plan->base_left[1] = false;
    if (mcts_force_left_tree)
    {
        int         pos;

        for (pos = 2; pos < n; pos++)
            plan->base_left[pos] = false;
    }
}

static Cost
ii_eval_single_kernel_plan(MctsContext *ctx,
                           const IterativeImprovementPlan *plan,
                           MemoryContext step_cxt,
                           List **merge_order_out)
{
    PlannerInfo *root = ctx->root;
    MemoryContext oldcxt;
    int         saved_len;
    struct HTAB *save_hash;
    RelOptInfo *current;
    RelOptInfo *second;
    RelOptInfo *jr;
    Cost        total_cost = (Cost) DBL_MAX;
    List       *merge_order = NIL;
    int         i;

    *merge_order_out = NIL;
    if (plan->nrels < 2)
        return (Cost) DBL_MAX;

    oldcxt = MemoryContextSwitchTo(root->planner_cxt);
    saved_len = list_length(root->join_rel_list);
    save_hash = root->join_rel_hash;
    root->join_rel_hash = NULL;

    current = (RelOptInfo *) list_nth(ctx->initial_rels, plan->order[0]);
    second = (RelOptInfo *) list_nth(ctx->initial_rels, plan->order[1]);
    jr = mcts_get_or_build_join(root, ctx->join_cost_cache,
                                current, second, &ctx->cache_hits, true);
    if (jr == NULL || !REL_HAS_CHEAPEST_PATH(jr))
        goto done;

    saio_append_merge_step(&merge_order, step_cxt,
                           current->relids, second->relids);
    current = jr;

    for (i = 2; i < plan->nrels; i++)
    {
        RelOptInfo *base = (RelOptInfo *) list_nth(ctx->initial_rels,
                                                   plan->order[i]);

        if (plan->base_left[i])
        {
            jr = mcts_get_or_build_join(root, ctx->join_cost_cache,
                                        base, current, &ctx->cache_hits, true);
            if (jr == NULL || !REL_HAS_CHEAPEST_PATH(jr))
                goto done;
            saio_append_merge_step(&merge_order, step_cxt,
                                   base->relids, current->relids);
        }
        else
        {
            jr = mcts_get_or_build_join(root, ctx->join_cost_cache,
                                        current, base, &ctx->cache_hits, true);
            if (jr == NULL || !REL_HAS_CHEAPEST_PATH(jr))
                goto done;
            saio_append_merge_step(&merge_order, step_cxt,
                                   current->relids, base->relids);
        }

        current = jr;
    }

    total_cost = REL_CHEAPEST_PATH(current)->total_cost;
    *merge_order_out = merge_order;

done:
    root->join_rel_list = list_truncate(root->join_rel_list, saved_len);
    root->join_rel_hash = save_hash;
    MemoryContextSwitchTo(oldcxt);
    return total_cost;
}

static void
ii_fill_last_stats(MctsContext *ctx, Cost best_cost,
                   int evals, int accepted_moves, int restarts,
                   int best_iteration, int cache_hits,
                   double search_ms, double replay_ms)
{
    int         total_lookups = mcts_cost_eval_count + cache_hits;
    double      hit_ratio = total_lookups > 0
        ? 100.0 * cache_hits / total_lookups : 0.0;

    mcts_last_stats.valid = true;
    mcts_last_stats.algorithm = "iterative_improvement";
    mcts_last_stats.mode = "one-kernel";
    mcts_last_stats.num_rels = ctx->num_rels;
    mcts_last_stats.phases = 1;
    mcts_last_stats.iterations = evals;
    mcts_last_stats.accepted_moves = accepted_moves;
    mcts_last_stats.depth_limit_hits = 0;
    mcts_last_stats.total_random_rollouts = 0;
    mcts_last_stats.rollout_failures = 0;
    mcts_last_stats.exhausted = restarts;
    mcts_last_stats.best_cost = best_cost;
    mcts_last_stats.best_cost_phase = 1;
    mcts_last_stats.best_cost_iteration = best_iteration;
    mcts_last_stats.cost_evals = mcts_cost_eval_count;
    mcts_last_stats.cache_on = mcts_cache_enabled;
    mcts_last_stats.cache_hits = cache_hits;
    mcts_last_stats.cache_size = (mcts_cache_enabled && ctx->join_cost_cache)
        ? (long long) hash_get_num_entries(ctx->join_cost_cache) : 0LL;
    mcts_last_stats.cache_max_size = mcts_cache_max_size;
    mcts_last_stats.cache_hit_ratio = hit_ratio;
    mcts_last_stats.selection_time_ms = 0.0;
    mcts_last_stats.expansion_time_ms = 0.0;
    mcts_last_stats.rollout_time_ms = 0.0;
    mcts_last_stats.backprop_time_ms = 0.0;
    mcts_last_stats.search_time_ms = search_ms;
    mcts_last_stats.replay_time_ms = replay_ms;
    mcts_last_stats.total_planning_mcts_time_ms = search_ms + replay_ms;
    mcts_last_stats.final_temperature = 0.0;
}

RelOptInfo *
mcts_iterative_improvement(PlannerInfo *root, List *initial_rels)
{
    MctsContext ctx;
    MemoryContext run_context;
    MemoryContext eval_context;
    IterativeImprovementPlan *current_plan = NULL;
    Cost        current_cost = (Cost) DBL_MAX;
    Cost        best_cost = (Cost) DBL_MAX;
    List       *best_merge_order = NIL;
    RelOptInfo *best_rel = NULL;
    int         saved_join_rel_len;
    int         evals = 0;
    int         accepted_moves = 0;
    int         restarts = 0;
    int         no_improve = 0;
    int         best_iteration = 0;
    int         restart_patience;
    instr_time search_start;
    instr_time search_end;
    instr_time replay_start;
    instr_time replay_end;
    double      search_ms;
    double      replay_ms;

    if (list_length(initial_rels) < 2)
        return NULL;

    memset(&ctx, 0, sizeof(ctx));
    ctx.root = root;
    ctx.initial_rels = initial_rels;
    ctx.num_rels = list_length(initial_rels);
    ctx.best_cost = (Cost) DBL_MAX;
    ctx.phase_best_cost = (Cost) DBL_MAX;
    ctx.cost_min = DBL_MAX;
    ctx.cost_max = 0.0;
    ctx.cache_hits = 0;

    run_context = AllocSetContextCreate(CurrentMemoryContext,
                                        "Iterative improvement run",
                                        ALLOCSET_DEFAULT_MINSIZE,
                                        ALLOCSET_DEFAULT_INITSIZE,
                                        ALLOCSET_DEFAULT_MAXSIZE);
    eval_context = AllocSetContextCreate(run_context,
                                         "Iterative improvement eval",
                                         ALLOCSET_DEFAULT_MINSIZE,
                                         ALLOCSET_DEFAULT_INITSIZE,
                                         ALLOCSET_DEFAULT_MAXSIZE);
    ctx.run_context = run_context;
    ctx.eval_context = eval_context;
    ctx.join_cost_cache = mcts_cache_enabled
        ? create_join_cost_cache(run_context, mcts_cache_size) : NULL;

    mcts_cost_eval_count = 0;
    mcts_mix_seedbuf(&ctx, 0, 0, 0);
    saved_join_rel_len = list_length(root->join_rel_list);
    restart_patience = (mcts_patience > 0)
        ? mcts_patience : Max(32, ctx.num_rels * ctx.num_rels * 2);

    INSTR_TIME_SET_CURRENT(search_start);

    while (evals < mcts_start_budget)
    {
        if (current_plan == NULL ||
            current_cost >= (Cost) DBL_MAX ||
            no_improve >= restart_patience)
        {
            List       *restart_order = NIL;

            current_plan = ii_random_single_kernel_plan(&ctx, run_context);
            MemoryContextReset(eval_context);
            current_cost = ii_eval_single_kernel_plan(&ctx, current_plan,
                                                      eval_context,
                                                      &restart_order);
            evals++;
            no_improve = 0;
            restarts++;

            if (current_cost < best_cost)
            {
                best_cost = current_cost;
                best_merge_order = saio_copy_merge_order(restart_order,
                                                         run_context);
                best_iteration = evals;
            }

            if (evals >= mcts_start_budget)
                break;
            if (current_cost >= (Cost) DBL_MAX)
                continue;
        }

        {
            IterativeImprovementPlan *proposal;
            List       *proposal_order = NIL;
            Cost        proposal_cost;

            proposal = ii_copy_plan(current_plan, run_context);
            ii_mutate_single_kernel_plan(&ctx, proposal);
            MemoryContextReset(eval_context);
            proposal_cost = ii_eval_single_kernel_plan(&ctx, proposal,
                                                       eval_context,
                                                       &proposal_order);
            evals++;

            if (proposal_cost < current_cost)
            {
                current_plan = proposal;
                current_cost = proposal_cost;
                accepted_moves++;
                no_improve = 0;

                if (proposal_cost < best_cost)
                {
                    best_cost = proposal_cost;
                    best_merge_order = saio_copy_merge_order(proposal_order,
                                                             run_context);
                    best_iteration = evals;
                }
            }
            else
            {
                no_improve++;
            }
        }
    }

    INSTR_TIME_SET_CURRENT(search_end);
    INSTR_TIME_SUBTRACT(search_end, search_start);
    search_ms = INSTR_TIME_GET_MILLISEC(search_end);

    INSTR_TIME_SET_CURRENT(replay_start);
    if (best_merge_order != NIL)
    {
        root->join_rel_list = list_truncate(root->join_rel_list, saved_join_rel_len);
        root->join_rel_hash = NULL;
        best_rel = mcts_replay_best_order(root, initial_rels, best_merge_order);
        if (best_rel && REL_HAS_CHEAPEST_PATH(best_rel))
            best_cost = REL_CHEAPEST_PATH(best_rel)->total_cost;
    }
    INSTR_TIME_SET_CURRENT(replay_end);
    INSTR_TIME_SUBTRACT(replay_end, replay_start);
    replay_ms = INSTR_TIME_GET_MILLISEC(replay_end);

    ii_fill_last_stats(&ctx, best_cost, evals, accepted_moves, restarts,
                       best_iteration, ctx.cache_hits, search_ms, replay_ms);

    MemoryContextDelete(run_context);
    return best_rel;
}

/* ----------
 *  GUC registration
 * ----------
 */

/*
 * local_search_register_gucs
 *	  Register the search-algorithm selector and the SAIO knobs.  Called
 *	  once from _PG_init_mcts_gucs() in mcts.c.
 */
void
local_search_register_gucs(void)
{
     DefineCustomBoolVariable("mcts_extreme.force_left_tree",
                              "Force single-kernel local search to always join the growing kernel on the left.",
                              NULL,
                              &mcts_force_left_tree,
                              false,
                              PGC_USERSET,
                              0, NULL, NULL, NULL);

     DefineCustomEnumVariable("mcts_extreme.search_algorithm",
                              "Join-order search algorithm: mcts, saio, or iterative_improvement. "
                              "SAIO and iterative_improvement are constrained to one kernel.",
                              NULL,
                              &mcts_search_algorithm,
                              MCTS_EXTREME_SEARCH_MCTS,
                              mcts_extreme_search_algorithm_options,
                              PGC_USERSET,
                              0, NULL, NULL, NULL);

     DefineCustomIntVariable("mcts_extreme.saio_equilibrium_factor",
                             "SAIO equilibrium iterations per relation before cooling.",
                             NULL,
                             &saio_equilibrium_factor,
                             16,
                             1, 1000000,
                             PGC_USERSET,
                             0, NULL, NULL, NULL);

     DefineCustomRealVariable("mcts_extreme.saio_initial_temperature_factor",
                              "SAIO initial temperature as a multiplier of the initial plan cost.",
                              NULL,
                              &saio_initial_temperature_factor,
                              2.0,
                              0.0, 1000000.0,
                              PGC_USERSET,
                              0, NULL, NULL, NULL);

     DefineCustomRealVariable("mcts_extreme.saio_temperature_reduction_factor",
                              "SAIO temperature multiplier applied after each equilibrium window.",
                              NULL,
                              &saio_temperature_reduction_factor,
                              0.9,
                              0.000001, 1.0,
                              PGC_USERSET,
                              0, NULL, NULL, NULL);

     DefineCustomIntVariable("mcts_extreme.saio_moves_before_frozen",
                             "SAIO stop threshold for consecutive rejected moves after temperature reaches 1.",
                             NULL,
                             &saio_moves_before_frozen,
                             4,
                             1, 1000000,
                             PGC_USERSET,
                             0, NULL, NULL, NULL);

     DefineCustomIntVariable("mcts_extreme.saio_max_iterations",
                             "Safety cap for SAIO candidate moves (0 = no cap).",
                             NULL,
                             &saio_max_iterations,
                             100000,
                             0, 100000000,
                             PGC_USERSET,
                             0, NULL, NULL, NULL);
}
