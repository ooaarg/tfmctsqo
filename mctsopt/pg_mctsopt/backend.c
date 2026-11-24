/* Native costing, memo lifetime, trace rendering and plan materialization stay
 * in PG. */
#include "native.inc"

RelOptInfo *pg_mctsopt(PlannerInfo *, List *);
bool mcts_is_enabled(void);
void _PG_init_mcts_gucs(void);

typedef struct {
    PlannerInfo *root;
    Bitmapset *all;
    MemoryContext run, eval;
    HTAB *cache;
    int hits;
} PgHost;
typedef struct {
    int length;
    HTAB *hash;
    MemoryContext context;
} PgScope;

static Relids ids(MctsoptRel r) { return ((RelOptInfo *)r)->relids; }
static void *pg_alloc(void *user, size_t n) { return palloc(n); }
static bool pg_desirable(void *user, MctsoptRel l, MctsoptRel r) {
    return desirable_join(((PgHost *)user)->root, l, r);
}
static bool pg_overlap(void *user, MctsoptRel l, MctsoptRel r) {
    return bms_overlap(ids(l), ids(r));
}
static bool pg_complete(void *user, MctsoptRel r) {
    return bms_equal(((PgHost *)user)->all, ids(r));
}
static double pg_cost(void *user, MctsoptRel r) {
    RelOptInfo *rel = r;
    return rel && REL_HAS_CHEAPEST_PATH(rel) ? REL_CHEAPEST_PATH(rel)->total_cost : DBL_MAX;
}
static double pg_rows(void *user, MctsoptRel r) { return ((RelOptInfo *)r)->rows; }
static void *pg_begin(void *user, bool temporary) {
    PgHost *h = user;
    PgScope *s = palloc(sizeof(*s));
    s->length = list_length(h->root->join_rel_list);
    s->hash = h->root->join_rel_hash;
    s->context = CurrentMemoryContext;
    h->root->join_rel_hash = NULL;
    if (temporary)
        MemoryContextSwitchTo(h->eval);
    return s;
}
static void pg_end(void *user, void *saved) {
    PgHost *h = user;
    PgScope *s = saved;
    h->root->join_rel_list = list_truncate(h->root->join_rel_list, s->length);
    h->root->join_rel_hash = s->hash;
    MemoryContextSwitchTo(s->context);
    pfree(s);
}
static void pg_rank(void *user, MctsoptRel left, MctsoptRel right, int ranking, double *score,
                    double *rows) {
    PgHost *h = user;
    RelOptInfo *l = left, *r = right, *jr = NULL;
    Cost c = DBL_MAX;
    double left_rows = Max(l->rows, 1.0), right_rows = Max(r->rows, 1.0);
    *rows = DBL_MAX;
    if (h->cache && join_cost_cache_lookup(h->cache, l->relids, r->relids, &c, &jr) &&
        (ranking == MCTSOPT_COST || jr != NULL))
        h->hits++;
    else {
        jr = make_join_rel(h->root, l, r);
        if (jr) {
            mcts_cost_eval_count++;
            mcts_finalize_joinrel_paths(h->root, jr);
            if (REL_HAS_CHEAPEST_PATH(jr)) {
                c = REL_CHEAPEST_PATH(jr)->total_cost;
                if (h->cache)
                    join_cost_cache_insert(h->cache, l->relids, r->relids, c, jr);
            }
        }
    }
    if (jr && REL_HAS_CHEAPEST_PATH(jr)) {
        c = REL_CHEAPEST_PATH(jr)->total_cost;
        *rows = Max(jr->rows, 1.0);
    }
    switch (ranking) {
    case MCTSOPT_ROWS:
        *score = *rows;
        break;
    case MCTSOPT_MIX025:
        *score = log(*rows + 1.0) + 0.25 * log(c + 1.0);
        break;
    case MCTSOPT_MIX050:
        *score = log(*rows + 1.0) + 0.50 * log(c + 1.0);
        break;
    case MCTSOPT_SELECTIVITY:
        *score = *rows / Max(left_rows * right_rows, 1.0);
        break;
    default:
        *score = c;
        break;
    }
}
static MctsoptRel pg_join(void *user, MctsoptRel l, MctsoptRel r, bool store) {
    PgHost *h = user;
    return mcts_get_or_build_join(h->root, h->cache, l, r, &h->hits, store);
}
static void pg_restart(void *user) {
    PgHost *h = user;
    if (h->cache) {
        hash_destroy(h->cache);
        h->cache = create_join_cost_cache(h->run, mcts_cache_size);
    }
    h->hits = 0;
}
static uint64_t pg_entropy(void *user) { return (uint64_t)GetCurrentTimestamp(); }
static void pg_interrupt(void *user) { CHECK_FOR_INTERRUPTS(); }
static double pg_clock(void *user) {
    instr_time t;
    INSTR_TIME_SET_CURRENT(t);
    return INSTR_TIME_GET_MILLISEC(t);
}
static void pg_iteration(void *user, int phase, int iter, double pb, double gb, int d) {
    mcts_itertrace_add(phase, iter, pb, gb, d);
}
static void pg_phase(void *user, int phase, int luby, int budget, int iters, double best,
                     bool improved, MctsoptOrder order) {
    PgHost *h = user;
    mcts_phasetrace_add(phase, luby, budget, iters, best, improved);
    for (int i = 0; i < order.count; i++) {
        Bitmapset *u = bms_union(ids(order.items[i].left), ids(order.items[i].right));
        mcts_phasesubtrace_add(phase, mcts_trace_relids_text(h->root, u), bms_num_members(u));
        bms_free(u);
    }
}
static int emit(PgHost *h, const MctsoptNode *node, int snap, int d, const MctsoptNode **spine,
                int spine_len, int emitted) {
    if (d >= 64)
        return emitted;
    for (int j = 0; j < node->child_count && emitted < 2000; j++) {
        const MctsoptNode *ch = node->children[j];
        bool chosen = false;
        int sid;
        for (int i = 0; i < spine_len; i++)
            if (spine[i] == ch)
                chosen = true;
        sid = mcts_searchtrace_add(
            snap, d + 1, mcts_trace_relids_text(h->root, ids(ch->creation.left)),
            mcts_trace_relids_text(h->root, ids(ch->creation.right)), ch->visits, ch->best_reward,
            ch->sum_reward, ch->best_cost, ch->immediate_cost, ch->rows, chosen, false);
        emitted = emit(h, ch, sid, d + 1, spine, spine_len, emitted + 1);
    }
    for (int j = 0; j < node->dropped_count && emitted < 2000; j++) {
        MctsoptDropped *a = &node->dropped[j];
        mcts_searchtrace_add(snap, d + 1, mcts_trace_relids_text(h->root, ids(a->left)),
                             mcts_trace_relids_text(h->root, ids(a->right)), 0, 0, 0, DBL_MAX,
                             a->cost, a->rows, false, true);
        emitted++;
    }
    return emitted;
}
static void pg_snapshot(void *user, const MctsoptNode *root, MctsoptOrder order) {
    PgHost *h = user;
    const MctsoptNode *spine[128], *cur = root;
    int len = 0, sid;
    for (int i = 0; i < order.count && len < 128; i++) {
        const MctsoptNode *next = NULL;
        MctsoptStep s = order.items[i];
        for (int j = 0; j < cur->child_count; j++) {
            const MctsoptNode *ch = cur->children[j];
            if ((bms_equal(ids(ch->creation.left), ids(s.left)) &&
                 bms_equal(ids(ch->creation.right), ids(s.right))) ||
                (bms_equal(ids(ch->creation.left), ids(s.right)) &&
                 bms_equal(ids(ch->creation.right), ids(s.left)))) {
                next = ch;
                break;
            }
        }
        if (!next)
            break;
        spine[len++] = next;
        cur = next;
    }
    mcts_searchtrace_begin();
    sid = mcts_searchtrace_add(-1, 0, NULL, NULL, root->visits, root->best_reward, root->sum_reward,
                               root->best_cost, DBL_MAX, root->rows, true, false);
    emit(h, root, sid, 0, spine, len, 0);
}
RelOptInfo *pg_mctsopt(PlannerInfo *root, List *initial_rels) {
    mcts_last_stats.valid = false;
    if (list_length(initial_rels) < pg_mctsopt_min_relations)
        return NULL;
    PgHost h = {0};
    MctsoptHost host = {&h,           pg_alloc,   pg_desirable, pg_overlap,   pg_complete,
                        pg_cost,      pg_rows,    pg_rank,      pg_join,      pg_begin,
                        pg_end,       pg_restart, pg_entropy,   pg_interrupt, pg_clock,
                        pg_iteration, pg_phase,   pg_snapshot};
    MctsoptConfig cfg = {.seed = (uint64_t)pg_mctsopt_random_seed,
                         .budget = mcts_start_budget,
                         .phases = mcts_phases,
                         .top_k = mcts_top_k,
                         .depth = depth,
                         .shape = mcts_plan_shape,
                         .rollouts = mcts_rollouts_per_leaf,
                         .patience = mcts_patience,
                         .ranking = mcts_expand_strategy,
                         .reward = mcts_reward_map,
                         .exploration = pg_mctsopt_exploration_constant,
                         .gamma = pg_mctsopt_gamma,
                         .average = mcts_uct_aggregation == MCTS_EXTREME_UCT_AGG_AVERAGE,
                         .full_budget = mcts_full_budget,
                         .luby_budget = mcts_luby_enabled,
                         .luby_rollout = mcts_rollout_mode == MCTS_EXTREME_ROLLOUT_LUBY,
                         .force_left = mcts_force_left_tree,
                         .trace = mcts_trace_search};
    MctsoptResult result;
    MctsoptRel *initial;
    MemoryContext old;
    ListCell *lc;
    List *order = NIL;
    int n = list_length(initial_rels), saved_len = list_length(root->join_rel_list), k = 0;
    double started, searched, replayed;
    if (n < 2)
        return NULL;
    foreach (lc, root->join_info_list) {
        SpecialJoinInfo *sji = lfirst(lc);
        if (sji->jointype == JOIN_LEFT || sji->jointype == JOIN_RIGHT || sji->jointype == JOIN_FULL)
            return NULL;
    }
    h.root = root;
    initial = palloc(sizeof(*initial) * n);
    foreach (lc, initial_rels) {
        RelOptInfo *rel = lfirst(lc);
        initial[k++] = rel;
        h.all = bms_union(h.all, rel->relids);
    }
    /* The inherited cache fingerprints one bitmap word. Decline larger IDs. */
    if (h.all->nwords > 1)
        return NULL;
    h.run = AllocSetContextCreate(CurrentMemoryContext, "mctsopt run", ALLOCSET_DEFAULT_SIZES);
    h.eval = AllocSetContextCreate(h.run, "mctsopt eval", ALLOCSET_DEFAULT_SIZES);
    h.cache = mcts_cache_enabled ? create_join_cost_cache(h.run, mcts_cache_size) : NULL;
    if (mcts_trace_search) {
        mcts_searchtrace_begin();
        mcts_phasetrace_begin();
        mcts_itertrace_begin();
        mcts_phasesubtrace_begin();
    }
    started = pg_clock(&h);
    old = MemoryContextSwitchTo(h.run);
    result = mctsopt_search(&cfg, &host, initial, n);
    MemoryContextSwitchTo(old);
    if (result.status == MCTSOPT_INVALID_ARGUMENT)
        ereport(ERROR, (errmsg("pg_mctsopt search configuration exceeds core limits")));
    searched = pg_clock(&h);
    if (result.order.count) {
        for (int i = 0; i < result.order.count; i++) {
            MctsMergeStep *step = palloc(sizeof(*step));
            step->left = ids(result.order.items[i].left);
            step->right = ids(result.order.items[i].right);
            order = lappend(order, step);
        }
        root->join_rel_list = list_truncate(root->join_rel_list, saved_len);
        root->join_rel_hash = NULL;
        (void)mcts_replay_best_order(root, initial_rels, order);
    }
    replayed = pg_clock(&h);
    mcts_last_stats = (MctsLastStats){
        .valid = true,
        .algorithm = "mcts",
        .mode = mcts_plan_shape == 0 ? "bushy" : (mcts_plan_shape == 1 ? "linear" : "K-component"),
        .num_rels = n,
        .phases = mcts_phases,
        .iterations = result.iterations,
        .depth_limit_hits = result.rollouts,
        .total_random_rollouts = result.rollouts * mcts_rollouts_per_leaf,
        .rollout_failures = result.failures,
        .exhausted = result.exhausted,
        .best_cost = result.best_cost,
        .best_cost_phase = result.best_phase,
        .best_cost_iteration = result.best_iteration,
        .cost_evals = mcts_cost_eval_count,
        .cache_on = mcts_cache_enabled,
        .cache_hits = h.hits,
        .cache_size = h.cache ? hash_get_num_entries(h.cache) : 0,
        .cache_max_size = mcts_cache_max_size,
        .cache_hit_ratio = mcts_cost_eval_count + h.hits > 0
                               ? 100.0 * h.hits / (mcts_cost_eval_count + h.hits)
                               : 0,
        .selection_time_ms = result.selection_ms,
        .expansion_time_ms = result.expansion_ms,
        .rollout_time_ms = result.rollout_ms,
        .backprop_time_ms = result.backprop_ms,
        .search_time_ms = searched - started,
        .replay_time_ms = replayed - searched,
        .total_planning_mcts_time_ms = replayed - started};
    /* Preserve the reference return value: it replays but returns its incumbent.
     */
    return result.best_rel;
}
#include "controls.inc"
