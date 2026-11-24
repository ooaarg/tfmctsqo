#include "mctsopt.h"

#include <assert.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <string.h>

typedef struct {
    const MctsoptConfig *cfg;
    const MctsoptHost *host;
    uint64_t rng;
    double low, high, phase_best;
    MctsoptResult result;
} Search;

static void *allocate(Search *s, size_t n) {
    void *p = s->host->alloc(s->host->user, n ? n : 1);
    assert(p);
    return p;
}
static double now(Search *s) { return s->host->clock_ms(s->host->user); }
static double cost(Search *s, MctsoptRel r) {
    return r ? s->host->cost(s->host->user, r) : DBL_MAX;
}
static double rows(Search *s, MctsoptRel r) { return s->host->rows(s->host->user, r); }
static void *scope(Search *s, bool temporary) {
    return s->host->begin_scope(s->host->user, temporary);
}
static void end_scope(Search *s, void *p) { s->host->end_scope(s->host->user, p); }
static void interrupt(Search *s) { s->host->interrupt(s->host->user); }

uint64_t mctsopt_seed(uint64_t base, uint64_t c0, uint64_t c1, uint64_t c2) {
    uint64_t mix = base * UINT64_C(0x9e3779b97f4a7c15) + c0 * UINT64_C(0xbf58476d1ce4e5b9) +
                   c1 * UINT64_C(0x94d049bb133111eb) + c2 * UINT64_C(0xc6bc279692b5c323);
    mix ^= mix >> 30;
    mix *= UINT64_C(0xbf58476d1ce4e5b9);
    mix ^= mix >> 27;
    mix *= UINT64_C(0x94d049bb133111eb);
    mix ^= mix >> 31;
    return mix & UINT64_C(0xffffffffffff);
}
double mctsopt_random(uint64_t *state) {
    *state = (*state * UINT64_C(0x5deece66d) + 11) & UINT64_C(0xffffffffffff);
    return ldexp((double)*state, -48);
}
int mctsopt_luby(int i) {
    int64_t size = 1, seq = 1;
    if (i <= 0)
        return 0;
    while (size < i) {
        size = 2 * size + 1;
        seq *= 2;
    }
    while (size > i) {
        seq /= 2;
        size /= 2;
        if (i > size)
            i -= size;
    }
    return (int)seq;
}
static void reseed(Search *s, int phase, int iter, int rollout) {
    uint64_t base = s->cfg->seed ? s->cfg->seed : s->host->entropy(s->host->user);
    s->rng = mctsopt_seed(base, (uint64_t)phase, (uint64_t)iter, (uint64_t)rollout);
}
static double reward(Search *s, double c) {
    if (s->cfg->reward == MCTSOPT_NEG_COST)
        return c >= DBL_MAX ? -DBL_MAX : -c;
    if (s->cfg->reward == MCTSOPT_NORM_NEG_LOG) {
        double lo = s->low, hi = s->high;
        if (c <= 0)
            return 1;
        if (c >= DBL_MAX)
            return 0;
        if (!isfinite(lo) || lo <= 0 || !isfinite(hi) || hi <= 0)
            return 1;
        if (c <= lo)
            return 1;
        if (c >= hi)
            return 0;
        double log_lo = log(lo), log_hi = log(hi), log_c = log(c);
        if (log_hi - log_lo <= 0)
            return 1;
        return (log_hi - log_c) / (log_hi - log_lo);
    }
    if (c <= 0)
        return -log(DBL_MIN);
    if (c >= DBL_MAX)
        return -log(DBL_MAX);
    return -log(c);
}
static bool shape(Search *s, MctsoptClump a, MctsoptClump b, int chains, int bases, bool relax) {
    int k = s->cfg->shape;
    if (k == 1) {
        if (a.size > 1 && b.size > 1)
            return false;
        if (!relax && chains > 0 && a.size == 1 && b.size == 1)
            return false;
    } else if (k >= 2) {
        bool both_base = a.size == 1 && b.size == 1;
        bool both_chain = a.size > 1 && b.size > 1;
        if (both_chain && bases > 0)
            return false;
        if (!relax) {
            if (both_base && chains >= k)
                return false;
            if (!both_base && !both_chain && chains < k && bases >= 2)
                return false;
        }
    }
    return true;
}
static bool orient(Search *s, MctsoptClump a, MctsoptClump b, int chains, int *i, int *j) {
    if (!s->cfg->force_left || s->cfg->shape != 1)
        return true;
    if (a.size > 1 && b.size > 1)
        return false;
    if (a.size == 1 && b.size == 1)
        return chains == 0;
    if (b.size > 1) {
        int temp = *i;
        *i = *j;
        *j = temp;
    }
    return true;
}
static MctsoptAction *candidates_with_pass(Search *s, MctsoptClump *c, int n, bool rollout,
                                            int *count, int *selected_pass) {
    size_t capacity = (size_t)n * (size_t)n;
    MctsoptAction *good = allocate(s, capacity * sizeof(*good));
    MctsoptAction *fallback = allocate(s, capacity * sizeof(*fallback));
    int chains = 0, bases = 0, ng = 0, nf = 0;
    for (int i = 0; i < n; i++) {
        if (c[i].size > 1)
            chains++;
        else
            bases++;
    }
    int passes = rollout && s->cfg->shape != 0 ? 3 : 1;
    for (int pass = 0; pass < passes && ng == 0; pass++) {
        if (selected_pass)
            *selected_pass = pass;
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++) {
                if (rollout && s->host->overlap(s->host->user, c[i].rel, c[j].rel))
                    continue;
                if ((!rollout || s->cfg->shape != 0) && pass < 2 &&
                    !shape(s, c[i], c[j], chains, bases, pass == 1))
                    continue;
                int left = i, right = j;
                if ((!rollout || s->cfg->shape != 0) &&
                    !orient(s, c[i], c[j], chains, &left, &right))
                    continue;
                MctsoptAction a = {left, right, DBL_MAX};
                if (pass < 2 && s->host->desirable(s->host->user, c[i].rel, c[j].rel))
                    good[ng++] = a;
                else
                    fallback[nf++] = a;
            }
        if (ng == 0) {
            memcpy(good, fallback, (size_t)nf * sizeof(*good));
            ng = nf;
        }
        nf = 0;
    }
    *count = ng;
    return good;
}
static MctsoptAction *candidates(Search *s, MctsoptClump *c, int n, bool rollout, int *count) {
    return candidates_with_pass(s, c, n, rollout, count, NULL);
}
static void dropped(Search *s, MctsoptNode *node, MctsoptAction a, double value, double r) {
    if (!s->cfg->trace)
        return;
    MctsoptDropped *d = &node->dropped[node->dropped_count++];
    *d = (MctsoptDropped){node->clumps[a.left].rel, node->clumps[a.right].rel, value, r};
}
static void enumerate(Search *s, MctsoptNode *node) {
    node->actions = candidates(s, node->clumps, node->count, false, &node->action_count);
    int n = node->action_count, k = s->cfg->top_k;
    if (k <= 0 || n <= k) {
        if (s->cfg->rank_all) {
            /* Preserve action order to isolate memo enrichment from pruning. */
            void *saved = scope(s, false);
            for (int i = 0; i < n; i++) {
                interrupt(s);
                MctsoptAction *a = &node->actions[i];
                double r;
                s->host->rank(s->host->user, node->clumps[a->left].rel,
                              node->clumps[a->right].rel, s->cfg->ranking, &a->rank, &r);
            }
            end_scope(s, saved);
        }
        return;
    }
    MctsoptAction *top = allocate(s, (size_t)k * sizeof(*top));
    double *top_rows = allocate(s, (size_t)k * sizeof(*top_rows));
    node->dropped = allocate(s, (size_t)n * sizeof(*node->dropped));
    int length = 0;
    void *saved = scope(s, false);
    for (int i = 0; i < n; i++) {
        interrupt(s);
        MctsoptAction a = node->actions[i];
        double value, r;
        s->host->rank(s->host->user, node->clumps[a.left].rel, node->clumps[a.right].rel,
                      s->cfg->ranking, &value, &r);
        a.rank = value;
        int pos;
        if (length < k)
            pos = length++;
        else if (value < top[length - 1].rank) {
            dropped(s, node, top[length - 1], top[length - 1].rank, top_rows[length - 1]);
            pos = length - 1;
        } else {
            dropped(s, node, a, value, r);
            continue;
        }
        while (pos > 0 && top[pos - 1].rank > value) {
            top[pos] = top[pos - 1];
            top_rows[pos] = top_rows[pos - 1];
            pos--;
        }
        top[pos] = a;
        top_rows[pos] = r;
    }
    end_scope(s, saved);
    node->actions = top;
    node->action_count = length;
}
static MctsoptNode *new_node(Search *s, MctsoptNode *parent, MctsoptClump *clumps, int n,
                             MctsoptStep step) {
    MctsoptNode *node = allocate(s, sizeof(*node));
    memset(node, 0, sizeof(*node));
    node->parent = parent;
    node->clumps = clumps;
    node->count = n;
    node->creation = step;
    node->depth = parent ? parent->depth + 1 : 0;
    node->best_reward = -DBL_MAX;
    node->best_cost = DBL_MAX;
    node->immediate_cost = DBL_MAX;
    node->rows = -1;
    node->terminal = n == 1 && (!parent || s->host->complete(s->host->user, clumps[0].rel));
    if (!node->terminal)
        enumerate(s, node);
    node->children = allocate(s, (size_t)node->action_count * sizeof(*node->children));
    return node;
}
static double normalized_average(Search *s, const MctsoptNode *node) {
    double positive_reward = node->positive_cost_visits;
    if (isfinite(s->low) && s->low > 0 && isfinite(s->high) && s->high > s->low) {
        double log_lo = log(s->low), log_hi = log(s->high);
        if (log_hi > log_lo) {
            positive_reward = ((double)node->positive_cost_visits * log_hi - node->sum_log_cost) /
                              (log_hi - log_lo);
            /* Cancellation at a boundary must not create rewards outside [0, 1]. */
            positive_reward = fmax(0, fmin(node->positive_cost_visits, positive_reward));
        }
    }
    return ((double)node->zero_cost_visits + positive_reward) / (double)node->visits;
}
static void record_cost(MctsoptNode *node, double value) {
    if (value >= DBL_MAX)
        return;
    if (value <= 0)
        node->zero_cost_visits++;
    else {
        node->positive_cost_visits++;
        node->sum_log_cost += log(value);
    }
}
static double score(Search *s, MctsoptNode *child, int visits, bool classic) {
    if (child->visits == 0)
        return 1e30;
    double q =
        classic || s->cfg->average ? child->sum_reward / (double)child->visits : child->best_reward;
    if (s->cfg->reward == MCTSOPT_NORM_NEG_LOG && !s->cfg->legacy_normalization)
        q = classic || s->cfg->average ? normalized_average(s, child) : reward(s, child->best_cost);
    if (classic) {
        if (visits <= 1)
            return q;
        return q + s->cfg->exploration * sqrt(log((double)visits) / (double)child->visits);
    }
    if (visits <= 0)
        return q;
    double explore = s->cfg->exploration * log((double)visits) / (double)child->visits;
    explore = pow(explore, s->cfg->gamma);
    return q + explore;
}
static MctsoptNode *select_node(Search *s, MctsoptNode *node, bool classic) {
    if (!node || (!classic && node->exhausted))
        return NULL;
    if (node->terminal || node->child_count == 0) {
        if (!classic && !node->terminal && node->next_action == node->action_count)
            return NULL;
        return node;
    }
    if (!classic && node->next_action < node->action_count)
        return node;
    for (;;) {
        MctsoptNode *best = NULL;
        double best_score = -1e30;
        for (int i = 0; i < node->child_count; i++) {
            MctsoptNode *ch = node->children[i];
            if (!classic &&
                (ch->exhausted || (ch->child_count == 0 && ch->next_action == ch->action_count)))
                continue;
            int visits = classic && node->visits < 1 ? 1 : node->visits;
            double value = score(s, ch, visits, classic);
            if (value > best_score || (!best && !classic && !s->cfg->legacy_selection)) {
                best_score = value;
                best = ch;
            }
        }
        if (!best) {
            if (!classic && !s->cfg->legacy_selection)
                node->exhausted = true;
            return NULL;
        }
        MctsoptNode *selected = select_node(s, best, classic);
        if (selected || classic || s->cfg->legacy_selection)
            return selected;
        /* A dead preferred subtree does not imply that its siblings are dead. */
        best->exhausted = true;
    }
}
static MctsoptNode *expand(Search *s, MctsoptNode *leaf) {
    if (leaf->next_action == leaf->action_count)
        return leaf;
    MctsoptAction a = leaf->actions[leaf->next_action++];
    MctsoptClump l = leaf->clumps[a.left], r = leaf->clumps[a.right];
    void *saved = scope(s, false);
    if (s->host->overlap(s->host->user, l.rel, r.rel)) {
        end_scope(s, saved);
        return leaf;
    }
    MctsoptRel rel = s->host->join(s->host->user, l.rel, r.rel, true);
    if (!rel) {
        end_scope(s, saved);
        return leaf;
    }
    MctsoptClump *clumps = allocate(s, (size_t)(leaf->count - 1) * sizeof(*clumps));
    for (int i = 0, j = 0; i < leaf->count; i++) {
        if (i == a.right)
            continue;
        clumps[j++] = i == a.left ? (MctsoptClump){rel, l.size + r.size} : leaf->clumps[i];
    }
    double rcount = rows(s, rel);
    end_scope(s, saved);
    MctsoptNode *child = new_node(s, leaf, clumps, leaf->count - 1, (MctsoptStep){l.rel, r.rel});
    child->immediate_cost = a.rank;
    child->rows = rcount;
    leaf->children[leaf->child_count++] = child;
    if (child->terminal) {
        double c = cost(s, rel), value = reward(s, c);
        child->best_reward = value;
        child->sum_reward = value;
        child->best_cost = c;
        child->visits = 1;
        if (s->cfg->reward == MCTSOPT_NORM_NEG_LOG)
            record_cost(child, c);
    }
    return child;
}
typedef struct {
    MctsoptRel rel;
    double cost;
    MctsoptOrder order;
} Rollout;
/* Mix uniform sampling with inverse cost midrank.
 * Ties share weight and every admitted action keeps positive probability. */
static int weighted_rollout_pick(Search *s, MctsoptAction *actions, int count) {
    double normalizer = 0;
    for (int i = 0; i < count; i++) {
        double rank = 1;
        for (int j = 0; j < count; j++) {
            if (actions[j].rank < actions[i].rank) rank += 1;
            else if (i != j && actions[j].rank == actions[i].rank) rank += 0.5;
        }
        normalizer += 1 / rank;
    }
    double draw = mctsopt_random(&s->rng), cumulative = 0;
    for (int i = 0; i < count; i++) {
        double rank = 1;
        for (int j = 0; j < count; j++) {
            if (actions[j].rank < actions[i].rank) rank += 1;
            else if (i != j && actions[j].rank == actions[i].rank) rank += 0.5;
        }
        cumulative += s->cfg->rollout_epsilon / count +
                      (1.0 - s->cfg->rollout_epsilon) / (rank * normalizer);
        if (draw < cumulative) return i;
    }
    return count - 1;
}
static Rollout rollout(Search *s, MctsoptNode *node) {
    s->result.rollout_calls++;
    /* Orders survive the temporary evaluation scope, even on hosts that reset it.
     */
    MctsoptOrder order = {allocate(s, (size_t)node->count * sizeof(MctsoptStep)), 0};
    void *saved = scope(s, true);
    int n = node->count;
    MctsoptClump *clumps = allocate(s, (size_t)n * sizeof(*clumps));
    memcpy(clumps, node->clumps, (size_t)n * sizeof(*clumps));
    while (n > 1) {
        interrupt(s);
        int count;
        MctsoptAction *actions = candidates(s, clumps, n, true, &count), a = {0};
        if (s->cfg->rank_biased_rollout) {
        /* Evaluate admitted alternatives without reordering or consuming RNG.
         * The DuckDB host retains useful subtrees in its native memo. */
        for (int i = 0; i < count; i++) {
            interrupt(s);
            double value, rows;
            s->host->rank(s->host->user, clumps[actions[i].left].rel,
                          clumps[actions[i].right].rel, s->cfg->ranking, &value, &rows);
            actions[i].rank = isfinite(value) ? value : DBL_MAX;
        }
        }
        MctsoptRel rel = NULL;
        while (count > 0) {
            int pick = s->cfg->rank_biased_rollout ? weighted_rollout_pick(s, actions, count)
                                                : (int)(mctsopt_random(&s->rng) * count);
            a = actions[pick];
            rel = s->host->join(s->host->user, clumps[a.left].rel, clumps[a.right].rel, false);
            if (rel)
                break;
            memmove(actions + pick, actions + pick + 1,
                    (size_t)(count - pick - 1) * sizeof(*actions));
            count--;
        }
        if (!rel) {
            s->result.failures++;
            end_scope(s, saved);
            return (Rollout){NULL, DBL_MAX, {NULL, 0}};
        }
        order.items[order.count++] = (MctsoptStep){clumps[a.left].rel, clumps[a.right].rel};
        clumps[a.left].rel = rel;
        clumps[a.left].size += clumps[a.right].size;
        memmove(clumps + a.right, clumps + a.right + 1,
                (size_t)(n - a.right - 1) * sizeof(*clumps));
        n--;
    }
    Rollout out = {clumps[0].rel, cost(s, clumps[0].rel), order};
    if (s->host->candidate_observation)
        s->host->candidate_observation(s->host->user, out.rel);
    end_scope(s, saved);
    return out;
}
static void backprop(Search *s, MctsoptNode *node, double value) {
    if (value >= DBL_MAX) {
        for (; node; node = node->parent)
            node->visits++;
        return;
    }
    if (value > 0 && value < s->low)
        s->low = value;
    if (value > s->high)
        s->high = value;
    double r = reward(s, value);
    for (; node; node = node->parent) {
        node->visits++;
        node->sum_reward += r;
        if (s->cfg->reward == MCTSOPT_NORM_NEG_LOG)
            record_cost(node, value);
        if (value < node->best_cost)
            node->best_cost = value;
        node->best_reward = reward(s, node->best_cost);
    }
}
static MctsoptOrder capture(Search *s, MctsoptNode *node, MctsoptOrder suffix) {
    MctsoptOrder out = {allocate(s, (size_t)(node->depth + suffix.count) * sizeof(MctsoptStep)),
                        node->depth + suffix.count};
    for (int i = node->depth - 1; i >= 0; i--, node = node->parent)
        out.items[i] = node->creation;
    if (suffix.count)
        memcpy(out.items + out.count - suffix.count, suffix.items,
               (size_t)suffix.count * sizeof(MctsoptStep));
    return out;
}
static void observe(Search *s, MctsoptNode *node, Rollout r, int phase, int iteration,
                    MctsoptOrder *phase_order) {
    if (r.cost < DBL_MAX && r.cost < s->phase_best) {
        s->phase_best = r.cost;
        if (s->cfg->trace)
            *phase_order = capture(s, node, r.order);
    }
    if (r.rel && cost(s, r.rel) < DBL_MAX && r.cost < s->result.best_cost) {
        s->result.best_cost = r.cost;
        s->result.best_rel = r.rel;
        s->result.order = capture(s, node, r.order);
        s->result.best_phase = phase + 1;
        s->result.best_iteration = iteration + 1;
    }
}
static bool valid(const MctsoptConfig *c, const MctsoptHost *h, const MctsoptRel *initial, int n) {
    if (!c || !h || !initial || n < 2 || (size_t)n > SIZE_MAX / (size_t)n / sizeof(MctsoptAction))
        return false;
    if (!h->alloc || !h->desirable || !h->overlap || !h->complete || !h->cost || !h->rows ||
        !h->rank || !h->join || !h->begin_scope || !h->end_scope || !h->restart || !h->entropy ||
        !h->interrupt || !h->clock_ms)
        return false;
    if (c->budget < 1 || c->phases < 1 || c->top_k < 0 || c->depth < 0 || c->shape < 0 ||
        c->rollouts < 1 || c->patience < 0 || c->ranking < MCTSOPT_COST ||
        c->ranking > MCTSOPT_SELECTIVITY || c->reward < MCTSOPT_NEG_COST ||
        c->reward > MCTSOPT_NORM_NEG_LOG || !isfinite(c->exploration) || c->exploration < 0 ||
        !isfinite(c->gamma) || c->gamma < 0)
        return false;
    if (c->rank_biased_rollout && (!isfinite(c->rollout_epsilon) ||
        c->rollout_epsilon < 0 || c->rollout_epsilon > 1))
        return false;
    /* Bound cumulative counters as well as each individual restart budget. */
    int64_t total = 0;
    for (int phase = 0; phase < c->phases; phase++) {
        total += (int64_t)c->budget * (c->luby_budget ? mctsopt_luby(phase + 1) : 1);
        if (total > INT_MAX / c->rollouts)
            return false;
    }
    for (int i = 0; i < n; i++)
        if (!initial[i])
            return false;
    return true;
}
MctsoptGreedyResult mctsopt_greedy(const MctsoptConfig *cfg, const MctsoptHost *host,
                                   const MctsoptRel *initial, int n) {
    MctsoptGreedyResult result = {.status = MCTSOPT_INVALID_ARGUMENT, .cost = DBL_MAX};
    if (!valid(cfg, host, initial, n))
        return result;
    Search s = {.cfg = cfg, .host = host};
    result.status = MCTSOPT_NO_PLAN;
    result.order.items = allocate(&s, (size_t)(n - 1) * sizeof(MctsoptStep));
    MctsoptClump *clumps = allocate(&s, (size_t)n * sizeof(*clumps));
    for (int i = 0; i < n; i++)
        clumps[i] = (MctsoptClump){initial[i], 1};
    while (n > 1) {
        interrupt(&s);
        int count, pass = 0;
        MctsoptAction *actions = candidates_with_pass(&s, clumps, n, true, &count, &pass);
        bool *desirable = allocate(&s, (size_t)count * sizeof(*desirable));
        /* Ranking may emit native cross-product edges. Classify all candidates first. */
        for (int i = 0; i < count; i++)
            desirable[i] = host->desirable(host->user, clumps[actions[i].left].rel,
                                          clumps[actions[i].right].rel);
        void *saved = scope(&s, false);
        for (int i = 0; i < count; i++) {
            interrupt(&s);
            MctsoptAction *a = &actions[i];
            double r;
            host->rank(host->user, clumps[a->left].rel, clumps[a->right].rel,
                       MCTSOPT_COST, &a->rank, &r);
            if (!isfinite(a->rank) || a->rank >= DBL_MAX) {
                a->rank = DBL_MAX;
                result.failures++;
            }
        }
        end_scope(&s, saved);
        MctsoptRel joined = NULL;
        MctsoptAction selected = {0};
        bool selected_desirable = false;
        for (;;) {
            int best = -1;
            double best_cost = DBL_MAX;
            for (int i = 0; i < count; i++) {
                if (actions[i].rank < best_cost) {
                    best = i;
                    best_cost = actions[i].rank;
                }
            }
            if (best < 0)
                break;
            selected = actions[best];
            selected_desirable = desirable[best];
            actions[best].rank = DBL_MAX;
            interrupt(&s);
            saved = scope(&s, false);
            joined = host->join(host->user, clumps[selected.left].rel,
                                clumps[selected.right].rel, true);
            end_scope(&s, saved);
            if (joined)
                break;
            result.failures++;
        }
        if (!joined) {
            /* Absence of candidates is distinct from a rejected candidate. */
            if (count == 0)
                result.failures++;
            return result;
        }
        result.order.items[result.order.count++] =
            (MctsoptStep){clumps[selected.left].rel, clumps[selected.right].rel};
        result.steps++;
        result.relaxed_steps += pass > 0;
        result.fallback_steps += !selected_desirable;
        clumps[selected.left] = (MctsoptClump){joined,
            clumps[selected.left].size + clumps[selected.right].size};
        memmove(clumps + selected.right, clumps + selected.right + 1,
                (size_t)(n - selected.right - 1) * sizeof(*clumps));
        n--;
    }
    double final_cost = cost(&s, clumps[0].rel);
    if (!host->complete(host->user, clumps[0].rel) || !isfinite(final_cost) || final_cost >= DBL_MAX) {
        result.failures++;
        return result;
    }
    result.status = MCTSOPT_OK;
    result.best_rel = clumps[0].rel;
    result.cost = final_cost;
    return result;
}
const char *mctsopt_stop_reason_name(int reason) {
    switch (reason) {
    case MCTSOPT_STOP_INVALID_ARGUMENT: return "invalid_argument";
    case MCTSOPT_STOP_BUDGET: return "budget";
    case MCTSOPT_STOP_TREE_EXHAUSTED: return "tree_exhausted";
    case MCTSOPT_STOP_PATIENCE: return "patience";
    default: return "unknown";
    }
}
MctsoptResult mctsopt_search(const MctsoptConfig *cfg, const MctsoptHost *host,
                             const MctsoptRel *initial, int n) {
    Search s = {0};
    s.cfg = cfg;
    s.host = host;
    s.low = DBL_MAX;
    s.result.best_cost = DBL_MAX;
    s.result.status = MCTSOPT_INVALID_ARGUMENT;
    if (!valid(cfg, host, initial, n))
        return s.result;
    s.result.status = MCTSOPT_NO_PLAN;
    if (host->initial_plan) {
        MctsoptRel initial_best = host->initial_plan(host->user);
        double initial_cost = cost(&s, initial_best);
        if (initial_best && host->complete(host->user, initial_best) &&
            isfinite(initial_cost) && initial_cost < DBL_MAX) {
            s.result.best_rel = initial_best;
            s.result.best_cost = initial_cost;
            s.result.best_phase = s.result.best_iteration = -1;
        }
    }
    double snapshot_best = DBL_MAX;
    int patience = 0;
    for (int phase = 0; phase < cfg->phases; phase++) {
        interrupt(&s);
        host->restart(host->user);
        reseed(&s, phase, 0, 0);
        s.result.stop_reason = MCTSOPT_STOP_BUDGET;
        s.phase_best = DBL_MAX;
        MctsoptClump *clumps = allocate(&s, (size_t)n * sizeof(*clumps));
        for (int i = 0; i < n; i++)
            clumps[i] = (MctsoptClump){initial[i], 1};
        MctsoptNode *root = new_node(&s, NULL, clumps, n, (MctsoptStep){NULL, NULL});
        MctsoptOrder phase_order = {NULL, 0};
        int multiplier = cfg->luby_budget ? mctsopt_luby(phase + 1) : 1;
        int budget = cfg->budget * multiplier, i;
        for (i = 0; i < budget; i++) {
            interrupt(&s);
            double t = now(&s);
            MctsoptNode *leaf = select_node(&s, root, false);
            s.result.selection_ms += now(&s) - t;
            bool replay = false;
            if (!leaf) {
                if (!cfg->full_budget) {
                    s.result.stop_reason = MCTSOPT_STOP_TREE_EXHAUSTED;
                    break;
                }
                leaf = select_node(&s, root, true);
                if (!leaf) {
                    s.result.stop_reason = MCTSOPT_STOP_TREE_EXHAUSTED;
                    break;
                }
                replay = true;
            }
            s.result.attempted_iterations++;
            if (cfg->trace && host->iteration)
                host->iteration(host->user, phase + 1, i + 1, s.phase_best, s.result.best_cost,
                                leaf->depth);
            bool limited = cfg->depth > 0 && leaf->depth >= cfg->depth && !leaf->terminal;
            MctsoptNode *child = leaf;
            if (!replay && !limited) {
                t = now(&s);
                child = expand(&s, leaf);
                s.result.expansion_ms += now(&s) - t;
                if (child == leaf)
                    continue;
            }
            Rollout r = {NULL, DBL_MAX, {NULL, 0}};
            if (child->terminal) {
                r.cost = child->best_cost;
                if (child->count == 1)
                    r.rel = child->clumps[0].rel;
                if (host->candidate_observation && r.rel)
                    host->candidate_observation(host->user, r.rel);
                /* The legacy driver scores a terminal's current native path for the
                 * global incumbent. */
                if (r.cost < s.phase_best) {
                    s.phase_best = r.cost;
                    if (cfg->trace)
                        phase_order = capture(&s, child, r.order);
                }
                Rollout native = r;
                native.cost = cost(&s, native.rel);
                if (native.rel && native.cost < s.result.best_cost) {
                    s.result.best_cost = native.cost;
                    s.result.best_rel = native.rel;
                    s.result.order = capture(&s, child, native.order);
                    s.result.best_phase = phase + 1;
                    s.result.best_iteration = i + 1;
                }
            } else {
                t = now(&s);
                if (replay || limited) {
                    reseed(&s, phase, i, 0);
                    for (int rr = 0; rr < cfg->rollouts; rr++) {
                        if (cfg->luby_rollout)
                            s.rng ^= (uint64_t)mctsopt_luby(rr + 1);
                        Rollout candidate = rollout(&s, child);
                        if (candidate.cost < r.cost)
                            r = candidate;
                        reseed(&s, phase, i, rr + 1);
                    }
                } else {
                    if (cfg->luby_rollout)
                        s.rng ^= (uint64_t)mctsopt_luby(s.result.rollouts + 1);
                    r = rollout(&s, child);
                }
                observe(&s, child, r, phase, i, &phase_order);
                s.result.rollout_ms += now(&s) - t;
                s.result.rollouts++;
            }
            if (limited && !replay) {
                child->exhausted = true;
                s.result.exhausted++;
            }
            t = now(&s);
            backprop(&s, child, r.cost);
            s.result.backprop_ms += now(&s) - t;
            s.result.iterations++;
        }
        s.result.completed_phases++;
        if (s.result.stop_reason == MCTSOPT_STOP_TREE_EXHAUSTED)
            s.result.exhausted_phases++;
        if (cfg->trace && host->phase)
            host->phase(host->user, phase + 1, multiplier, budget, i, s.phase_best,
                        s.result.best_phase == phase + 1, phase_order);
        if (cfg->trace && host->snapshot && root->best_cost <= snapshot_best) {
            snapshot_best = root->best_cost;
            host->snapshot(host->user, root, s.result.order);
        }
        patience = s.result.best_phase == phase + 1 ? 0 : patience + 1;
        if (cfg->patience > 0 && patience >= cfg->patience && phase + 1 < cfg->phases) {
            s.result.stop_reason = MCTSOPT_STOP_PATIENCE;
            break;
        }
    }
    if (s.result.best_rel)
        s.result.status = MCTSOPT_OK;
    return s.result;
}
