#ifndef MCTSOPT_H
#define MCTSOPT_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Native relations are opaque and remain owned by the adapter's search arena.
 */
typedef void *MctsoptRel;
typedef struct {
    MctsoptRel rel;
    int size;
} MctsoptClump;
typedef struct {
    MctsoptRel left, right;
} MctsoptStep;
typedef struct {
    MctsoptStep *items;
    int count;
} MctsoptOrder;
typedef struct {
    int left, right;
    double rank;
} MctsoptAction;
typedef struct {
    MctsoptRel left, right;
    double cost, rows;
} MctsoptDropped;

typedef struct MctsoptNode {
    struct MctsoptNode *parent;
    struct MctsoptNode **children;
    int child_count;
    MctsoptClump *clumps;
    int count;
    MctsoptAction *actions;
    int action_count, next_action;
    MctsoptDropped *dropped;
    int dropped_count;
    MctsoptStep creation;
    double immediate_cost, rows, best_reward, sum_reward, best_cost;
    int visits, depth;
    int positive_cost_visits, zero_cost_visits;
    double sum_log_cost;
    bool terminal, exhausted;
} MctsoptNode;

enum { MCTSOPT_COST, MCTSOPT_ROWS, MCTSOPT_MIX025, MCTSOPT_MIX050, MCTSOPT_SELECTIVITY };
enum { MCTSOPT_NEG_COST, MCTSOPT_NEG_LOG, MCTSOPT_NORM_NEG_LOG };
typedef struct {
    uint64_t seed;
    int budget, phases, top_k, depth, shape, rollouts, patience;
    int ranking, reward;
    double exploration, gamma;
    bool average, full_budget, luby_budget, luby_rollout, force_left, trace;
    bool legacy_selection, rank_all, legacy_normalization;
    bool rank_biased_rollout;
    double rollout_epsilon;
} MctsoptConfig;

enum { MCTSOPT_OK, MCTSOPT_NO_PLAN, MCTSOPT_INVALID_ARGUMENT };
enum {
    MCTSOPT_STOP_INVALID_ARGUMENT, MCTSOPT_STOP_BUDGET,
    MCTSOPT_STOP_TREE_EXHAUSTED, MCTSOPT_STOP_PATIENCE
};
typedef struct {
    int status;
    int iterations, rollouts, exhausted, failures, best_phase, best_iteration;
    double best_cost, selection_ms, expansion_ms, rollout_ms, backprop_ms;
    MctsoptRel best_rel;
    MctsoptOrder order;
    /* Existing iterations/rollouts count evaluated iterations/rollout batches. */
    int attempted_iterations, rollout_calls, completed_phases, exhausted_phases, stop_reason;
} MctsoptResult;

typedef struct {
    int status;
    MctsoptRel best_rel;
    double cost;
    MctsoptOrder order;
    int steps, relaxed_steps, fallback_steps, failures;
} MctsoptGreedyResult;

typedef struct {
    void *user;
    /* Allocations live until the search arena is released by the adapter. */
    void *(*alloc)(void *, size_t);
    bool (*desirable)(void *, MctsoptRel, MctsoptRel);
    bool (*overlap)(void *, MctsoptRel, MctsoptRel);
    bool (*complete)(void *, MctsoptRel);
    double (*cost)(void *, MctsoptRel);
    double (*rows)(void *, MctsoptRel);
    /* Preserve native memo/cache effects: ranking and building are distinct. */
    void (*rank)(void *, MctsoptRel, MctsoptRel, int, double *, double *);
    MctsoptRel (*join)(void *, MctsoptRel, MctsoptRel, bool);
    void *(*begin_scope)(void *, bool);
    void (*end_scope)(void *, void *);
    void (*restart)(void *);
    uint64_t (*entropy)(void *);
    void (*interrupt)(void *);
    double (*clock_ms)(void *);
    void (*iteration)(void *, int, int, double, double, int);
    void (*phase)(void *, int, int, int, int, double, bool, MctsoptOrder);
    void (*snapshot)(void *, const MctsoptNode *, MctsoptOrder);
    /* Optional, explicit incumbent. Pure search must leave this callback NULL. */
    MctsoptRel (*initial_plan)(void *);
    /* Observe completed candidates without changing search rewards. */
    void (*candidate_observation)(void *, MctsoptRel);
} MctsoptHost;

/* No DBMS headers or global search state; separate calls may run concurrently.
 */
#ifdef __cplusplus
extern "C" {
#endif

/* Explicit initialization helper; never called implicitly by mctsopt_search. */
MctsoptGreedyResult mctsopt_greedy(const MctsoptConfig *, const MctsoptHost *, const MctsoptRel *, int);
MctsoptResult mctsopt_search(const MctsoptConfig *, const MctsoptHost *, const MctsoptRel *, int);
uint64_t mctsopt_seed(uint64_t, uint64_t, uint64_t, uint64_t);
double mctsopt_random(uint64_t *);
int mctsopt_luby(int);
const char *mctsopt_stop_reason_name(int);

#ifdef __cplusplus
}
#endif

#endif
