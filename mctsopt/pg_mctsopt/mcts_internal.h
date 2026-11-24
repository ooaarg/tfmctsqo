/* Native PostgreSQL join replay, cache helpers and MCTS statistics. */
#ifndef MCTS_INTERNAL_H
#define MCTS_INTERNAL_H

#include "nodes/pathnodes.h"
#include "nodes/pg_list.h"
#include "utils/hsearch.h"

/* In PG, cheapest_total_path is a single Path *, not a List. */
#define REL_CHEAPEST_PATH(rel) ((rel)->cheapest_total_path)
#define REL_HAS_CHEAPEST_PATH(rel) ((rel)->cheapest_total_path != NULL)

/*
 * MctsClump -- a group of joined relations, analogous to GEQO's Clump.
 * Each clump tracks how many base relations have been folded into it.
 */
typedef struct MctsClump
{
	RelOptInfo *joinrel;
	int			size;
} MctsClump;

/*
 * MctsMergeStep -- one step in a recorded join order, identifying the
 * left and right relid sets that were joined.
 */
typedef struct MctsMergeStep
{
	Relids		left;
	Relids		right;
} MctsMergeStep;

/*
 * MctsLastStats -- statistics from the most recent run, reported via EXPLAIN
 * and the debug log.
 */
typedef struct MctsLastStats
{
	bool		valid;
	const char *algorithm;
	const char *mode;
	int			num_rels;
	int			phases;
	int			iterations;
	int			depth_limit_hits;
	int			total_random_rollouts;
	int			rollout_failures;
	int			exhausted;
	double		best_cost;
	int			best_cost_phase;
	int			best_cost_iteration;
	int			cost_evals;
	bool		cache_on;
	int			cache_hits;
	long long	cache_size;
	int			cache_max_size;
	double		cache_hit_ratio;
	double		selection_time_ms;
	double		expansion_time_ms;
	double		rollout_time_ms;
	double		backprop_time_ms;
	double		search_time_ms;
	double		replay_time_ms;
	double		total_planning_mcts_time_ms;
} MctsLastStats;

/* Run statistics (defined in native.inc). */
extern MctsLastStats mcts_last_stats;

/* Native adapter settings and counters. */
extern int	mcts_start_budget;
extern int	mcts_patience;
extern bool mcts_cache_enabled;
extern int	mcts_cache_size;
extern int	mcts_cache_max_size;
extern int	mcts_cost_eval_count;

/* Native helpers (defined in native.inc). */
extern HTAB *create_join_cost_cache(MemoryContext cxt, int nentries);
extern RelOptInfo *mcts_get_or_build_join(PlannerInfo *root, HTAB *cache,
										  RelOptInfo *outer, RelOptInfo *inner,
										  int *cache_hits_out, bool store_joinrel);
extern RelOptInfo *mcts_replay_best_order(PlannerInfo *root, List *initial_rels,
										  List *merge_order);


#endif							/* MCTS_INTERNAL_H */
