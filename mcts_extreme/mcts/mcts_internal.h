/*-------------------------------------------------------------------------
 *
 * mcts_internal.h
 *	  Internal interface shared between the MCTS search (mcts.c) and the
 *	  non-tree local-search baselines (local_search.c).
 *
 *	  These declarations let local_search.c reuse mcts.c's join-building,
 *	  cost cache, replay and run statistics without duplicating them, and
 *	  let mcts.c dispatch to the local-search entry points.  They are not
 *	  part of the extension's public (SQL) interface.
 *
 *-------------------------------------------------------------------------
 */
#ifndef MCTS_INTERNAL_H
#define MCTS_INTERNAL_H

#include "nodes/pathnodes.h"
#include "nodes/pg_list.h"
#include "utils/hsearch.h"

/* In PG, cheapest_total_path is a single Path *, not a List. */
#define REL_CHEAPEST_PATH(rel) ((rel)->cheapest_total_path)
#define REL_HAS_CHEAPEST_PATH(rel) ((rel)->cheapest_total_path != NULL)

/*
 * Join-order search algorithm, selected by mcts_extreme.search_algorithm.
 * MCTS is the default; SAIO and iterative improvement are the non-tree
 * local-search controls implemented in local_search.c.
 */
typedef enum
{
	MCTS_EXTREME_SEARCH_MCTS,
	MCTS_EXTREME_SEARCH_SAIO,
	MCTS_EXTREME_SEARCH_ITERATIVE_IMPROVEMENT
} MctsExtremeSearchAlgorithm;

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
 * MctsAction -- a candidate join of two clumps by their indices in the
 * current clump list.
 */
typedef struct MctsAction
{
	int			left_idx;
	int			right_idx;
	Cost		rank_cost;		/* immediate join cost top-k ranked it by (trace) */
} MctsAction;

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
 * MctsContext -- per-search state shared across all MCTS phases (and reused
 * by the local-search baselines).  Holds planner state, memory contexts for
 * tree vs. ephemeral allocation, the join cost cache, and running counters.
 */
typedef struct MctsContext
{
	PlannerInfo *root;
	List	   *initial_rels;
	Bitmapset  *all_query_relids;
	MemoryContext run_context;	/* long-lived: tree nodes, merge orders */
	MemoryContext eval_context;	/* short-lived: rollout temporaries */
	int			num_rels;
	double		exploration_c;
	double		gamma;
	unsigned short seedbuf[3];	/* state for erand48 */
	int			iterations_done;
	int			rollout_failures;
	Cost		best_cost;		/* best across the whole search */
	Cost		phase_best_cost;	/* best seen in the current phase only */
	double		cost_min;		/* min finite rollout cost seen so far */
	double		cost_max;		/* max finite rollout cost seen so far */
	List	   *best_merge_order;
	HTAB	   *join_cost_cache;
	int			cache_hits;
	List	   *last_topk_dropped;	/* dropped actions from the last enumerate */
} MctsContext;

/*
 * MctsLastStats -- statistics from the most recent run, reported via EXPLAIN
 * and the debug log.  Written by both the MCTS search and the local-search
 * baselines.
 */
typedef struct MctsLastStats
{
	bool		valid;
	const char *algorithm;
	const char *mode;
	int			num_rels;
	int			phases;
	int			iterations;
	int			accepted_moves;
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
	double		final_temperature;
} MctsLastStats;

/* Run statistics (defined in mcts.c). */
extern MctsLastStats mcts_last_stats;

/*
 * GUC variables shared with local_search.c.  search_algorithm and
 * force_left_tree are owned by local_search.c; the rest are owned by mcts.c.
 */
extern int	mcts_search_algorithm;
extern bool mcts_force_left_tree;
extern int	mcts_start_budget;
extern int	mcts_patience;
extern bool mcts_cache_enabled;
extern int	mcts_cache_size;
extern int	mcts_cache_max_size;
extern int	mcts_cost_eval_count;

/* Shared helpers (defined in mcts.c). */
extern HTAB *create_join_cost_cache(MemoryContext cxt, int nentries);
extern RelOptInfo *mcts_get_or_build_join(PlannerInfo *root, HTAB *cache,
										  RelOptInfo *outer, RelOptInfo *inner,
										  int *cache_hits_out, bool store_joinrel);
extern RelOptInfo *mcts_replay_best_order(PlannerInfo *root, List *initial_rels,
										  List *merge_order);
extern void mcts_mix_seedbuf(MctsContext *ctx, uint64 c0, uint64 c1, uint64 c2);

/* Local-search baselines (defined in local_search.c). */
extern const char *mcts_search_algorithm_name(void);
extern RelOptInfo *saio_one_kernel(PlannerInfo *root, List *initial_rels);
extern RelOptInfo *mcts_iterative_improvement(PlannerInfo *root, List *initial_rels);
extern void local_search_register_gucs(void);

#endif							/* MCTS_INTERNAL_H */
