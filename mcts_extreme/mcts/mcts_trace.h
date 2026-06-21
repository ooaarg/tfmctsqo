/*-------------------------------------------------------------------------
 *
 * mcts_trace.h
 *	  Search-decision tracing for mcts_extreme.
 *
 *	  Records why MCTS chose the join order it did, across four independent
 *	  traces (the decision tree, the Luby phases, the per-iteration progress,
 *	  and each phase's subplans), each exposed through its own SRF.  All are
 *	  gated by the mcts_extreme.trace_search GUC.
 *
 *-------------------------------------------------------------------------
 */
#ifndef MCTS_TRACE_H
#define MCTS_TRACE_H

#include "nodes/pathnodes.h"

/* GUC registration helper, called from _PG_init_mcts_gucs(). */
extern void mcts_trace_define_gucs(void);

/* ----------
 *  Search-decision trace (why MCTS chose this join order)
 *
 *  When mcts_extreme.trace_search is on,
 *  mcts.c snapshots the MCTS decision tree along the best join order every time
 *  a new global best is committed: for each step, the chosen join action plus
 *  its sibling alternatives, with their UCT statistics.  The final snapshot
 *  (the winning order) is exposed through the SRF mcts_search_trace().
 * ----------
 */
extern bool mcts_trace_search;

/* Reset the search-trace buffer (call before refilling, and at search start). */
extern void mcts_searchtrace_begin(void);

/* Append one decision node; returns its id (use as parent_id of its children). */
extern int	mcts_searchtrace_add(int parent_id, int depth,
								 const char *left, const char *right,
								 int visits, double best_reward,
								 double sum_reward, double best_cost,
								 double topk_cost, double est_rows,
								 bool chosen, bool dropped);

/* Render a Relids set as text (relation aliases), for callers in mcts.c. */
extern char *mcts_trace_relids_text(PlannerInfo *root, Bitmapset *relids);

/* ----------
 *  Per-phase (Luby restart) trace
 *
 *  One row per MCTS phase: its Luby multiplier and the resulting iteration
 *  budget (which changes per restart), how many iterations actually ran, the
 *  best cost found in that phase, and whether it improved the global best.
 *  Gated by mcts_extreme.trace_search; exposed via mcts_phase_trace().
 * ----------
 */
extern void mcts_phasetrace_begin(void);
extern void mcts_phasetrace_add(int phase, int luby, int budget,
								int iterations, double phase_best_cost,
								bool improved);

/*
 *  Per-iteration trace: one row per MCTS iteration (within a phase), recording
 *  the best cost so far (phase and global) and the depth the selection reached.
 *  Gated by mcts_extreme.trace_search; exposed via mcts_iter_trace().
 */
extern void mcts_itertrace_begin(void);
extern void mcts_itertrace_add(int phase, int iteration,
							   double phase_best_cost, double global_best_cost,
							   int depth);

/*
 *  Per-phase subplan trace: the joinrels (building blocks) of each phase's best
 *  plan, so subplan reuse across phases can be compared.  Gated by trace_search;
 *  exposed via mcts_phasesub_trace().
 */
extern void mcts_phasesubtrace_begin(void);
extern void mcts_phasesubtrace_add(int phase, const char *relids, int size);

#endif							/* MCTS_TRACE_H */
