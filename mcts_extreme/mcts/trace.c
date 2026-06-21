/*-------------------------------------------------------------------------
 *
 * trace.c
 *	  Search-decision tracing for mcts_extreme.
 *
 *	  When mcts_extreme.trace_search is on, mcts.c records why MCTS chose the
 *	  join order it did: the decision tree along the winning order, the Luby
 *	  restart phases, the per-iteration best-so-far cost, and the subplans of
 *	  each phase's best plan.  Each trace is exposed through its own SRF
 *	  (mcts_search_trace / mcts_phase_trace / mcts_iter_trace /
 *	  mcts_phasesub_trace).
 *
 *	  See mcts_trace.h for the entry points mcts.c calls.
 *
 *-------------------------------------------------------------------------
 */
#include "postgres.h"

#include <float.h>

#include "funcapi.h"
#include "miscadmin.h"
#include "nodes/bitmapset.h"
#include "nodes/parsenodes.h"
#include "nodes/pg_list.h"
#include "utils/builtins.h"
#include "utils/guc.h"
#include "utils/memutils.h"
#include "utils/tuplestore.h"

#include "mcts_trace.h"

/* Render a Relids set as a space-separated list of relation aliases. */
static char *
trace_relids_text(PlannerInfo *root, Relids relids)
{
	StringInfoData buf;
	int			x = -1;
	bool		first = true;

	if (relids == NULL)
		return pstrdup("");

	initStringInfo(&buf);
	while ((x = bms_next_member(relids, x)) >= 0)
	{
		const char *name = NULL;

		if (root != NULL && x > 0 && x < root->simple_rel_array_size &&
			root->simple_rte_array[x] != NULL &&
			root->simple_rte_array[x]->eref != NULL)
			name = root->simple_rte_array[x]->eref->aliasname;

		if (!first)
			appendStringInfoChar(&buf, ' ');
		first = false;
		if (name != NULL)
			appendStringInfoString(&buf, name);
		else
			appendStringInfo(&buf, "rel%d", x);
	}
	return buf.data;
}

/* Public wrapper so mcts.c can format relids the same way. */
char *
mcts_trace_relids_text(PlannerInfo *root, Bitmapset *relids)
{
	return trace_relids_text(root, relids);
}

/* ----------
 *  GUC
 * ----------
 */
void
mcts_trace_define_gucs(void)
{
	DefineCustomBoolVariable("mcts_extreme.trace_search",
							 "Record the MCTS decision tree for the best join order",
							 "When on, mcts_search_trace() returns, for each step "
							 "of the chosen join order, the picked join action and "
							 "its sibling alternatives with their UCT statistics "
							 "(visits, reward, best cost) -- i.e. why MCTS chose "
							 "this order.",
							 &mcts_trace_search,
							 false,
							 PGC_USERSET,
							 0, NULL, NULL, NULL);
}

/* ----------
 *  Search-decision trace (why MCTS chose this join order)
 * ----------
 */
bool		mcts_trace_search = false;	/* GUC */

typedef struct SearchNode
{
	int			id;
	int			parent_id;
	int			depth;
	char	   *left;
	char	   *right;
	int			visits;
	double		best_reward;
	double		sum_reward;
	double		best_cost;
	double		topk_cost;		/* immediate join cost top-k ranked it by */
	double		est_rows;		/* estimated cardinality of the join (-1 = n/a) */
	bool		chosen;
	bool		dropped;		/* evaluated then dropped by top-k filtering */
} SearchNode;

static MemoryContext mcts_search_cxt = NULL;
static List *search_nodes = NIL;
static int	search_next_id = 0;

void
mcts_searchtrace_begin(void)
{
	if (mcts_search_cxt == NULL)
		mcts_search_cxt = AllocSetContextCreate(TopMemoryContext,
												"mcts_extreme search trace",
												ALLOCSET_SMALL_SIZES);
	else
		MemoryContextReset(mcts_search_cxt);
	search_nodes = NIL;
	search_next_id = 0;
}

int
mcts_searchtrace_add(int parent_id, int depth,
					 const char *left, const char *right,
					 int visits, double best_reward, double sum_reward,
					 double best_cost, double topk_cost, double est_rows,
					 bool chosen, bool dropped)
{
	MemoryContext old = MemoryContextSwitchTo(mcts_search_cxt);
	SearchNode *n = (SearchNode *) palloc0(sizeof(SearchNode));

	n->id = search_next_id++;
	n->parent_id = parent_id;
	n->depth = depth;
	n->left = left ? pstrdup(left) : NULL;
	n->right = right ? pstrdup(right) : NULL;
	n->visits = visits;
	n->best_reward = best_reward;
	n->sum_reward = sum_reward;
	n->best_cost = best_cost;
	n->topk_cost = topk_cost;
	n->est_rows = est_rows;
	n->chosen = chosen;
	n->dropped = dropped;
	search_nodes = lappend(search_nodes, n);
	MemoryContextSwitchTo(old);
	return n->id;
}

PG_FUNCTION_INFO_V1(mcts_search_trace);

Datum
mcts_search_trace(PG_FUNCTION_ARGS)
{
	ReturnSetInfo *rsinfo = (ReturnSetInfo *) fcinfo->resultinfo;
	ListCell   *lc;

	InitMaterializedSRF(fcinfo, 0);

	foreach(lc, search_nodes)
	{
		SearchNode *n = (SearchNode *) lfirst(lc);
		Datum		values[13];
		bool		nulls[13];

		memset(nulls, false, sizeof(nulls));
		values[0] = Int32GetDatum(n->id);
		if (n->parent_id < 0)
			nulls[1] = true;
		else
			values[1] = Int32GetDatum(n->parent_id);
		values[2] = Int32GetDatum(n->depth);
		if (n->left)
			values[3] = CStringGetTextDatum(n->left);
		else
			nulls[3] = true;
		if (n->right)
			values[4] = CStringGetTextDatum(n->right);
		else
			nulls[4] = true;
		values[5] = Int32GetDatum(n->visits);
		/* dropped actions were never rolled out -> no reward */
		if (n->dropped)
		{
			nulls[6] = true;
			nulls[7] = true;
		}
		else
		{
			values[6] = Float8GetDatum(n->best_reward);
			values[7] = Float8GetDatum(n->sum_reward);
		}
		if (n->best_cost >= (double) DBL_MAX)
			nulls[8] = true;
		else
			values[8] = Float8GetDatum(n->best_cost);
		if (n->topk_cost >= (double) DBL_MAX)
			nulls[9] = true;
		else
			values[9] = Float8GetDatum(n->topk_cost);
		if (n->est_rows < 0)
			nulls[10] = true;
		else
			values[10] = Float8GetDatum(n->est_rows);
		values[11] = BoolGetDatum(n->chosen);
		values[12] = BoolGetDatum(n->dropped);
		tuplestore_putvalues(rsinfo->setResult, rsinfo->setDesc, values, nulls);
	}
	return (Datum) 0;
}


/* ----------
 *  Per-phase (Luby restart) trace
 * ----------
 */
typedef struct PhaseNode
{
	int			phase;
	int			luby;
	int			budget;
	int			iterations;
	double		phase_best_cost;
	bool		improved;
} PhaseNode;

static MemoryContext mcts_phase_cxt = NULL;
static List *phase_nodes = NIL;

void
mcts_phasetrace_begin(void)
{
	if (mcts_phase_cxt == NULL)
		mcts_phase_cxt = AllocSetContextCreate(TopMemoryContext,
											   "mcts_extreme phase trace",
											   ALLOCSET_SMALL_SIZES);
	else
		MemoryContextReset(mcts_phase_cxt);
	phase_nodes = NIL;
}

void
mcts_phasetrace_add(int phase, int luby, int budget, int iterations,
					double phase_best_cost, bool improved)
{
	MemoryContext old = MemoryContextSwitchTo(mcts_phase_cxt);
	PhaseNode  *n = (PhaseNode *) palloc0(sizeof(PhaseNode));

	n->phase = phase;
	n->luby = luby;
	n->budget = budget;
	n->iterations = iterations;
	n->phase_best_cost = phase_best_cost;
	n->improved = improved;
	phase_nodes = lappend(phase_nodes, n);
	MemoryContextSwitchTo(old);
}

PG_FUNCTION_INFO_V1(mcts_phase_trace);

Datum
mcts_phase_trace(PG_FUNCTION_ARGS)
{
	ReturnSetInfo *rsinfo = (ReturnSetInfo *) fcinfo->resultinfo;
	ListCell   *lc;

	InitMaterializedSRF(fcinfo, 0);

	foreach(lc, phase_nodes)
	{
		PhaseNode  *n = (PhaseNode *) lfirst(lc);
		Datum		values[6];
		bool		nulls[6];

		memset(nulls, false, sizeof(nulls));
		values[0] = Int32GetDatum(n->phase);
		values[1] = Int32GetDatum(n->luby);
		values[2] = Int32GetDatum(n->budget);
		values[3] = Int32GetDatum(n->iterations);
		if (n->phase_best_cost >= (double) DBL_MAX)
			nulls[4] = true;
		else
			values[4] = Float8GetDatum(n->phase_best_cost);
		values[5] = BoolGetDatum(n->improved);
		tuplestore_putvalues(rsinfo->setResult, rsinfo->setDesc, values, nulls);
	}
	return (Datum) 0;
}

/* ----------
 *  Per-iteration trace (within a phase): best-so-far cost and search depth.
 * ----------
 */
typedef struct IterNode
{
	int			phase;
	int			iteration;
	double		phase_best_cost;
	double		global_best_cost;
	int			depth;
} IterNode;

static MemoryContext mcts_iter_cxt = NULL;
static List *iter_nodes = NIL;

void
mcts_itertrace_begin(void)
{
	if (mcts_iter_cxt == NULL)
		mcts_iter_cxt = AllocSetContextCreate(TopMemoryContext,
											  "mcts_extreme iter trace",
											  ALLOCSET_SMALL_SIZES);
	else
		MemoryContextReset(mcts_iter_cxt);
	iter_nodes = NIL;
}

void
mcts_itertrace_add(int phase, int iteration, double phase_best_cost,
				   double global_best_cost, int depth)
{
	MemoryContext old = MemoryContextSwitchTo(mcts_iter_cxt);
	IterNode   *n = (IterNode *) palloc0(sizeof(IterNode));

	n->phase = phase;
	n->iteration = iteration;
	n->phase_best_cost = phase_best_cost;
	n->global_best_cost = global_best_cost;
	n->depth = depth;
	iter_nodes = lappend(iter_nodes, n);
	MemoryContextSwitchTo(old);
}

PG_FUNCTION_INFO_V1(mcts_iter_trace);

Datum
mcts_iter_trace(PG_FUNCTION_ARGS)
{
	ReturnSetInfo *rsinfo = (ReturnSetInfo *) fcinfo->resultinfo;
	ListCell   *lc;

	InitMaterializedSRF(fcinfo, 0);

	foreach(lc, iter_nodes)
	{
		IterNode   *n = (IterNode *) lfirst(lc);
		Datum		values[5];
		bool		nulls[5];

		memset(nulls, false, sizeof(nulls));
		values[0] = Int32GetDatum(n->phase);
		values[1] = Int32GetDatum(n->iteration);
		if (n->phase_best_cost >= (double) DBL_MAX)
			nulls[2] = true;
		else
			values[2] = Float8GetDatum(n->phase_best_cost);
		if (n->global_best_cost >= (double) DBL_MAX)
			nulls[3] = true;
		else
			values[3] = Float8GetDatum(n->global_best_cost);
		values[4] = Int32GetDatum(n->depth);
		tuplestore_putvalues(rsinfo->setResult, rsinfo->setDesc, values, nulls);
	}
	return (Datum) 0;
}

/* ----------
 *  Per-phase subplan trace: the joinrels of each phase's best plan, so the
 *  subplans (building blocks) can be compared across phases.
 * ----------
 */
typedef struct PhaseSubNode
{
	int			phase;
	char	   *relids;
	int			size;
} PhaseSubNode;

static MemoryContext mcts_phasesub_cxt = NULL;
static List *phasesub_nodes = NIL;

void
mcts_phasesubtrace_begin(void)
{
	if (mcts_phasesub_cxt == NULL)
		mcts_phasesub_cxt = AllocSetContextCreate(TopMemoryContext,
												  "mcts_extreme phasesub trace",
												  ALLOCSET_SMALL_SIZES);
	else
		MemoryContextReset(mcts_phasesub_cxt);
	phasesub_nodes = NIL;
}

void
mcts_phasesubtrace_add(int phase, const char *relids, int size)
{
	MemoryContext old = MemoryContextSwitchTo(mcts_phasesub_cxt);
	PhaseSubNode *n = (PhaseSubNode *) palloc0(sizeof(PhaseSubNode));

	n->phase = phase;
	n->relids = relids ? pstrdup(relids) : NULL;
	n->size = size;
	phasesub_nodes = lappend(phasesub_nodes, n);
	MemoryContextSwitchTo(old);
}

PG_FUNCTION_INFO_V1(mcts_phasesub_trace);

Datum
mcts_phasesub_trace(PG_FUNCTION_ARGS)
{
	ReturnSetInfo *rsinfo = (ReturnSetInfo *) fcinfo->resultinfo;
	ListCell   *lc;

	InitMaterializedSRF(fcinfo, 0);

	foreach(lc, phasesub_nodes)
	{
		PhaseSubNode *n = (PhaseSubNode *) lfirst(lc);
		Datum		values[3];
		bool		nulls[3];

		memset(nulls, false, sizeof(nulls));
		values[0] = Int32GetDatum(n->phase);
		if (n->relids)
			values[1] = CStringGetTextDatum(n->relids);
		else
			nulls[1] = true;
		values[2] = Int32GetDatum(n->size);
		tuplestore_putvalues(rsinfo->setResult, rsinfo->setDesc, values, nulls);
	}
	return (Datum) 0;
}
