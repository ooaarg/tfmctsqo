/*-------------------------------------------------------------------------
 *
 * mcts_extreme.c
 *	  Entry point for the mcts_extreme PostgreSQL extension.
 *
 *	  Installs join_search_hook and forwards every call to the MCTS
 *	  search.  When the GUC mcts_extreme.enabled is off, or MCTS
 *	  declines a particular query (e.g. fewer than min_relations base
 *	  rels, see mcts_is_enabled() in mcts/mcts.c), control falls through
 *	  to the previous hook (if any), or to GEQO / standard_join_search()
 *	  per enable_geqo and geqo_threshold (mirroring core's
 *	  make_rel_from_joinlist(), which a join_search_hook bypasses).
 *
 *-------------------------------------------------------------------------
 */
#include "postgres.h"

#include "fmgr.h"
#include "optimizer/geqo.h"
#include "optimizer/paths.h"
#include "utils/guc.h"

PG_MODULE_MAGIC;

void		_PG_init(void);
void		_PG_fini(void);

/* Exposed by mcts/mcts.c */
extern void			_PG_init_mcts_gucs(void);
extern RelOptInfo *mcts_extreme(PlannerInfo *root, List *initial_rels);
extern bool			mcts_is_enabled(void);

static join_search_hook_type prev_join_search_hook = NULL;

static RelOptInfo *
mcts_extreme_join_search(PlannerInfo *root, int levels_needed,
						 List *initial_rels)
{
	if (mcts_is_enabled())
	{
		RelOptInfo *rel = mcts_extreme(root, initial_rels);

		if (rel != NULL)
			return rel;
		/* MCTS declined (e.g. too few rels): fall through. */
	}

	if (prev_join_search_hook)
		return prev_join_search_hook(root, levels_needed, initial_rels);

	/*
	 * Core's make_rel_from_joinlist() chooses GEQO vs DP only in the branch
	 * that runs when no join_search_hook is installed.  Since we install one,
	 * that choice never happens in core, so replicate it here.  Without this,
	 * a "geqo=on" baseline (MCTS disabled) silently runs DP instead of GEQO.
	 */
	if (enable_geqo && levels_needed >= geqo_threshold)
		return geqo(root, levels_needed, initial_rels);
	return standard_join_search(root, levels_needed, initial_rels);
}

void
_PG_init(void)
{
	_PG_init_mcts_gucs();

	MarkGUCPrefixReserved("mcts_extreme");

	prev_join_search_hook = join_search_hook;
	join_search_hook = mcts_extreme_join_search;
}

void
_PG_fini(void)
{
	join_search_hook = prev_join_search_hook;
}
