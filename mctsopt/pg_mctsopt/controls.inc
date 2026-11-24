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

     ExplainOpenGroup("pg_mctsopt.mcts", "pg_mctsopt.mcts", true, es);
     ExplainPropertyText("MCTS Search Algorithm",
                         mcts_last_stats.algorithm ? mcts_last_stats.algorithm : "mcts", es);
     if (mcts_last_stats.mode)
         ExplainPropertyText("MCTS Mode", mcts_last_stats.mode, es);
     ExplainPropertyInteger("MCTS Relations", NULL, mcts_last_stats.num_rels, es);
    ExplainPropertyInteger("MCTS Phases", NULL, mcts_last_stats.phases, es);
    ExplainPropertyInteger("MCTS Iterations", NULL, mcts_last_stats.iterations, es);
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
     ExplainCloseGroup("pg_mctsopt.mcts", "pg_mctsopt.mcts", true, es);

     mcts_last_stats.valid = false;
 }

 /* ----------
  *  GUC registration + hook install
  * ----------
  */

 /*
  * _PG_init_mcts_gucs
  *	  Register all pg_mctsopt.* GUC variables and install the
  *	  EXPLAIN hook. Called from _PG_init in core/pg_mctsopt.c.
  */
 bool
 mcts_is_enabled(void)
 {
     if (!pg_mctsopt_enabled)
         mcts_last_stats.valid = false;
     return pg_mctsopt_enabled;
 }

 void
 _PG_init_mcts_gucs(void)
 {
     DefineCustomBoolVariable("pg_mctsopt.enabled",
                              "Enable MCTS Extreme join ordering",
                              NULL,
                              &pg_mctsopt_enabled,
                              true,
                              PGC_USERSET,
                              0, NULL, NULL, NULL);

     DefineCustomIntVariable("pg_mctsopt.min_relations",
                             "Minimum relations to trigger MCTS",
                             NULL,
                             &pg_mctsopt_min_relations,
                             3,
                             3, BITS_PER_BITMAPWORD,
                             PGC_USERSET,
                             0, NULL, NULL, NULL);

     DefineCustomRealVariable("pg_mctsopt.exploration_constant",
                              "UCT exploration constant",
                              NULL,
                              &pg_mctsopt_exploration_constant,
                              1.4,
                              0.0, 1000.0,
                              PGC_USERSET,
                              0, NULL, NULL, NULL);

     DefineCustomRealVariable("pg_mctsopt.gamma",
                              "UCT Extreme exploration exponent",
                              NULL,
                              &pg_mctsopt_gamma,
                              0.5,
                              0.01, 2.0,
                              PGC_USERSET,
                              0, NULL, NULL, NULL);

     mcts_trace_define_gucs();

     DefineCustomIntVariable("pg_mctsopt.random_seed",
                             "Random seed (0 = time-based)",
                             NULL,
                             &pg_mctsopt_random_seed,
                             0,
                             0, 2147483647,
                             PGC_USERSET,
                             0, NULL, NULL, NULL);

     DefineCustomIntVariable("pg_mctsopt.depth",
                             "MCTS tree depth limit",
                             NULL,
                             &depth,
                             10,
                             2, 100,
                             PGC_USERSET,
                             0, NULL, NULL, NULL);

     DefineCustomIntVariable("pg_mctsopt.start_budget",
                             "Start budget for MCTS (iterations in phase 1)",
                             NULL,
                             &mcts_start_budget,
                             480,
                             1, 100000000,
                             PGC_USERSET,
                             0, NULL, NULL, NULL);

     DefineCustomIntVariable("pg_mctsopt.phases",
                             "Number of Luby-sequence restart phases",
                             NULL,
                             &mcts_phases,
                             1,
                             1, 1000,
                             PGC_USERSET,
                             0, NULL, NULL, NULL);

     DefineCustomIntVariable("pg_mctsopt.patience",
                             "Stop after this many phases without improvement (0 = disabled)",
                             NULL,
                             &mcts_patience,
                             0,
                             0, 1000,
                             PGC_USERSET,
                             0, NULL, NULL, NULL);

     DefineCustomIntVariable("pg_mctsopt.top_k",
                             "Limit actions to top-k cheapest (0 = all)",
                             NULL,
                             &mcts_top_k,
                             10,
                             0, 10000,
                             PGC_USERSET,
                             0, NULL, NULL, NULL);

     DefineCustomEnumVariable("pg_mctsopt.expand_strategy",
                              "Top-k expansion ranking: cost, row, mixed_025, "
                              "mixed_050, or selectivity.",
                              NULL,
                              &mcts_expand_strategy,
                              MCTS_EXTREME_EXPAND_COST,
                              pg_mctsopt_expand_strategy_options,
                              PGC_USERSET,
                              0, NULL, NULL, NULL);

     DefineCustomIntVariable("pg_mctsopt.rollouts_per_leaf",
                             "Number of random rollouts per depth-limited leaf",
                             NULL,
                             &mcts_rollouts_per_leaf,
                             1,
                             1, 10000,
                             PGC_USERSET,
                             0, NULL, NULL, NULL);

     DefineCustomBoolVariable("pg_mctsopt.full_budget",
                              "Continue after the expandable frontier is exhausted, "
                              "spending the whole iteration budget with classic UCB1.",
                              NULL,
                              &mcts_full_budget,
                              false,
                              PGC_USERSET,
                              0, NULL, NULL, NULL);

     DefineCustomBoolVariable("pg_mctsopt.cache_enabled",
                              "Enable join cost caching",
                              NULL,
                              &mcts_cache_enabled,
                              true,
                              PGC_USERSET,
                              0, NULL, NULL, NULL);

     DefineCustomIntVariable("pg_mctsopt.cache_size",
                             "Initial hash table entries for cost cache",
                             NULL,
                             &mcts_cache_size,
                             256,
                             256, 1000000,
                             PGC_USERSET,
                             0, NULL, NULL, NULL);

     DefineCustomIntVariable("pg_mctsopt.cache_max_size",
                             "Max cache entries (0 = unlimited)",
                             NULL,
                             &mcts_cache_max_size,
                             50000,
                             0, INT_MAX,
                             PGC_USERSET,
                             0, NULL, NULL, NULL);

     DefineCustomBoolVariable("pg_mctsopt.log_debug",
                              "Emit detailed WARNING messages during MCTS",
                              NULL,
                              &mcts_log_debug,
                              false,
                              PGC_USERSET,
                              0, NULL, NULL, NULL);
     DefineCustomBoolVariable("pg_mctsopt.log_steps",
                              "Log each join step of the best merge order with sizes",
                              NULL,
                              &mcts_log_steps,
                              false,
                              PGC_USERSET,
                              0, NULL, NULL, NULL);
     DefineCustomIntVariable("pg_mctsopt.plan_shape",
                             "Plan-shape parameter K: 0 = bushy, "
                             "1 = linear (zig-zag), >=2 = K-component bushy",
                             NULL,
                             &mcts_plan_shape,
                             1,
                             0, 100,
                             PGC_USERSET,
                             0, NULL, NULL, NULL);

     DefineCustomBoolVariable("pg_mctsopt.force_left_tree",
                              "Keep the growing kernel on the left in single-kernel MCTS.",
                              NULL,
                              &mcts_force_left_tree,
                              false,
                              PGC_USERSET,
                              0, NULL, NULL, NULL);

     DefineCustomEnumVariable("pg_mctsopt.rollout",
                              "Rollout strategy: 'random' (fixed PRNG seed per "
                              "phase) or 'luby' (re-seed the PRNG per rollout to "
                              "decorrelate consecutive rollouts).",
                              NULL,
                              &mcts_rollout_mode,
                              MCTS_EXTREME_ROLLOUT_RANDOM,
                              pg_mctsopt_rollout_options,
                              PGC_USERSET,
                              0, NULL, NULL, NULL);

     DefineCustomEnumVariable("pg_mctsopt.uct_aggregation",
                              "UCT Q-value aggregation: 'best' (UCT-Extreme, best "
                              "reward in subtree) or 'average' (standard UCT, "
                              "mean reward).",
                              NULL,
                              &mcts_uct_aggregation,
                              MCTS_EXTREME_UCT_AGG_BEST,
                              pg_mctsopt_uct_aggregation_options,
                              PGC_USERSET,
                              0, NULL, NULL, NULL);

     DefineCustomEnumVariable("pg_mctsopt.reward_map",
                              "Reward map phi (cost -> UCT-Extreme reward): "
                              "neg_log (-log(cost), default), "
                              "neg_cost (-cost), or "
                              "norm_neg_log (log-cost normalized into [0,1] "
                              "over the observed cost envelope).",
                              NULL,
                              &mcts_reward_map,
                              MCTS_EXTREME_REWARD_NEG_LOG,
                              pg_mctsopt_reward_options,
                              PGC_USERSET,
                              0, NULL, NULL, NULL);

     DefineCustomBoolVariable("pg_mctsopt.luby_enabled",
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
