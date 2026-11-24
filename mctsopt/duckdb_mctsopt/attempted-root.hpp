bool AttemptedRootTargets(DuckHost &state, JoinRelationSet &left, JoinRelationSet &right,
                          const vector<reference<NeighborInfo>> &connections, double &rows) {
    auto &context = state.context;
    vector<reference<JoinPredicate>> predicates;
    optional_ptr<JoinOrderOperator> group;
    bool has_group = false;
    for (auto &entry : connections) {
        auto &connection = entry.get();
        if (!context.IsConnectionApplicable(connection, left, right))
            continue;
        if (connection.generated_cross_product || connection.predicates.empty() ||
            (connection.join_operator &&
             connection.join_operator->type != JoinOrderOperatorType::INNER))
            return false;
        if (has_group && group != connection.join_operator)
            return false;
        group = connection.join_operator;
        has_group = true;
        for (auto &pred : connection.predicates) {
            auto &filter = pred.get().GetFilter();
            if (pred.get().GetPredicateClass() != JoinPredicateClass::INNER_EQUALITY ||
                filter.must_remain_filter || filter.must_remain_at_root)
                return false;
            predicates.push_back(pred);
        }
    }
    if (!has_group)
        return false;
    vector<bool> targets(context.RelationCount(), false);
    RetentionTargets(state, left, targets);
    RetentionTargets(state, right, targets);
    for (bool small_left : {true, false}) {
        auto &small_set = small_left ? left : right;
        auto &other_set = small_left ? right : left;
        auto &small = context.GetPlan(small_set);
        auto &other = context.GetPlan(other_set);
        if (!small.is_leaf || small.cardinality == 0 ||
            small.cardinality > state.retention_leaf_cap)
            continue;
        idx_t sid = small_set.relations[0].index;
        if (state.retention_scan_evidence && !state.retention_reduced_scans[sid])
            continue;
        for (auto &entry : predicates) {
            auto &pred = entry.get();
            if (!pred.HasValidEqualityBindings())
                continue;
            auto a = pred.GetEqualityBinding(true).table_index.index;
            auto b = pred.GetEqualityBinding(false).table_index.index;
            auto target = a == sid ? b : b == sid ? a : DConstants::INVALID_INDEX;
            if (target < targets.size() && RetentionInnerPath(context, other_set, target) &&
                (other.cardinality == 0 ||
                 (state.retention_path_risk && EstimatedBuildBarrier(context, other_set, target))) &&
                RetentionTargetPrunable(state, pred, target))
                targets[target] = true;
        }
    }
    rows = 0;
    for (idx_t i = 0; i < targets.size(); i++)
        if (targets[i]) {
            auto &leaf = context.GetPlan(context.GetRelation(RelationIndex(i)));
            if (leaf.cardinality != DConstants::INVALID_INDEX)
                rows += double(leaf.cardinality);
        }
    return true;
}
void RetainAttemptedRoot(DuckHost &state, JoinRelationSet &left, JoinRelationSet &right,
                         const vector<reference<NeighborInfo>> &connections,
                         JoinRelationSet &complete, const JoinOrderSampledCost &sampled) {
    state.attempted_observations++;
    double rows = 0;
    if (!AttemptedRootTargets(state, left, right, connections, rows)) {
        state.attempted_unsupported++;
        return;
    }
    if (!std::isfinite(sampled.cost) || !std::isfinite(RetentionScore(state, sampled.cost, rows)))
        throw SearchAbort("Nonfinite attempted-root score");
    for (const auto &candidate : state.retention_frontier)
        if (candidate.cost <= sampled.cost && candidate.target_rows <= rows)
            return;
    auto l = state.context.DiagnoseMemoCost(left);
    auto r = state.context.DiagnoseMemoCost(right);
    double recost =
        sampled.cardinality + sampled.left_input_cost + l.recosted_cost + r.recosted_cost;
    if (!l.complete || !r.complete || !std::isfinite(recost))
        throw SearchAbort("Attempted root has no reproducible child costs");
    auto *fl = FreezeRetained(state, left);
    auto *fr = FreezeRetained(state, right);
    auto *root = Freeze(state, complete, fl, fr, sampled.cost);
    auto &frontier = state.retention_frontier;
    frontier.erase(std::remove_if(frontier.begin(), frontier.end(),
                                  [&](const RetentionCandidate &old) {
                                      return sampled.cost <= old.cost && rows <= old.target_rows;
                                  }),
                   frontier.end());
    frontier.push_back(RetentionCandidate{root, sampled.cost, recost, rows});
    state.attempted_captures++;
}
