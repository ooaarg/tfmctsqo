bool RetentionContains(const JoinRelationSet &set, idx_t target) {
    for (idx_t i = 0; i < set.count; i++)
        if (set.relations[i].index == target)
            return true;
    return false;
}
bool RetentionInnerPath(JoinOrderSearchContext &context, JoinRelationSet &set, idx_t target) {
    if (!RetentionContains(set, target))
        return false;
    auto &plan = context.GetPlan(set);
    if (plan.is_leaf)
        return true;
    if (plan.generated_cross_product || plan.predicates.empty())
        return false;
    for (auto &entry : plan.predicates) {
        auto &predicate = entry.get();
        auto &filter = predicate.GetFilter();
        if (predicate.GetPredicateClass() != JoinPredicateClass::INNER_EQUALITY ||
            filter.must_remain_filter || filter.must_remain_at_root)
            return false;
    }
    return RetentionInnerPath(context, plan.left_set, target) ||
           RetentionInnerPath(context, plan.right_set, target);
}
bool EstimatedBuildBarrier(JoinOrderSearchContext &context, JoinRelationSet &set, idx_t target) {
    auto &node = context.GetPlan(set);
    if (node.is_leaf)
        return false;
    bool in_left = RetentionContains(node.left_set, target);
    auto &branch = in_left ? node.left_set : node.right_set;
    auto &sibling = in_left ? node.right_set : node.left_set;
    auto a = context.GetPlan(branch).cardinality;
    auto b = context.GetPlan(sibling).cardinality;
    return (a != DConstants::INVALID_INDEX && b != DConstants::INVALID_INDEX && a < b) ||
           EstimatedBuildBarrier(context, branch, target);
}
void RetentionTargets(DuckHost &state, JoinRelationSet &set, vector<bool> &targets) {
    auto &context = state.context;
    auto &plan = context.GetPlan(set);
    if (plan.is_leaf)
        return;
    if (!plan.generated_cross_product) {
        for (bool left : {true, false}) {
            auto &small_set = left ? plan.left_set : plan.right_set;
            auto &other_set = left ? plan.right_set : plan.left_set;
            auto &small = context.GetPlan(small_set);
            auto &other = context.GetPlan(other_set);
            if (!small.is_leaf || small.cardinality == 0 ||
                small.cardinality > state.retention_leaf_cap)
                continue;
            idx_t sid = small_set.relations[0].index;
            if (state.retention_scan_evidence && !state.retention_reduced_scans[sid])
                continue;
            for (auto &entry : plan.predicates) {
                auto &predicate = entry.get();
                auto &filter = predicate.GetFilter();
                if (predicate.GetPredicateClass() != JoinPredicateClass::INNER_EQUALITY ||
                    !predicate.HasValidEqualityBindings() || filter.must_remain_filter ||
                    filter.must_remain_at_root)
                    continue;
                auto a = predicate.GetEqualityBinding(true).table_index.index;
                auto b = predicate.GetEqualityBinding(false).table_index.index;
                auto target = a == sid ? b : b == sid ? a : DConstants::INVALID_INDEX;
                if (target < targets.size() && RetentionInnerPath(context, other_set, target) &&
                    (other.cardinality == 0 || (state.retention_path_risk &&
                                                EstimatedBuildBarrier(context, other_set, target))) &&
                    RetentionTargetPrunable(state, predicate, target))
                    targets[target] = true;
            }
        }
    }
    RetentionTargets(state, plan.left_set, targets);
    RetentionTargets(state, plan.right_set, targets);
}
double RetentionTargetRows(DuckHost &state, JoinRelationSet &set) {
    auto &context = state.context;
    vector<bool> targets(context.RelationCount(), false);
    RetentionTargets(state, set, targets);
    double sum = 0;
    for (idx_t i = 0; i < targets.size(); i++) {
        if (!targets[i])
            continue;
        auto &leaf = context.GetPlan(context.GetRelation(RelationIndex(i)));
        if (leaf.cardinality != DConstants::INVALID_INDEX)
            sum += double(leaf.cardinality);
    }
    return sum;
}
FrozenRelation *FreezeRetained(DuckHost &state, JoinRelationSet &set) {
    auto &plan = state.context.GetPlan(set);
    if (plan.is_leaf)
        return Freeze(state, set, nullptr, nullptr, plan.cost);
    auto *left = FreezeRetained(state, plan.left_set);
    auto *right = FreezeRetained(state, plan.right_set);
    return Freeze(state, set, left, right, plan.cost);
}
double RetentionScore(DuckHost &state, double cost, double target_rows) {
    return state.retention_log_score ? cost * (1.0 + std::log1p(target_rows))
                                     : cost + state.retention_lambda * target_rows;
}
void SelectRetention(DuckHost &state, const RetentionCandidate &candidate) {
    state.retention_plan = candidate.plan;
    state.retention_stored_cost = candidate.cost;
    state.retention_recosted_cost = candidate.recosted;
    state.retention_target_rows = candidate.target_rows;
    state.retention_score = RetentionScore(state, candidate.cost, candidate.target_rows);
}
void ObserveRetention(DuckHost &state, bool final = false) {
    if (!state.retention_enabled)
        return;
    auto complete = state.context.CompleteRelation();
    if (!complete)
        return;
    state.retention_observations++;
    double native_cost = state.context.GetPlan(*complete).cost;
    double target_rows = RetentionTargetRows(state, *complete);
    double score = RetentionScore(state, native_cost, target_rows);
    if (!std::isfinite(score))
        throw SearchAbort("Non-finite retained candidate score");
    auto capture = [&]() {
        auto diagnostic = state.context.DiagnoseMemoCost(*complete);
        if (!diagnostic.complete || !std::isfinite(diagnostic.recosted_cost))
            throw SearchAbort("Retained candidate has no reproducible native cost");
        return RetentionCandidate{FreezeRetained(state, *complete), native_cost,
                                  diagnostic.recosted_cost, target_rows};
    };
    if (final) {
        // Start with the final native tree so exact score ties preserve it.
        SelectRetention(state, capture());
        double ceiling = state.retention_cost_guard > 0
                             ? native_cost * state.retention_cost_guard
                             : (state.retention_log_score ? std::numeric_limits<double>::max()
                                                          : native_cost * 1.10);
        state.retention_eligible = 1;
        state.retention_rejected = 0;
        for (const auto &candidate : state.retention_frontier) {
            state.retention_eligible += candidate.cost <= ceiling;
            state.retention_rejected += candidate.cost > ceiling;
            double candidate_score = RetentionScore(state, candidate.cost, candidate.target_rows);
            if (candidate.cost <= ceiling && candidate_score < state.retention_score)
                SelectRetention(state, candidate);
        }
        if (state.retention_stored_cost > ceiling)
            throw SearchAbort("Retained candidate violates native cost guard");
        return;
    }
    // A dominated candidate cannot win for any nonnegative penalty or cost ceiling.
    for (const auto &candidate : state.retention_frontier) {
        if (candidate.cost <= native_cost && candidate.target_rows <= target_rows)
            return;
    }
    auto candidate = capture();
    auto &frontier = state.retention_frontier;
    frontier.erase(std::remove_if(frontier.begin(), frontier.end(),
                                  [&](const RetentionCandidate &old) {
                                      return native_cost <= old.cost &&
                                             target_rows <= old.target_rows;
                                  }),
                   frontier.end());
    frontier.push_back(candidate);
    state.retention_changes++;
}
