#include "core_adapter.hpp"
#include "duckdb/optimizer/join_order/join_order_operator.hpp"
#include "duckdb/optimizer/join_order/join_predicate.hpp"
#include "leaf_metadata.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include <iomanip>
#include <sstream>
#include <new>
#include "duckdb/main/client_context.hpp"
#include "duckdb/storage/arena_allocator.hpp"
#include "duckdb/storage/data_table.hpp"
#include "duckdb/storage/table/row_group.hpp"
#include "duckdb/storage/table/row_group_collection.hpp"
#include "duckdb/storage/statistics/numeric_stats.hpp"
#include <map>

#include <array>
#include <chrono>
#include <cmath>
#include <limits>

namespace duckdb {
namespace {
string MemoNumber(double value) {
    if (!std::isfinite(value)) {
        return "null";
    }
    std::ostringstream output;
    output << std::setprecision(17) << value;
    return output.str();
}
struct FrozenRelation {
    JoinRelationSet &set;
    const FrozenRelation *left;
    const FrozenRelation *right;
    double cost;
};
string FrozenSignature(const FrozenRelation &node) {
    if (!node.left) {
        return node.set.ToString();
    }
    return "(" + FrozenSignature(*node.left) + "," + FrozenSignature(*node.right) + ")";
}
struct RetentionCandidate {
    const FrozenRelation *plan;
    double cost, recosted, target_rows;
};
struct PairCacheEntry {
    optional_ptr<JoinRelationSet> result;
    optional_ptr<JoinRelationSet> result_left;
    optional_ptr<JoinRelationSet> result_right;
    std::array<double, 6> values;
    vector<reference<NeighborInfo>> connections;
};
struct DuckHost {
    explicit DuckHost(JoinOrderSearchContext &context)
        : context(context), arena(Allocator::Get(context.GetContext())) {}
    JoinOrderSearchContext &context;
    ArenaAllocator arena;
    idx_t join_calls = 0;
    idx_t candidate_evaluations = 0;
    bool retention_enabled = false, retain_attempted_roots = false;
    idx_t attempted_observations = 0, attempted_unsupported = 0, attempted_captures = 0,
          attempted_uncosted = 0;
    double retention_lambda = 0, retention_score = std::numeric_limits<double>::max();
    double retention_stored_cost = 0, retention_recosted_cost = 0, retention_target_rows = 0;
    const FrozenRelation *retention_plan = nullptr;
    idx_t retention_observations = 0, retention_changes = 0;
    vector<RetentionCandidate> retention_frontier;
    bool retention_scan_evidence = false, retention_log_score = false;
    idx_t retention_leaf_cap = 4;
    double retention_cost_guard = 0, retention_max_leaf_fraction = 1;
    double retention_max_group_fraction = 1;
    //! Relation index -> scan table index, as resolved by InitializeScanEvidence.
    vector<idx_t> retention_scan_source;
    std::map<std::pair<idx_t, idx_t>, double> retention_group_fraction;
    std::map<string, double> retention_group_fraction_named;
    idx_t retention_prune_checks = 0, retention_prune_rejected = 0, retention_prune_unknown = 0;
    bool retention_path_risk = false;
    vector<bool> retention_reduced_scans;
    idx_t retention_mapped_scans = 0, retention_eligible_scans = 0;
    idx_t retention_eligible = 0, retention_rejected = 0;
    bool cache_connections = false;
    bool pair_cache_enabled = false;
    idx_t connection_cache_hits = 0;
    idx_t connection_cache_misses = 0;
    idx_t connection_cache_entries = 0;
    idx_t cache_invalidations = 0;
    idx_t pair_cache_hits = 0;
    idx_t pair_cache_misses = 0;
    idx_t pair_cache_entries = 0;
    reference_map_t<JoinRelationSet,
                    reference_map_t<JoinRelationSet, vector<reference<NeighborInfo>>>>
        connections;
    reference_map_t<JoinRelationSet, reference_map_t<JoinRelationSet, PairCacheEntry>> pairs;
};

constexpr idx_t CACHE_ENTRY_LIMIT = 50000;
vector<reference<NeighborInfo>> Connections(DuckHost &state, JoinRelationSet &left,
                                            JoinRelationSet &right) {
    if (!state.cache_connections) {
        return state.context.GetQueryGraph().GetConnections(left, right);
    }
    auto l = state.connections.find(left);
    if (l != state.connections.end()) {
        auto r = l->second.find(right);
        if (r != l->second.end()) {
            state.connection_cache_hits++;
            return r->second;
        }
    }
    state.connection_cache_misses++;
    auto result = state.context.GetQueryGraph().GetConnections(left, right);
    if (state.connection_cache_entries < CACHE_ENTRY_LIMIT) {
        state.connections[left][right] = result;
        state.connection_cache_entries++;
    }
    return result;
}
void InvalidateCaches(DuckHost &state) {
    // Emitting an empty pair may add graph edges, including for other cached pairs.
    state.connections.clear();
    state.pairs.clear();
    state.connection_cache_entries = 0;
    state.pair_cache_entries = 0;
    if (state.cache_connections || state.pair_cache_enabled) {
        state.cache_invalidations++;
    }
}
DuckHost &Host(void *user) { return *static_cast<DuckHost *>(user); }
JoinRelationSet &AsRelation(MctsoptRel relation) {
    return *static_cast<JoinRelationSet *>(relation);
}
FrozenRelation *Freeze(DuckHost &state, JoinRelationSet &set, const FrozenRelation *left,
                       const FrozenRelation *right, double cost) {
    static_assert(alignof(FrozenRelation) <= 8, "DuckDB arena must align frozen relations");
    return new (state.arena.AllocateAligned(sizeof(FrozenRelation)))
        FrozenRelation{set, left, right, cost};
}
void *Allocate(void *user, size_t size) {
    static_assert(alignof(MctsoptNode) <= 8, "DuckDB arena must align core nodes");
    return Host(user).arena.AllocateAligned(size);
}
#include "leaf-selectivity.hpp"
#include "retention-selection.hpp"
#include "attempted-root.hpp"

bool Overlap(void *, MctsoptRel left, MctsoptRel right) {
    return JoinRelationSet::Intersects(AsRelation(left), AsRelation(right));
}
bool Complete(void *user, MctsoptRel relation) {
    return AsRelation(relation).count == Host(user).context.RelationCount();
}
bool Desirable(void *user, MctsoptRel left, MctsoptRel right) {
    auto &context = Host(user).context;
    auto &l = AsRelation(left);
    auto &r = AsRelation(right);
    for (auto &edge : Connections(Host(user), l, r)) {
        if (context.IsConnectionApplicable(edge, l, r)) {
            return true;
        }
    }
    return false;
}
double Cost(void *user, MctsoptRel relation) {
    return Host(user).context.GetPlan(AsRelation(relation)).cost;
}
double Rows(void *user, MctsoptRel relation) {
    return double(Host(user).context.GetPlan(AsRelation(relation)).cardinality);
}
std::array<double, 6> PairInputs(JoinOrderSearchContext &context, JoinRelationSet &left,
                                 JoinRelationSet &right, JoinRelationSet &result) {
    auto &l = context.GetPlan(left);
    auto &r = context.GetPlan(right);
    auto &out = context.GetPlan(result);
    return {l.cost,
            r.cost,
            double(l.cardinality),
            double(r.cardinality),
            out.cost,
            double(context.GetPlan(out.right_set).cardinality)};
}
bool SameConnections(const vector<reference<NeighborInfo>> &left,
                     const vector<reference<NeighborInfo>> &right) {
    if (left.size() != right.size()) {
        return false;
    }
    for (idx_t i = 0; i < left.size(); i++) {
        if (&left[i].get() != &right[i].get()) {
            return false;
        }
    }
    return true;
}
MctsoptRel Join(void *user, MctsoptRel left, MctsoptRel right, bool) {
    Host(user).join_calls++;
    if (Overlap(user, left, right)) {
        return nullptr;
    }
    auto &context = Host(user).context;
    auto &l = AsRelation(left);
    auto &r = AsRelation(right);
    auto &state = Host(user);
    auto connections = Connections(state, l, r);
    bool cacheable = state.pair_cache_enabled && !connections.empty() &&
                     !(state.retention_enabled && l.count + r.count == context.RelationCount());
    for (auto &connection : connections) {
        cacheable = cacheable && !connection.get().generated_cross_product;
    }
    optional_ptr<PairCacheEntry> cached;
    if (cacheable) {
        auto left_entry = state.pairs.find(l);
        if (left_entry != state.pairs.end()) {
            auto entry = left_entry->second.find(r);
            if (entry != left_entry->second.end()) {
                cached = entry->second;
                auto &out = context.GetPlan(*cached->result);
                // Memo entries can improve through other splits, including equal-cost ties.
                if (cached->result_left.get() == &out.left_set &&
                    cached->result_right.get() == &out.right_set &&
                    cached->values == PairInputs(context, l, r, *cached->result) &&
                    SameConnections(cached->connections, connections)) {
                    state.pair_cache_hits++;
                    return cached->result.get();
                }
            }
        }
        state.pair_cache_misses++;
    }
    if (connections.empty()) {
        InvalidateCaches(state);
    }
    optional_ptr<DPJoinNode> plan;
    if (state.retention_enabled && state.retain_attempted_roots &&
        l.count + r.count == context.RelationCount()) {
        JoinOrderSampledCost sampled;
        plan = context.EmitPairWithSampledCost(l, r, connections, context.GetPlan(l).cost,
                                               context.GetPlan(r).cost, sampled);
        if (plan && sampled.valid)
            RetainAttemptedRoot(state, l, r, connections, plan->set, sampled);
        else if (plan)
            state.attempted_uncosted++;
    } else {
        plan = context.EmitPair(l, r, connections);
    }
    if (cacheable && plan && (cached || state.pair_cache_entries < CACHE_ENTRY_LIMIT)) {
        if (!cached) {
            cached = state.pairs[l][r];
            state.pair_cache_entries++;
        }
        cached->result = plan->set;
        cached->result_left = plan->left_set;
        cached->result_right = plan->right_set;
        cached->values = PairInputs(context, l, r, plan->set);
        cached->connections = connections;
    }
    // Native memo nodes can be replaced. Only interned relation sets are stable handles.
    return plan ? &plan->set : nullptr;
}
void Rank(void *user, MctsoptRel left, MctsoptRel right, int ranking, double *score, double *rows) {
    Host(user).candidate_evaluations++;
    auto joined = Join(user, left, right, false);
    double cost = joined ? Cost(user, joined) : std::numeric_limits<double>::max();
    *rows = joined ? MaxValue(Rows(user, joined), 1.0) : std::numeric_limits<double>::max();
    switch (ranking) {
    case MCTSOPT_ROWS:
        *score = *rows;
        break;
    case MCTSOPT_MIX025:
        *score = std::log(*rows + 1.0) + 0.25 * std::log(cost + 1.0);
        break;
    case MCTSOPT_MIX050:
        *score = std::log(*rows + 1.0) + 0.50 * std::log(cost + 1.0);
        break;
    case MCTSOPT_SELECTIVITY:
        *score = *rows / (MaxValue(Rows(user, left), 1.0) * MaxValue(Rows(user, right), 1.0));
        break;
    default:
        *score = cost;
        break;
    }
}
MctsoptRel InitialPlan(void *user) {
    auto plan = Host(user).context.CompleteRelation();
    return plan ? plan.get() : nullptr;
}
void *BeginScope(void *, bool) { return nullptr; }
void EndScope(void *, void *) {}
void Restart(void *) {}
uint64_t Entropy(void *) {
    return uint64_t(std::chrono::high_resolution_clock::now().time_since_epoch().count());
}
void Interrupt(void *user) { Host(user).context.GetContext().InterruptCheck(); }
double Clock(void *) {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}
void ObserveCandidate(void *user, MctsoptRel) { ObserveRetention(Host(user)); }
MctsoptHost MakeHost(DuckHost &state, bool use_initial_plan) {
    MctsoptHost host{};
    host.user = &state;
    host.alloc = Allocate;
    host.desirable = Desirable;
    host.overlap = Overlap;
    host.complete = Complete;
    host.cost = Cost;
    host.rows = Rows;
    host.rank = Rank;
    host.join = Join;
    host.begin_scope = BeginScope;
    host.end_scope = EndScope;
    host.restart = Restart;
    host.entropy = Entropy;
    host.interrupt = Interrupt;
    host.clock_ms = Clock;
    host.candidate_observation = state.retention_enabled ? ObserveCandidate : nullptr;
    host.initial_plan = use_initial_plan ? InitialPlan : nullptr;
    return host;
}
vector<MctsoptRel> InitialRelations(JoinOrderSearchContext &context) {
    vector<MctsoptRel> initial;
    for (idx_t i = 0; i < context.RelationCount(); i++) {
        initial.push_back(&context.GetRelation(RelationIndex(i)));
    }
    return initial;
}
void ReplayFrozen(DuckHost &state, const FrozenRelation &node, idx_t &replayed) {
    if (!node.left) {
        if (node.right || node.set.count != 1 || !state.context.GetPlan(node.set).is_leaf) {
            throw SearchAbort("Invalid sampled incumbent leaf during replay");
        }
        return;
    }
    if (!node.right) {
        throw SearchAbort("Invalid sampled incumbent branch during replay");
    }
    ReplayFrozen(state, *node.left, replayed);
    ReplayFrozen(state, *node.right, replayed);
    auto &left = node.left->set;
    auto &right = node.right->set;
    // Replay can create graph edges too, so do not borrow a cache entry across emission.
    auto connections = state.cache_connections
                           ? Connections(state, left, right)
                           : state.context.GetQueryGraph().GetConnections(left, right);
    state.join_calls++;
    replayed++;
    auto plan = state.context.EmitPair(left, right, connections);
    if (!plan || &plan->set != &node.set) {
        throw SearchAbort("Cannot replay immutable sampled incumbent through native DuckDB");
    }
}
} // namespace

CoreGreedyResult RunCoreGreedy(JoinOrderSearchContext &context, const MctsoptConfig &config) {
    DuckHost state(context);
    auto host = MakeHost(state, false);
    auto initial = InitialRelations(context);
    auto result = mctsopt_greedy(&config, &host, initial.data(), NumericCast<int>(initial.size()));
    if (result.status == MCTSOPT_INVALID_ARGUMENT) {
        throw InvalidInputException("Invalid shape-aware greedy initialization configuration");
    }
    string order_json = "[";
    for (int i = 0; i < result.order.count; i++) {
        if (i) {
            order_json += ",";
        }
        auto &step = result.order.items[i];
        order_json += "{\"left\":" + AsRelation(step.left).ToString() +
                      ",\"right\":" + AsRelation(step.right).ToString() + "}";
    }
    order_json += "]";
    // Relation handles survive in DuckDB, but the merge sequence belongs to this arena.
    result.order = {nullptr, 0};
    return {result, order_json, state.join_calls, state.candidate_evaluations};
}

CoreSearchResult RunCoreSearch(JoinOrderSearchContext &context, const MctsoptConfig &config,
                               bool cache_connections, bool pair_cache,
                               const RetentionConfig &retention) {
    DuckHost state(context);
    state.cache_connections = cache_connections;
    state.pair_cache_enabled = pair_cache;
    state.retention_enabled = retention.enabled;
    auto leaf_cap = retention.leaf_cap;
    auto path_risk = retention.path_risk;
    auto cost_guard = retention.cost_guard;
    auto max_leaf_fraction = retention.max_leaf_fraction;
    auto scan_evidence = retention.scan_evidence;
    auto log_score = retention.log_score;
    auto attempted_roots = retention.attempted_roots;
    auto retention_lambda = retention.lambda;
    state.retention_lambda = retention_lambda;
    if (leaf_cap < 1 || leaf_cap > 64)
        throw InvalidInputException("Retention leaf cap must be between 1 and 64");
    state.retention_leaf_cap = leaf_cap;
    state.retention_path_risk = path_risk;
    if (!std::isfinite(cost_guard) || (cost_guard != 0 && cost_guard < 1))
        throw InvalidInputException("Retention cost guard must be zero or at least one");
    state.retention_cost_guard = cost_guard;
    if (!std::isfinite(max_leaf_fraction) || max_leaf_fraction <= 0 || max_leaf_fraction > 1)
        throw InvalidInputException("Retention maximum leaf fraction must be in (0, 1]");
    state.retention_max_leaf_fraction = max_leaf_fraction;
    if (!std::isfinite(retention.max_group_fraction) || retention.max_group_fraction <= 0 ||
        retention.max_group_fraction > 1)
        throw InvalidInputException("Retention maximum row-group fraction must be in (0, 1]");
    state.retention_max_group_fraction = retention.max_group_fraction;
    state.retention_scan_evidence = scan_evidence;
    state.retention_log_score = log_score;
    state.retain_attempted_roots = attempted_roots;
    if (!std::isfinite(retention_lambda) || retention_lambda < 0)
        throw InvalidInputException("Invalid retention lambda");

    auto host = MakeHost(state, context.HasCompletePlan());
    auto initial = InitialRelations(context);
    if (state.retention_enabled)
        InitializeScanEvidence(state);
    auto result = mctsopt_search(&config, &host, initial.data(), NumericCast<int>(initial.size()));
    if (result.status == MCTSOPT_INVALID_ARGUMENT) {
        throw InvalidInputException(
            "Invalid mctsopt_core configuration or cumulative iteration budget");
    }
    auto pre_replay_cost = context.BestCost();
    auto pre_replay_plan = context.BestPlanSignature();
    if (result.status == MCTSOPT_OK) {
        for (int i = 0; i < result.order.count; i++) {
            auto &step = result.order.items[i];
            if (!Join(&state, step.left, step.right, true)) {
                throw SearchAbort(
                    "Cannot replay mctsopt_core join order through native DuckDB");
            }
        }
    }
    string retention_json = "null";
    if (result.status == MCTSOPT_OK && state.retention_enabled) {
        auto native_signature = context.BestPlanSignature();
        auto native_final_cost = context.BestCost();
        auto native_target_rows = RetentionTargetRows(state, *context.CompleteRelation());
        ObserveRetention(state, true);
        if (!state.retention_plan)
            throw SearchAbort("Missing retained candidate");
        auto signature = FrozenSignature(*state.retention_plan);
        if (!log_score && retention_lambda == 0 && signature != native_signature)
            throw SearchAbort("Zero penalty changed native final topology");
        InvalidateCaches(state);
        context.PrepareSearch();
        idx_t replayed = 0;
        ReplayFrozen(state, *state.retention_plan, replayed);
        double actual_cost = context.BestCost();
        double scale = MaxValue(1.0, std::abs(state.retention_recosted_cost));
        if (context.BestPlanSignature() != signature || !std::isfinite(actual_cost) ||
            std::abs(actual_cost - state.retention_recosted_cost) > 1e-10 * scale)
            throw SearchAbort("Retained candidate replay changed topology or recosted cost");
        auto actual_target_rows = RetentionTargetRows(state, *context.CompleteRelation());
        if (actual_target_rows != state.retention_target_rows)
            throw SearchAbort("Retained candidate replay changed target-row feature");
        std::ostringstream out;
        out << "{\"lambda\":" << MemoNumber(retention_lambda)
            << ",\"scan_evidence\":" << (scan_evidence ? "true" : "false")
            << ",\"attempted_roots\":" << (attempted_roots ? "true" : "false")
            << ",\"attempted_observations\":" << state.attempted_observations
            << ",\"attempted_unsupported\":" << state.attempted_unsupported
            << ",\"attempted_captures\":" << state.attempted_captures
            << ",\"attempted_uncosted\":" << state.attempted_uncosted
            << ",\"mapped_scans\":" << state.retention_mapped_scans
            << ",\"eligible_scans\":" << state.retention_eligible_scans
            << ",\"max_group_fraction\":" << MemoNumber(state.retention_max_group_fraction)
            << ",\"prune_checks\":" << state.retention_prune_checks
            << ",\"prune_rejected\":" << state.retention_prune_rejected
            << ",\"prune_unknown\":" << state.retention_prune_unknown
            << ",\"group_fractions\":" << GroupFractionsJson(state) << ",\"score_mode\":\""
            << (log_score ? "cost_log_multiplier" : "additive_guard") << "\""
            << ",\"cost_guard\":"
            << (cost_guard > 0 ? MemoNumber(cost_guard) : (log_score ? "null" : "1.1"))
            << ",\"frontier_size\":" << state.retention_frontier.size()
            << ",\"eligible_with_final\":" << state.retention_eligible
            << ",\"rejected_by_guard\":" << state.retention_rejected
            << ",\"selected_native_cost_ratio\":"
            << MemoNumber(native_final_cost == 0 ? 1
                                                 : state.retention_stored_cost / native_final_cost)
            << ",\"max_leaf_fraction\":" << MemoNumber(max_leaf_fraction)
            << ",\"small_leaf_cap\":" << leaf_cap
            << ",\"path_risk\":" << (path_risk ? "true" : "false")
            << ",\"observations\":" << state.retention_observations
            << ",\"updates\":" << state.retention_changes
            << ",\"selected_earlier\":" << (signature != native_signature ? "true" : "false")
            << ",\"native_final_cost\":" << MemoNumber(native_final_cost)
            << ",\"native_target_rows\":" << MemoNumber(native_target_rows)
            << ",\"selected_stored_cost\":" << MemoNumber(state.retention_stored_cost)
            << ",\"selected_recosted_cost\":" << MemoNumber(state.retention_recosted_cost)
            << ",\"selected_target_rows\":" << MemoNumber(state.retention_target_rows)
            << ",\"score\":" << MemoNumber(state.retention_score)
            << ",\"replay_join_calls\":" << replayed << ",\"replay_verified\":true}";
        retention_json = out.str();
    }
    result.order = {nullptr, 0};
    return {result,
            pre_replay_cost,
            pre_replay_plan,
            state.join_calls,
            state.candidate_evaluations,
            state.connection_cache_hits,
            state.connection_cache_misses,
            state.cache_invalidations,
            state.pair_cache_hits,
            state.pair_cache_misses,
            state.pair_cache_entries,
            retention_json};
}
} // namespace duckdb
