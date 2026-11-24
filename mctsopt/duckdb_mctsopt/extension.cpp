#include "core_adapter.hpp"
#include "leaf_metadata.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/storage/statistics/node_statistics.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/execution/expression_executor_state.hpp"
#include "duckdb/logging/logger.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/client_context_state.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/optimizer/optimizer_extension.hpp"

#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>

namespace duckdb {
struct MctsoptState : public ClientContextState {
    idx_t searches = 0;
    idx_t core_searches = 0;
    idx_t core_iterations = 0;
    idx_t core_no_plan = 0;
    idx_t fallbacks = 0;
    idx_t original_plans = 0;
    bool has_cost = false;
    double cost = 0;
    vector<string> diagnostics;
};

template <class T> static T Option(ClientContext &context, const char *name) {
    Value value;
    if (!context.TryGetCurrentSetting(name, value)) {
        throw InternalException("Missing duckdb_mctsopt setting: %s", name);
    }
    return value.GetValue<T>();
}

static void FilterBindings(const Expression &expr, std::unordered_set<idx_t> &bindings,
                           bool &unknown) {
    if (expr.GetExpressionClass() == ExpressionClass::BOUND_COLUMN_REF) {
        auto &column = expr.Cast<BoundColumnRefExpression>();
        if (column.Depth() || !column.Binding().table_index.IsValid())
            unknown = true;
        else
            bindings.insert(column.Binding().table_index.index);
    }
    ExpressionIterator::EnumerateChildren(
        expr, [&](const Expression &child) { FilterBindings(child, bindings, unknown); });
}
static void CollectFilterEvidence(const Expression &expr, LeafMetadataState::Frame &frame) {
    if (expr.GetExpressionType() == ExpressionType::CONJUNCTION_AND) {
        ExpressionIterator::EnumerateChildren(
            expr, [&](const Expression &child) { CollectFilterEvidence(child, frame); });
        return;
    }
    std::unordered_set<idx_t> bindings;
    bool unknown = false;
    FilterBindings(expr, bindings, unknown);
    if (!unknown && bindings.size() == 1)
        frame.filtered.insert(*bindings.begin());
}
static void CollectLeafMetadata(ClientContext &context, LogicalOperator &op,
                                LeafMetadataState::Frame &frame) {
    if (op.type == LogicalOperatorType::LOGICAL_FILTER)
        for (auto &expr : op.expressions)
            CollectFilterEvidence(*expr, frame);
    if (op.type == LogicalOperatorType::LOGICAL_GET) {
        auto &get = op.Cast<LogicalGet>();
        LeafMetadataState::Entry entry;
        entry.function = get.function.name.GetIdentifierName();
        entry.source = "unknown";
        // Use scan statistics directly; EstimateCardinality can return a fallback or prior
        // estimate.
        if (get.function.cardinality) {
            auto stats = get.function.cardinality(context, get.bind_data.get());
            if (stats && stats->has_estimated_cardinality) {
                entry.cardinality = stats->estimated_cardinality;
                entry.known = true;
                entry.source = "scan_statistics";
            }
        }
        entry.table = get.GetTable();
        entry.column_ids = get.GetColumnIds();
        entry.projection_ids = get.projection_ids;
        frame.scans[get.table_index.index] = entry;
        if (get.table_filters.HasFilters())
            frame.filtered.insert(get.table_index.index);
    }
    for (auto &child : op.children)
        CollectLeafMetadata(context, *child, frame);
}
static void FinishLeafMetadata(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &) {
    if (!Option<bool>(input.context, "duckdb_mctsopt_enabled") ||
        !Option<bool>(input.context, "duckdb_mctsopt_retention_enabled") ||
        !Option<bool>(input.context, "duckdb_mctsopt_retention_scan_evidence"))
        return;
    auto state =
        input.context.registered_state->GetOrCreate<LeafMetadataState>("mctsopt_leaf_metadata");
    if (state->frames.empty() || state->frames.back().optimizer != &input.optimizer)
        throw InternalException("Mismatched optimizer metadata scope");
    state->frames.pop_back();
}
static void BeginLeafMetadata(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan) {
    if (!Option<bool>(input.context, "duckdb_mctsopt_enabled") ||
        !Option<bool>(input.context, "duckdb_mctsopt_retention_enabled") ||
        !Option<bool>(input.context, "duckdb_mctsopt_retention_scan_evidence"))
        return;
    auto metadata =
        input.context.registered_state->GetOrCreate<LeafMetadataState>("mctsopt_leaf_metadata");
    metadata->frames.push_back({&input.optimizer, {}, {}});
    CollectLeafMetadata(input.context, *plan, metadata->frames.back());
}

static int IntegerOption(ClientContext &context, const char *name, int minimum, int maximum) {
    auto value = Option<int64_t>(context, name);
    if (value < minimum || value > maximum) {
        throw InvalidInputException("%s must be between %d and %d", name, minimum, maximum);
    }
    return int(value);
}
static int EnumOption(ClientContext &context, const char *name, const vector<string> &values) {
    auto value = Option<string>(context, name);
    for (idx_t i = 0; i < values.size(); i++) {
        if (value == values[i]) {
            return int(i);
        }
    }
    throw InvalidInputException("Invalid %s: %s", name, value);
}
static double ClockMilliseconds() {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}
static string JsonNumber(double value) {
    if (!std::isfinite(value) || value >= std::numeric_limits<double>::max()) {
        return "null";
    }
    std::ostringstream output;
    output << std::setprecision(17) << value;
    return output.str();
}
static string JsonString(const string &text) {
    string out = "\"";
    for (char c : text) {
        if (c == '"' || c == '\\')
            out += '\\';
        out += c;
    }
    return out + "\"";
}
class CoreSearch : public JoinOrderSearchAlgorithm {
  public:
    CoreSearch(MctsoptConfig config, bool greedy_init, bool shape_greedy, bool cache_connections,
               bool pair_cache)
        : config(config), greedy_init(greedy_init), shape_greedy(shape_greedy),
          cache_connections(cache_connections), pair_cache(pair_cache) {}
    bool Search(JoinOrderSearchContext &context) override {
        auto state =
            context.GetContext().registered_state->GetOrCreate<MctsoptState>("duckdb_mctsopt");
        auto start = ClockMilliseconds();
        context.PrepareSearch();
        if (context.HasCompletePlan() && context.RelationCount() > 1) {
            throw InternalException("Pure MCTS preparation retained a non-leaf complete plan");
        }
        CoreGreedyResult greedy_run{};
        bool initialized = false;
        if (greedy_init) {
            if (shape_greedy) {
                greedy_run = RunCoreGreedy(context, config);
                initialized = greedy_run.core.status == MCTSOPT_OK && context.HasCompletePlan();
            } else {
                initialized = context.InitializeGreedy();
            }
        }
        bool initialization_failed = greedy_init && !initialized;
        auto initial_cost = context.BestCost();
        auto initial_plan = context.BestPlanSignature();
        auto initialization_ms = ClockMilliseconds() - start;
        auto search_start = ClockMilliseconds();
        CoreSearchResult run{};
        string abort_reason;
        if (!initialization_failed) {
            RetentionConfig retention;
            auto &client = context.GetContext();
            retention.enabled = Option<bool>(client, "duckdb_mctsopt_retention_enabled");
            retention.attempted_roots =
                Option<bool>(client, "duckdb_mctsopt_retain_attempted_roots");
            retention.scan_evidence =
                Option<bool>(client, "duckdb_mctsopt_retention_scan_evidence");
            retention.log_score = Option<bool>(client, "duckdb_mctsopt_retention_log_score");
            retention.path_risk = Option<bool>(client, "duckdb_mctsopt_retention_path_risk");
            retention.leaf_cap = IntegerOption(client, "duckdb_mctsopt_retention_leaf_cap", 1, 64);
            retention.cost_guard = Option<double>(client, "duckdb_mctsopt_retention_cost_guard");
            retention.max_leaf_fraction =
                Option<double>(client, "duckdb_mctsopt_retention_max_leaf_fraction");
            retention.lambda = Option<double>(client, "duckdb_mctsopt_retention_lambda");
            retention.max_group_fraction =
                Option<double>(client, "duckdb_mctsopt_retention_max_group_fraction");
            try {
                run = RunCoreSearch(context, config, cache_connections, pair_cache, retention);
            } catch (SearchAbort &abort) {
                // The search could not reproduce its own result. Plan natively; do not fail the query.
                abort_reason = abort.what();
                run = CoreSearchResult{};
                run.core.status = MCTSOPT_NO_PLAN;
                run.core.best_cost = std::numeric_limits<double>::max();
                run.pre_replay_cost = initial_cost;
                run.pre_replay_plan = initial_plan;
            }
            state->core_searches++;
        } else {
            // A failed initialization is a registered fallback, not a greedy MCTS success.
            run.core.status = MCTSOPT_NO_PLAN;
            run.core.best_cost = std::numeric_limits<double>::max();
            run.pre_replay_cost = initial_cost;
            run.pre_replay_plan = initial_plan;
        }
        auto search_ms = initialization_failed ? 0 : ClockMilliseconds() - search_start;
        auto &result = run.core;
        state->core_iterations += result.iterations;
        bool solved = true;
        bool fallback = result.status == MCTSOPT_NO_PLAN;
        if (fallback) {
            if (!initialization_failed) {
                state->core_no_plan++;
            }
            state->fallbacks++;
            DUCKDB_LOG_WARNING(context.GetContext(),
                               "duckdb_mctsopt: reason=%s seed=%llu budget=%d iterations=%d "
                               "relations=%llu fallback=native_search",
                               initialization_failed    ? "greedy_initialization_failed"
                               : !abort_reason.empty() ? "search_aborted"
                                                       : "no_plan",
                               config.seed, config.budget, result.iterations,
                               context.RelationCount());
            solved = context.FallbackToNative();
            if (!solved) {
                state->original_plans++;
            }
        }
        state->searches++;
        state->has_cost = solved;
        state->cost = context.BestCost();
        std::ostringstream diagnostic;
        int64_t requested = 0;
        for (int phase = 1; phase <= config.phases; phase++) {
            requested += int64_t(config.budget) * (config.luby_budget ? mctsopt_luby(phase) : 1);
        }
        diagnostic
            << "{\"relation_count\":" << context.RelationCount()
            << ",\"selected\":true,\"seed\":" << config.seed
            << ",\"greedy_init\":" << (greedy_init ? "true" : "false") << ",\"greedy_init_mode\":\""
            << (greedy_init ? (shape_greedy ? "shape" : "native") : "none") << "\""
            << ",\"greedy_invocations\":" << (greedy_init ? 1 : 0)
            << ",\"greedy_initialization_failed\":" << (initialization_failed ? "true" : "false")
            << ",\"greedy_initial_order\":"
            << (greedy_init && shape_greedy ? greedy_run.order_json : "null")
            << ",\"greedy_steps\":"
            << (greedy_init && shape_greedy ? std::to_string(greedy_run.core.steps) : "null")
            << ",\"greedy_relaxed_steps\":"
            << (greedy_init && shape_greedy ? std::to_string(greedy_run.core.relaxed_steps)
                                            : "null")
            << ",\"greedy_fallback_steps\":"
            << (greedy_init && shape_greedy ? std::to_string(greedy_run.core.fallback_steps)
                                            : "null")
            << ",\"greedy_failures\":"
            << (greedy_init && shape_greedy ? std::to_string(greedy_run.core.failures) : "null")
            << ",\"greedy_candidate_evaluations\":"
            << (greedy_init && shape_greedy ? std::to_string(greedy_run.candidate_evaluations)
                                            : "null")
            << ",\"greedy_join_calls\":"
            << (greedy_init && shape_greedy ? std::to_string(greedy_run.join_calls) : "null")
            << ",\"initial_has_plan\":" << (initialized ? "true" : "false")
            << ",\"initial_cost\":" << JsonNumber(initial_cost) << ",\"initial_plan\":\""
            << initial_plan << "\""
            << ",\"cost\":" << JsonNumber(state->cost)
            << ",\"observed_cost\":" << JsonNumber(result.best_cost)
            << ",\"pre_replay_cost\":" << JsonNumber(run.pre_replay_cost)
            << ",\"pre_replay_plan\":\"" << run.pre_replay_plan << "\""
            << ",\"final_plan\":\"" << context.BestPlanSignature() << "\""
            << ",\"initialization_ms\":" << JsonNumber(initialization_ms)
            << ",\"search_ms\":" << JsonNumber(search_ms)
            << ",\"selection_ms\":" << JsonNumber(result.selection_ms)
            << ",\"expansion_ms\":" << JsonNumber(result.expansion_ms)
            << ",\"rollout_ms\":" << JsonNumber(result.rollout_ms)
            << ",\"backprop_ms\":" << JsonNumber(result.backprop_ms)
            << ",\"iterations\":" << result.iterations
            << ",\"attempted_iterations\":" << result.attempted_iterations
            << ",\"requested_iterations\":" << requested
            << ",\"rollout_calls\":" << result.rollout_calls
            << ",\"phases_completed\":" << result.completed_phases
            << ",\"exhausted_phases\":" << result.exhausted_phases
            << ",\"best_phase\":" << result.best_phase
            << ",\"best_iteration\":" << result.best_iteration
            << ",\"failures\":" << result.failures
            << ",\"candidate_evaluations\":" << run.candidate_evaluations
            << ",\"cache_connections\":" << (cache_connections ? "true" : "false")
            << ",\"connection_cache_hits\":" << run.connection_cache_hits
            << ",\"connection_cache_misses\":" << run.connection_cache_misses
            << ",\"cache_invalidations\":" << run.cache_invalidations
            << ",\"retention\":" << run.retention_json << ",\"rollout\":\""
            << (config.rank_biased_rollout ? "rank_biased"
                : config.luby_rollout      ? "luby"
                                           : "random")
            << "\""
            << ",\"rollout_epsilon\":" << config.rollout_epsilon
            << ",\"pair_cache_enabled\":" << (pair_cache ? "true" : "false")
            << ",\"pair_cache_hits\":" << run.pair_cache_hits
            << ",\"pair_cache_misses\":" << run.pair_cache_misses
            << ",\"pair_cache_entries\":" << run.pair_cache_entries
            << ",\"join_calls\":" << run.join_calls << ",\"stop_reason\":\""
            << (initialization_failed ? "initialization_failed"
                                      : mctsopt_stop_reason_name(result.stop_reason))
            << "\""
            << ",\"fallback\":" << (fallback ? "true" : "false") << ",\"fallback_reason\":"
            << (fallback ? (initialization_failed    ? "\"greedy_initialization_failed\""
                            : !abort_reason.empty() ? "\"search_aborted\""
                                                    : "\"no_plan\"")
                         : "null")
            << ",\"abort_detail\":" << (abort_reason.empty() ? string("null") : JsonString(abort_reason))
            << ",\"fallback_result\":"
            << (fallback ? (solved ? "\"native_plan\"" : "\"original_plan\"") : "null") << "}";
        state->diagnostics.push_back(diagnostic.str());
        return solved;
    }

  private:
    MctsoptConfig config;
    bool greedy_init;
    bool shape_greedy;
    bool cache_connections;
    bool pair_cache;
};

static unique_ptr<JoinOrderSearchAlgorithm> SelectSearch(JoinOrderExtensionInput &input) {
    if (!Option<bool>(input.context, "duckdb_mctsopt_enabled")) {
        return nullptr;
    }
    auto scope = Option<string>(input.context, "duckdb_mctsopt_scope");
    if (scope != "approximate" && scope != "all") {
        throw InvalidInputException("duckdb_mctsopt_scope must be 'approximate' or 'all'");
    }
    if ((scope == "all") != (input.stage == JoinOrderSearchStage::COMPLETE)) {
        return nullptr;
    }
    auto &context = input.context;
    auto minimum = IntegerOption(context, "duckdb_mctsopt_min_relations", 0, 1000);
    if (input.relation_count < idx_t(minimum)) {
        auto state = context.registered_state->GetOrCreate<MctsoptState>("duckdb_mctsopt");
        state->diagnostics.push_back("{\"relation_count\":" + std::to_string(input.relation_count) +
                                     ",\"selected\":false,\"reason\":\"below_min_relations\"}");
        return nullptr;
    }
    MctsoptConfig config{};
    config.seed = Option<uint64_t>(context, "duckdb_mctsopt_seed");
    auto budget = Option<uint64_t>(context, "duckdb_mctsopt_iterations");
    if (budget == 0 || budget > 100000000) {
        throw InvalidInputException("duckdb_mctsopt_iterations must be between 1 and 100000000");
    }
    config.budget = int(budget);
    config.phases = IntegerOption(context, "duckdb_mctsopt_phases", 1, 1000);
    config.top_k = IntegerOption(context, "duckdb_mctsopt_top_k", 0, 10000);
    config.depth = IntegerOption(context, "duckdb_mctsopt_depth", 0, 100);
    config.shape = IntegerOption(context, "duckdb_mctsopt_plan_shape", 0, 100);
    config.rollouts = IntegerOption(context, "duckdb_mctsopt_rollouts_per_leaf", 1, 10000);
    config.patience = IntegerOption(context, "duckdb_mctsopt_patience", 0, 1000);
    config.ranking = EnumOption(context, "duckdb_mctsopt_expand_strategy",
                                {"cost", "row", "mixed_025", "mixed_050", "selectivity"});
    config.reward =
        EnumOption(context, "duckdb_mctsopt_reward_map", {"neg_cost", "neg_log", "norm_neg_log"});
    config.average =
        EnumOption(context, "duckdb_mctsopt_uct_aggregation", {"best", "average"}) == 1;
    auto rollout = EnumOption(context, "duckdb_mctsopt_rollout", {"random", "luby", "rank_biased"});
    config.luby_rollout = rollout == 1;
    config.rank_biased_rollout = rollout == 2;
    config.rollout_epsilon = Option<double>(context, "duckdb_mctsopt_rollout_epsilon");
    if (!std::isfinite(config.rollout_epsilon) || config.rollout_epsilon < 0 ||
        config.rollout_epsilon > 1)
        throw InvalidInputException("duckdb_mctsopt_rollout_epsilon must be in [0, 1]");
    config.exploration = Option<double>(context, "duckdb_mctsopt_exploration_constant");
    config.gamma = Option<double>(context, "duckdb_mctsopt_gamma");
    config.full_budget = Option<bool>(context, "duckdb_mctsopt_full_budget");
    config.luby_budget = Option<bool>(context, "duckdb_mctsopt_luby_enabled");
    config.force_left = Option<bool>(context, "duckdb_mctsopt_force_left_tree");
    config.legacy_selection = Option<bool>(context, "duckdb_mctsopt_legacy_selection");
    config.rank_all = Option<bool>(context, "duckdb_mctsopt_rank_all");
    config.legacy_normalization = Option<bool>(context, "duckdb_mctsopt_legacy_normalization");
    auto greedy_mode = EnumOption(context, "duckdb_mctsopt_greedy_init_mode", {"native", "shape"});
    auto greedy_init = Option<bool>(context, "duckdb_mctsopt_greedy_init");
    auto pair_cache = Option<bool>(context, "duckdb_mctsopt_pair_cache");
    if (pair_cache && greedy_init) {
        throw InvalidInputException("Pair cache requires pure MCTS (greedy_init=false)");
    }
    return make_uniq<CoreSearch>(config, greedy_init, greedy_mode == 1,
                                 Option<bool>(context, "duckdb_mctsopt_cache_connections"),
                                 pair_cache);
}

static void Diagnostics(DataChunk &args, ExpressionState &state, Vector &result) {
    auto stats = state.GetContext().registered_state->Get<MctsoptState>("duckdb_mctsopt");
    string output = "[";
    if (stats) {
        for (idx_t i = 0; i < stats->diagnostics.size(); i++) {
            if (i) {
                output += ",";
            }
            output += stats->diagnostics[i];
        }
    }
    output += "]";
    result.Reference(Value(output), count_t(args.size()));
}
static void LastCost(DataChunk &args, ExpressionState &state, Vector &result) {
    auto stats = state.GetContext().registered_state->Get<MctsoptState>("duckdb_mctsopt");
    result.Reference(stats && stats->has_cost ? Value::DOUBLE(stats->cost)
                                              : Value(LogicalType::DOUBLE),
                     count_t(args.size()));
}
static void SearchCount(DataChunk &args, ExpressionState &state, Vector &result) {
    auto stats = state.GetContext().registered_state->Get<MctsoptState>("duckdb_mctsopt");
    result.Reference(Value::UBIGINT(stats ? stats->searches : 0), count_t(args.size()));
}
static void CoreSearchCount(DataChunk &args, ExpressionState &state, Vector &result) {
    auto stats = state.GetContext().registered_state->Get<MctsoptState>("duckdb_mctsopt");
    result.Reference(Value::UBIGINT(stats ? stats->core_searches : 0), count_t(args.size()));
}
static void CoreIterations(DataChunk &args, ExpressionState &state, Vector &result) {
    auto stats = state.GetContext().registered_state->Get<MctsoptState>("duckdb_mctsopt");
    result.Reference(Value::UBIGINT(stats ? stats->core_iterations : 0), count_t(args.size()));
}
static void CoreNoPlanCount(DataChunk &args, ExpressionState &state, Vector &result) {
    auto stats = state.GetContext().registered_state->Get<MctsoptState>("duckdb_mctsopt");
    result.Reference(Value::UBIGINT(stats ? stats->core_no_plan : 0), count_t(args.size()));
}
static void FallbackCount(DataChunk &args, ExpressionState &state, Vector &result) {
    auto stats = state.GetContext().registered_state->Get<MctsoptState>("duckdb_mctsopt");
    result.Reference(Value::UBIGINT(stats ? stats->fallbacks : 0), count_t(args.size()));
}
static void OriginalPlanCount(DataChunk &args, ExpressionState &state, Vector &result) {
    auto stats = state.GetContext().registered_state->Get<MctsoptState>("duckdb_mctsopt");
    result.Reference(Value::UBIGINT(stats ? stats->original_plans : 0), count_t(args.size()));
}
} // namespace duckdb

extern "C" {
DUCKDB_CPP_EXTENSION_ENTRY(duckdb_mctsopt, loader) {
    using namespace duckdb;
    auto &config = DBConfig::GetConfig(loader.GetDatabaseInstance());
    config.AddExtensionOption("duckdb_mctsopt_rollout_epsilon",
                              "Uniform weight in rank-biased rollout", LogicalType::DOUBLE,
                              Value(0.25));
    config.AddExtensionOption("duckdb_mctsopt_retention_enabled",
                              "Retain and select alternative complete trees", LogicalType::BOOLEAN,
                              Value(true));
    config.AddExtensionOption(
        "duckdb_mctsopt_retention_max_group_fraction",
        "Count a retained-plan risk target only if an equality filter on its join column is expected to "
        "read at most this share of its row groups (min/max statistics); 1 disables the check",
        LogicalType::DOUBLE, Value::DOUBLE(0.5));
    config.AddExtensionOption(
        "duckdb_mctsopt_retention_max_leaf_fraction",
        "Maximum estimated filtered/base dimension fraction for retained-plan risk",
        LogicalType::DOUBLE, Value::DOUBLE(0.05));
    config.AddExtensionOption(
        "duckdb_mctsopt_retention_cost_guard",
        "Maximum retained cost divided by final native memo cost; zero disables the extra guard",
        LogicalType::DOUBLE, Value::DOUBLE(1.1));
    config.AddExtensionOption(
        "duckdb_mctsopt_retention_leaf_cap",
        "Maximum estimated cardinality of a filtered dimension for retained-plan risk",
        LogicalType::UBIGINT, Value::UBIGINT(4));
    config.AddExtensionOption("duckdb_mctsopt_retention_path_risk",
                              "Include estimated build barriers inside an eligible target path",
                              LogicalType::BOOLEAN, Value(true));
    config.AddExtensionOption(
        "duckdb_mctsopt_retain_attempted_roots",
        "Retain eligible attempted complete roots before native-cost rejection",
        LogicalType::BOOLEAN, Value(true));
    config.AddExtensionOption("duckdb_mctsopt_retention_log_score",
                              "Use native cost times one plus log of scan-risk rows",
                              LogicalType::BOOLEAN, Value(true));
    config.AddExtensionOption("duckdb_mctsopt_retention_scan_evidence",
                              "Require estimated scan reduction and local filter evidence",
                              LogicalType::BOOLEAN, Value(true));
    config.AddExtensionOption("duckdb_mctsopt_retention_lambda",
                              "Experimental retained-plan zero-side penalty", LogicalType::DOUBLE,
                              Value::DOUBLE(0.01));
    config.AddExtensionOption("duckdb_mctsopt_enabled", "Enable external MCTS join search",
                              LogicalType::BOOLEAN, Value(false));
    config.AddExtensionOption("duckdb_mctsopt_scope",
                              "Search all joins or only approximate-search fallback",
                              LogicalType::VARCHAR, Value("all"));
    config.AddExtensionOption("duckdb_mctsopt_seed",
                              "Shared core seed; zero uses time-based entropy",
                              LogicalType::UBIGINT, Value::UBIGINT(42));
    config.AddExtensionOption("duckdb_mctsopt_iterations", "Positive shared core per-phase budget",
                              LogicalType::UBIGINT, Value::UBIGINT(250));
    config.AddExtensionOption("duckdb_mctsopt_phases", "Number of search phases",
                              LogicalType::INTEGER, Value(1));
    config.AddExtensionOption("duckdb_mctsopt_top_k", "Expansion candidate limit; zero keeps all",
                              LogicalType::INTEGER, Value(20));
    config.AddExtensionOption("duckdb_mctsopt_depth", "Tree depth limit; zero is unlimited",
                              LogicalType::INTEGER, Value(10));
    config.AddExtensionOption("duckdb_mctsopt_plan_shape",
                              "Search shape: zero bushy, one linear, higher K-component",
                              LogicalType::INTEGER, Value(1));
    config.AddExtensionOption("duckdb_mctsopt_rollouts_per_leaf", "Rollouts per depth-limited leaf",
                              LogicalType::INTEGER, Value(1));
    config.AddExtensionOption("duckdb_mctsopt_patience",
                              "Phases without improvement before stopping; zero disables",
                              LogicalType::INTEGER, Value(0));
    config.AddExtensionOption("duckdb_mctsopt_expand_strategy", "Candidate ranking strategy",
                              LogicalType::VARCHAR, Value("cost"));
    config.AddExtensionOption("duckdb_mctsopt_reward_map", "Cost-to-reward mapping",
                              LogicalType::VARCHAR, Value("neg_log"));
    config.AddExtensionOption("duckdb_mctsopt_uct_aggregation", "UCT aggregation: best or average",
                              LogicalType::VARCHAR, Value("best"));
    config.AddExtensionOption("duckdb_mctsopt_rollout",
                              "Rollout policy: random, luby, or rank_biased", LogicalType::VARCHAR,
                              Value("rank_biased"));
    config.AddExtensionOption("duckdb_mctsopt_exploration_constant", "UCT exploration constant",
                              LogicalType::DOUBLE, Value(0.05));
    config.AddExtensionOption("duckdb_mctsopt_gamma", "UCT Extreme exploration exponent",
                              LogicalType::DOUBLE, Value(0.5));
    config.AddExtensionOption("duckdb_mctsopt_full_budget",
                              "Continue after expandable frontier exhaustion", LogicalType::BOOLEAN,
                              Value(false));
    config.AddExtensionOption("duckdb_mctsopt_luby_enabled",
                              "Weight phase budgets by the Luby sequence", LogicalType::BOOLEAN,
                              Value(false));
    config.AddExtensionOption("duckdb_mctsopt_force_left_tree",
                              "Keep the growing kernel left in linear search", LogicalType::BOOLEAN,
                              Value(false));
    config.AddExtensionOption("duckdb_mctsopt_greedy_init",
                              "Initialize an explicit greedy incumbent before MCTS",
                              LogicalType::BOOLEAN, Value(false));
    config.AddExtensionOption("duckdb_mctsopt_greedy_init_mode",
                              "Greedy initialization: native or shape-aware merge sequence",
                              LogicalType::VARCHAR, Value("native"));
    config.AddExtensionOption("duckdb_mctsopt_min_relations",
                              "Minimum native join-region relation count", LogicalType::INTEGER,
                              Value(12));
    config.AddExtensionOption("duckdb_mctsopt_legacy_selection",
                              "Reproduce historical premature subtree exhaustion",
                              LogicalType::BOOLEAN, Value(false));
    config.AddExtensionOption("duckdb_mctsopt_rank_all",
                              "Rank all candidates even when top-k does not prune",
                              LogicalType::BOOLEAN, Value(false));
    config.AddExtensionOption("duckdb_mctsopt_legacy_normalization",
                              "Use historical cached reward normalization", LogicalType::BOOLEAN,
                              Value(false));
    config.AddExtensionOption("duckdb_mctsopt_cache_connections",
                              "Cache ordered connection lookups within the adapter search",
                              LogicalType::BOOLEAN, Value(true));
    config.AddExtensionOption("duckdb_mctsopt_pair_cache",
                              "Reuse unchanged native pair emissions in pure MCTS",
                              LogicalType::BOOLEAN, Value(false));
    ScalarFunction diagnostics("duckdb_mctsopt_diagnostics", {}, LogicalType::VARCHAR, Diagnostics);
    diagnostics.SetStability(FunctionStability::VOLATILE);
    loader.RegisterFunction(diagnostics);
    OptimizerExtension extension;
    extension.pre_optimize_function = BeginLeafMetadata;
    extension.optimize_function = FinishLeafMetadata;
    extension.join_order_function = SelectSearch;
    OptimizerExtension::Register(config, std::move(extension));
    ScalarFunction cost("duckdb_mctsopt_last_cost", {}, LogicalType::DOUBLE, LastCost);
    cost.SetStability(FunctionStability::VOLATILE);
    loader.RegisterFunction(cost);
    ScalarFunction count("duckdb_mctsopt_search_count", {}, LogicalType::UBIGINT, SearchCount);
    count.SetStability(FunctionStability::VOLATILE);
    loader.RegisterFunction(count);

    ScalarFunction core_search_count("duckdb_mctsopt_core_search_count", {}, LogicalType::UBIGINT,
                                     CoreSearchCount);
    core_search_count.SetStability(FunctionStability::VOLATILE);
    loader.RegisterFunction(core_search_count);
    ScalarFunction core_iterations("duckdb_mctsopt_core_iterations", {}, LogicalType::UBIGINT,
                                   CoreIterations);
    core_iterations.SetStability(FunctionStability::VOLATILE);
    loader.RegisterFunction(core_iterations);
    ScalarFunction core_no_plan_count("duckdb_mctsopt_core_no_plan_count", {}, LogicalType::UBIGINT,
                                      CoreNoPlanCount);
    core_no_plan_count.SetStability(FunctionStability::VOLATILE);
    loader.RegisterFunction(core_no_plan_count);
    ScalarFunction fallback_count("duckdb_mctsopt_fallback_count", {}, LogicalType::UBIGINT,
                                  FallbackCount);
    fallback_count.SetStability(FunctionStability::VOLATILE);
    loader.RegisterFunction(fallback_count);
    ScalarFunction original_plan_count("duckdb_mctsopt_original_plan_count", {},
                                       LogicalType::UBIGINT, OriginalPlanCount);
    original_plan_count.SetStability(FunctionStability::VOLATILE);
    loader.RegisterFunction(original_plan_count);
}
}
