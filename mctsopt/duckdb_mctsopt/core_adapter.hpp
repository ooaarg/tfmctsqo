#pragma once

#include "duckdb/optimizer/join_order/join_order_extension.hpp"
#include "mctsopt.h"

#include <stdexcept>

namespace duckdb {
//! A search whose result cannot be reproduced or trusted. The caller discards it and plans natively.
//! Not an InternalException on purpose: DuckDB invalidates the whole database for those, which is the
//! wrong response to one plan that an experimental search could not replay.
class SearchAbort : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};
struct RetentionConfig {
    bool enabled = true, attempted_roots = true, scan_evidence = true, log_score = true,
         path_risk = true;
    idx_t leaf_cap = 4;
    double cost_guard = 1.1, max_leaf_fraction = 0.05, lambda = 0.01, max_group_fraction = 0.5;
};
struct CoreSearchResult {
    MctsoptResult core;
    double pre_replay_cost;
    string pre_replay_plan;
    idx_t join_calls;
    idx_t candidate_evaluations;
    idx_t connection_cache_hits;
    idx_t connection_cache_misses;
    idx_t cache_invalidations;
    idx_t pair_cache_hits;
    idx_t pair_cache_misses;
    idx_t pair_cache_entries;
    string retention_json = "null";
};
struct CoreGreedyResult {
    MctsoptGreedyResult core;
    string order_json;
    idx_t join_calls;
    idx_t candidate_evaluations;
};
CoreSearchResult RunCoreSearch(JoinOrderSearchContext &context, const MctsoptConfig &config,
                               bool cache_connections, bool pair_cache,
                               const RetentionConfig &retention);
CoreGreedyResult RunCoreGreedy(JoinOrderSearchContext &context, const MctsoptConfig &config);
} // namespace duckdb
