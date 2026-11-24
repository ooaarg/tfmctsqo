#pragma once
#include "duckdb/main/client_context_state.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/common/column_index.hpp"
#include "duckdb/common/projection_index.hpp"
#include <unordered_map>
#include <unordered_set>
namespace duckdb {
struct LeafMetadataState : public ClientContextState {
    struct Entry {
        idx_t cardinality = 0;
        bool known = false;
        string source, function;
        //! The scanned table and its bound columns, for row-group statistics. Empty when not a table.
        optional_ptr<TableCatalogEntry> table;
        vector<ColumnIndex> column_ids;
        vector<ProjectionIndex> projection_ids;
    };
    struct Frame {
        const void *optimizer;
        std::unordered_map<idx_t, Entry> scans;
        std::unordered_set<idx_t> filtered;
    };
    vector<Frame> frames;
    void QueryBegin(ClientContext &) override { frames.clear(); }
    void QueryEnd() override { frames.clear(); }
};
} // namespace duckdb
