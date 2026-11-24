const BoundColumnRefExpression *ScanBinding(const Expression &expr) {
    if (expr.GetExpressionClass() == ExpressionClass::BOUND_COLUMN_REF) {
        auto &column = expr.Cast<BoundColumnRefExpression>();
        return column.Depth() == 0 && column.Binding().table_index.IsValid() ? &column : nullptr;
    }
    if (expr.GetExpressionClass() != ExpressionClass::BOUND_FUNCTION ||
        !BoundCastExpression::IsCast(expr))
        return nullptr;
    auto &cast = expr.Cast<BoundFunctionExpression>();
    if (BoundCastExpression::IsTryCast(cast) ||
        !BoundCastExpression::CastIsInvertible(BoundCastExpression::SourceType(cast),
                                               cast.GetReturnType()))
        return nullptr;
    return ScanBinding(BoundCastExpression::Child(cast));
}
void InitializeScanEvidence(DuckHost &state) {
    auto &context = state.context;
    auto count = context.RelationCount();
    state.retention_reduced_scans.assign(count, false);
    if (!state.retention_scan_evidence)
        return;
    auto metadata = context.GetContext().registered_state->GetOrCreate<LeafMetadataState>(
        "mctsopt_leaf_metadata");
    if (metadata->frames.empty())
        return;
    auto &frame = metadata->frames.back();
    vector<idx_t> source(count, DConstants::INVALID_INDEX);
    vector<bool> ambiguous(count, false);
    for (idx_t i = 0; i < count; i++)
        for (idx_t j = i + 1; j < count; j++) {
            auto &a = context.GetRelation(RelationIndex(i));
            auto &b = context.GetRelation(RelationIndex(j));
            for (auto &edge : context.GetQueryGraph().GetConnections(a, b))
                for (auto entry : edge.get().predicates) {
                    auto &predicate = entry.get();
                    auto &filter = predicate.GetFilter();
                    if (!predicate.HasValidEqualityBindings() || !filter.filter ||
                        !BoundComparisonExpression::IsComparison(*filter.filter))
                        continue;
                    auto &expr = filter.filter->Cast<BoundFunctionExpression>();
                    for (bool left : {true, false}) {
                        auto local = predicate.GetEqualityBinding(left).table_index.index;
                        if (local >= count)
                            continue;
                        auto column = ScanBinding(left ? BoundComparisonExpression::Left(expr)
                                                       : BoundComparisonExpression::Right(expr));
                        if (!column) {
                            ambiguous[local] = true;
                            continue;
                        }
                        auto global = column->Binding().table_index.index;
                        if (source[local] != DConstants::INVALID_INDEX && source[local] != global)
                            ambiguous[local] = true;
                        source[local] = global;
                    }
                }
        }
    state.retention_scan_source = source;
    for (idx_t i = 0; i < count; i++)
        if (ambiguous[i])
            state.retention_scan_source[i] = DConstants::INVALID_INDEX;
    for (idx_t i = 0; i < count; i++) {
        auto entry = frame.scans.find(source[i]);
        if (ambiguous[i] || entry == frame.scans.end() || !entry->second.known)
            continue;
        state.retention_mapped_scans++;
        auto rows = context.GetPlan(context.GetRelation(RelationIndex(i))).cardinality;
        // A consistent scan lineage, local predicate and estimate reduction are evidence, not a
        // filter-effect guarantee.
        bool reduced =
            frame.filtered.count(source[i]) && rows != DConstants::INVALID_INDEX && rows > 0 &&
            rows < entry->second.cardinality &&
            double(rows) <= state.retention_max_leaf_fraction * double(entry->second.cardinality);
        state.retention_reduced_scans[i] = reduced;
        state.retention_eligible_scans += reduced && rows <= state.retention_leaf_cap;
    }
}

//! Row-weighted expected share of a table's row groups whose min/max range admits one value of the
//! column, a value drawn uniformly from the column's range. 1 means an equality filter on this column
//! cannot skip a single row group; a small value means the groups separate the column's values.
//! Negative when the column has no usable integral statistics.
double ExpectedGroupFraction(DataTable &storage, idx_t column) {
    auto &collection = *storage.GetRowGroupCollection();
    auto groups = collection.GetRowGroupCount();
    vector<std::array<double, 3>> spans;
    double low = 0, high = 0, rows = 0;
    for (idx_t i = 0; i < groups; i++) {
        auto group = collection.GetRowGroup(int64_t(i));
        if (!group)
            return -1;
        auto stats = group->GetStatistics(column);
        if (!stats || !stats->GetType().IsIntegral() || !NumericStats::HasMinMax(*stats))
            return -1;
        double minimum = NumericStats::Min(*stats).GetValue<double>();
        double maximum = NumericStats::Max(*stats).GetValue<double>();
        double count = double(group->count.load());
        low = spans.empty() ? minimum : MinValue(low, minimum);
        high = spans.empty() ? maximum : MaxValue(high, maximum);
        rows += count;
        spans.push_back({minimum, maximum, count});
    }
    if (spans.empty() || rows <= 0)
        return -1;
    double domain = high - low + 1, weighted = 0;
    for (auto &span : spans)
        weighted += span[2] * MinValue(1.0, (span[1] - span[0] + 1) / domain);
    return weighted / rows;
}
string GroupFractionsJson(DuckHost &state) {
    std::ostringstream out;
    out << "{";
    bool first = true;
    for (auto &entry : state.retention_group_fraction_named) {
        out << (first ? "" : ",") << "\"" << entry.first << "\":" << MemoNumber(entry.second);
        first = false;
    }
    return out.str() + "}";
}
//! Whether an early equality filter reaching `target` through this predicate could let its scan skip
//! row groups. A small filtered dimension only makes a cheap scan when the fact table's join column is
//! physically separated by value; with mixed values every group admits the key and nothing is skipped,
//! while an early join can cost the pruning other join keys would have given. Unknown layouts keep the
//! previous behaviour and are counted.
bool RetentionTargetPrunable(DuckHost &state, const JoinPredicate &predicate, idx_t target) {
    if (state.retention_max_group_fraction >= 1)
        return true;
    state.retention_prune_checks++;
    auto unknown = [&]() {
        state.retention_prune_unknown++;
        return true;
    };
    auto &context = state.context;
    auto metadata = context.GetContext().registered_state->GetOrCreate<LeafMetadataState>(
        "mctsopt_leaf_metadata");
    auto &filter = predicate.GetFilter();
    if (metadata->frames.empty() || target >= state.retention_scan_source.size() || !filter.filter ||
        !BoundComparisonExpression::IsComparison(*filter.filter))
        return unknown();
    auto &expr = filter.filter->Cast<BoundFunctionExpression>();
    bool left = predicate.GetEqualityBinding(true).table_index.index == target;
    auto column = ScanBinding(left ? BoundComparisonExpression::Left(expr)
                                   : BoundComparisonExpression::Right(expr));
    auto global = state.retention_scan_source[target];
    if (!column || global == DConstants::INVALID_INDEX || column->Binding().table_index.index != global)
        return unknown();
    auto &frame = metadata->frames.back();
    auto entry = frame.scans.find(global);
    if (entry == frame.scans.end() || !entry->second.table || !entry->second.table->IsDuckTable())
        return unknown();
    auto &scan = entry->second;
    idx_t position = column->Binding().column_index.GetIndex();
    if (!scan.projection_ids.empty()) {
        if (position >= scan.projection_ids.size())
            return unknown();
        position = scan.projection_ids[position].GetIndex();
    }
    if (position >= scan.column_ids.size() || scan.column_ids[position].IsVirtualColumn())
        return unknown();
    auto physical = scan.table->GetStorageIndex(scan.column_ids[position]).GetPrimaryIndex();
    auto key = std::make_pair(global, physical);
    auto cached = state.retention_group_fraction.find(key);
    double fraction;
    if (cached != state.retention_group_fraction.end()) {
        fraction = cached->second;
    } else {
        fraction = ExpectedGroupFraction(scan.table->GetStorage(), physical);
        state.retention_group_fraction[key] = fraction;
        state.retention_group_fraction_named[scan.table->name + "." +
                                             scan.table->GetColumns().GetColumn(PhysicalIndex(physical)).Name()] =
            fraction;
    }
    if (fraction < 0)
        return unknown();
    if (fraction > state.retention_max_group_fraction) {
        state.retention_prune_rejected++;
        return false;
    }
    return true;
}
