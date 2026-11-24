"""Unit tests pinning lib's aggregation, relative-ratio, coverage, and plan parsing."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import lib  # noqa: E402  — needs the sys.path entry above


def _df(records):
    rows = []
    for config, query, seed, exec_ms in records:
        is_base = config in ("dp", "geqo")
        rows.append(
            {
                "source": "job",
                "config": config,
                "reward": "" if is_base else "norm_neg_log",
                "agg": "" if is_base else "best",
                "combo_id": "",
                "query": query,
                "seed": seed,
                "exec_time_ms": exec_ms,
                "plan_time_ms": 1.0,
                "mcts_best_cost": None if is_base else exec_ms,
                "plan_cost": exec_ms,
                "timed_out": exec_ms is None,
            }
        )
    return pd.DataFrame(rows)


_RATIO_FIXTURE = [
    ("luby", "qA", 1, 100.0),
    ("luby", "qA", 2, 120.0),  # luby qA mean = 110
    ("luby", "qB", 1, 200.0),
    ("luby", "qB", 2, 240.0),  # luby qB mean = 220
    ("geqo", "qA", 1, 200.0),
    ("geqo", "qA", 2, 200.0),  # geqo qA mean = 200
    ("geqo", "qB", 1, 100.0),
    ("geqo", "qB", 2, 100.0),  # geqo qB mean = 100
]


def test_relative_mean_of_ratios():
    out = lib.relative_workload(_df(_RATIO_FIXTURE), "exec_time_ms", "geqo", "mean_of_ratios")
    luby = out[out["config"] == "luby"].iloc[0]
    assert luby["n_queries"] == 2
    assert abs(luby["relative__mean"] - 1.375) < 1e-9  # mean(0.55, 2.20)
    assert abs(luby["relative__best"] - 0.55) < 1e-9
    assert abs(luby["relative__worst"] - 2.20) < 1e-9
    geqo = out[out["config"] == "geqo"].iloc[0]
    assert abs(geqo["relative__mean"] - 1.0) < 1e-9  # baseline vs itself


def test_relative_ratio_of_sums_is_weighted_and_within_whiskers():
    out = lib.relative_workload(_df(_RATIO_FIXTURE), "exec_time_ms", "geqo", "ratio_of_sums")
    luby = out[out["config"] == "luby"].iloc[0]
    assert abs(luby["relative__mean"] - 1.10) < 1e-9  # (110+220)/(200+100)
    assert luby["relative__best"] <= luby["relative__mean"] <= luby["relative__worst"]


def test_relative_drops_queries_baseline_lacks():
    # geqo has no qB -> inner join drops qB for every cell
    df = _df([("luby", "qA", 1, 100.0), ("luby", "qB", 1, 200.0), ("geqo", "qA", 1, 200.0)])
    luby = lib.relative_workload(df, "exec_time_ms", "geqo", "mean_of_ratios")
    luby = luby[luby["config"] == "luby"].iloc[0]
    assert luby["n_queries"] == 1
    assert abs(luby["relative__mean"] - 0.5) < 1e-9  # 100/200, qB gone


def test_aggregate_workload_sums_per_seed_totals():
    df = _df(
        [
            ("luby", "qA", 1, 100.0),
            ("luby", "qB", 1, 200.0),  # seed 1 workload total = 300
            ("luby", "qA", 2, 110.0),
            ("luby", "qB", 2, 210.0),  # seed 2 workload total = 320
        ]
    )
    row = lib.aggregate_workload(df).iloc[0]
    assert abs(row["exec_time_ms__mean"] - 310.0) < 1e-9  # mean of seed totals
    assert abs(row["exec_time_ms__best"] - 300.0) < 1e-9  # best whole-seed total
    assert abs(row["exec_time_ms__worst"] - 320.0) < 1e-9


def test_query_coverage_flags_timeout_mismatch():
    # luby times out on qB; geqo completes both
    df = _df(
        [
            ("luby", "qA", 1, 100.0),
            ("luby", "qB", 1, None),
            ("geqo", "qA", 1, 200.0),
            ("geqo", "qB", 1, 100.0),
        ]
    )
    _coverage, union, common = lib.query_coverage(df)
    assert union == {("job", "qA"), ("job", "qB")}
    assert common == {("job", "qA")}  # luby qB unusable
    restricted = lib.restrict_to_common_queries(df)
    assert set(restricted["query"].unique()) == {"qA"}


_PLAN = (
    " Aggregate  (cost=10.00..10.01 rows=1 width=8)"
    " (actual time=5.000..5.000 rows=1.00 loops=1)\n"
    "   ->  Seq Scan on t  (cost=0.00..9.00 rows=100 width=4)"
    " (actual time=1.000..2.000 rows=100.00 loops=1)\n"
    "   ->  Index Scan using i on u  (cost=0.00..1.00 rows=1 width=4)"
    " (actual time=0.100..0.200 rows=1.00 loops=10)\n"
    " Planning Time: 3.000 ms\n"
    " Execution Time: 5.100 ms\n"
)


def test_parse_plan_tree_structure_and_self_time():
    nodes = lib.parse_plan_tree(_PLAN)
    assert len(nodes) == 3
    root, scan, idx = nodes
    assert root["op"] == "Aggregate" and root["depth"] == 0
    assert scan["op"] == "Seq Scan on t" and scan["depth"] == 1
    # loops multiply wall time: index scan total = 0.200 * 10 = 2.0
    assert abs(idx["total_ms"] - 2.0) < 1e-9
    # root self = 5.0 - (seq 2.0 + idx 2.0) = 1.0
    assert abs(root["self_ms"] - 1.0) < 1e-9
    assert abs(scan["self_ms"] - 2.0) < 1e-9


def test_parse_explain_scalars():
    meta = lib.parse_explain(_PLAN)
    assert abs(meta["plan_time_ms"] - 3.0) < 1e-9
    assert abs(meta["exec_time_ms"] - 5.1) < 1e-9
    assert abs(meta["plan_cost"] - 10.01) < 1e-9


def test_plan_scan_methods():
    methods = lib.plan_scan_methods(lib.parse_plan_tree(_PLAN))
    assert methods == {"t": "Seq Scan", "u": "Index Scan"}


_JOIN_PLAN = (
    " Aggregate  (cost=10.00..10.01 rows=1 width=8) (actual time=5.000..5.000 rows=1.00 loops=1)\n"
    "   ->  Nested Loop  (cost=0.00..9.00 rows=1 width=4) (actual time=1.000..2.000 rows=1.00 loops=1)\n"
    "         ->  Seq Scan on title t  (cost=0.00..5.00 rows=1 width=4)"
    " (actual time=0.100..0.200 rows=1.00 loops=1)\n"
    "         ->  Index Scan using ix on movie_keyword mk  (cost=0.00..4.00 rows=1 width=4)"
    " (actual time=0.100..0.200 rows=1.00 loops=1)\n"
    " Execution Time: 5.000 ms\n"
)


def test_rows_removed_scaled_by_loops():
    plan = (
        " Seq Scan on t  (cost=0.00..5.00 rows=1 width=4)"
        " (actual time=0.100..0.200 rows=1.00 loops=10)\n"
        "   Filter: (x > 0)\n"
        "   Rows Removed by Filter: 7\n"
        " Execution Time: 1.000 ms\n"
    )
    node = lib.parse_plan_tree(plan)[0]
    assert node["rows_removed"] == 70  # 7 per loop * 10 loops


def test_join_order_leading():
    nodes = lib.parse_plan_tree(_JOIN_PLAN)
    assert lib.join_order_leading(nodes) == "Leading((t mk))"


def test_join_order_tree():
    tree = lib.join_order_tree(lib.parse_plan_tree(_JOIN_PLAN))
    assert tree[0] == (0, "⋈ join")
    assert (1, "t") in tree
    assert (1, "mk") in tree


def test_align_plans_pairs_and_marks_diffs():
    plan_a = lib.parse_plan_tree(_PLAN)
    plan_b = lib.parse_plan_tree(_PLAN.replace("Seq Scan on t", "Index Scan using ti on t"))
    rows = lib.align_plans(plan_a, plan_b)
    assert [tag for _a, _b, tag in rows] == ["same", "diff", "same"]


def _main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for test in tests:
        try:
            test()
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {test.__name__}: {exc}")
        else:
            print(f"ok    {test.__name__}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
