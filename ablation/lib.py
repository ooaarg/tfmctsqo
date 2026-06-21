"""Parsing, aggregation, and small stats helpers shared by sweep.py / app.py / sharing.py."""

from __future__ import annotations

import difflib
import json
import re
import statistics
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

_TOP_COST_RE = re.compile(r"\(cost=([\d.]+)\.\.([\d.]+)")
_PLAN_TIME_RE = re.compile(r"^\s*Planning Time:\s+([\d.]+)\s*ms", re.M)
_EXEC_TIME_RE = re.compile(r"^\s*Execution Time:\s+([\d.]+)\s*ms", re.M)
_MCTS_FIELD_RE = re.compile(r"^\s*MCTS ([\w\- ]+?):\s+([\d.]+)", re.M)
_TIMEOUT_RE = re.compile(r"canceling statement due to statement timeout", re.I)

_FIELD_MAP = {
    "Iterations": "iters",
    "Best Cost": "mcts_best_cost",
    "Relations": "rels",
    "Phases": "phases",
    "Depth-Limit Rollouts": "depth_limit_hits",
    "Total Random Rollouts": "total_random_rollouts",
    "Rollout Failures": "rollout_failures",
    "Exhausted": "exhausted",
}


def parse_explain(text: str) -> dict:
    """Pull MCTS stats + plan/exec time + top plan cost from a psql EXPLAIN capture.
    Missing fields are omitted; callers treat absence as None."""
    out: dict = {}

    m = _TOP_COST_RE.search(text)
    if m:
        out["plan_cost"] = float(m.group(2))

    m = _PLAN_TIME_RE.search(text)
    if m:
        out["plan_time_ms"] = float(m.group(1))

    m = _EXEC_TIME_RE.search(text)
    if m:
        out["exec_time_ms"] = float(m.group(1))

    for m in _MCTS_FIELD_RE.finditer(text):
        label = m.group(1).strip()
        key = _FIELD_MAP.get(label)
        if key is None:
            continue
        val_str = m.group(2)
        try:
            val = float(val_str) if "." in val_str else int(val_str)
        except ValueError:
            continue
        out[key] = val

    if _TIMEOUT_RE.search(text):
        out["timed_out"] = True

    return out


_PLAN_NODE_RE = re.compile(
    r"^(?P<lead>\s*)(?P<arrow>->\s+)?(?P<op>.*?)\s+"
    r"\(cost=(?P<cs>[\d.]+)\.\.(?P<ct>[\d.]+)\s+rows=(?P<er>\d+)\s+width=(?P<w>\d+)\)"
    r"(?:\s+\(actual time=(?P<as_>[\d.]+)\.\.(?P<at>[\d.]+)\s+"
    r"rows=(?P<ar>[\d.]+)\s+loops=(?P<lp>\d+)\)|\s+\((?P<never>never executed)\))?\s*$"
)
# The plan tree ends where the planner/executor/MCTS footer begins.
_PLAN_FOOTER_RE = re.compile(r"^\s*(Planning|Execution Time:|JIT:|Settings:|Trigger|MCTS\b)")
_ROWS_REMOVED_RE = re.compile(r"Rows Removed by [\w ]+?:\s*(\d+)")


def _count_rows_removed(details: list[str]) -> int:
    return sum(
        int(value) for detail in details for value in _ROWS_REMOVED_RE.findall(detail)
    )


def _plan_node_from_match(m: re.Match) -> dict:
    loops = int(m.group("lp")) if m.group("lp") else None
    act_total = float(m.group("at")) if m.group("at") else None
    total_ms = act_total * loops if (act_total is not None and loops) else 0.0
    return {
        "op": m.group("op").strip(),
        "cost_total": float(m.group("ct")),
        "est_rows": int(m.group("er")),
        "act_total_ms": act_total,
        "act_rows": float(m.group("ar")) if m.group("ar") else None,
        "loops": loops,
        "never": bool(m.group("never")),
        "total_ms": total_ms,
        "self_ms": total_ms,  # children subtracted in a second pass
        "details": [],
        "_children": [],
    }


def parse_plan_tree(text: str) -> list[dict]:
    """Parse a psql text ``EXPLAIN (ANALYZE)`` capture into a flat pre-order list
    of node dicts.  Header (LOAD/SET/QUERY PLAN/---) and footer (Planning/
    Execution/MCTS) lines are skipped; non-node lines attach as ``details`` of the
    current node.  Depth comes from the column where each node's name starts, via
    an indent stack, so it survives arbitrary nesting.  ``total_ms`` is wall time
    including repeats (actual_total * loops); ``self_ms`` subtracts children's
    ``total_ms`` so the true hot node stands out.  Returns ``[]`` if no node parses
    (e.g. an EXPLAIN-without-ANALYZE or a timed-out capture)."""
    nodes: list[dict] = []
    stack: list[tuple[int, dict]] = []  # (name_start_col, node)
    started = False
    for raw in text.splitlines():
        if _PLAN_FOOTER_RE.match(raw):
            if started:
                break
            continue
        m = _PLAN_NODE_RE.match(raw)
        if m is None:
            if started and raw.strip() and stack:
                stack[-1][1]["details"].append(raw.strip())
            continue
        started = True
        node = _plan_node_from_match(m)
        name_col = len(m.group("lead")) + (len(m.group("arrow")) if m.group("arrow") else 0)
        while stack and stack[-1][0] >= name_col:
            stack.pop()
        node["depth"] = len(stack)
        if stack:
            stack[-1][1]["_children"].append(node)
        stack.append((name_col, node))
        nodes.append(node)
    for node in nodes:
        child_total = sum(child["total_ms"] for child in node["_children"])
        node["self_ms"] = max(0.0, node["total_ms"] - child_total)
        # Rows Removed is per-loop, like row counts -> scale to a total.
        node["rows_removed"] = _count_rows_removed(node["details"]) * (node["loops"] or 1)
        del node["_children"]
    return nodes


_PLAN_SCAN_RE = re.compile(r"^(?P<scan>[A-Za-z][\w ]*?Scan)(?: using \S+)? on (?P<rel>\S+)")


def plan_scan_methods(nodes: list[dict]) -> dict:
    """Map base relation -> scan operator for every scan node (e.g. ``title`` ->
    ``Index Scan``, ``keyword`` -> ``Seq Scan``).  Comparing two plans' maps shows
    operator-choice differences ("bad operator") for the same query."""
    methods: dict = {}
    for node in nodes:
        match = _PLAN_SCAN_RE.match(node["op"])
        if match:
            methods[match.group("rel")] = match.group("scan")
    return methods


def align_plans(nodes_a: list[dict], nodes_b: list[dict]) -> list[tuple]:
    """Row-align two parsed plans for side-by-side display via a sequence diff on
    indented operator signatures.  Returns ``(node_a | None, node_b | None, tag)``
    rows where tag is ``same`` | ``diff`` | ``a_only`` | ``b_only`` — so matching
    operators line up and a single table can show both sides."""
    a_sig = ["  " * node["depth"] + node["op"] for node in nodes_a]
    b_sig = ["  " * node["depth"] + node["op"] for node in nodes_b]
    matcher = difflib.SequenceMatcher(a=a_sig, b=b_sig, autojunk=False)
    rows: list[tuple] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            rows.extend((nodes_a[i1 + k], nodes_b[j1 + k], "same") for k in range(i2 - i1))
        elif tag == "replace":
            for k in range(max(i2 - i1, j2 - j1)):
                node_a = nodes_a[i1 + k] if i1 + k < i2 else None
                node_b = nodes_b[j1 + k] if j1 + k < j2 else None
                rows.append((node_a, node_b, "diff"))
        elif tag == "delete":
            rows.extend((nodes_a[k], None, "a_only") for k in range(i1, i2))
        else:  # insert
            rows.extend((None, nodes_b[k], "b_only") for k in range(j1, j2))
    return rows


_LEADING_ALIAS_RE = re.compile(r"\bon\s+(?P<table>\w+)(?:\s+(?P<alias>\w+))?\s*$")


def _scan_alias(op: str) -> str | None:
    """Alias (or table) a scan node reads, for Leading hints; None if not a scan."""
    if "Scan" not in op:
        return None
    match = _LEADING_ALIAS_RE.search(op)
    if not match:
        return None
    return match.group("alias") or match.group("table")


def _plan_children(nodes: list[dict]) -> dict:
    """Map node index -> direct child indices, reconstructed from the depth column."""
    children: dict[int, list[int]] = {}
    stack: list[int] = []
    for idx, node in enumerate(nodes):
        while stack and nodes[stack[-1]]["depth"] >= node["depth"]:
            stack.pop()
        if stack:
            children.setdefault(stack[-1], []).append(idx)
        stack.append(idx)
    return children


def _join_structure(nodes: list[dict], children: dict, idx: int):
    """Nested join order: a leaf alias (str), a list of sub-structures (a join), or
    None.  Base scans are leaves; single-child wrappers (Aggregate/Hash/Sort/…) are
    unwrapped; join nodes keep their >=2 inputs."""
    alias = _scan_alias(nodes[idx]["op"])
    if alias:
        return alias
    parts = []
    for child in children.get(idx, []):
        sub = _join_structure(nodes, children, child)
        if sub is not None:
            parts.append(sub)
    if len(parts) <= 1:
        return parts[0] if parts else None
    return parts


def _structure_to_leading(struct) -> str:
    if struct is None:
        return ""
    if isinstance(struct, str):
        return struct
    return f"({' '.join(_structure_to_leading(part) for part in struct)})"


def _join_tree_lines(struct, depth: int, out: list) -> None:
    if struct is None:
        return
    if isinstance(struct, str):
        out.append((depth, struct))
        return
    out.append((depth, "⋈ join"))
    for part in struct:
        _join_tree_lines(part, depth + 1, out)


def join_order_leading(nodes: list[dict]) -> str:
    """Plan's join order as a pg_hint_plan ``Leading(...)`` hint using table aliases.
    Returns '' for an empty plan."""
    if not nodes:
        return ""
    inner = _structure_to_leading(_join_structure(nodes, _plan_children(nodes), 0))
    return f"Leading({inner})" if inner else ""


def join_order_tree(nodes: list[dict]) -> list[tuple]:
    """Join order as ``(depth, label)`` rows for an indented tree view — ``⋈ join``
    for a join node, the table alias for a leaf."""
    if not nodes:
        return []
    out: list[tuple] = []
    _join_tree_lines(_join_structure(nodes, _plan_children(nodes), 0), 0, out)
    return out


def load_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return pd.DataFrame(rows)


def iter_jsonl(path: Path, start_offset: int = 0) -> tuple[list[dict], int]:
    """Incremental tail-style read; returns (new_rows, new_offset) for the live view."""
    if not path.exists():
        return [], start_offset
    with path.open("rb") as f:
        f.seek(start_offset)
        data = f.read()
    if not data:
        return [], start_offset
    text = data.decode("utf-8", errors="replace")
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    return rows, start_offset + len(data)


METRICS = ("mcts_best_cost", "plan_cost", "plan_time_ms", "exec_time_ms", "e2e_time_ms", "iters")


def _ensure_combo_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "combo_id" not in df.columns:
        df["combo_id"] = ""
    else:
        df["combo_id"] = df["combo_id"].fillna("").astype(str)
    return df


def aggregate_cells(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (config, reward, agg, combo_id, query) with best/mean/median/worst
    of each metric across seeds.  Baselines have no `mcts_best_cost`, so we fall back
    to `plan_cost` so they still appear on the cost panel."""
    if df.empty:
        return df
    df = _ensure_combo_id(df)
    if "mcts_best_cost" not in df.columns:
        df["mcts_best_cost"] = pd.NA
    if "plan_cost" in df.columns:
        df["mcts_best_cost"] = df["mcts_best_cost"].fillna(df["plan_cost"])
    keys = ["config", "reward", "agg", "combo_id", "query"]
    grouped = df.groupby(keys, as_index=False)
    rows = []
    for vals, sub in grouped:
        row = dict(zip(keys, vals, strict=False))
        row["n_seeds"] = len(sub)
        for m in METRICS:
            if m not in sub.columns:
                continue
            s = sub[m].dropna()
            if s.empty:
                continue
            row[f"{m}__best"] = float(s.min())
            row[f"{m}__mean"] = float(s.mean())
            row[f"{m}__median"] = float(s.median())
            row[f"{m}__worst"] = float(s.max())
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_workload(df: pd.DataFrame) -> pd.DataFrame:
    """Per-cell workload totals across queries.

    Mean remains the mean workload total across seeds.  Best/median/worst are
    computed from whole-seed workload totals, not by summing per-query extrema.
    That keeps "best" achievable: it is one actual seed for that cell.
    """
    if df.empty:
        return df
    df = _ensure_combo_id(df)
    if "mcts_best_cost" not in df.columns:
        df["mcts_best_cost"] = pd.NA
    if "plan_cost" in df.columns:
        df["mcts_best_cost"] = df["mcts_best_cost"].fillna(df["plan_cost"])
    if "seed" not in df.columns:
        df["seed"] = 0

    keys = ["config", "reward", "agg", "combo_id"]
    grouped = df.groupby(keys, dropna=False)
    rows = []
    for vals, sub in grouped:
        row = dict(zip(keys, vals, strict=False))
        query_keys = ["source", "query"] if "source" in sub.columns else ["query"]
        row["n_queries"] = sub[query_keys].drop_duplicates().shape[0]
        row["n_seeds"] = sub["seed"].nunique(dropna=True)
        for m in METRICS:
            if m not in sub.columns:
                continue
            seed_totals = sub.groupby("seed")[m].sum(min_count=1).dropna()
            if seed_totals.empty:
                continue
            row[f"{m}__best"] = float(seed_totals.min())
            row[f"{m}__mean"] = float(seed_totals.mean())
            row[f"{m}__median"] = float(seed_totals.median())
            row[f"{m}__worst"] = float(seed_totals.max())
        rows.append(row)
    return pd.DataFrame(rows)


def relative_workload(df: pd.DataFrame, metric: str, baseline: str, formula: str) -> pd.DataFrame:
    """Per-cell metric relative to a ``baseline`` config, as a ratio.

    Per query, ``cell_value`` is the mean over seeds; ``relative`` is
    ``cell_value / baseline_value`` for the same query.  Across queries, per cell,
    ``relative__best/median/worst`` summarize the per-query ratio distribution,
    while ``relative__mean`` is either the plain mean of those ratios (default) or
    the pooled ratio ``Σ cell_value / Σ baseline_value`` (``formula ==
    "ratio_of_sums"``).  The pooled ratio is a baseline-weighted mean of the
    per-query ratios, so it always lies within best..worst.

    Queries the baseline lacks are dropped by the inner join, so a cell's ratio may
    cover fewer queries than the absolute workload — ``n_queries`` reports how many,
    and callers should surface mismatches.
    """
    if df.empty:
        return pd.DataFrame()
    query_keys = ["source", "query"] if "source" in df.columns else ["query"]
    cell_keys = ["config", "reward", "agg", "combo_id"]

    sub = df.copy()
    value_col = "__relative_metric_value"
    if metric == "mcts_best_cost":
        sub[value_col] = sub["mcts_best_cost"] if "mcts_best_cost" in sub.columns else pd.NA
        if "plan_cost" in sub.columns:
            sub[value_col] = sub[value_col].fillna(sub["plan_cost"])
    elif metric in sub.columns:
        sub[value_col] = sub[metric]
    else:
        return pd.DataFrame()
    sub = sub[sub[value_col].notna()]
    if sub.empty:
        return pd.DataFrame()

    per_query = (
        sub.groupby(query_keys + cell_keys, dropna=False, as_index=False)[value_col]
        .mean()
        .rename(columns={value_col: "cell_value"})
    )
    base = per_query[per_query["config"] == baseline][[*query_keys, "cell_value"]].rename(
        columns={"cell_value": "baseline_value"}
    )
    if base.empty:
        return pd.DataFrame()

    merged = per_query.merge(base, on=query_keys, how="inner")
    merged = merged[merged["baseline_value"].notna() & (merged["baseline_value"] != 0)]
    if merged.empty:
        return pd.DataFrame()
    merged["relative"] = merged["cell_value"] / merged["baseline_value"]
    inf = float("inf")
    merged = merged.replace([inf, -inf], float("nan")).dropna(subset=["relative"])
    if merged.empty:
        return pd.DataFrame()

    rows = []
    for vals, group in merged.groupby(cell_keys, dropna=False):
        ratios = group["relative"].dropna()
        if ratios.empty:
            continue
        row = dict(zip(cell_keys, vals, strict=False))
        row["n_queries"] = int(ratios.count())
        row["relative__best"] = float(ratios.min())
        if formula == "ratio_of_sums":
            row["relative__mean"] = float(group["cell_value"].sum() / group["baseline_value"].sum())
        else:
            row["relative__mean"] = float(ratios.mean())
        row["relative__median"] = float(ratios.median())
        row["relative__worst"] = float(ratios.max())
        rows.append(row)
    return pd.DataFrame(rows)


def query_coverage(df: pd.DataFrame) -> tuple[dict, set, set]:
    """Per-cell usable-query coverage, for cross-cell comparability checks.

    Returns ``(coverage, union, common)``.  ``coverage`` maps each cell key
    ``(config, reward, agg, combo_id)`` to the set of queries it has USABLE data
    for (at least one row that isn't ``timed_out``); ``union`` is every query any
    cell covers; ``common`` is the intersection present for every cell.  When
    ``union != common`` the cells' workload totals / ratios are summed over
    different query sets and are not directly comparable.
    """
    if df.empty:
        return {}, set(), set()
    work = _ensure_combo_id(df)
    if "timed_out" in work.columns:
        work = work[~work["timed_out"].fillna(False).astype(bool)]
    q_cols = ["source", "query"] if "source" in work.columns else ["query"]
    coverage: dict = {}
    for vals, group in work.groupby(["config", "reward", "agg", "combo_id"], dropna=False):
        coverage[tuple(vals)] = {tuple(row) for row in group[q_cols].drop_duplicates().to_numpy()}
    if not coverage:
        return {}, set(), set()
    union = set().union(*coverage.values())
    common = set.intersection(*coverage.values())
    return coverage, union, common


def restrict_to_common_queries(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to queries every cell has usable data for (apples-to-apples totals)."""
    _coverage, _union, common = query_coverage(df)
    if df.empty or not common:
        return df
    q_cols = ["source", "query"] if "source" in df.columns else ["query"]
    keep = df[q_cols].apply(lambda row: tuple(row) in common, axis=1)
    return df[keep]


def coef_var(values: Iterable[float]) -> float:
    arr = [float(v) for v in values if v is not None]
    if len(arr) < 2:
        return float("nan")
    mu = statistics.mean(arr)
    if mu == 0:
        return float("nan")
    return statistics.pstdev(arr) / abs(mu)


_REWARD_SHORT = {
    "norm_neg_log": "nnl",
    "phase_ratio": "pr",
    "normalized": "norm",
    "neg_cost": "nc",
    "neg_log": "nl",
}


_BASELINE_CONFIGS = {"dp", "geqo"}


def short_label(config: str, reward: str, agg: str, combo_id: str = "") -> str:
    if config in _BASELINE_CONFIGS:
        return config
    cfg = "luby" if config == "luby" else "sing"
    rw = _REWARD_SHORT.get(reward, reward[:6])
    ag = "best" if agg == "best" else "avg"
    base = f"{cfg}/{rw}/{ag}"
    return f"{base}/{combo_id}" if combo_id else base


STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_STOPPED = "stopped"


def read_status(run_dir: Path) -> str | None:
    p = run_dir / "status"
    if not p.exists():
        return None
    return p.read_text().strip() or None


def write_status(run_dir: Path, status: str) -> None:
    (run_dir / "status").write_text(status + "\n")
