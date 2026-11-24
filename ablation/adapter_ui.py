"""Shared-core experiment controls for the ablation UI."""

import json
import subprocess
import sys
from pathlib import Path

import streamlit as st


def render(config_path, config, implementation, launch):
    root = Path(__file__).resolve().parent.parent
    st.caption(
        "Each query runs in a fresh connection. Settings below override the selected TOML config."
    )
    name = st.text_input("Name", value=config.get("name", "shared-core"))
    queries = st.text_area("Queries", value=" ".join(config.get("queries", [])))
    seeds = st.number_input("Seeds (1 through N)", min_value=1, value=int(config.get("n_seeds", 3)))
    modes = (
        ["single", "luby", "duckdb_default"]
        if implementation == "duckdb_mctsopt"
        else ["single", "luby", "pg_default", "dp", "geqo"]
    )
    selected = st.multiselect(
        "Configs",
        modes,
        default=[c for c in config.get("configs", []) if c in modes] or ["single", modes[2]],
    )
    rewards = st.multiselect(
        "Reward",
        ["neg_log", "neg_cost", "norm_neg_log"],
        default=config.get("reward_modes") or ["neg_log"],
    )
    aggs = st.multiselect(
        "Aggregation", ["best", "average"], default=config.get("uct_aggregations") or ["best"]
    )
    measurement = st.selectbox(
        "Measurement",
        ["full_protocol", "cost_only"],
        index=int(config.get("measurement") == "cost_only"),
    )
    st.caption(
        "GUC arrays define sweep axes. Use shared, single, and luby groups; budget is start_budget (or DuckDB iterations)."
    )
    gucs_text = st.text_area(
        "GUC overrides (JSON)", value=json.dumps(config.get("gucs", {}), indent=2), height=240
    )
    runtime = dict(config.get("runtime", {}))
    runtime.setdefault("cache", "cold")
    runtime.setdefault("timeout_seconds", 30)
    if implementation == "duckdb_mctsopt":
        runtime.setdefault("binary", "duckdb")
        runtime.setdefault("database", "/path/to/imdb.duckdb")
        runtime.setdefault("extension", "/path/to/duckdb_mctsopt.duckdb_extension")
    else:
        runtime.setdefault("extension", "pg_mctsopt")
    runtime_text = st.text_area("Runtime (JSON)", value=json.dumps(runtime, indent=2), height=200)
    st.caption(
        "Cold DuckDB: evict and verify database pages. Cold PostgreSQL: restart the configured cluster and drop OS caches. Warm uses fresh connections without eviction."
    )
    try:
        gucs = json.loads(gucs_text)
        runtime = json.loads(runtime_text)
        if not isinstance(gucs, dict) or not isinstance(runtime, dict):
            raise ValueError("Expected JSON objects")
    except (ValueError, TypeError) as exc:
        st.error(str(exc))
        return
    cmd = [
        sys.executable,
        str(root / "ablation/sweep.py"),
        "-c",
        str(config_path),
        "--implementation",
        implementation,
        "--name",
        name,
        "--queries",
        queries,
        "--seeds",
        str(seeds),
        "--configs",
        " ".join(selected),
        "--rewards",
        " ".join(rewards),
        "--aggs",
        " ".join(aggs),
        "--measurement",
        measurement,
        "--gucs",
        json.dumps(gucs),
        "--runtime",
        json.dumps(runtime),
        "--mcts-cells",
        "[]",
    ]
    preview, start = st.columns(2)
    valid = bool(
        name.strip()
        and queries.strip()
        and selected
        and (not set(selected) & {"single", "luby"} or (rewards and aggs))
    )
    if preview.button("Preview", disabled=not valid):
        result = subprocess.run([*cmd, "--dry-run"], capture_output=True, text=True, check=False)
        st.code(result.stdout or result.stderr)
    if start.button("Launch", disabled=not valid, type="primary"):
        ok, message = launch(cmd)
        if ok:
            st.success(message)
            st.rerun()
        else:
            st.error(message)
