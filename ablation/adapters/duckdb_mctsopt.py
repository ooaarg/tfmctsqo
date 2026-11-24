"""DuckDB CLI measurements with query-owned profiles and diagnostics."""

import json
import math
import re
from pathlib import Path

from adapters.common import documents, execute, literal, validate_settings

ENGINE = "duckdb"
BASELINES = {"duckdb_default"}
SETTINGS = {
    "retention_enabled",
    "retention_max_group_fraction",
    "iterations",
    "phases",
    "luby_enabled",
    "patience",
    "top_k",
    "depth",
    "plan_shape",
    "force_left_tree",
    "rollouts_per_leaf",
    "rollout",
    "expand_strategy",
    "reward_map",
    "uct_aggregation",
    "exploration_constant",
    "gamma",
    "full_budget",
    "min_relations",
    "greedy_init",
    "greedy_init_mode",
    "scope",
    "legacy_selection",
    "legacy_normalization",
    "rank_all",
}


def settings_for(mode, values, reward, agg, seed):
    values = dict(values)
    # The shared sweep UI expresses phase budgets as start_budget.
    if "start_budget" in values:
        if "iterations" in values:
            raise ValueError("Use iterations or start_budget, not both")
        values["iterations"] = values.pop("start_budget")
    validate_settings(values, SETTINGS)
    result = {
        "enabled": True,
        "scope": "all",
        "min_relations": 12,
        "greedy_init": False,
        "iterations": 250,
        "phases": 1,
        "luby_enabled": mode == "luby",
        "patience": 0,
        "top_k": 10,
        "depth": 10,
        "plan_shape": 1,
        "rollout": "random",
        "rollouts_per_leaf": 1,
        "exploration_constant": 1.4,
        "gamma": 0.5,
        "full_budget": False,
        **values,
        "reward_map": reward,
        "uct_aggregation": agg,
        "seed": seed,
    }
    if mode == "single" and (result["phases"] != 1 or result["luby_enabled"]):
        raise ValueError("single requires phases=1 and luby_enabled=false")
    return result


def adapt_sql(sql):
    # JOB uses AT as a table alias; DuckDB reserves it. Preserve literals/comments.
    tokens = r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|--[^\r\n]*|/\*[\s\S]*?\*/|\bAT\b"
    return re.sub(tokens, lambda m: '"at"' if m[0].lower() == "at" else m[0], sql, flags=re.I)


def profile_metrics(profile, sql):
    if profile["query"]["sql"].strip() != sql.strip():
        raise ValueError("Profile belongs to another statement")
    total = 1000 * profile["query"]["total_time"]
    planning = 1000 * sum(
        profile[key]["total_time"] for key in ("parser", "planner", "optimizer", "physical_planner")
    )
    if not all(math.isfinite(v) and v >= 0 for v in (total, planning)) or planning > total:
        raise ValueError("Invalid DuckDB profile timing")
    return {"e2e_time_ms": total, "plan_time_ms": planning, "exec_time_ms": total - planning}


def measure(cfg, mode, settings, sql, output):
    runtime = cfg.runtime
    binary = runtime.get("binary", "duckdb")
    database = runtime["database"]
    timeout = runtime.get("timeout_seconds", 30)
    baseline = mode in BASELINES
    query = adapt_sql(sql).strip().rstrip(";") + ";"
    profile_path = output.with_suffix(".profile.json").resolve()
    setup = (
        f"SET threads={int(runtime.get('threads', 1))};\n"
        f"SET memory_limit={literal(runtime.get('memory_limit', '8192MB'))};\n"
        "SET enable_progress_bar=false;\n"
        f"SET max_execution_time={int(timeout * 1000)};\n"
    )
    if not baseline:
        setup += f"LOAD {literal(runtime['extension'])};\n"
        setup += "".join(f"SET duckdb_mctsopt_{k}={literal(v)};\n" for k, v in settings.items())
    setup += "SET enable_logging=true; SET logging_level='warning'; SET logging_storage='memory';\n"
    if cfg.measurement == "cost_only":
        statement = "EXPLAIN (FORMAT JSON) " + query
    else:
        setup += (
            f"SET enable_profiling='json'; SET profiling_output={literal(str(profile_path))};\n"
        )
        statement = query
    script = setup + statement + "\nPRAGMA disable_profiling;\n"
    if not baseline:
        script += (
            "SET duckdb_mctsopt_enabled=false;\n"
            "SELECT duckdb_mctsopt_diagnostics() AS regions, "
            "duckdb_mctsopt_last_cost() AS cost, "
            "duckdb_mctsopt_fallback_count() AS fallbacks;\n"
        )
    command = [binary, "-init", "/dev/null", "-readonly", database, "-unsigned", "-bail", "-json"]
    text = execute(command, script, timeout + 15, output)
    docs = documents(text)
    if len(docs) != (2 if baseline else 3) or docs[1] != []:
        raise ValueError("Unexpected DuckDB statement results")
    metrics = (
        {}
        if cfg.measurement == "cost_only"
        else profile_metrics(json.loads(profile_path.read_text()), query)
    )
    metrics["answer"] = docs[0] if cfg.measurement != "cost_only" else None
    if cfg.measurement == "cost_only":
        output.with_suffix(".plan").write_text(json.dumps(docs[0], indent=2))
    else:
        output.with_suffix(".plan").write_text(profile_path.read_text())
    if not baseline:
        stats = docs[2][0]
        regions = json.loads(stats["regions"])
        output.with_suffix(".diagnostics.json").write_text(json.dumps(regions, indent=2))
        selected = [r for r in regions if r.get("selected")]
        metrics.update(
            mcts_best_cost=stats["cost"],
            fallback_count=int(stats["fallbacks"]),
            regions=regions,
            iters=sum(r.get("iterations", 0) for r in selected),
        )
        metrics["search_time_ms"] = sum(r.get("search_ms", 0) for r in selected)
    return metrics


def validate_runtime(cfg):
    if set(cfg.configs) - BASELINES and not Path(cfg.runtime.get("extension", "")).is_file():
        raise ValueError("runtime.extension must name the DuckDB module")
    if not Path(cfg.runtime.get("database", "")).is_file():
        raise ValueError("runtime.database must name an existing DuckDB database")
    if Path(cfg.runtime["database"] + ".wal").exists():
        raise ValueError("Checkpoint and close the DuckDB database before a sweep")
