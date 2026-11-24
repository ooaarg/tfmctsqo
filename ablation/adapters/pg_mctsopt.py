"""PostgreSQL shared-core adapter; each measurement uses a fresh backend."""

import json
import os
from pathlib import Path

import lib
from adapters.common import execute, literal, validate_settings

ENGINE = "postgres"
BASELINES = {"pg_default", "dp", "geqo"}
SETTINGS = {
    "start_budget",
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
    "cache_enabled",
    "cache_size",
    "cache_max_size",
    "log_debug",
    "log_steps",
}


def settings_for(mode, values, reward, agg, seed):
    validate_settings(values, SETTINGS)
    result = {
        "enabled": True,
        "min_relations": 12,
        "start_budget": 250,
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
        "random_seed": seed,
    }
    if mode == "single" and (result["phases"] != 1 or result["luby_enabled"]):
        raise ValueError("single requires phases=1 and luby_enabled=false")
    return result


def measure(cfg, mode, settings, sql, output):
    runtime = cfg.runtime
    timeout = runtime.get("timeout_seconds", 30)
    setup = (
        "DO $$ BEGIN IF current_setting('shared_preload_libraries') || ',' || "
        "current_setting('session_preload_libraries') ~ '(mcts_extreme|pg_mctsopt)' "
        "THEN RAISE EXCEPTION 'Use a server without preloaded MCTS extensions'; END IF; END $$;\n"
        f"SET statement_timeout={int(timeout * 1000)};\n"
        "SET max_parallel_workers_per_gather=0;\n"
        f"SET work_mem={literal(runtime.get('work_mem', '64MB'))};\n"
        f"SET from_collapse_limit={int(runtime.get('collapse_limit', 64))};\n"
        f"SET join_collapse_limit={int(runtime.get('collapse_limit', 64))};\n"
        "SET geqo=on; SET geqo_threshold=12; SET geqo_seed=0;\n"
    )
    if mode not in BASELINES:
        setup += f"LOAD {literal(runtime.get('extension', 'pg_mctsopt'))};\n"
        setup += "".join(f"SET pg_mctsopt.{k}={literal(v)};\n" for k, v in settings.items())
    elif mode == "dp":
        setup += "SET geqo=off;\n"
    elif mode == "geqo":
        setup += (
            f"SET geqo_threshold=2; SET geqo_seed={(settings['seed'] * 0.6180339887498949) % 1};\n"
        )
    query = sql.strip().rstrip(";")
    explain = (
        "EXPLAIN (COSTS, SUMMARY) "
        if cfg.measurement == "cost_only"
        else "EXPLAIN (ANALYZE, BUFFERS, COSTS, SUMMARY) "
    )
    script = setup + explain + query + ";\n"
    if cfg.measurement != "cost_only":
        # EXPLAIN ANALYZE discards rows. Check answers in a separate, untimed execution.
        script += "\\echo __MCTSOPT_ANSWER__\n"
        script += "SELECT coalesce(json_agg(t), '[]'::json) FROM (" + query + ") t;\n"
    binary = runtime.get("binary", str(Path(cfg.pgbin) / "psql") if cfg.pgbin else "psql")
    command = [
        binary,
        "-X",
        "-qAt",
        "-v",
        "ON_ERROR_STOP=1",
        "-h",
        os.environ.get("PGHOST", "/tmp"),
        "-d",
        cfg.db,
        "-U",
        cfg.pguser,
    ]
    if cfg.docker_container:
        command = [
            "docker",
            "exec",
            "-i",
            cfg.docker_container,
            "psql",
            "-X",
            "-qAt",
            "-v",
            "ON_ERROR_STOP=1",
            "-U",
            cfg.pguser,
            "-d",
            cfg.db,
        ]
    text = execute(command, script, 2 * timeout + 15, output)
    plan, _, answer = text.partition("__MCTSOPT_ANSWER__\n")
    output.with_suffix(".plan").write_text(plan)
    metrics = lib.parse_explain(plan)
    if cfg.measurement != "cost_only":
        if "plan_time_ms" not in metrics or "exec_time_ms" not in metrics:
            raise ValueError("Missing PostgreSQL EXPLAIN timing")
        metrics["e2e_time_ms"] = metrics["plan_time_ms"] + metrics["exec_time_ms"]
        metrics["answer"] = json.loads(answer.strip())
    else:
        metrics["answer"] = None
    metrics["fallback_count"] = int("no plan from MCTS (fallback used)" in plan)
    return metrics


def validate_runtime(cfg):
    if cfg.runtime.get("cache", "cold") == "cold" and not (cfg.pgdata or cfg.docker_container):
        raise ValueError("Cold PostgreSQL runs require PGDATA or docker_container for restart")
