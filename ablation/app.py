"""Streamlit UI for the mcts_extreme ablation runner.  `uv run streamlit run app.py`."""

from __future__ import annotations

import datetime as _dt
import json
import os
import signal
import subprocess
import sys
import time
import tomllib
from collections import defaultdict
from pathlib import Path


def dt_now_aware():
    return _dt.datetime.now(tz=_dt.UTC)


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as psub
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"
CONFIG_DIR = ROOT / "ablation" / "configs"
DEFAULT_PGBIN = os.environ.get("PGBIN", "")
DEFAULT_PGDATA = os.environ.get("PGDATA", "")  # set PGDATA env or enter it in the UI
DEFAULT_PGLOG = os.environ.get("PGLOG", "/tmp/mcts_extreme_pg.log")
DEFAULT_DOCKER_CONTAINER = os.environ.get("DOCKER_CONTAINER", "")
DROP_CACHES_HELPER = ROOT / "ablation" / "drop_caches.sh"

SOURCES = {
    "job": str(ROOT / "source" / "resource" / "imdb_original_queries"),
    "jobcomplex": str(ROOT / "source" / "resource" / "job_complex"),
    "imdb-ceb": str(ROOT / "source" / "resource" / "imdb-ceb" / "queries" / "rel9_seed42_200_flat"),
    "tpch": str(ROOT / "source" / "resource" / "tpch"),
}

QUERY_PRESET_PREDICATES = {
    "job": {
        "17 rels": lambda r: r == 17,
        "≥14 rels": lambda r: r >= 14,
        "≥12 rels": lambda r: r >= 12,
        "<12 rels": lambda r: r < 12,
    },
    "jobcomplex": {
        "≥12 rels": lambda r: r >= 12,
        "<12 rels": lambda r: r < 12,
    },
    "tpch": {
        "joins ≥6": lambda r: r >= 6,
        "joins ≥3": lambda r: r >= 3,
        "single-table": lambda r: r <= 1,
    },
}

REWARD_OPTIONS = ["neg_log", "neg_cost", "norm_neg_log"]
AGG_OPTIONS = ["best", "average"]
CONFIG_OPTIONS = ["luby", "single", "dp", "geqo"]
EXPAND_OPTIONS = ["cost", "row", "mixed_025", "mixed_050", "selectivity"]
BASELINE_CONFIGS = {"dp", "geqo"}

_LUBY_WEIGHTS = (1, 1, 2, 1, 1, 2, 4, 1, 1, 2, 1, 1, 2, 4, 8, 1)


def luby_cap(start_budget: int, phases: int) -> int:
    phases = max(1, min(phases, len(_LUBY_WEIGHTS)))
    return int(start_budget) * sum(_LUBY_WEIGHTS[:phases])


METRIC_LABELS = {
    "mcts_best_cost": "Cost (lower = better)",
    "exec_time_ms": "Exec Time ms (lower = better)",
    "plan_time_ms": "Plan Time ms (lower = better)",
    "e2e_time_ms": "End-to-End ms = plan + exec (lower = better)",
    "iters": "MCTS Iterations",
}

PLOTLY_CONFIG = {
    "displaylogo": False,
    "displayModeBar": True,
    "scrollZoom": True,
    "doubleClick": "autosize",
    "modeBarButtonsToAdd": ["autoScale2d", "resetScale2d"],
}

st.set_page_config(page_title="MCTS Ablation", layout="wide")


def list_runs() -> list[Path]:
    if not RUNS_DIR.exists():
        return []
    return sorted([p for p in RUNS_DIR.iterdir() if p.is_dir()], reverse=True)


RUN_GROUP_ORDER = {
    "Running": 0,
    "PG / JOB-Complex": 10,
    "PG / JOB high-alias reward": 20,
    "PG / JOB high-alias top-k": 30,
    "PG / cache and baselines": 40,
    "Imported / other": 80,
    "Smoke / test": 90,
    "Other": 99,
}


def config_run_group(run_dir: Path) -> str:
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        return ""
    try:
        cfg = json.loads(cfg_path.read_text())
    except Exception:
        return ""
    return str(cfg.get("run_group") or cfg.get("group") or "").strip()


def run_group(run_dir: Path) -> str:
    name = run_dir.name.lower()
    status = lib.read_status(run_dir) or ""
    if status == "running":
        return "Running"
    explicit_group = config_run_group(run_dir)
    if explicit_group:
        return explicit_group
    if "smoke" in name or "test" in name:
        return "Smoke / test"
    if "jobcomplex" in name:
        return "PG / JOB-Complex"
    if "reward" in name:
        return "PG / JOB high-alias reward"
    if "topk" in name or "top-k" in name:
        return "PG / JOB high-alias top-k"
    if "algorithm" in name or "cache" in name:
        return "PG / cache and baselines"
    if name.startswith("imported"):
        return "Imported / other"
    return "Other"


def group_runs(runs: list[Path]) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for run in runs:
        grouped[run_group(run)].append(run)
    return dict(
        sorted(
            grouped.items(),
            key=lambda item: (RUN_GROUP_ORDER.get(item[0], 50), item[0]),
        )
    )


def list_configs() -> list[Path]:
    if not CONFIG_DIR.exists():
        return []
    return sorted(CONFIG_DIR.glob("*.toml"))


def sanitize_run_name(name: str) -> str:
    import re

    cleaned = name.strip().replace(" ", "-")
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", cleaned)
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip(".-")
    return cleaned


def _source_query_dir(source: str) -> Path:
    if source in SOURCES:
        return Path(SOURCES[source])

    p = Path(source)
    try:
        resolved = p.resolve()
    except OSError:
        resolved = p

    for known in SOURCES.values():
        kp = Path(known)
        try:
            if kp.resolve() == resolved:
                return kp
        except OSError:
            if kp == p:
                return kp
    return p


def list_source_queries(source: str) -> list[str]:
    p = _source_query_dir(source)
    if not p.is_dir():
        return []
    return sorted([f.stem for f in p.glob("*.sql")])


@st.cache_data(show_spinner=False)
def query_rel_count(source: str) -> dict[str, int]:
    """Rel count ≈ comma-separated refs between FROM and WHERE."""
    p = _source_query_dir(source)
    if not p.is_dir():
        return {}
    import re as _re

    out: dict[str, int] = {}
    for f in p.glob("*.sql"):
        text = f.read_text()
        m = _re.search(r"\bFROM\b(.*?)\bWHERE\b", text, _re.I | _re.S)
        out[f.stem] = (m.group(1).count(",") + 1) if m else 0
    return out


def _row_query_rel_count(row) -> int | None:
    src = row.get("source", "")
    q = str(row.get("query", ""))
    if not src:
        return None
    return query_rel_count(src).get(q)


def status_badge(run_dir: Path) -> str:
    status = lib.read_status(run_dir) or "?"
    color = {
        "running": "🟡",
        "done": "🟢",
        "failed": "🔴",
        "stopped": "⏹️",
    }.get(status, "⚪")
    return f"{color} {status}"


def _json_encode_unhashable_cols(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == object and df[col].map(lambda v: isinstance(v, (dict, list))).any():
            df[col] = df[col].map(
                lambda v: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
            )
    return df


@st.cache_data(show_spinner=False)
def _load_jsonl_cached(path_str: str, size: int, mtime: float) -> pd.DataFrame:
    return _json_encode_unhashable_cols(lib.load_jsonl(Path(path_str)))


def load_results(run_dir: Path) -> pd.DataFrame:
    p = run_dir / "results.jsonl"
    try:
        stt = p.stat()
    except OSError:
        return lib.load_jsonl(p)
    return _load_jsonl_cached(str(p), stt.st_size, stt.st_mtime)


@st.cache_data(show_spinner=False)
def _agg_workload(df: pd.DataFrame) -> pd.DataFrame:
    return lib.aggregate_workload(df)


_TIMEOUT_MARKER_RE = lib._TIMEOUT_RE


@st.cache_data(show_spinner=False)
def _scan_plan_for_timeout(plan_abs_path: str, mtime: float) -> bool:
    """Cached on (path, mtime) so unchanged files don't get re-read."""
    try:
        text = Path(plan_abs_path).read_text(errors="replace")
    except OSError:
        return False
    return bool(_TIMEOUT_MARKER_RE.search(text))


def backfill_timed_out(run_dir: Path, df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        df = df.copy()
        df["timed_out"] = pd.Series(dtype=bool)
        return df
    df = df.copy()
    if "timed_out" not in df.columns:
        df["timed_out"] = pd.NA
    missing = df["timed_out"].isna()
    if missing.any() and "plan_path" in df.columns:
        vals = []
        for _, row in df.loc[missing].iterrows():
            pp = row.get("plan_path")
            if not pp:
                vals.append(False)
                continue
            ap = run_dir / pp
            try:
                mt = ap.stat().st_mtime
            except OSError:
                vals.append(False)
                continue
            vals.append(_scan_plan_for_timeout(str(ap), mt))
        df.loc[missing, "timed_out"] = vals
    df["timed_out"] = df["timed_out"].fillna(False).astype(bool)
    return df


def expected_run_count(run_dir: Path) -> int | None:
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        return None
    try:
        cfg = json.loads(cfg_path.read_text())
    except Exception:
        return None
    try:
        if cfg.get("imported_from") and _has_collapsed_import_rows(run_dir):
            return _jsonl_line_count(run_dir / "results.jsonl")
        n_reward = len(cfg["reward_modes"])
        n_agg = len(cfg["uct_aggregations"])
        n_combos = int(cfg.get("n_combos", 1) or 1)
        selected_mcts = cfg.get("mcts_cells", []) or []
        cells_per_config = sum(
            1
            if c in BASELINE_CONFIGS
            else sum(1 for cell in selected_mcts if cell.get("config", "single") == c)
            if selected_mcts
            else n_reward * n_agg * n_combos
            for c in cfg["configs"]
        )
        return len(cfg["queries"]) * cells_per_config * int(cfg["n_seeds"])
    except (KeyError, TypeError):
        return None


def _jsonl_line_count(path: Path) -> int | None:
    try:
        with path.open() as f:
            return sum(1 for _ in f)
    except OSError:
        return None


def _has_collapsed_import_rows(run_dir: Path) -> bool:
    results_path = run_dir / "results.jsonl"
    try:
        with results_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                return "n_seeds_collapsed" in row or "imported_status" in row
    except (OSError, json.JSONDecodeError):
        return False
    return False


def read_pid(run_dir: Path) -> int | None:
    p = run_dir / "pid"
    if not p.exists():
        return None
    try:
        return int(p.read_text().strip())
    except (ValueError, OSError):
        return None


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False
    except OSError:
        return False


def stop_run(run_dir: Path) -> tuple[bool, str]:
    pid = read_pid(run_dir)
    if pid is None:
        return False, "no pid file (already finished?)"
    if not pid_alive(pid):
        return False, f"pid {pid} not alive"
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError as e:
        return False, f"kill failed: {e}"
    return True, f"sent SIGTERM to pid {pid}"


def _pg_tool(tool: str) -> str:
    return str(Path(DEFAULT_PGBIN) / tool) if DEFAULT_PGBIN else tool


def clear_os_cache_now() -> tuple[bool, str]:
    cmd = [str(DROP_CACHES_HELPER)] if os.geteuid() == 0 else ["sudo", "-n", str(DROP_CACHES_HELPER)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        msg = (result.stderr or result.stdout or "").strip()
        return False, msg or f"drop_caches failed rc={result.returncode}"
    return True, "OS caches cleared."


def restart_pg_now() -> tuple[bool, str]:
    if DEFAULT_DOCKER_CONTAINER:
        cmd = ["docker", "restart", DEFAULT_DOCKER_CONTAINER]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            msg = (result.stderr or result.stdout or "").strip()
            return False, msg or f"docker restart failed rc={result.returncode}"
        return True, f"Docker container restarted: {DEFAULT_DOCKER_CONTAINER}"
    if not DEFAULT_PGDATA:
        return False, "PGDATA is empty."
    if not Path(DEFAULT_PGDATA).is_dir():
        return False, f"PGDATA does not exist: {DEFAULT_PGDATA}"
    cmd = [
        _pg_tool("pg_ctl"),
        "-D",
        DEFAULT_PGDATA,
        "-l",
        DEFAULT_PGLOG,
        "restart",
        "-m",
        "fast",
        "-w",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        msg = (result.stderr or result.stdout or "").strip()
        return False, msg or f"pg_ctl restart failed rc={result.returncode}"
    return True, "PostgreSQL restarted."


def launch_sweep(cmd: list[str]) -> tuple[bool, str]:
    RUNS_DIR.mkdir(exist_ok=True)
    log_path = RUNS_DIR / f"_launch_{int(time.time())}.log"
    log_file = Path(log_path).open("w")
    try:
        popen = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except Exception as e:
        return False, f"failed to launch: {e}"
    time.sleep(1.2)
    rc = popen.poll()
    if rc is not None and rc != 0:
        tail = log_path.read_text()[-2000:] if log_path.exists() else ""
        return False, f"sweep.py exited rc={rc} immediately. Tail of {log_path.name}:\n{tail}"
    return True, f"Launched (PID {popen.pid})."


# --- sidebar ---


@st.dialog("New experiment", width="large")
def _new_experiment_dialog():
    """Full-width modal for configuring + launching a sweep."""
    config_files = list_configs()
    if not config_files:
        st.warning("No `ablation/configs/*.toml` — create one first.")
        return

    base_idx = st.selectbox(
        "Base config",
        range(len(config_files)),
        format_func=lambda i: config_files[i].name,
        key="dlg_base_cfg",
    )
    base_cfg_path = config_files[base_idx]
    try:
        base_cfg = tomllib.loads(base_cfg_path.read_text())
    except Exception as e:
        st.error(f"failed to parse {base_cfg_path.name}: {e}")
        return

    if st.session_state.get("dlg_loaded_cfg") != str(base_cfg_path):
        for k in list(st.session_state):
            if k.startswith("dlg_") and k not in ("dlg_base_cfg", "dlg_loaded_cfg"):
                del st.session_state[k]

        shared = base_cfg.get("gucs", {}).get("shared", {}) or {}
        luby_d = base_cfg.get("gucs", {}).get("luby", {}) or {}
        single_d = base_cfg.get("gucs", {}).get("single", {}) or {}
        toml_sources = base_cfg.get("sources") or base_cfg.get("source") or ["job"]
        if isinstance(toml_sources, str):
            toml_sources = [toml_sources]

        ss = st.session_state
        ss["dlg_name"] = base_cfg.get("name", "exp")
        ss["dlg_run_group"] = base_cfg.get("run_group") or base_cfg.get("group") or ""
        ss["dlg_seeds"] = int(base_cfg.get("n_seeds", 20))
        ss["dlg_sources"] = [s for s in toml_sources if s in SOURCES] or ["job"]
        ss["dlg_rewards"] = base_cfg.get("reward_modes", ["neg_log", "norm_neg_log"])
        ss["dlg_aggs"] = base_cfg.get("uct_aggregations", ["best", "average"])
        ss["dlg_configs"] = base_cfg.get("configs", ["luby", "single"])
        plan_shape_default = int(shared.get("plan_shape", shared.get("kernels", 1)))
        ss["dlg_shape_mode"] = (
            "bushy" if plan_shape_default == 0
            else "linear" if plan_shape_default == 1
            else "K-component"
        )
        ss["dlg_kernels"] = max(2, plan_shape_default)
        ss["dlg_depth"] = int(shared.get("depth", 4))
        ss["dlg_top_k"] = int(shared.get("top_k", 5))
        ss["dlg_pat"] = int(shared.get("patience", 0))
        ss["dlg_lubysb"] = int(luby_d.get("start_budget", 20))
        ss["dlg_lubyph"] = int(luby_d.get("phases", 8))
        ss["dlg_rpl"] = int(shared.get("rollouts_per_leaf", 1))
        ss["dlg_expC"] = float(shared.get("exploration_constant", 1.4))
        ss["dlg_single_auto"] = "start_budget" not in single_d
        ss["dlg_parallel"] = 1
        ss["dlg_seeds"] = 3
        ss["dlg_restart_pg"] = True
        ss["dlg_prewarm"] = False
        ss["dlg_drop_os_cache"] = True
        ss["dlg_clean_per_run"] = True

        ss["dlg_loaded_cfg"] = str(base_cfg_path)

    c_name, c_group, c_seeds = st.columns([2, 2, 1])
    with c_name:
        name = st.text_input("Name", value=base_cfg.get("name", "exp"), key="dlg_name")
    with c_group:
        run_group_name = st.text_input(
            "Run group",
            value=base_cfg.get("run_group") or base_cfg.get("group") or "",
            key="dlg_run_group",
            placeholder="PG / JOB-Complex",
        ).strip()
    with c_seeds:
        n_seeds = st.number_input(
            "n_seeds",
            1,
            200,
            3,
            key="dlg_seeds",
        )

    toml_sources = base_cfg.get("sources") or base_cfg.get("source") or ["job"]
    if isinstance(toml_sources, str):
        toml_sources = [toml_sources]
    source_default = [s for s in toml_sources if s in SOURCES] or ["job"]
    sources_picked = st.multiselect(
        "Sources (merged; queries resolved in order)",
        options=list(SOURCES),
        default=source_default,
        key="dlg_sources",
    )

    # --- per-source query pickers (one st.pills group per rel-count) ---
    toml_queries = list(base_cfg.get("queries", []))

    def _rel_state_key(src: str, rel: int) -> str:
        return f"dlg_q_{src}_r{rel}"

    def _set_preset(src: str, predicate, rels: dict[str, int]) -> None:
        by_rels: dict[int, list[str]] = defaultdict(list)
        for q, r in rels.items():
            by_rels[r].append(q)
        for r, qs in by_rels.items():
            st.session_state[_rel_state_key(src, r)] = sorted(qs) if predicate(r) else []

    def _gather_selection(src: str, rels: dict[str, int]) -> list[str]:
        rel_values = sorted(set(rels.values()), reverse=True)
        out: list[str] = []
        for r in rel_values:
            out.extend(st.session_state.get(_rel_state_key(src, r), []))
        return out

    st.markdown("**Queries**  *(click chips to toggle; tags pick subsets)*")
    queries_box = st.container(height=360, border=True)
    with queries_box:
        for src in sources_picked:
            available = list_source_queries(src)
            rels = query_rel_count(src)

            by_rels: dict[int, list[str]] = defaultdict(list)
            for q in available:
                by_rels[rels.get(q, 0)].append(q)
            for r in by_rels:
                by_rels[r] = sorted(by_rels[r])
                k = _rel_state_key(src, r)
                if k not in st.session_state:
                    st.session_state[k] = [q for q in toml_queries if q in by_rels[r]]

            presets = QUERY_PRESET_PREDICATES.get(src, {})
            n_btns = len(presets) + 2
            head_cols = st.columns([3] + [2] * n_btns)
            title_slot = head_cols[0].empty()
            for i, (label, pred) in enumerate(presets.items(), start=1):
                with head_cols[i]:
                    if st.button(label, key=f"dlg_preset_{src}_{i}", use_container_width=True):
                        _set_preset(src, pred, rels)
            with head_cols[-2]:
                if st.button("All", key=f"dlg_all_{src}", use_container_width=True):
                    _set_preset(src, lambda r: True, rels)
            with head_cols[-1]:
                if st.button("Clear", key=f"dlg_clear_{src}", use_container_width=True):
                    _set_preset(src, lambda r: False, rels)

            current = _gather_selection(src, rels)
            title_slot.markdown(f"### {src}  ·  *{len(current)}/{len(available)}*")

            for r in sorted(by_rels, reverse=True):
                qs = by_rels[r]
                k = _rel_state_key(src, r)
                n_sel = len(st.session_state.get(k, []))
                st.markdown(f"**{r} rels** — {n_sel}/{len(qs)}")
                st.pills(
                    label=f"{src} {r}-rels queries",
                    options=qs,
                    selection_mode="multi",
                    key=k,
                    label_visibility="collapsed",
                )
            st.divider()

    shared = base_cfg.get("gucs", {}).get("shared", {})
    luby_d = base_cfg.get("gucs", {}).get("luby", {})
    single_d = base_cfg.get("gucs", {}).get("single", {})

    def _as_bool(v, default=False) -> bool:
        if isinstance(v, list):
            v = v[0] if v else default
        if isinstance(v, bool):
            return v
        if v is None:
            return bool(default)
        return str(v).strip().lower() in {"1", "true", "t", "yes", "y", "on"}

    # --- modes / aggs / configs / search variants ---
    cells_mode = (
        st.session_state.get(
            "dlg_guc_mode", "Selected cells" if base_cfg.get("mcts_cells") else "Quick"
        )
        == "Selected cells"
    )
    cm_rewards, cm_aggs, cm_configs, cm_budget, cm_expand, cm_measure = st.columns(
        [1.2, 1, 1, 1.1, 1.1, 1.2]
    )
    with cm_rewards:
        rewards = st.multiselect(
            "Reward modes",
            options=REWARD_OPTIONS,
            default=base_cfg.get("reward_modes", ["neg_log", "norm_neg_log"]),
            key="dlg_rewards",
            disabled=cells_mode,
            help="Ignored in Selected-cells mode — each line sets its own reward.",
        )
    with cm_aggs:
        aggs = st.multiselect(
            "Aggregations",
            options=AGG_OPTIONS,
            default=base_cfg.get("uct_aggregations", ["best", "average"]),
            key="dlg_aggs",
            disabled=cells_mode,
            help="Ignored in Selected-cells mode — each line sets its own agg.",
        )
    with cm_configs:
        configs_to_run = st.multiselect(
            "Configs",
            options=CONFIG_OPTIONS,
            default=base_cfg.get("configs", ["luby", "single"]),
            key="dlg_configs",
        )
    with cm_budget:
        budget_mode_to_full = {"flex budget": False, "full budget": True}
        default_budget_mode = (
            "full budget" if _as_bool(shared.get("full_budget"), False) else "flex budget"
        )
        budget_modes = st.multiselect(
            "Budget mode",
            options=list(budget_mode_to_full),
            default=[default_budget_mode],
            key="dlg_full_budget",
            help=(
                "flex budget: iteration budget is an upper bound and MCTS can stop "
                "when no expandable leaf remains. full budget: after expansion is "
                "exhausted, remaining iterations are spent by classic UCB1."
            ),
        )
    budget_full_values = [budget_mode_to_full[m] for m in budget_modes]
    with cm_expand:
        raw_expand_default = shared.get("expand_strategy", "cost")
        expand_default = raw_expand_default if isinstance(raw_expand_default, list) else [raw_expand_default]
        expand_default = ["mixed_050" if str(v) == "mix" else v for v in expand_default]
        expand_default = [str(v) for v in expand_default if str(v) in EXPAND_OPTIONS] or ["cost"]
        expand_scenarios = st.multiselect(
            "Expand scenario",
            options=EXPAND_OPTIONS,
            default=expand_default,
            key="dlg_expand_strategy",
            help=(
                "Controls top-k expansion ranking: cost = planner total_cost, "
                "row = estimated join rows, mixed_* = log(rows + 1) plus a "
                "weighted log(cost + 1), selectivity = rows / left_rows / right_rows."
            ),
        )
    with cm_measure:
        measurement_label = st.selectbox(
            "Measurement",
            ["full protocol", "cost only"],
            index=1 if base_cfg.get("measurement") == "cost_only" else 0,
            key="dlg_measurement",
            help=(
                "full protocol runs EXPLAIN ANALYZE and measures execution. "
                "cost only runs EXPLAIN without executing the query and uses one seed."
            ),
        )
    measurement_mode = "cost_only" if measurement_label == "cost only" else "full_protocol"

    st.markdown("**MCTS GUCs**")
    default_guc_mode = "Selected cells" if base_cfg.get("mcts_cells") else "Quick"
    mode = st.segmented_control(
        label="GUC mode",
        options=["Quick", "Grid", "Selected cells"],
        default=default_guc_mode,
        key="dlg_guc_mode",
        label_visibility="collapsed",
        help=(
            "Quick: one shared setting. Grid: Cartesian-product sweep. "
            "Selected cells: exact mode tuples, no square grid."
        ),
    )
    if mode == "Selected cells":
        st.caption(
            "Selected-cells mode uses each line's own reward & agg (those two pickers are "
            "disabled above). **Configs** (MCTS configs to keep + any dp/geqo baselines), "
            "**Budget mode**, **Expand scenario** and **Measurement** still apply to every cell."
        )

    def _parse_list(s: str, cast) -> list:
        if s is None:
            return []
        out = []
        for part in s.split(","):
            t = part.strip()
            if not t:
                continue
            try:
                out.append(cast(t))
            except ValueError:
                raise ValueError(f"bad number {t!r}") from None
        return out

    def _as_list(v, default) -> list:
        if isinstance(v, list):
            return list(v)
        return [v] if v is not None else list(default)

    def _budget_guc_value():
        if len(budget_full_values) == 1:
            return budget_full_values[0]
        return budget_full_values

    def _guc_value(values: list):
        if len(values) == 1:
            return values[0]
        return values

    def _shape_mode_from_k(value: int) -> str:
        if value == 0:
            return "bushy"
        if value == 1:
            return "linear"
        return "K-component"

    def _k_from_shape_mode(mode: str, count: int) -> int:
        if mode == "bushy":
            return 0
        if mode == "linear":
            return 1
        return max(2, int(count))

    def _variant_cells(cells: list[dict]) -> list[dict]:
        if not budget_full_values or not expand_scenarios:
            return []
        out: list[dict] = []
        for cell in cells:
            base_cid = str(cell.get("combo_id") or "").strip()
            for value in budget_full_values:
                budget_suffix = "full" if value else "flex"
                for expand in expand_scenarios:
                    parts = []
                    if len(budget_full_values) > 1:
                        parts.append(budget_suffix)
                    if len(expand_scenarios) > 1:
                        parts.append(f"exp{expand}")
                    suffix = "__".join(parts)
                    cid = f"{base_cid}__{suffix}" if base_cid and suffix else base_cid or suffix
                    out.append(
                        {
                            **cell,
                            "full_budget": value,
                            "expand_strategy": expand,
                            "combo_id": cid,
                        }
                    )
        return out

    def _cell_line(cell: dict) -> str:
        return ",".join(
            str(cell.get(k, default))
            for k, default in (
                ("config", "single"),
                ("reward", "neg_log"),
                ("agg", "best"),
                ("depth", 4),
                ("top_k", 5),
                ("start_budget", 160),
                ("plan_shape", 1),
                ("combo_id", ""),
            )
        )

    def _parse_selected_cells(text: str) -> list[dict]:
        cells: list[dict] = []
        for lineno, raw in enumerate(text.splitlines(), start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) not in (6, 7, 8):
                raise ValueError(
                    f"line {lineno}: expected 6, 7, or 8 comma-separated fields, got {len(parts)}"
                )
            config, reward, agg, depth_s, topk_s, budget_s = parts[:6]
            kernels_s = parts[6] if len(parts) == 8 else ""
            combo_id = parts[7] if len(parts) == 8 else parts[6] if len(parts) == 7 else ""
            depth_i = int(depth_s)
            topk_i = int(topk_s)
            budget_i = int(budget_s)
            kernels_i = int(kernels_s) if kernels_s else 1
            cell = {
                "config": config or "single",
                "reward": reward,
                "agg": agg or "best",
                "depth": depth_i,
                "top_k": topk_i,
                "plan_shape": kernels_i,
                "start_budget": budget_i,
            }
            kernel_suffix = "" if kernels_i == 1 else f"_k{kernels_i}"
            cell["combo_id"] = combo_id or f"d{depth_i}_tk{topk_i}{kernel_suffix}_S_sb{budget_i}"
            cells.append(cell)
        if not cells:
            raise ValueError("selected-cell list is empty")
        return cells

    selected_cells: list[dict] = []
    selected_parse_err = None

    if mode == "Grid":
        st.caption(
            "Enter comma-separated values per GUC.  Each combination becomes an "
            "extra cell — beware of grid blow-up."
        )
        g1, g2, g3 = st.columns(3)
        with g1:
            search_algorithm_s = st.text_input(
                "search_algorithm",
                value=", ".join(str(v) for v in _as_list(shared.get("search_algorithm"), ["mcts"])),
                key="dlg_search_algorithm_grid",
            )
            depth_s = st.text_input(
                "depth",
                value=", ".join(str(v) for v in _as_list(shared.get("depth"), [4])),
                key="dlg_depth_grid",
            )
            top_k_s = st.text_input(
                "top_k",
                value=", ".join(str(v) for v in _as_list(shared.get("top_k"), [5])),
                key="dlg_top_k_grid",
            )
            patience_s = st.text_input(
                "patience",
                value=", ".join(str(v) for v in _as_list(shared.get("patience"), [0])),
                key="dlg_pat_grid",
            )
            kernels_s = st.text_input(
                "plan_shape (K)",
                value=", ".join(
                    str(v)
                    for v in _as_list(shared.get("plan_shape", shared.get("kernels")), [1])
                ),
                key="dlg_kernels_grid",
                help="Plan-shape K: 0 = bushy, 1 = linear/zig-zag (default), ≥2 = K-component bushy.",
            )
        with g2:
            lubysb_s = st.text_input(
                "luby budget/phase",
                value=", ".join(str(v) for v in _as_list(luby_d.get("start_budget"), [20])),
                key="dlg_lubysb_grid",
            )
            lubyph_s = st.text_input(
                "luby phases",
                value=", ".join(str(v) for v in _as_list(luby_d.get("phases"), [8])),
                key="dlg_lubyph_grid",
            )
            rpl_s = st.text_input(
                "rollouts/leaf",
                value=", ".join(str(v) for v in _as_list(shared.get("rollouts_per_leaf"), [1])),
                key="dlg_rpl_grid",
            )
        with g3:
            expC_s = st.text_input(
                "exploration_constant",
                value=", ".join(
                    str(v) for v in _as_list(shared.get("exploration_constant"), [1.4])
                ),
                key="dlg_expC_grid",
            )
            auto_default = "start_budget" not in single_d
            single_auto = st.toggle(
                "Auto-derive single start_budget",
                value=auto_default,
                key="dlg_single_auto_grid",
                help="Off → take an explicit list in the field below.",
            )
            single_sb_s = (
                ""
                if single_auto
                else st.text_input(
                    "single start_budget",
                    value=", ".join(str(v) for v in _as_list(single_d.get("start_budget"), [260])),
                    key="dlg_single_sb_grid",
                )
            )

        try:
            search_algorithm_l = _parse_list(search_algorithm_s, str)
            kernels_l = _parse_list(kernels_s, int)
            depth_l = _parse_list(depth_s, int)
            top_k_l = _parse_list(top_k_s, int)
            pat_l = _parse_list(patience_s, int)
            lubysb_l = _parse_list(lubysb_s, int)
            lubyph_l = _parse_list(lubyph_s, int)
            rpl_l = _parse_list(rpl_s, int)
            expC_l = _parse_list(expC_s, float)
            single_sb_l = [] if single_auto else _parse_list(single_sb_s, int)
            parse_err = None
        except ValueError as e:
            search_algorithm_l = []
            kernels_l = []
            depth_l = []
            top_k_l = []
            pat_l = []
            lubysb_l = []
            lubyph_l = []
            rpl_l = []
            expC_l = []
            single_sb_l = []
            parse_err = str(e)

        # Derive the cli_gucs payload as a nested dict with list-valued leaves.
        # Empty inputs are dropped (apply_gucs will fall back to the toml).
        cli_gucs_grid: dict = {}
        if search_algorithm_l:
            cli_gucs_grid["search_algorithm"] = search_algorithm_l
        if kernels_l:
            cli_gucs_grid["plan_shape"] = kernels_l
        if depth_l:
            cli_gucs_grid["depth"] = depth_l
        if top_k_l:
            cli_gucs_grid["top_k"] = top_k_l
        if pat_l:
            cli_gucs_grid["patience"] = pat_l
        if rpl_l:
            cli_gucs_grid["rollouts_per_leaf"] = rpl_l
        if budget_full_values:
            cli_gucs_grid["full_budget"] = _budget_guc_value()
        if expand_scenarios:
            cli_gucs_grid["expand_strategy"] = _guc_value(expand_scenarios)
        if expC_l:
            cli_gucs_grid["exploration_constant"] = expC_l
        luby_block = {}
        if lubysb_l:
            luby_block["start_budget"] = lubysb_l
        if lubyph_l:
            luby_block["phases"] = lubyph_l
        if luby_block:
            cli_gucs_grid["luby"] = luby_block
        if single_sb_l:
            cli_gucs_grid["single"] = {"start_budget": single_sb_l}

        # Live grid-size preview.
        n_combos_est = 1
        for vlist in (search_algorithm_l, kernels_l, depth_l, top_k_l, pat_l, lubysb_l, lubyph_l, rpl_l, expC_l):
            if vlist:
                n_combos_est *= len(vlist)
        if single_sb_l:
            n_combos_est *= len(single_sb_l)
        if budget_full_values:
            n_combos_est *= len(budget_full_values)
        else:
            n_combos_est = 0
        if expand_scenarios:
            n_combos_est *= len(expand_scenarios)
        else:
            n_combos_est = 0
        if parse_err:
            st.error(f"grid parse error: {parse_err}")
        elif not budget_full_values:
            st.error("Pick at least one budget mode.")
        elif not expand_scenarios:
            st.error("Pick at least one expand scenario.")
        else:
            st.metric("grid combos", n_combos_est)
    elif mode == "Selected cells":
        default_cells = base_cfg.get("mcts_cells") or [
            {
                "config": "single",
                "reward": "neg_log",
                "agg": "best",
                "depth": 10,
                "top_k": 10,
                "start_budget": 480,
                "combo_id": "d10_tk10_S_sb480",
            },
            {
                "config": "single",
                "reward": "neg_log",
                "agg": "best",
                "depth": 10,
                "top_k": 10,
                "start_budget": 160,
                "combo_id": "d10_tk10_S_sb160",
            },
            {
                "config": "single",
                "reward": "neg_log",
                "agg": "best",
                "depth": 4,
                "top_k": 3,
                "start_budget": 480,
                "combo_id": "d4_tk3_S_sb480",
            },
            {
                "config": "single",
                "reward": "neg_log",
                "agg": "best",
                "depth": 4,
                "top_k": 3,
                "start_budget": 240,
                "combo_id": "d4_tk3_S_sb240",
            },
            {
                "config": "single",
                "reward": "neg_log",
                "agg": "best",
                "depth": 10,
                "top_k": 10,
                "start_budget": 240,
                "combo_id": "d10_tk10_S_sb240",
            },
            {
                "config": "single",
                "reward": "norm_neg_log",
                "agg": "best",
                "depth": 4,
                "top_k": 10,
                "start_budget": 240,
                "combo_id": "d4_tk10_S_sb240",
            },
        ]
        st.markdown(
            "**Each line is one MCTS cell** — "
            "`config, reward, agg, depth, top_k, start_budget[, combo_id]`"
        )
        st.caption(
            "config: `single` or `luby` · reward: `neg_log` / `neg_cost` / `norm_neg_log` · "
            "agg: `best` / `average` · depth, top_k, start_budget: integers · "
            "combo_id: optional label (auto `d<depth>_tk<top_k>_S_sb<budget>`). "
            "Each cell runs for every Budget-mode × Expand-scenario chosen above; "
            "blank lines and lines starting with `#` are skipped."
        )
        selected_text = st.text_area(
            "Selected cells",
            value="\n".join(_cell_line(cell) for cell in default_cells),
            height=180,
            key="dlg_selected_cells",
            help=(
                "One cell per line: config,reward,agg,depth,top_k,start_budget,combo_id. "
                "Newer form also accepts config,reward,agg,depth,top_k,start_budget,kernels,combo_id. "
                "combo_id may be blank; lines starting with # are ignored."
            ),
        )
        try:
            selected_cells = _parse_selected_cells(selected_text)
            selected_parse_err = None
            selected_df = pd.DataFrame(selected_cells)
            st.dataframe(
                selected_df[
                    [
                        "config",
                        "reward",
                        "agg",
                        "depth",
                        "top_k",
                        "plan_shape",
                        "start_budget",
                        "combo_id",
                    ]
                ],
                hide_index=True,
                use_container_width=True,
            )
            st.metric("selected MCTS cells", len(selected_cells))
        except Exception as e:
            selected_parse_err = str(e)
            st.error(f"selected-cell parse error: {selected_parse_err}")
        n_combos_est = len(budget_full_values) * len(expand_scenarios)
        parse_err = None
    else:
        # --- quick mode: single-point inputs ---
        g1, g2, g3 = st.columns(3)
        with g1:
            search_algorithm = st.segmented_control(
                "Search algorithm",
                ["mcts", "saio", "iterative_improvement"],
                default=str(_as_list(shared.get("search_algorithm"), ["mcts"])[0]),
                key="dlg_search_algorithm",
                help="SAIO and iterative_improvement are run with one kernel regardless of plan-shape controls.",
            )
            plan_shape_default = int(
                _as_list(shared.get("plan_shape", shared.get("kernels")), [1])[0]
            )
            shape_mode = st.segmented_control(
                "Plan shape",
                ["bushy", "linear", "K-component"],
                default=_shape_mode_from_k(plan_shape_default),
                key="dlg_shape_mode",
                help="Plan-shape K (mcts_extreme.plan_shape): 0=bushy, "
                "1=linear/zig-zag, >=2=K-component bushy.",
            )
            if shape_mode == "K-component":
                kernels_count = st.number_input(
                    "K (components)",
                    2,
                    100,
                    max(2, plan_shape_default),
                    key="dlg_kernels",
                )
            else:
                kernels_count = 1
            depth = st.number_input(
                "depth", 2, 8, int(_as_list(shared.get("depth"), [4])[0]), key="dlg_depth"
            )
            top_k = st.number_input(
                "top_k", 2, 10, int(_as_list(shared.get("top_k"), [5])[0]), key="dlg_top_k"
            )
            patience = st.number_input(
                "patience", 0, 10, int(_as_list(shared.get("patience"), [0])[0]), key="dlg_pat"
            )
        with g2:
            luby_start_budget = st.number_input(
                "luby budget/phase",
                5,
                500,
                int(_as_list(luby_d.get("start_budget"), [20])[0]),
                key="dlg_lubysb",
            )
            luby_phases = st.number_input(
                "luby phases",
                1,
                16,
                int(_as_list(luby_d.get("phases"), [8])[0]),
                key="dlg_lubyph",
            )
            rollouts_per_leaf = st.number_input(
                "rollouts/leaf",
                1,
                8,
                int(_as_list(shared.get("rollouts_per_leaf"), [1])[0]),
                key="dlg_rpl",
            )
        with g3:
            exploration_constant = st.number_input(
                "exploration_constant",
                0.1,
                5.0,
                float(_as_list(shared.get("exploration_constant"), [1.4])[0]),
                0.1,
                key="dlg_expC",
            )
            auto_default = "start_budget" not in single_d
            single_auto = st.toggle(
                "Auto-derive single start_budget",
                value=auto_default,
                key="dlg_single_auto",
                help="Off → enter an explicit value below.",
            )
            derived_single = luby_cap(int(luby_start_budget), int(luby_phases))
            if single_auto:
                st.metric("single budget (auto-derived)", derived_single)
                single_start_budget = None
            else:
                single_start_budget = st.number_input(
                    "single start_budget",
                    1,
                    100000,
                    int(_as_list(single_d.get("start_budget"), [derived_single])[0]),
                    key="dlg_single_sb",
                )
        n_combos_est = len(budget_full_values) * len(expand_scenarios)
        parse_err = None

    queries: list[str] = []
    for src in sources_picked:
        queries.extend(_gather_selection(src, query_rel_count(src)))

    n_baseline = sum(1 for c in configs_to_run if c in BASELINE_CONFIGS)
    n_mcts = len(configs_to_run) - n_baseline
    selected_mcts_cells = _variant_cells(selected_cells) if mode == "Selected cells" else []
    if selected_mcts_cells:
        n_selected = sum(
            1
            for cell in selected_mcts_cells
            if cell.get("config", "single") in set(configs_to_run) - BASELINE_CONFIGS
        )
        n_cells = n_selected + n_baseline
        extra = f" · selected MCTS cells={n_selected}"
    else:
        n_cells = n_mcts * len(rewards) * len(aggs) * n_combos_est + n_baseline
        extra = (
            f" · grid combos={n_combos_est}"
            if mode == "Grid"
            else f" · budget modes={len(budget_full_values)}"
            if len(budget_full_values) > 1
            else ""
        )
    effective_n_seeds = 1 if measurement_mode == "cost_only" else int(n_seeds)
    total = len(queries) * n_cells * effective_n_seeds
    st.caption(
        f"Plan: **{len(queries)}** queries × **{n_cells}** cells "
        f"× **{effective_n_seeds}** seeds = **{total}** runs{extra}"
    )
    if measurement_mode == "cost_only" and int(n_seeds) != 1:
        st.caption("Cost-only mode uses one seed because the query is not executed.")

    st.markdown("**Execution**")
    c_par, c_restart, c_warm, c_drop, c_clean = st.columns([2, 1, 1, 1, 1])
    with c_par:
        parallel = st.number_input(
            "Parallel cells per batch (-j)",
            min_value=0,
            max_value=20,
            value=1,
            key="dlg_parallel",
            help="0 = all cells in parallel, 1 = sequential default.",
        )
        if parallel > n_cells:
            st.caption(f"capped at {n_cells} (len(cells))")
    with c_restart:
        restart_pg = st.toggle(
            "Restart PG",
            value=True,
            key="dlg_restart_pg",
            help="Restart PostgreSQL before every (query, seed) parallel batch.",
        )
    with c_warm:
        prewarm = st.toggle(
            "Catalog prewarm",
            value=False,
            key="dlg_prewarm",
            help="Run one EXPLAIN before each batch to warm shared_buffers. Doesn't move the numbers — leave off unless you want it.",
        )
    with c_drop:
        drop_os_cache = st.toggle(
            "Drop OS cache",
            value=True,
            key="dlg_drop_os_cache",
            help="Drop Linux page cache after each PG restart and before launching the synchronized batch. Requires passwordless sudo for ablation/drop_caches.sh.",
        )
    with c_clean:
        clean_per_run = st.toggle(
            "Clean each run",
            value=True,
            key="dlg_clean_per_run",
            help="Restart/drop before every individual cell measurement. Forces -j 1.",
        )
        if clean_per_run:
            st.caption("forces -j 1")
    docker_container = st.text_input(
        "Docker container",
        value=DEFAULT_DOCKER_CONTAINER,
        key="dlg_docker_container",
        help="When set, sweep.py uses docker restart/docker exec psql instead of local pg_ctl/psql.",
    ).strip()

    if st.button("Launch sweep", type="primary", use_container_width=True, key="dlg_launch"):
        if not queries:
            st.error("Pick at least one query.")
            return
        if not configs_to_run:
            st.error("Pick at least one config.")
            return
        if not budget_full_values:
            st.error("Pick at least one budget mode.")
            return
        if not expand_scenarios:
            st.error("Pick at least one expand scenario.")
            return
        has_mcts_config = any(c not in BASELINE_CONFIGS for c in configs_to_run)
        if has_mcts_config and mode != "Selected cells" and not (rewards and aggs):
            st.error("Pick at least one reward and aggregation for MCTS configs.")
            return
        mcts_cells_payload: list[dict] = []
        if mode == "Grid":
            if parse_err:
                st.error(f"grid parse error: {parse_err}")
                return
            if n_combos_est == 0:
                st.error("Grid mode: every GUC field is empty.")
                return
            cli_gucs = cli_gucs_grid
        elif mode == "Selected cells":
            if selected_parse_err:
                st.error(f"selected-cell parse error: {selected_parse_err}")
                return
            mcts_config_set = set(configs_to_run) - BASELINE_CONFIGS
            if mcts_config_set and not any(
                cell.get("config", "single") in mcts_config_set for cell in selected_cells
            ):
                st.error("Selected cells do not match any selected MCTS config.")
                return
            cli_gucs = {}
            mcts_cells_payload = [
                cell
                for cell in _variant_cells(selected_cells)
                if cell.get("config", "single") in mcts_config_set
            ]
        else:
            cli_gucs = {
                "search_algorithm": str(search_algorithm),
                "plan_shape": (
                    1
                    if str(search_algorithm) in {"saio", "iterative_improvement"}
                    else _k_from_shape_mode(str(shape_mode), int(kernels_count))
                ),
                "depth": int(depth),
                "top_k": int(top_k),
                "patience": int(patience),
                "rollouts_per_leaf": int(rollouts_per_leaf),
                "full_budget": _budget_guc_value(),
                "expand_strategy": _guc_value(expand_scenarios),
                "exploration_constant": float(exploration_constant),
                "luby": {
                    "start_budget": int(luby_start_budget),
                    "phases": int(luby_phases),
                },
            }
            if not single_auto and single_start_budget is not None:
                cli_gucs["single"] = {"start_budget": int(single_start_budget)}
        cmd = [
            sys.executable,
            str(ROOT / "ablation" / "sweep.py"),
            "-c",
            str(base_cfg_path),
            "-n",
            name,
            "--run-group",
            run_group_name,
            "--seeds",
            str(effective_n_seeds),
            "--queries",
            " ".join(queries),
            "--rewards",
            " ".join(rewards),
            "--aggs",
            " ".join(aggs),
            "--configs",
            " ".join(configs_to_run),
            "--measurement",
            measurement_mode,
            "--source",
            ",".join(sources_picked) if sources_picked else "job",
            "--gucs",
            json.dumps(cli_gucs),
            "--mcts-cells",
            json.dumps(mcts_cells_payload),
            "-j",
            str(int(parallel)),
        ]
        if prewarm:
            cmd.append("--prewarm")
        if drop_os_cache:
            cmd.append("--drop-os-cache")
        if not restart_pg:
            cmd.append("--no-restart-pg-per-batch")
        if clean_per_run:
            cmd.append("--clean-per-run")
        if docker_container:
            cmd.extend(["--docker-container", docker_container])
        ok, msg = launch_sweep(cmd)
        if not ok:
            st.error(msg.split("\n", 1)[0])
            if "\n" in msg:
                st.code(msg.split("\n", 1)[1], language="text")
            return
        st.success(f"{msg} Closing modal — the live counter tracks progress.")
        time.sleep(0.5)
        st.rerun()


with st.sidebar:
    st.header("Run")
    runs = list_runs()
    if not runs:
        st.info("No runs yet — start one with the button below.")
        selected_run = None
    else:
        runs_by_group = group_runs(runs)
        group_names = list(runs_by_group)
        selected_group = st.selectbox(
            "Run group",
            group_names,
            format_func=lambda g: f"{g} ({len(runs_by_group[g])})",
        )
        group_run_list = runs_by_group[selected_group]
        selected_idx = st.selectbox(
            "Run",
            range(len(group_run_list)),
            format_func=lambda i: f"{status_badge(group_run_list[i])}  {group_run_list[i].name}",
            index=0,
            label_visibility="collapsed",
        )
        selected_run = group_run_list[selected_idx]

    auto_reload_plots = st.checkbox(
        "Auto-reload plots (2 s)",
        value=False,
        help=(
            "While a sweep runs, the live counter always updates on its own.  Turn "
            "this ON to also reload the plots every 2 s — this resets the Plans/Image "
            "tabs (the old behavior).  Leave OFF to read tabs undisturbed."
        ),
    )

    if st.button("+ New experiment", use_container_width=True, type="primary"):
        _new_experiment_dialog()

    with st.expander("🧹 Maintenance"):
        st.caption(f"PGDATA: `{DEFAULT_PGDATA}`")
        st.caption(f"Docker container: `{DEFAULT_DOCKER_CONTAINER or '(off)'}`")
        m1, m2 = st.columns(2)
        with m1:
            if st.button("Clear OS caches", use_container_width=True):
                ok, msg = clear_os_cache_now()
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)
        with m2:
            if st.button("Restart PG", use_container_width=True):
                ok, msg = restart_pg_now()
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)

    if selected_run is not None:
        with st.expander("🔁 Rerun experiment"):
            cfg_path = selected_run / "config.json"
            if not cfg_path.exists():
                st.caption("No config.json in this run.")
            else:
                try:
                    rerun_cfg = json.loads(cfg_path.read_text())
                    default_name = sanitize_run_name(f"rerun-{selected_run.name}")
                    rerun_name = st.text_input(
                        "New run name",
                        value=default_name,
                        key=f"rerun_name_{selected_run.name}",
                    )
                    rerun_group = st.text_input(
                        "Run group",
                        value=str(rerun_cfg.get("run_group") or config_run_group(selected_run) or ""),
                        key=f"rerun_group_{selected_run.name}",
                        placeholder="PG / JOB-Complex",
                    ).strip()
                    rerun_parallel = st.number_input(
                        "Parallel cells per batch (-j)",
                        min_value=0,
                        max_value=20,
                        value=1,
                        key=f"rerun_parallel_{selected_run.name}",
                    )
                    rerun_restart_pg = st.toggle(
                        "Restart PG before each batch",
                        value=True,
                        key=f"rerun_restart_pg_{selected_run.name}",
                    )
                    rerun_prewarm = st.toggle(
                        "Catalog prewarm",
                        value=False,
                        key=f"rerun_prewarm_{selected_run.name}",
                    )
                    rerun_drop_os_cache = st.toggle(
                        "Drop OS cache",
                        value=False,
                        key=f"rerun_drop_os_cache_{selected_run.name}",
                        help="Requires passwordless sudo for ablation/drop_caches.sh.",
                    )
                    rerun_clean_per_run = st.toggle(
                        "Clean before every run",
                        value=False,
                        key=f"rerun_clean_per_run_{selected_run.name}",
                        help="Restart/drop before each individual cell measurement. Forces -j 1.",
                    )
                    rerun_docker_container = st.text_input(
                        "Docker container",
                        value=rerun_cfg.get("docker_container") or DEFAULT_DOCKER_CONTAINER,
                        key=f"rerun_docker_container_{selected_run.name}",
                    ).strip()
                    queries = [str(q).split(":", 1)[-1] for q in rerun_cfg.get("queries", [])]
                    sources = rerun_cfg.get("sources", ["job"])
                    configs = rerun_cfg.get("configs", [])
                    selected_cells = rerun_cfg.get("mcts_cells", []) or []
                    n_cells = sum(1 for c in configs if c in BASELINE_CONFIGS) + (
                        len(selected_cells)
                        if selected_cells
                        else sum(1 for c in configs if c not in BASELINE_CONFIGS)
                        * len(rerun_cfg.get("reward_modes", []))
                        * len(rerun_cfg.get("uct_aggregations", []))
                        * int(rerun_cfg.get("n_combos", 1) or 1)
                    )
                    total = len(queries) * n_cells * int(rerun_cfg.get("n_seeds", 1))
                    st.caption(
                        f"{len(queries)} queries × {n_cells} cells × "
                        f"{int(rerun_cfg.get('n_seeds', 1))} seeds = {total} runs"
                    )
                    if st.button(
                        "Rerun selected run",
                        key=f"rerun_btn_{selected_run.name}",
                        use_container_width=True,
                    ):
                        if not queries or not configs:
                            st.error("Cannot rerun: config has no queries or configs.")
                        else:
                            cmd = [
                                sys.executable,
                                str(ROOT / "ablation" / "sweep.py"),
                                "-c",
                                str(CONFIG_DIR / "job_jobcomplex_geqo.toml"),
                                "-n",
                                sanitize_run_name(rerun_name),
                                "--run-group",
                                rerun_group,
                                "--seeds",
                                str(int(rerun_cfg.get("n_seeds", 1))),
                                "--queries",
                                " ".join(queries),
                                "--rewards",
                                " ".join(rerun_cfg.get("reward_modes", [])),
                                "--aggs",
                                " ".join(rerun_cfg.get("uct_aggregations", [])),
                                "--configs",
                                " ".join(configs),
                                "--source",
                                ",".join(sources),
                                "--gucs",
                                json.dumps(rerun_cfg.get("cli_gucs", {}) or {}),
                                "--mcts-cells",
                                json.dumps(selected_cells),
                                "-j",
                                str(int(rerun_parallel)),
                            ]
                            if rerun_prewarm:
                                cmd.append("--prewarm")
                            if rerun_drop_os_cache:
                                cmd.append("--drop-os-cache")
                            if not rerun_restart_pg:
                                cmd.append("--no-restart-pg-per-batch")
                            if rerun_clean_per_run:
                                cmd.append("--clean-per-run")
                            if rerun_docker_container:
                                cmd.extend(["--docker-container", rerun_docker_container])
                            ok, msg = launch_sweep(cmd)
                            if not ok:
                                st.error(msg.split("\n", 1)[0])
                                if "\n" in msg:
                                    st.code(msg.split("\n", 1)[1], language="text")
                            else:
                                st.success(msg)
                                time.sleep(0.5)
                                st.rerun()
                except Exception as e:
                    st.error(f"Cannot read rerun config: {e}")

        with st.expander("✏️ Rename experiment"):
            new_name_raw = st.text_input(
                "Run directory name",
                value=selected_run.name,
                key=f"rename_input_{selected_run.name}",
            )
            new_group_raw = st.text_input(
                "Run group",
                value=config_run_group(selected_run),
                key=f"rename_group_{selected_run.name}",
                placeholder="PG / JOB-Complex",
            )
            new_group = new_group_raw.strip()
            new_name = sanitize_run_name(new_name_raw)
            if new_name != new_name_raw.strip():
                st.caption(f"Will be saved as `{new_name}`.")
            if st.button(
                "Save name/group",
                key=f"rename_btn_{selected_run.name}",
                use_container_width=True,
            ):
                status = lib.read_status(selected_run) or "?"
                if status == "running":
                    st.warning("Stop the run before changing name or group.")
                elif not new_name:
                    st.warning("Name cannot be empty.")
                else:
                    target = RUNS_DIR / new_name
                    if target != selected_run and target.exists():
                        st.warning(f"`{target.name}` already exists.")
                    else:
                        try:
                            if target != selected_run:
                                selected_run.rename(target)
                            cfg_path = target / "config.json"
                            if cfg_path.exists():
                                cfg = json.loads(cfg_path.read_text())
                                cfg["name"] = new_name
                                cfg["run_group"] = new_group
                                cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
                            readme_path = target / "README.md"
                            if readme_path.exists():
                                lines = readme_path.read_text().splitlines()
                                if lines and lines[0].startswith("# "):
                                    lines[0] = f"# {new_name}"
                                group_line_seen = False
                                for i, line in enumerate(lines):
                                    if line.startswith("- **Run group**:"):
                                        lines[i] = f"- **Run group**: {new_group or '(none)'}"
                                        group_line_seen = True
                                        break
                                if not group_line_seen:
                                    insert_at = next(
                                        (
                                            i + 1
                                            for i, line in enumerate(lines)
                                            if line.startswith("- **Git revision**:")
                                        ),
                                        min(4, len(lines)),
                                    )
                                    lines.insert(insert_at, f"- **Run group**: {new_group or '(none)'}")
                                readme_path.write_text("\n".join(lines) + "\n")
                            st.success(f"Saved `{new_name}`.")
                            time.sleep(0.5)
                            st.rerun()
                        except Exception as e:
                            st.error(f"rename failed: {e}")

    with st.expander("📥 Import xlsx workbook"):
        uploaded = st.file_uploader(
            "Colleague-format .xlsx",
            type=["xlsx"],
            key="sidebar_import_xlsx",
            label_visibility="collapsed",
            help="Materialise an xlsx into a synthetic runs/ entry.",
        )
        if uploaded is not None and st.button("Import", key="sidebar_import_btn"):
            try:
                import re
                import tempfile

                import sharing

                stem = re.sub(r"\s+\(\d+\)", "", Path(uploaded.name).stem)
                stem = stem.strip().replace(" ", "_") or "uploaded"
                out_dir = RUNS_DIR / f"imported-{stem}"
                with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
                    tmp.write(uploaded.getvalue())
                    tmp_path = Path(tmp.name)
                sharing.import_workbook(tmp_path, out_dir)
                st.success(f"Imported → {out_dir.name}")
                time.sleep(0.5)
                st.rerun()
            except Exception as e:
                st.error(f"import failed: {e}")


# --- main area ---

if selected_run is None:
    st.title("MCTS Ablation")
    st.write("Pick a run on the left, or start a new experiment.")
    st.stop()

st.title(selected_run.name)
status = lib.read_status(selected_run) or "?"
cols = st.columns([1, 1, 1, 3])
with cols[0]:
    st.metric("Status", status)
with cols[1]:
    if status == "running" and st.button("⏹️ Stop", type="secondary"):
        ok, msg = stop_run(selected_run)
        if ok:
            st.success(msg)
            time.sleep(1.0)
            st.rerun()
        else:
            st.warning(msg)
with cols[2]:
    pid = read_pid(selected_run)
    if pid is not None:
        alive = "alive" if pid_alive(pid) else "dead"
        st.caption(f"pid {pid} ({alive})")
with cols[3]:
    _spacer, _btn_xlsx, _btn_exp = st.columns([2, 2, 2])
    with _btn_xlsx:
        n_rows = (
            sum(1 for _ in (selected_run / "results.jsonl").open())
            if (selected_run / "results.jsonl").exists()
            else 0
        )
        cache_key = f"_xlsx::{selected_run.name}::{n_rows}"
        if st.button(
            "📊 Export workbook", help="Build colleague-compatible xlsx", use_container_width=True
        ):
            try:
                import tempfile

                import sharing

                with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
                    out_path = sharing.export_workbook(selected_run, Path(tmp.name))
                st.session_state[cache_key] = Path(out_path).read_bytes()
            except Exception as e:
                st.error(f"export failed: {e}")
        cached_xlsx = st.session_state.get(cache_key)
        if cached_xlsx is not None:
            st.download_button(
                "💾 Download .xlsx",
                data=cached_xlsx,
                file_name=f"{selected_run.name}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"dl_{cache_key}",
                use_container_width=True,
            )
    with _btn_exp:
        slim_key = f"expdata_slim_{selected_run.name}"
        if st.button(
            "📤 Export to expdata",
            help="Copy this run into expdata/<name>/ at the repo root, ready to git add.",
            use_container_width=True,
        ):
            try:
                import sharing

                info = sharing.export_expdata(
                    selected_run, slim=st.session_state.get(slim_key, False), overwrite=True
                )
                rel = info["path"].relative_to(ROOT)
                kb = info["size_bytes"] / 1024
                slim_note = f" (slim — skipped {info['skipped_plans']} plans)" if info["slim"] else ""
                st.session_state[f"_exp_msg_{selected_run.name}"] = (
                    f"✅ Wrote `{rel}/`  ·  {info['n_files']} files, {kb:.1f} KiB{slim_note}\n\n"
                    f"```bash\ngit add {rel}\n```"
                )
            except Exception as e:
                st.session_state[f"_exp_msg_{selected_run.name}"] = f"❌ export failed: {e}"
        st.toggle(
            "slim",
            value=False,
            key=slim_key,
            help="Skip *.plan files (~95% smaller). Plots still work from results.jsonl.",
        )
        msg = st.session_state.get(f"_exp_msg_{selected_run.name}")
        if msg:
            (st.error if msg.startswith("❌") else st.success)(msg)
readme_path = selected_run / "README.md"
if readme_path.exists():
    with st.expander("README"):
        st.markdown(readme_path.read_text())

df = load_results(selected_run)
df = backfill_timed_out(selected_run, df)
if not df.empty:
    if "combo_id" not in df.columns:
        df["combo_id"] = ""
    else:
        df["combo_id"] = df["combo_id"].fillna("").astype(str)
    # NaN propagates so timeout rows (missing one half) drop out of the plots.
    if "plan_time_ms" in df.columns and "exec_time_ms" in df.columns:
        df["e2e_time_ms"] = df["plan_time_ms"] + df["exec_time_ms"]

if not df.empty and df["timed_out"].any():
    n_timeout = int(df["timed_out"].sum())
    st.warning(
        f"⏱️ {n_timeout} of {len(df)} runs hit the statement_timeout cap "
        "— see the **Timeouts** tab for which (cell, query, seed) combos failed."
    )

expected = expected_run_count(selected_run)
rendered_n = len(df)
first_ts = df["ts"].iloc[0] if (not df.empty and "ts" in df.columns) else None


def _eta_text(done_n: int) -> str:
    if done_n <= 0 or first_ts is None:
        return ""
    try:
        elapsed = (dt_now_aware() - pd.to_datetime(first_ts)).total_seconds()
        remaining = (expected - done_n) * (elapsed / done_n)
    except Exception:
        return ""
    return f"  ·  ~{int(remaining // 60)}m {int(remaining % 60)}s left"


def _render_progress(done_n: int) -> None:
    if not expected:
        if done_n:
            st.caption(f"{done_n} runs logged (no progress estimate)")
        return
    frac = min(1.0, done_n / expected)
    eta = _eta_text(done_n) if status == "running" else ""
    st.progress(frac, text=f"{done_n}/{expected} runs ({frac * 100:.1f}%){eta}")


def _live_row_count() -> int:
    return _jsonl_line_count(selected_run / "results.jsonl") or 0


if df.empty:
    if status == "running":

        @st.fragment(run_every=2.0)
        def _wait_for_rows() -> None:
            live_n = _live_row_count()
            _render_progress(live_n)
            st.info("No rows yet — waiting for the runner to write data.")
            if live_n > 0:
                st.rerun()

        _wait_for_rows()
    else:
        _render_progress(0)
        st.info("No rows in `results.jsonl`.")
    st.stop()

if status == "running":

    @st.fragment(run_every=2.0)
    def _live_monitor() -> None:
        live_n = _live_row_count()
        _render_progress(live_n)
        new_rows = live_n - rendered_n
        if new_rows <= 0:
            return
        if auto_reload_plots:
            st.rerun()
        else:
            lc, rc = st.columns([4, 1])
            lc.caption(f"🔄 {new_rows} new row(s) since this view loaded — plots paused.")
            if rc.button("Reload", use_container_width=True, key="_live_reload"):
                st.rerun()

    _live_monitor()
else:
    _render_progress(rendered_n)

if "source" not in df.columns:
    df["source"] = ""

df["rels"] = pd.to_numeric(df["rels"], errors="coerce") if "rels" in df.columns else pd.NA
missing_rels = df["rels"].isna()
if missing_rels.any() and "source" in df.columns:
    inferred_rels = df.loc[missing_rels].apply(_row_query_rel_count, axis=1)
    df.loc[missing_rels, "rels"] = pd.to_numeric(inferred_rels, errors="coerce")

with st.expander("Filters", expanded=True):
    filters_enabled = st.toggle(
        "Enable query filters",
        value=status != "running",
        key=f"filters_enabled_{selected_run.name}",
        help="Turn off while a run is active to see all incoming rows in real time.",
    )
    source_options = sorted(s for s in df["source"].dropna().astype(str).unique() if s)
    if filters_enabled:
        f_bench, f_rels, f_q = st.columns([2, 2, 3])
        with f_bench:
            picked_sources = st.multiselect(
                "Benchmark",
                options=source_options,
                default=source_options,
                key=f"filter_sources_{selected_run.name}",
            )
        rel_values = sorted(int(v) for v in df["rels"].dropna().unique())
        with f_rels:
            if len(rel_values) > 1:
                rel_min, rel_max = st.slider(
                    "Relations",
                    min_value=min(rel_values),
                    max_value=max(rel_values),
                    value=(min(rel_values), max(rel_values)),
                    key=f"filter_rels_{selected_run.name}",
                )
            elif len(rel_values) == 1:
                rel_min = rel_max = rel_values[0]
                st.metric("Relations", rel_values[0])
            else:
                rel_min = rel_max = None
                st.caption("No relation-count metadata.")
        filter_base = df.copy()
        if picked_sources:
            filter_base = filter_base[filter_base["source"].isin(picked_sources)]
        if rel_min is not None and rel_max is not None:
            filter_base = filter_base[
                filter_base["rels"].isna()
                | ((filter_base["rels"] >= rel_min) & (filter_base["rels"] <= rel_max))
            ]
        query_options = sorted(filter_base["query"].dropna().astype(str).unique())
        with f_q:
            picked_queries = st.multiselect(
                "Queries",
                options=query_options,
                default=query_options,
                key=f"filter_queries_{selected_run.name}",
            )
        if picked_sources:
            df = df[df["source"].isin(picked_sources)]
        if rel_min is not None and rel_max is not None:
            df = df[
                df["rels"].isna()
                | ((df["rels"] >= rel_min) & (df["rels"] <= rel_max))
            ]
        if picked_queries:
            df = df[df["query"].astype(str).isin(picked_queries)]
    else:
        st.caption("Filters disabled. Showing every logged row.")
    n_filter_queries = (
        df[["source", "query"]].drop_duplicates().shape[0]
        if "source" in df.columns
        else df["query"].nunique()
    )
    st.caption(f"Showing {len(df)} rows across {n_filter_queries} queries.")

if df.empty:
    st.info("No rows match the active filters.")
    st.stop()

tab_summary, tab_winners, tab_traj, tab_image, tab_timeouts, tab_plans = st.tabs(
    ["Summary", "Winners", "Trajectory", "Image", "Timeouts", "Plans"]
)


# --- plotly helpers ---


def _attach_compact_labels(df: pd.DataFrame) -> pd.DataFrame:
    mcts = df[~df["config"].isin(BASELINE_CONFIGS)]
    show_reward = mcts["reward"].nunique() > 1 if not mcts.empty else False
    show_agg = mcts["agg"].nunique() > 1 if not mcts.empty else False

    def _build(r):
        if r["config"] in BASELINE_CONFIGS:
            return r["config"]
        parts = ["luby" if r["config"] == "luby" else "sing"]
        if show_reward:
            parts.append(lib._REWARD_SHORT.get(r["reward"], str(r["reward"])[:6]))
        if show_agg:
            parts.append("best" if r["agg"] == "best" else "avg")
        cid = r.get("combo_id", "") or ""
        if cid:
            parts.append(cid)
        return "/".join(parts)

    df["label"] = df.apply(_build, axis=1)
    return df


def _plotly_panel(
    workload: pd.DataFrame,
    metric: str,
    *,
    title: str,
    log_y: bool = False,
    zoom: bool = False,
    mean_only: bool = False,
    sort_metric: str | None = None,
    y_scale: str = "Adaptive",
) -> go.Figure:
    wanted_stats = ("mean",) if mean_only else ("best", "mean", "median", "worst")
    cols = {f"{metric}__{s}": s for s in wanted_stats if f"{metric}__{s}" in workload.columns}
    if not cols:
        return go.Figure().update_layout(title=f"{title} (no data)")

    w = workload.copy()
    # Drop cells with no data for this metric (e.g. baselines on the MCTS Iters panel).
    w = w[w[f"{metric}__mean"].notna()]
    if w.empty:
        return go.Figure().update_layout(title=f"{title} (no data)")
    w = _attach_compact_labels(w)
    sort_col = f"{sort_metric or metric}__mean"
    if sort_col not in w.columns:
        sort_col = f"{metric}__mean"
    w = w.sort_values(sort_col).reset_index(drop=True)

    mean = w[f"{metric}__mean"].astype(float).values
    best = w.get(f"{metric}__best", pd.Series([np.nan] * len(w))).astype(float).values
    worst = w.get(f"{metric}__worst", pd.Series([np.nan] * len(w))).astype(float).values
    median = w.get(f"{metric}__median", pd.Series([np.nan] * len(w))).astype(float).values

    yerr_lo = np.clip(np.nan_to_num(mean - best, nan=0.0), 0, None)
    yerr_hi = np.clip(np.nan_to_num(worst - mean, nan=0.0), 0, None)

    # Color: luby=blue, single=orange, dp=green, geqo=red.
    colors_by_config = {
        "luby": "#1f77b4",
        "single": "#ff7f0e",
        "dp": "#2ca02c",
        "geqo": "#d62728",
    }
    colors = [colors_by_config.get(c, "#7f7f7f") for c in w["config"]]
    patterns = [
        "/" if (a == "average" and cfg not in BASELINE_CONFIGS) else ""
        for a, cfg in zip(w["agg"], w["config"], strict=False)
    ]
    hovertemplate = (
        "%{x}<br>mean: %{y:.2f}<extra></extra>"
        if mean_only
        else (
            "%{x}<br>"
            "mean: %{y:.2f}<br>"
            "best: %{customdata[0]:.2f}<br>"
            "median: %{customdata[1]:.2f}<br>"
            "worst: %{customdata[2]:.2f}"
            "<extra></extra>"
        )
    )

    fig = go.Figure()
    fig.add_bar(
        x=w["label"],
        y=mean,
        marker_color=colors,
        marker_pattern_shape=patterns,
        marker_line_color="black",
        marker_line_width=1,
        opacity=0.65,
        error_y=None
        if mean_only
        else {
            "type": "data",
            "symmetric": False,
            "array": yerr_hi,
            "arrayminus": yerr_lo,
            "color": "black",
            "thickness": 1,
        },
        hovertemplate=hovertemplate,
        customdata=np.column_stack([best, median, worst]),
        showlegend=False,
    )
    if not mean_only:
        fig.add_scatter(
            x=w["label"],
            y=median,
            mode="markers",
            marker={"symbol": "diamond", "color": "black", "size": 8},
            showlegend=False,
            hoverinfo="skip",
        )

    use_zero_range = y_scale == "0 to max"
    actual_log_y = log_y and not use_zero_range
    fig.update_layout(
        title={"text": title, "font": {"size": 12}},
        margin={"l": 40, "r": 10, "t": 40, "b": 40},
        yaxis={"type": "log" if actual_log_y else "linear"},
        xaxis={"showticklabels": False, "ticks": ""},
    )
    if use_zero_range and len(mean):
        y_values = mean if mean_only else np.concatenate([mean, best, worst])
        y_values = y_values[np.isfinite(y_values)]
        if len(y_values):
            hi = float(np.nanmax(y_values))
            fig.update_yaxes(range=[0, max(hi * 1.05, 1.0)])
    elif zoom and not actual_log_y and len(mean):
        lo, hi = (
            (float(np.nanmin(mean)), float(np.nanmax(mean)))
            if mean_only
            else (float(np.nanmin(best)), float(np.nanmax(worst)))
        )
        if hi > lo:
            pad = (hi - lo) * 0.15
            fig.update_yaxes(range=[lo - pad, hi + pad])
    return fig


def _summary_y_values(workload: pd.DataFrame, metric: str, mean_only: bool) -> np.ndarray:
    mean_col = f"{metric}__mean"
    if mean_col not in workload.columns:
        return np.array([], dtype=float)
    valid = workload[workload[mean_col].notna()]
    if valid.empty:
        return np.array([], dtype=float)
    cols = [mean_col] if mean_only else [
        c for c in (f"{metric}__best", mean_col, f"{metric}__worst") if c in valid.columns
    ]
    values = valid[cols].to_numpy(dtype=float).ravel()
    return values[np.isfinite(values)]


def _apply_summary_yaxis(
    fig: go.Figure,
    workload: pd.DataFrame,
    metric: str,
    *,
    row: int,
    col: int,
    mean_only: bool,
    log_y: bool,
    zoom: bool,
    y_scale: str,
) -> None:
    if y_scale == "0 to max":
        values = _summary_y_values(workload, metric, mean_only)
        if len(values):
            hi = float(np.nanmax(values))
            fig.update_yaxes(type="linear", range=[0, max(hi * 1.05, 1.0)], row=row, col=col)
        else:
            fig.update_yaxes(type="linear", row=row, col=col)
        return
    if log_y:
        fig.update_yaxes(type="log", row=row, col=col)
        return
    if zoom:
        values = _summary_y_values(workload, metric, mean_only)
        if len(values):
            lo = float(np.nanmin(values))
            hi = float(np.nanmax(values))
            if hi > lo:
                pad = (hi - lo) * 0.15
                fig.update_yaxes(range=[lo - pad, hi + pad], row=row, col=col)


# --- summary tab ---


def panel(ax, agg, metric, title, log_y=False, zoom_pad_frac=None):
    """One panel; sorted by sum-of-mean across queries (best on left)."""
    cols = {
        f"{metric}__{s}": s
        for s in ("best", "mean", "median", "worst")
        if f"{metric}__{s}" in agg.columns
    }
    if not cols:
        ax.set_title(f"{title} (no data)")
        return
    agg = _attach_compact_labels(agg.copy())
    agg = agg.sort_values(f"{metric}__mean").reset_index(drop=True)

    x = np.arange(len(agg))
    mean = agg[f"{metric}__mean"].values
    best = agg.get(f"{metric}__best", pd.Series([np.nan] * len(agg))).values
    worst = agg.get(f"{metric}__worst", pd.Series([np.nan] * len(agg))).values
    median = agg.get(f"{metric}__median", pd.Series([np.nan] * len(agg))).values
    colors = ["#1f77b4" if c == "luby" else "#ff7f0e" for c in agg["config"]]
    hatches = ["" if a == "best" else "//" for a in agg["agg"]]

    bars = ax.bar(x, mean, color=colors, alpha=0.5, edgecolor="black", linewidth=0.8)
    for b, h in zip(bars, hatches, strict=False):
        b.set_hatch(h)
    yerr_lo = np.clip(np.nan_to_num(mean - best, nan=0.0), 0, None)
    yerr_hi = np.clip(np.nan_to_num(worst - mean, nan=0.0), 0, None)
    ax.errorbar(
        x, mean, yerr=[yerr_lo, yerr_hi], fmt="none", ecolor="black", capsize=4, elinewidth=1
    )
    ax.scatter(x, median, marker="D", color="black", zorder=5, s=18)
    ax.set_xticks(x)
    ax.set_xticklabels(agg["label"], rotation=35, ha="right", fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    if log_y:
        ax.set_yscale("log")
    if zoom_pad_frac is not None and len(best) and not np.all(np.isnan(best)):
        lo, hi = np.nanmin(best), np.nanmax(worst)
        pad = (hi - lo) * zoom_pad_frac if hi > lo else max(abs(hi) * 0.01, 1.0)
        ax.set_ylim(lo - pad, hi + pad)


def _render_coverage_audit(coverage: dict, union: set, common: set) -> bool:
    """Warn when cells cover different query sets; return the common-set toggle state."""
    if not union or union == common:
        return False
    c_warn, c_tog = st.columns([4, 1])
    with c_warn:
        st.warning(
            f"⚠️ Cells cover different query sets — totals/ratios sum over different queries "
            f"and aren't directly comparable (union {len(union)}, common {len(common)})."
        )
    with c_tog:
        common_only = st.checkbox(
            f"Common set only ({len(common)})",
            key="summary_common_only",
            help="Restrict every panel to queries all cells have non-timed-out data for.",
        )
    with st.expander(f"Which of the {len(union) - len(common)} differing query(ies)?"):
        for q in sorted(union - common):
            qlabel = "/".join(str(x) for x in q)
            missing = [lib.short_label(*cell) for cell, qs in coverage.items() if q not in qs]
            st.caption(f"**{qlabel}** — missing from: {', '.join(missing)}")
    return common_only


@st.fragment
def _render_summary():
    coverage, union, common = lib.query_coverage(df)
    common_only = _render_coverage_audit(coverage, union, common)
    view_df = lib.restrict_to_common_queries(df) if common_only else df
    workload = _agg_workload(view_df)
    if workload.empty:
        st.info("Not enough rows for the summary view yet.")
    else:
        n_q = int(workload["n_queries"].max())
        baseline_options = sorted(c for c in df["config"].dropna().unique() if c in BASELINE_CONFIGS)
        c_view, c_y, c_sort, c_mean, c_base, c_formula = st.columns([2, 1.4, 2, 1, 1, 2])
        with c_view:
            summary_view = st.radio(
                "Summary scale",
                ["Absolute totals", "Relative to baseline"],
                horizontal=True,
                key="summary_scale",
            )
        with c_y:
            summary_y_scale = st.radio(
                "Y range",
                ["Adaptive", "0 to max"],
                horizontal=True,
                key="summary_y_scale",
            )
        summary_sort_options = [
            (m, METRIC_LABELS[m])
            for m in ("e2e_time_ms", "exec_time_ms", "plan_time_ms", "mcts_best_cost", "iters")
            if f"{m}__mean" in workload.columns and workload[f"{m}__mean"].notna().any()
        ]
        with c_sort:
            summary_sort_metric = st.selectbox(
                "Sort modes by",
                [m for m, _ in summary_sort_options],
                index=next(
                    (
                        i
                        for i, (m, _) in enumerate(summary_sort_options)
                        if m == "e2e_time_ms"
                    ),
                    0,
                ),
                format_func=lambda m: dict(summary_sort_options).get(m, m),
                key="summary_sort_metric",
                help="Applies the same mode order to every Summary figure.",
            )
        with c_mean:
            summary_mean_only = st.toggle(
                "Mean only",
                value=False,
                key="summary_mean_only",
                help="Hide best/worst ranges and median diamonds in Summary plots.",
            )
        with c_base:
            rel_baseline = st.selectbox(
                "Baseline",
                baseline_options,
                index=baseline_options.index("geqo") if "geqo" in baseline_options else 0,
                key="summary_rel_baseline",
                disabled=summary_view == "Absolute totals" or not baseline_options,
            ) if baseline_options else None
        with c_formula:
            rel_formula = st.selectbox(
                "Relative formula",
                ["Mean of query ratios", "Ratio of workload totals"],
                key="summary_rel_formula",
                disabled=summary_view == "Absolute totals",
                help=(
                    "Mean of query ratios = average_i(algorithm_i / baseline_i). "
                    "Ratio of workload totals = sum_i algorithm_i / sum_i baseline_i."
                ),
            )

        if summary_view == "Relative to baseline":
            if not rel_baseline:
                st.info("Relative view needs a GEQO or DP baseline in this run.")
            else:
                formula_key = (
                    "ratio_of_sums"
                    if rel_formula == "Ratio of workload totals"
                    else "mean_query_ratios"
                )
                formula_text = (
                    f"`Σ algorithm_metric / Σ {rel_baseline}_metric`"
                    if formula_key == "ratio_of_sums"
                    else f"`average over queries(algorithm_metric / {rel_baseline}_metric)`"
                )
                st.caption(
                    f"{len(df)} runs · relative bar = {formula_text}; lower is better, "
                    "`1.0` equals the selected baseline."
                    + (
                        ""
                        if summary_mean_only
                        else "  Range = min–max per-query ratio; ♦ = median per-query ratio."
                    )
                )
                rel_metrics = [
                    (m, f"{METRIC_LABELS[m]} / {rel_baseline.upper()}", False, True)
                    for m in ("mcts_best_cost", "e2e_time_ms", "exec_time_ms", "plan_time_ms")
                    if (
                        (m in df.columns and df[m].notna().any())
                        or (m == "mcts_best_cost" and "plan_cost" in df.columns and df["plan_cost"].notna().any())
                    )
                ]
                if not rel_metrics:
                    st.info("No cost, E2E, execution-time, or planning-time metrics have data for relative view.")
                else:
                    n = len(rel_metrics)
                    n_cols = 2 if n > 1 else 1
                    n_rows = (n + n_cols - 1) // n_cols
                    big = psub.make_subplots(
                        rows=n_rows,
                        cols=n_cols,
                        subplot_titles=[m[1] for m in rel_metrics]
                        + [""] * (n_rows * n_cols - n),
                        horizontal_spacing=0.12,
                        vertical_spacing=min(0.16, 0.1 + 0.03 * max(0, n_rows - 2)),
                    )
                    for i, (metric, _, _, zoom) in enumerate(rel_metrics):
                        r, c = (i // n_cols) + 1, (i % n_cols) + 1
                        rel_df = lib.relative_workload(view_df, metric, rel_baseline, formula_key)
                        sub_fig = _plotly_panel(
                            rel_df,
                            "relative",
                            title="",
                            log_y=False,
                            zoom=zoom,
                            mean_only=summary_mean_only,
                            sort_metric="relative",
                            y_scale=summary_y_scale,
                        )
                        for trace in sub_fig.data:
                            big.add_trace(trace, row=r, col=c)
                        big.update_yaxes(
                            title_text=f"{metric} / {rel_baseline}",
                            row=r,
                            col=c,
                        )
                        _apply_summary_yaxis(
                            big,
                            rel_df,
                            "relative",
                            row=r,
                            col=c,
                            mean_only=summary_mean_only,
                            log_y=False,
                            zoom=zoom,
                            y_scale=summary_y_scale,
                        )
                        big.add_hline(
                            y=1.0,
                            line_dash="dot",
                            line_color="#666",
                            row=r,
                            col=c,
                        )
                    big.update_annotations(font_size=12)
                    big.update_xaxes(showticklabels=False, ticks="")
                    big.update_layout(
                        height=340 * n_rows,
                        margin={"l": 40, "r": 10, "t": 40, "b": 45},
                        showlegend=False,
                        bargap=0.18,
                    )
                    st.plotly_chart(
                        big,
                        use_container_width=True,
                        theme="streamlit",
                        config=PLOTLY_CONFIG,
                    )
        else:
            st.caption(
                f"{len(df)} runs · {n_q} queries · bar = Σ mean; "
                + (
                    ""
                    if summary_mean_only
                    else "range = best–worst whole-seed workload total; ♦ = median seed.  "
                )
                + "🟦 luby  🟧 single  🟩 dp  🟥 geqo  · hatched = agg=average."
            )
            all_metrics = [
                ("mcts_best_cost", METRIC_LABELS["mcts_best_cost"], False, True),
                ("e2e_time_ms", METRIC_LABELS["e2e_time_ms"], True, False),
                ("exec_time_ms", METRIC_LABELS["exec_time_ms"], True, False),
                ("plan_time_ms", METRIC_LABELS["plan_time_ms"], False, False),
                ("iters", METRIC_LABELS["iters"], False, False),
            ]

            def _has_data(metric: str) -> bool:
                col = f"{metric}__mean"
                return col in workload.columns and workload[col].notna().any()

            metrics = [m for m in all_metrics if _has_data(m[0])]
            if not metrics:
                st.info("No numeric metrics have data yet.")
            else:
                n = len(metrics)
                n_cols = 2 if n > 1 else 1
                n_rows = (n + n_cols - 1) // n_cols
                titles = [m[1] for m in metrics]
                titles += [""] * (n_rows * n_cols - n)
                v_space = min(0.16, 0.1 + 0.03 * max(0, n_rows - 2))
                big = psub.make_subplots(
                    rows=n_rows,
                    cols=n_cols,
                    subplot_titles=titles,
                    horizontal_spacing=0.1,
                    vertical_spacing=v_space,
                )
                for i, (metric, _title, log_y, zoom) in enumerate(metrics):
                    r, c = (i // n_cols) + 1, (i % n_cols) + 1
                    sub_fig = _plotly_panel(
                        workload,
                        metric,
                        title="",
                        log_y=log_y,
                        zoom=zoom,
                        mean_only=summary_mean_only,
                        sort_metric=summary_sort_metric,
                        y_scale=summary_y_scale,
                    )
                    for trace in sub_fig.data:
                        big.add_trace(trace, row=r, col=c)
                    _apply_summary_yaxis(
                        big,
                        workload,
                        metric,
                        row=r,
                        col=c,
                        mean_only=summary_mean_only,
                        log_y=log_y,
                        zoom=zoom,
                        y_scale=summary_y_scale,
                    )
                big.update_annotations(font_size=12)
                big.update_xaxes(showticklabels=False, ticks="")
                big.update_layout(
                    height=340 * n_rows,
                    margin={"l": 40, "r": 10, "t": 40, "b": 45},
                    showlegend=False,
                    bargap=0.18,
                )
                st.plotly_chart(
                    big, use_container_width=True, theme="streamlit", config=PLOTLY_CONFIG
                )


# --- winners tab ---


def _seed_aggregate(series: pd.Series, how: str) -> float | None:
    s = series.dropna()
    if s.empty:
        return None
    if how == "min":
        return float(s.min())
    if how == "mean":
        return float(s.mean())
    return float(s.median())


def _metric_series(frame: pd.DataFrame, metric: str) -> pd.Series:
    if metric == "mcts_best_cost":
        if "mcts_best_cost" in frame.columns:
            values = frame["mcts_best_cost"]
        else:
            values = pd.Series(pd.NA, index=frame.index)
        if "plan_cost" in frame.columns:
            values = values.fillna(frame["plan_cost"])
        return values
    if metric in frame.columns:
        return frame[metric]
    return pd.Series(pd.NA, index=frame.index)


def _winner_mode_table(
    df: pd.DataFrame, metrics: list[str], tie_band: float, seed_how: str
) -> pd.DataFrame:
    query_keys = ["query"]
    if "source" in df.columns:
        query_keys = ["source", "query"]
    cell_keys = ["config", "reward", "agg", "combo_id"]
    labels = {
        "mcts_best_cost": "COST",
        "exec_time_ms": "RUNTIME",
        "e2e_time_ms": "E2E",
        "plan_time_ms": "PLAN",
    }
    rows: dict[str, dict[str, int]] = {}
    tie_hi = 1.0 + tie_band / 100.0

    for metric in metrics:
        col = labels.get(metric, metric.upper())
        sub = df.copy()
        sub["__winner_metric"] = _metric_series(sub, metric)
        sub = sub[sub["__winner_metric"].notna()]
        if sub.empty:
            continue
        per_cell = (
            sub.groupby(query_keys + cell_keys, dropna=False)["__winner_metric"]
            .agg(lambda s: _seed_aggregate(s, seed_how))
            .reset_index()
            .rename(columns={"__winner_metric": "value"})
        )
        per_cell["mode"] = per_cell.apply(
            lambda r: (
                str(r["config"]).upper()
                if str(r["config"]) in BASELINE_CONFIGS
                else lib.short_label(
                    r["config"], r["reward"], r["agg"], r.get("combo_id", "") or ""
                )
            ),
            axis=1,
        )

        for mode in per_cell["mode"].dropna().astype(str).unique():
            rows.setdefault(f"{mode} WINS", {})
            rows[f"{mode} WINS"].setdefault(col, 0)

        for _, qdf in per_cell.groupby(query_keys, dropna=False):
            qdf = qdf[qdf["value"].notna()]
            if qdf.empty:
                continue
            best = float(qdf["value"].min())
            if not np.isfinite(best):
                continue
            winners = qdf[qdf["value"] <= best * tie_hi]
            for mode in winners["mode"].dropna().astype(str).unique():
                rows.setdefault(f"{mode} WINS", {})
                rows[f"{mode} WINS"][col] = rows[f"{mode} WINS"].get(col, 0) + 1

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame.from_dict(rows, orient="index").fillna(0).astype(int)
    preferred = ["DP WINS", "GEQO WINS"]
    rest = sorted([idx for idx in out.index if idx not in preferred])
    out = out.reindex([idx for idx in preferred if idx in out.index] + rest)
    out.index.name = "winner"
    return out.reset_index()


@st.fragment
def _render_winners():
    mcts_rows = df[~df["config"].isin(BASELINE_CONFIGS)]
    baseline_rows = df[df["config"].isin(BASELINE_CONFIGS)]

    if mcts_rows.empty:
        st.info("No MCTS cells in this run.")
    elif baseline_rows.empty:
        st.info("No baseline cells (dp/geqo) — Winners view needs at least one.")
    else:
        st.caption(
            "Per-query head-to-head: MCTS vs each baseline.  A query counts as a tie if the "
            "MCTS metric is within ±X% of the baseline."
        )
        winner_views = st.multiselect(
            "Show",
            ["Mode winner counts", "MCTS vs baseline", "Mode ranking"],
            default=["Mode winner counts", "MCTS vs baseline", "Mode ranking"],
            key="winners_show_sections",
        )

        # Distinct MCTS cells for the selector.
        mcts_cell_combos = (
            mcts_rows[["config", "reward", "agg", "combo_id"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        cell_labels = [
            lib.short_label(r["config"], r["reward"], r["agg"], r.get("combo_id", "") or "")
            for _, r in mcts_cell_combos.iterrows()
        ]

        c_mode, c_metric, c_tie, c_seed = st.columns([3, 2, 1, 1])
        with c_mode:
            mode = st.radio(
                "MCTS candidate",
                ["Best per query (any MCTS)", "Pick one MCTS cell"],
                key="win_mode",
                horizontal=False,
            )
            chosen_combo = None
            chosen_label = "best MCTS"
            if mode == "Pick one MCTS cell":
                sel = st.selectbox(
                    "Cell",
                    options=range(len(cell_labels)),
                    format_func=lambda i: cell_labels[i],
                    key="win_cell",
                )
                chosen_combo = mcts_cell_combos.iloc[sel]
                chosen_label = cell_labels[sel]
        with c_metric:
            metric_choices = [
                m
                for m in ("exec_time_ms", "e2e_time_ms", "plan_time_ms", "mcts_best_cost")
                if m in df.columns and df[m].notna().any()
            ]
            if not metric_choices:
                st.warning("No numeric metrics with data.")
                st.stop()
            metric = st.selectbox(
                "Metric",
                metric_choices,
                index=metric_choices.index("e2e_time_ms") if "e2e_time_ms" in metric_choices else 0,
                format_func=lambda k: METRIC_LABELS.get(k, k),
                key="win_metric",
            )
        with c_tie:
            tie_band = st.number_input(
                "Tie ±%", 0.0, 50.0, 5.0, 0.5, key="win_tie",
                help="Within this band of the baseline counts as a tie.",
            )
        with c_seed:
            seed_how = st.selectbox(
                "Across seeds",
                ["mean", "median", "min"],
                key="win_seed_agg",
            )

        table_metric_choices = [
            m
            for m in ("mcts_best_cost", "exec_time_ms", "e2e_time_ms", "plan_time_ms")
            if (
                (m in df.columns and df[m].notna().any())
                or (
                    m == "mcts_best_cost"
                    and "plan_cost" in df.columns
                    and df["plan_cost"].notna().any()
                )
            )
        ]
        if table_metric_choices and "Mode winner counts" in winner_views:
            st.markdown("**Mode winner counts**")
            st.caption(
                f"For each query, every cell is aggregated over seeds with `{seed_how}`, "
                "the minimum value is found separately for each metric column, and every used "
                "mode inside the tie band of that metric's minimum gets one win. "
                "Ties can make several modes win the same query."
            )
            mode_table = _winner_mode_table(df, table_metric_choices, tie_band, seed_how)
            st.dataframe(mode_table, hide_index=True, use_container_width=True)
            st.download_button(
                "💾 mode_winners.csv",
                data=mode_table.to_csv(index=False).encode(),
                file_name=f"{selected_run.name}__mode_winners.csv",
                mime="text/csv",
            )

        all_queries = sorted(df["query"].unique())

        # MCTS value per query
        if mode == "Best per query (any MCTS)":
            mcts_per_q = {
                q: _seed_aggregate(mcts_rows.loc[mcts_rows["query"] == q, metric], "min")
                for q in all_queries
            }
        else:
            mask = (
                (mcts_rows["config"] == chosen_combo["config"])
                & (mcts_rows["reward"] == chosen_combo["reward"])
                & (mcts_rows["agg"] == chosen_combo["agg"])
                & (mcts_rows["combo_id"] == (chosen_combo.get("combo_id", "") or ""))
            )
            sub = mcts_rows[mask]
            mcts_per_q = {
                q: _seed_aggregate(sub.loc[sub["query"] == q, metric], seed_how)
                for q in all_queries
            }

        tie_lo = 1.0 - tie_band / 100.0
        tie_hi = 1.0 + tie_band / 100.0

        results: list[dict] = []
        breakdown_rows: list[dict] = []
        for baseline in sorted(baseline_rows["config"].unique()):
            base_sub = baseline_rows[baseline_rows["config"] == baseline]
            better = same = worse = 0
            for q in all_queries:
                m = mcts_per_q.get(q)
                b = _seed_aggregate(base_sub.loc[base_sub["query"] == q, metric], seed_how)
                if m is None or b is None or b == 0:
                    verdict = "missing"
                else:
                    ratio = m / b
                    if ratio < tie_lo:
                        verdict = "better"
                        better += 1
                    elif ratio > tie_hi:
                        verdict = "worse"
                        worse += 1
                    else:
                        verdict = "same"
                        same += 1
                    breakdown_rows.append(
                        {
                            "query": q,
                            "baseline": baseline,
                            f"mcts_{metric}": m,
                            f"base_{metric}": b,
                            "ratio_mcts/base": round(ratio, 3),
                            "verdict": verdict,
                        }
                    )
            results.append(
                {"baseline": baseline, "better": better, "same": same, "worse": worse}
            )

        n_compared = max(
            (r["better"] + r["same"] + r["worse"] for r in results), default=0
        )

        colors = {"better": "#3a8fdc", "same": "#b8b8b8", "worse": "#e85a5a"}
        if "MCTS vs baseline" in winner_views:
            # Stacked bar chart.
            x = [f"vs {r['baseline'].upper()}" for r in results]
            fig = go.Figure()
            for cat in ("better", "same", "worse"):
                ys = [r[cat] for r in results]
                fig.add_bar(
                    name=cat,
                    x=x,
                    y=ys,
                    marker_color=colors[cat],
                    text=[str(y) if y else "" for y in ys],
                    textposition="inside",
                    textfont={"color": "white", "size": 14},
                    hovertemplate=f"%{{x}}<br>{cat}: %{{y}}<extra></extra>",
                )
            fig.update_layout(
                barmode="stack",
                title={
                    "text": (
                        f"{chosen_label}  ·  metric={METRIC_LABELS.get(metric, metric)}  ·  "
                        f"tie ±{tie_band:g}%  ·  n={n_compared}"
                    ),
                    "font": {"size": 13},
                },
                yaxis_title=f"Queries (n={n_compared})",
                height=440,
                margin={"l": 60, "r": 10, "t": 60, "b": 40},
                legend={
                    "orientation": "v", "yanchor": "middle", "y": 0.5,
                    "xanchor": "left", "x": 1.02,
                },
            )
            st.plotly_chart(
                fig, use_container_width=True, theme="streamlit", config=PLOTLY_CONFIG
            )

            with st.expander(f"Per-query breakdown ({len(breakdown_rows)} rows)"):
                if breakdown_rows:
                    bdf = pd.DataFrame(breakdown_rows).sort_values(
                        ["baseline", "verdict", "query"]
                    )
                    st.dataframe(bdf, hide_index=True, use_container_width=True)
                    st.download_button(
                        "💾 winners_breakdown.csv",
                        data=bdf.to_csv(index=False).encode(),
                        file_name=f"{selected_run.name}__winners.csv",
                        mime="text/csv",
                    )

        baseline_names = sorted(baseline_rows["config"].unique())
        if "Mode ranking" in winner_views:
            # --- Per-mode win-rate ranking ---
            st.divider()
            st.markdown(
                "**Mode ranking** — every MCTS cell scored against all baselines combined.  "
                "Bars are sorted by win-rate (best mode on top)."
            )
            ranking_rows: list[dict] = []
            for _, combo in mcts_cell_combos.iterrows():
                cid = combo.get("combo_id", "") or ""
                cell_sub = mcts_rows[
                    (mcts_rows["config"] == combo["config"])
                    & (mcts_rows["reward"] == combo["reward"])
                    & (mcts_rows["agg"] == combo["agg"])
                    & (mcts_rows["combo_id"] == cid)
                ]
                label = lib.short_label(combo["config"], combo["reward"], combo["agg"], cid)
                pq_mcts = {
                    q: _seed_aggregate(cell_sub.loc[cell_sub["query"] == q, metric], seed_how)
                    for q in all_queries
                }

                total_b = total_s = total_w = 0
                per_base: dict[str, tuple[int, int, int]] = {}
                for baseline in baseline_names:
                    base_sub = baseline_rows[baseline_rows["config"] == baseline]
                    b = s = w = 0
                    for q in all_queries:
                        m = pq_mcts.get(q)
                        bv = _seed_aggregate(base_sub.loc[base_sub["query"] == q, metric], seed_how)
                        if m is None or bv is None or bv == 0:
                            continue
                        ratio = m / bv
                        if ratio < tie_lo:
                            b += 1
                        elif ratio > tie_hi:
                            w += 1
                        else:
                            s += 1
                    per_base[baseline] = (b, s, w)
                    total_b += b
                    total_s += s
                    total_w += w

                total = total_b + total_s + total_w
                win_rate = (total_b / total) if total else 0.0
                row = {
                    "label": label,
                    "config": combo["config"],
                    "better": total_b,
                    "same": total_s,
                    "worse": total_w,
                    "total": total,
                    "win_rate": win_rate,
                }
                for base, (b, s, w) in per_base.items():
                    row[f"better_vs_{base}"] = b
                    row[f"same_vs_{base}"] = s
                    row[f"worse_vs_{base}"] = w
                ranking_rows.append(row)

            # Sort: highest win-rate first (and ties broken by fewer losses).
            ranking_rows.sort(key=lambda r: (-r["win_rate"], r["worse"]))

            labels = [r["label"] for r in ranking_rows]
            fig_rank = go.Figure()
            for cat in ("better", "same", "worse"):
                vals = [r[cat] for r in ranking_rows]
                fig_rank.add_bar(
                    name=cat,
                    x=vals,
                    y=labels,
                    orientation="h",
                    marker_color=colors[cat],
                    text=[str(v) if v else "" for v in vals],
                    textposition="inside",
                    textfont={"color": "white", "size": 12},
                    hovertemplate=f"%{{y}}<br>{cat}: %{{x}}<extra></extra>",
                )
            fig_rank.update_layout(
                barmode="stack",
                title={
                    "text": (
                        f"Win/tie/loss across {len(baseline_names)} baseline(s) — "
                        f"metric={METRIC_LABELS.get(metric, metric)}, tie ±{tie_band:g}%"
                    ),
                    "font": {"size": 12},
                },
                xaxis_title=f"Queries (summed across {len(baseline_names)} baseline(s))",
                yaxis={"autorange": "reversed"},  # best on top
                height=max(280, 28 * len(labels) + 90),
                margin={"l": 220, "r": 10, "t": 60, "b": 50},
                legend={
                    "orientation": "v", "yanchor": "middle", "y": 0.5,
                    "xanchor": "left", "x": 1.02,
                },
            )
            st.plotly_chart(
                fig_rank, use_container_width=True, theme="streamlit", config=PLOTLY_CONFIG
            )

            rank_df = pd.DataFrame(ranking_rows)
            rank_df["win_rate %"] = (rank_df["win_rate"] * 100).round(1)
            display_cols = ["label", "better", "same", "worse", "total", "win_rate %"] + [
                c for c in rank_df.columns if c.startswith(("better_vs_", "worse_vs_"))
            ]
            st.dataframe(
                rank_df[display_cols], hide_index=True, use_container_width=True
            )
            st.download_button(
                "💾 mode_ranking.csv",
                data=rank_df.to_csv(index=False).encode(),
                file_name=f"{selected_run.name}__mode_ranking.csv",
                mime="text/csv",
            )


# --- trajectory tab ---

@st.fragment
def _render_traj():
    # Hide metrics that have no data anywhere in this run (e.g. `iters`
    # for all-baseline runs).
    traj_metric_options = [m for m in METRIC_LABELS if m in df.columns and df[m].notna().any()]
    if not traj_metric_options:
        st.info("No numeric metrics with data yet.")
        st.stop()
    metric_choice = st.selectbox(
        "Metric",
        options=traj_metric_options,
        format_func=lambda k: METRIC_LABELS[k],
        index=0,
        key="traj_metric",
    )

    # Rank queries by per-query cross-cell variance on the chosen metric.
    # Most "interesting" = biggest spread between cells & seeds.
    queries_all = sorted(df["query"].unique())
    spread_by_q = {}
    for q in queries_all:
        s = df[df["query"] == q][metric_choice].dropna()
        if len(s) >= 2:
            spread_by_q[q] = float(s.max() - s.min())
        else:
            spread_by_q[q] = 0.0
    queries_by_spread = sorted(queries_all, key=lambda q: -spread_by_q[q])

    col_q, col_summary = st.columns([1, 2])
    with col_q:
        query_choice = st.selectbox(
            "Query (sorted by max-min spread, most interesting first)",
            options=queries_by_spread,
            format_func=lambda q: f"{q}  (Δ={spread_by_q[q]:.2f})",
            index=0,
            key="traj_query",
        )
    with col_summary:
        st.caption(
            f"Spread = max−min of `{metric_choice}` over all (cell, seed) pairs for the query. "
            "Larger spread → cells disagree more (Luby's seed randomness matters there). "
            "Flat queries are typically tree-exhaustion cases."
        )

    sub = df[df["query"] == query_choice].copy()
    if sub.empty:
        st.info("No data for this query yet.")
    else:
        fig = go.Figure()
        for (config, reward, agg, cid), g in sub.groupby(["config", "reward", "agg", "combo_id"]):
            g = g.sort_values("seed")
            label = lib.short_label(config, reward, agg, cid)
            dash = "solid" if config == "luby" else "dash"
            color = "#1f77b4" if config == "luby" else "#ff7f0e"
            # Cell flavor by line/marker: same color for all luby/single, but
            # we still vary marker + line opacity via the label.
            fig.add_scatter(
                x=g["seed"],
                y=g[metric_choice],
                mode="lines+markers",
                name=label,
                line={"dash": dash, "width": 1.5, "color": color},
                marker={"size": 6},
                hovertemplate=(
                    f"{label}<br>seed: %{{x}}<br>{metric_choice}: %{{y:.2f}}<extra></extra>"
                ),
            )
        fig.update_layout(
            title={
                "text": f"{METRIC_LABELS[metric_choice]}  ·  query={query_choice}",
                "font": {"size": 12},
            },
            xaxis_title="seed",
            yaxis_title=metric_choice,
            yaxis={"tickformat": ",.6~g"},  # no SI offset
            legend={"orientation": "h", "yanchor": "bottom", "y": -0.25},
            margin={"l": 50, "r": 10, "t": 40, "b": 80},
            height=420,
        )
        st.plotly_chart(
            fig, use_container_width=True, theme="streamlit", config=PLOTLY_CONFIG
        )

        # Per-cell variance table on the chosen metric (for this query only).
        stats = (
            sub.groupby(["config", "reward", "agg", "combo_id"])[metric_choice]
            .agg(n="count", mean="mean", std="std", min="min", max="max")
            .reset_index()
        )
        stats["cell"] = stats.apply(
            lambda r: lib.short_label(r["config"], r["reward"], r["agg"], r.get("combo_id", "")),
            axis=1,
        )
        stats["spread"] = stats["max"] - stats["min"]
        stats = stats[["cell", "n", "mean", "std", "min", "max", "spread"]]
        stats = stats.sort_values("mean").reset_index(drop=True)
        st.dataframe(
            stats.style.format(dict.fromkeys(("mean", "std", "min", "max", "spread"), "{:.2f}")),
            hide_index=True,
        )


# --- image tab ---

@st.fragment
def _render_image():
    st.caption(
        "Static matplotlib snapshot suitable for paper figures / PNG export.  "
        "Click **Render** to (re)build — auto-refresh does not redraw this tab."
    )
    cols_img = st.columns([1, 1, 4])
    with cols_img[0]:
        render_clicked = st.button("📷 Render", type="primary", key="img_render_btn")
    with cols_img[1]:
        download_slot = st.empty()

    # Cache key combines the run name and the row count so that a "Render"
    # click on the same data reuses the prior image.
    cache_key = f"_img::{selected_run.name}::{len(df)}"

    if render_clicked or cache_key in st.session_state:
        workload = _agg_workload(df)
        if workload.empty:
            st.info("No data to render yet.")
        else:
            if cache_key not in st.session_state or render_clicked:
                import io

                from matplotlib.lines import Line2D
                from matplotlib.patches import Patch

                fig, axes = plt.subplots(3, 2, figsize=(11, 12), dpi=130)
                panel(
                    axes[0, 0],
                    workload,
                    "mcts_best_cost",
                    METRIC_LABELS["mcts_best_cost"],
                    zoom_pad_frac=0.15,
                )
                panel(axes[0, 1], workload, "e2e_time_ms", METRIC_LABELS["e2e_time_ms"], log_y=True)
                panel(
                    axes[1, 0], workload, "exec_time_ms", METRIC_LABELS["exec_time_ms"], log_y=True
                )
                panel(axes[1, 1], workload, "plan_time_ms", METRIC_LABELS["plan_time_ms"])
                panel(axes[2, 0], workload, "iters", METRIC_LABELS["iters"])
                axes[2, 1].axis("off")
                legend = [
                    Patch(facecolor="#1f77b4", alpha=0.5, edgecolor="black", label="config=luby"),
                    Patch(facecolor="#ff7f0e", alpha=0.5, edgecolor="black", label="config=single"),
                    Patch(facecolor="white", edgecolor="black", hatch="//", label="agg=average"),
                    Patch(facecolor="white", edgecolor="black", label="agg=best"),
                    Line2D(
                        [0],
                        [0],
                        marker="D",
                        color="black",
                        linestyle="",
                        markersize=5,
                        label="Σ median",
                    ),
                    Line2D(
                        [0],
                        [0],
                        color="black",
                        marker="_",
                        markersize=8,
                        linestyle="-",
                        label="Σ best – Σ worst",
                    ),
                ]
                fig.legend(
                    handles=legend,
                    loc="lower center",
                    ncol=6,
                    fontsize=8,
                    frameon=False,
                    bbox_to_anchor=(0.5, -0.01),
                )
                fig.suptitle(selected_run.name, fontsize=11)
                fig.tight_layout(rect=[0, 0.03, 1, 0.96])
                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight")
                plt.close(fig)
                # Evict any older cache entries for this run to save memory.
                for k in list(st.session_state):
                    if k.startswith(f"_img::{selected_run.name}::"):
                        del st.session_state[k]
                st.session_state[cache_key] = buf.getvalue()
            png_bytes = st.session_state[cache_key]
            st.image(
                png_bytes,
                caption=f"{selected_run.name}  ({len(df)} runs)",
                use_container_width=False,
            )
            download_slot.download_button(
                "💾 PNG",
                data=png_bytes,
                file_name=f"{selected_run.name}.png",
                mime="image/png",
                key="img_dl_btn",
            )
    else:
        st.info("Press **Render** to generate a snapshot from the current data.")


# --- timeouts tab ---

@st.fragment
def _render_timeouts():
    st.caption("Runs that hit `statement_timeout` (5 min cap in the SQL templates).")
    timeouts = df[df["timed_out"]].copy() if not df.empty else df
    if timeouts.empty:
        st.success("No timeouts in this run.")
    else:
        timeouts["cell"] = timeouts.apply(
            lambda r: lib.short_label(r["config"], r["reward"], r["agg"], r.get("combo_id", "")),
            axis=1,
        )

        by_cell = (
            timeouts.groupby(["config", "reward", "agg"]).size().reset_index(name="n_timeouts")
        )
        by_cell["cell"] = by_cell.apply(
            lambda r: lib.short_label(r["config"], r["reward"], r["agg"], r.get("combo_id", "")),
            axis=1,
        )
        by_cell = by_cell.sort_values("n_timeouts", ascending=False)
        st.markdown("**Per-cell counts**")
        st.dataframe(
            by_cell[["cell", "config", "reward", "agg", "n_timeouts"]],
            hide_index=True,
            use_container_width=True,
        )

        by_query = (
            timeouts.groupby("query")
            .size()
            .reset_index(name="n_timeouts")
            .sort_values("n_timeouts", ascending=False)
        )
        st.markdown("**Per-query counts**")
        st.dataframe(by_query, hide_index=True, use_container_width=True)

        st.markdown("**All timed-out runs**")
        cols_show = ["cell", "query", "seed", "config", "reward", "agg", "plan_path"]
        cols_show = [c for c in cols_show if c in timeouts.columns]
        st.dataframe(
            timeouts[cols_show].sort_values(["cell", "query", "seed"]),
            hide_index=True,
            use_container_width=True,
        )

        st.download_button(
            "💾 timeouts.csv",
            data=timeouts[cols_show].to_csv(index=False).encode(),
            file_name=f"{selected_run.name}__timeouts.csv",
            mime="text/csv",
        )


# --- plans tab ---

def _fmt_ms(value) -> str:
    return f"{value:.1f} ms" if isinstance(value, (int, float)) else "—"


def _plan_tree_df(nodes: list[dict]) -> pd.DataFrame:
    """Flat pre-order table; depth shown as a box-drawing indent in `plan node`."""
    rows = []
    for node in nodes:
        depth = node["depth"]
        prefix = "│  " * (depth - 1) + "└─ " if depth else ""
        act, est = node["act_rows"], node["est_rows"]
        rows.append(
            {
                "plan node": prefix + node["op"],
                "self ms": round(node["self_ms"], 2),
                "total ms": round(node["total_ms"], 2),
                "loops": node["loops"],
                "act rows": act,
                "est rows": est,
                "act/est": round(act / est, 1) if (act is not None and est) else None,
                "rows removed": node.get("rows_removed", 0),
                "cost": node["cost_total"],
            }
        )
    return pd.DataFrame(rows)


def _cell_dirname(row) -> str:  # mirror of sweep.cell_dirname
    if row["config"] in BASELINE_CONFIGS:
        return row["config"]
    base = f"{row['config']}__{row['reward']}__{row['agg']}"
    cid = row.get("combo_id", "") or ""
    return f"{base}__{cid}" if cid else base


_PLAN_PARSE_VERSION = 2


@st.cache_data(show_spinner=False)
def _parse_plan_cached(path_str: str, mtime: float, parse_version: int) -> tuple:
    """Cached on (path, mtime, parse_version): (raw_text, parse_explain meta, nodes)."""
    text = Path(path_str).read_text()
    return text, lib.parse_explain(text), lib.parse_plan_tree(text)


def _load_plan(path: Path):
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    return _parse_plan_cached(str(path), mtime, _PLAN_PARSE_VERSION)


def _cost_str(meta: dict) -> str:
    return f"{meta['plan_cost']:.0f}" if "plan_cost" in meta else "—"


def _cell_label_map() -> dict:
    """Map each cell dirname to its compact short label."""
    return {
        _cell_dirname(row): lib.short_label(
            row["config"], row["reward"], row["agg"], row.get("combo_id", "") or ""
        )
        for _, row in df.iterrows()
    }


def _plan_summary_row(label: str, path: Path) -> dict:
    loaded = _load_plan(path)
    if loaded is None:
        return {"cell": label, "nodes": 0}
    _text, meta, nodes = loaded
    slow = max(nodes, key=lambda node: node["self_ms"], default=None)
    return {
        "cell": label,
        "exec ms": round(meta["exec_time_ms"], 1) if "exec_time_ms" in meta else None,
        "plan ms": round(meta["plan_time_ms"], 1) if "plan_time_ms" in meta else None,
        "top cost": round(meta["plan_cost"]) if "plan_cost" in meta else None,
        "nodes": len(nodes),
        "param": sum(1 for node in nodes if (node["loops"] or 1) > 1),
        "rows removed": sum(node.get("rows_removed", 0) for node in nodes),
        "slowest node": slow["op"] if slow else None,
        "slowest ms": round(slow["self_ms"], 1) if slow else None,
    }


def _node_indent(node) -> str:
    if node is None:
        return ""
    depth = node["depth"]
    prefix = "│  " * (depth - 1) + "└─ " if depth else ""
    return prefix + node["op"]


def _join_tree_text(nodes: list[dict]) -> str:
    """Indented join-order tree (⋈ join / leaf aliases), mirroring the Leading hint."""
    rows = []
    for depth, label in lib.join_order_tree(nodes):
        prefix = "│  " * (depth - 1) + "└─ " if depth else ""
        rows.append(prefix + label)
    return "\n".join(rows) or "—"


def _node_metric(node, metric: str):
    if node is None:
        return None
    if metric == "loops":
        return node["loops"]
    if metric == "rows removed":
        return node.get("rows_removed", 0)
    if metric == "act rows":
        return node["act_rows"]
    if metric == "total ms":
        return round(node["total_ms"], 1)
    return round(node["self_ms"], 1)


def _render_aligned_compare(nodes_a: list[dict], nodes_b: list[dict], metric: str) -> None:
    """One row-aligned table for Plan A vs B — a single scrollbar, so both sides
    scroll together; diverging rows are highlighted and marked in the Δ column.
    `metric` chooses which per-node value to show (self ms / loops / rows removed / …)."""
    if not (nodes_a and nodes_b):
        st.info("Need two executable plans to compare side by side.")
        return
    marker = {"same": "", "diff": "≠", "a_only": "◀ A", "b_only": "B ▶"}
    table = pd.DataFrame(
        [
            {
                "A: plan node": _node_indent(na),
                f"A {metric}": _node_metric(na, metric),
                "Δ": marker[tag],
                f"B {metric}": _node_metric(nb, metric),
                "B: plan node": _node_indent(nb),
            }
            for na, nb, tag in lib.align_plans(nodes_a, nodes_b)
        ]
    )

    def _highlight(row):
        shade = "background-color: #fff3cd" if row["Δ"] else ""
        return [shade] * len(row)

    st.dataframe(
        table.style.apply(_highlight, axis=1),
        use_container_width=True,
        hide_index=True,
        height=min(680, 32 * len(table) + 40),
    )


def _render_scan_diff(nodes_a: list[dict], nodes_b: list[dict]) -> None:
    scans_a = lib.plan_scan_methods(nodes_a)
    scans_b = lib.plan_scan_methods(nodes_b)
    scan_rows = [
        {"relation": rel, "Plan A": scans_a.get(rel, "—"), "Plan B": scans_b.get(rel, "—")}
        for rel in sorted(set(scans_a) | set(scans_b))
        if scans_a.get(rel) != scans_b.get(rel)
    ]
    if scan_rows:
        st.caption(f"Different scan/operator choice on {len(scan_rows)} relation(s):")
        st.dataframe(pd.DataFrame(scan_rows), hide_index=True, use_container_width=True)
    else:
        st.caption("Same scan method on every shared relation.")


def _render_plan_table(nodes: list[dict]) -> None:
    """Full per-plan node table with self-ms heat — the rich single-plan view."""
    if not nodes:
        st.info("No executable plan tree.")
        return
    table = _plan_tree_df(nodes)
    st.dataframe(
        table.style.background_gradient(subset=["self ms"], cmap="Reds"),
        use_container_width=True,
        hide_index=True,
        height=min(640, 30 * len(table) + 40),
    )


def _render_compare_plans() -> None:
    cell_labels = sorted({_cell_dirname(row) for _, row in df.iterrows()})
    short = _cell_label_map()
    q_options = sorted(df["query"].astype(str).unique())
    c_q, c_seed = st.columns([2, 1])
    with c_q:
        q_choice = st.selectbox("Query", q_options, key="cmp_query") if q_options else None
    if not q_choice:
        st.info("No queries available.")
        return
    seed_stems = sorted(
        {p.stem for cl in cell_labels for p in (selected_run / cl / q_choice).glob("seed-*.plan")}
    )
    with c_seed:
        seed_stem = (
            st.selectbox("Seed", seed_stems, key=f"cmp_seed::{q_choice}") if seed_stems else None
        )
    if not seed_stem:
        st.info(f"No saved plans for query {q_choice}.")
        return
    avail = [
        cl for cl in cell_labels if (selected_run / cl / q_choice / f"{seed_stem}.plan").exists()
    ]
    if not avail:
        st.info("No plans for that query / seed.")
        return

    st.caption(f"{len(avail)} plans for **{q_choice}** ({seed_stem}) — sorted by exec time.")
    rows = [
        _plan_summary_row(short.get(cl, cl), selected_run / cl / q_choice / f"{seed_stem}.plan")
        for cl in avail
    ]
    cmp_df = pd.DataFrame(rows).sort_values("exec ms", na_position="last").reset_index(drop=True)
    grad = [c for c in ("exec ms", "top cost") if c in cmp_df.columns and cmp_df[c].notna().any()]
    st.dataframe(
        cmp_df.style.background_gradient(subset=grad, cmap="Reds") if grad else cmp_df,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("**Side-by-side (aligned — scrolls together)**")

    def _fmt(cl: str) -> str:
        return short.get(cl, cl)

    a_col, b_col = st.columns(2)
    with a_col:
        cell_a = st.selectbox("Plan A", avail, index=0, format_func=_fmt, key=f"cmp_a::{q_choice}")
    with b_col:
        b_index = min(1, len(avail) - 1)
        cell_b = st.selectbox(
            "Plan B", avail, index=b_index, format_func=_fmt, key=f"cmp_b::{q_choice}"
        )
    load_a = _load_plan(selected_run / cell_a / q_choice / f"{seed_stem}.plan")
    load_b = _load_plan(selected_run / cell_b / q_choice / f"{seed_stem}.plan")
    nodes_a = load_a[2] if load_a else []
    nodes_b = load_b[2] if load_b else []

    lead_a, lead_b = st.columns(2)
    lead_a.caption(f"A · {_fmt(cell_a)} — join order")
    lead_a.code(_join_tree_text(nodes_a))
    lead_b.caption(f"B · {_fmt(cell_b)} — join order")
    lead_b.code(_join_tree_text(nodes_b))

    cmp_metric = st.radio(
        "Side-by-side column",
        ["self ms", "loops", "rows removed", "act rows", "total ms"],
        horizontal=True,
        key=f"cmp_metric::{q_choice}",
        help="'loops' > 1 marks parameterized operators; 'rows removed' shows filter waste.",
    )
    _render_aligned_compare(nodes_a, nodes_b, cmp_metric)
    _render_scan_diff(nodes_a, nodes_b)

    with st.expander(f"📋 Plan A — full node tree ({_fmt(cell_a)})"):
        _render_plan_table(nodes_a)
    with st.expander(f"📋 Plan B — full node tree ({_fmt(cell_b)})"):
        _render_plan_table(nodes_b)


def _render_single_plan() -> None:
    cell_labels = sorted({_cell_dirname(row) for _, row in df.iterrows()})
    c1, c2, c3 = st.columns(3)
    with c1:
        cell = st.selectbox("Cell", cell_labels, key="plans_cell")
    cell_root = selected_run / cell
    q_options = (
        sorted(p.name for p in cell_root.iterdir() if p.is_dir())
        if cell_root.is_dir()
        else sorted(df["query"].astype(str).unique())
    )
    with c2:
        q_choice = (
            st.selectbox("Query", q_options, key=f"plans_query::{cell}") if q_options else None
        )
    with c3:
        plans = sorted((cell_root / q_choice).glob("seed-*.plan")) if q_choice else []
        seed_choice = (
            st.selectbox(
                "Seed", plans, format_func=lambda p: p.stem, key=f"plans_seed::{cell}::{q_choice}"
            )
            if plans
            else None
        )

    if not (seed_choice and Path(seed_choice).exists()):
        st.info("No saved plan for that combination.")
        return
    loaded = _load_plan(Path(seed_choice))
    if loaded is None:
        st.info("Couldn't read that plan.")
        return
    text, meta, nodes = loaded

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Execution", _fmt_ms(meta.get("exec_time_ms")))
    m2.metric("Planning", _fmt_ms(meta.get("plan_time_ms")))
    m3.metric("Top plan cost", _cost_str(meta))
    m4.metric("Plan nodes", len(nodes))

    if not nodes:
        st.warning("No executable plan tree (cost-only or timed-out capture) — raw text below.")
        st.code(text, language="text")
        return

    slow = max(nodes, key=lambda node: node["self_ms"])
    st.caption(f"⏱️ Slowest node by self-time: **{slow['op']}** — {slow['self_ms']:.1f} ms")
    st.caption("🔗 Join order (tree):")
    st.code(_join_tree_text(nodes))
    with st.expander("As pg_hint_plan Leading hint"):
        st.code(lib.join_order_leading(nodes) or "—", language="sql")

    table = _plan_tree_df(nodes)
    flt = st.text_input("Filter plan nodes", "", placeholder="e.g. Seq Scan, cast_info, Hash Join")
    view = table[table["plan node"].str.contains(flt, case=False, na=False)] if flt else table
    st.dataframe(
        view.style.background_gradient(subset=["self ms"], cmap="Reds"),
        use_container_width=True,
        hide_index=True,
        height=min(640, 38 * len(view) + 40),
    )
    with st.expander("Raw plan text"):
        st.code(text, language="text")


@st.fragment
def _render_plans():
    mode = st.radio(
        "Plans mode",
        ["Single plan", "Compare plans"],
        horizontal=True,
        key="plans_mode",
        label_visibility="collapsed",
    )
    if mode == "Compare plans":
        _render_compare_plans()
    else:
        _render_single_plan()


# --- render tabs (each body is an isolated st.fragment) ---

with tab_summary:
    _render_summary()
with tab_winners:
    _render_winners()
with tab_traj:
    _render_traj()
with tab_image:
    _render_image()
with tab_timeouts:
    _render_timeouts()
with tab_plans:
    _render_plans()


# Liveness during a running sweep is handled by the `_live_monitor` fragment
# above (polls every 2 s without rerunning the whole app).
