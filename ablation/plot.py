import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HEADER_RE = re.compile(
    r"^===== "
    r"(?:\[[^\]]*\]\s+)?"  # optional [cell N/M elapsed Xs] prefix
    r"(?:query=(?P<query>\S+)\s+)?"
    r"reward_mode=(?P<reward>\S+)\s+"
    r"uct_aggregation=(?P<agg>\S+)\s+"
    r"config=(?P<config>\S+)\s+"
    r"n=(?P<n>\d+) runs ====="
)
METRIC_RE = re.compile(
    r"^\s*(?P<name>[A-Za-z][\w ]*?)\s*:\s*"
    r"(?P<best>[\d.]+)\s*/\s*(?P<mean>[\d.]+)\s*/\s*"
    r"(?P<median>[\d.]+)\s*/\s*(?P<worst>[\d.]+)"
)


def parse(path):
    """Return list of section dicts."""
    sections = []
    cur = None
    for line in path.read_text().splitlines():
        m = HEADER_RE.match(line)
        if m:
            if cur:
                sections.append(cur)
            cur = {
                "query": m["query"],
                "reward": m["reward"],
                "agg": m["agg"],
                "config": m["config"],
                "n": int(m["n"]),
                "metrics": {},
            }
            continue
        if cur is None:
            continue
        m = METRIC_RE.match(line)
        if m:
            cur["metrics"][m["name"].strip()] = (
                float(m["best"]),
                float(m["mean"]),
                float(m["median"]),
                float(m["worst"]),
            )
    if cur:
        sections.append(cur)
    return sections


def aggregate(sections):
    groups = defaultdict(list)
    for s in sections:
        groups[(s["config"], s["reward"], s["agg"])].append(s)
    cells = []
    for (config, reward, agg), sub in groups.items():
        agg_metrics = {}
        metric_names = set()
        for s in sub:
            metric_names.update(s["metrics"].keys())
        for metric in metric_names:
            tuples = [s["metrics"][metric] for s in sub if metric in s["metrics"]]
            if not tuples:
                continue
            arr = np.array(tuples)  # shape (n_queries, 4)
            totals = arr.sum(axis=0)
            agg_metrics[metric] = (
                float(totals[0]),
                float(totals[1]),
                float(totals[2]),
                float(totals[3]),
            )
        cells.append(
            {
                "config": config,
                "reward": reward,
                "agg_axis": agg,
                "n_queries": len(sub),
                "metrics": agg_metrics,
            }
        )
    return cells


def short_label(cell):
    cfg = "luby" if cell["config"] == "luby" else "sing"
    rw = (
        "nnl"
        if cell["reward"] == "norm_neg_log"
        else ("pr" if cell["reward"] == "phase_ratio" else cell["reward"][:6])
    )
    ag = "best" if cell["agg_axis"] == "best" else "avg"
    return f"{cfg}/{rw}/{ag}"


def panel(ax, cells, metric, title, log_y=False, zoom_pad_frac=None):
    rows = [(c, c["metrics"].get(metric)) for c in cells]
    rows = [r for r in rows if r[1] is not None]
    if not rows:
        ax.set_title(f"{title} (no data)")
        return
    mn = np.array([r[1][0] for r in rows])
    mean = np.array([r[1][1] for r in rows])
    med = np.array([r[1][2] for r in rows])
    mx = np.array([r[1][3] for r in rows])

    # Sort by mean ascending (best on left).
    order = np.argsort(mean)
    rows = [rows[i] for i in order]
    mn, mean, med, mx = mn[order], mean[order], med[order], mx[order]
    labels = [short_label(r[0]) for r in rows]

    x = np.arange(len(rows))
    colors = ["#1f77b4" if r[0]["config"] == "luby" else "#ff7f0e" for r in rows]
    hatches = ["" if r[0]["agg_axis"] == "best" else "//" for r in rows]

    bars = ax.bar(x, mean, color=colors, alpha=0.5, edgecolor="black", linewidth=0.8)
    for bar, h in zip(bars, hatches, strict=False):
        bar.set_hatch(h)
    yerr_lo = mean - mn
    yerr_hi = mx - mean
    ax.errorbar(
        x,
        mean,
        yerr=[yerr_lo, yerr_hi],
        fmt="none",
        ecolor="black",
        capsize=4,
        capthick=1,
        elinewidth=1,
    )
    ax.scatter(x, med, marker="D", color="black", zorder=5, s=18)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    if log_y:
        ax.set_yscale("log")
    if zoom_pad_frac is not None:
        lo, hi = mn.min(), mx.max()
        pad = (hi - lo) * zoom_pad_frac if hi > lo else max(abs(hi) * 0.01, 1.0)
        ax.set_ylim(lo - pad, hi + pad)


def main():
    log_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    sections = parse(log_path)
    if not sections:
        sys.exit(f"no sections parsed from {log_path}")
    cells = aggregate(sections)

    n_queries = max(c["n_queries"] for c in cells)
    is_multi = n_queries > 1

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    panel(
        axes[0, 0], cells, "MCTS Best Cost", "MCTS Best Cost (lower = better)", zoom_pad_frac=0.15
    )
    panel(axes[0, 1], cells, "Exec Time ms", "Exec Time ms (lower = better)", log_y=True)
    panel(axes[1, 0], cells, "Plan Time ms", "Plan Time ms (lower = better)")
    panel(axes[1, 1], cells, "MCTS Iters", "MCTS Iterations")

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    legend_handles = [
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
            label="Σ median over queries",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            marker="_",
            markersize=8,
            linestyle="-",
            label="Σ best – Σ worst over queries",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=6,
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )

    subtitle = (
        (
            f"each cell summed over {n_queries} queries; "
            "bar = Σ mean, range = Σ best – Σ worst, marker = Σ median"
        )
        if is_multi
        else ("bar = mean across runs; range = best–worst over runs")
    )
    fig.suptitle(
        f"Ablation comparison — {log_path.name}  (each panel sorted by mean, best→worst)\n"
        f"{subtitle}\n"
        "x-tick: config / reward_mode (nnl=norm_neg_log, pr=phase_ratio) / aggregation",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(
        f"saved {out_path}  ({len(cells)} cells, {n_queries} {'queries' if is_multi else 'query'})"
    )


if __name__ == "__main__":
    main()
