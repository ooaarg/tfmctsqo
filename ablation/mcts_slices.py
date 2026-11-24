import csv
import io
import math
import subprocess
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GREEN_D = "#1e8449"
RED_D = "#c0392b"
EXPLOIT_C = "#2e86c1"
EXPLORE_C = "#e67e22"
DP_C = "#16a085"


# ----------------------------------------------------------------------------
def parse_iter(csv_text):
    rows = []
    csv_text = csv_text.strip()
    if not csv_text:
        return rows
    for r in csv.DictReader(io.StringIO(csv_text)):
        rows.append({
            "phase": int(r["phase"]),
            "iteration": int(r["iteration"]),
            "phase_best": float(r["phase_best_cost"]) if r["phase_best_cost"] else None,
            "global_best": float(r["global_best_cost"]) if r["global_best_cost"] else None,
            "depth": int(r["depth"]),
        })
    return rows


def parse_phasesub(csv_text):
    rows = []
    csv_text = csv_text.strip()
    if not csv_text:
        return rows
    for r in csv.DictReader(io.StringIO(csv_text)):
        rows.append({"phase": int(r["phase"]),
                     "relids": r["relids"] or "",
                     "size": int(r["size"])})
    return rows


def _relset(n):
    return frozenset((n.left + " " + n.right).split())


def best_vs_iter(iters, out_png):
    if not iters:
        return None
    by_phase = defaultdict(list)
    for r in iters:
        by_phase[r["phase"]].append(r)
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap("viridis")
    phases = sorted(by_phase)
    for pi, ph in enumerate(phases):
        rs = sorted(by_phase[ph], key=lambda r: r["iteration"])
        xs = [r["iteration"] for r in rs]
        ys = [r["phase_best"] for r in rs]
        col = cmap(pi / max(1, len(phases) - 1))
        ax.step(xs, ys, where="post", color=col, lw=1.8,
                label=f"phase {ph}")
        # mark the iteration where this phase first hit its best
        good = [(x, y) for x, y in zip(xs, ys) if y is not None]
        if good:
            bx, by = min(good, key=lambda t: t[1])
            ax.scatter([bx], [by], color=col, s=45, zorder=5,
                       edgecolors="white", linewidths=0.6)
    ax.set_yscale("log")
    ax.set_xlabel("iteration within phase")
    ax.set_ylabel("best cost so far (within phase)")
    ax.legend(fontsize=8, ncol=2)
    ax.set_title("MCTS best cost vs iteration, per phase\n"
                 "(dot = where the phase found its best; far-right dot = it kept "
                 "improving late)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_png


def depth_per_iter(iters, out_png):
    if not iters:
        return None
    by_phase = defaultdict(list)
    for r in iters:
        by_phase[r["phase"]].append(r)
    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.get_cmap("viridis")
    phases = sorted(by_phase)
    for pi, ph in enumerate(phases):
        rs = sorted(by_phase[ph], key=lambda r: r["iteration"])
        xs = [r["iteration"] for r in rs]
        ys = [r["depth"] for r in rs]
        ax.plot(xs, ys, ".", color=cmap(pi / max(1, len(phases) - 1)),
                ms=5, label=f"phase {ph}", alpha=0.8)
    ax.set_xlabel("iteration within phase")
    ax.set_ylabel("join depth reached by selection")
    ax.legend(fontsize=8, ncol=2)
    ax.set_title("MCTS search depth per iteration\n"
                 "(rising to the full join count = tree is being driven to "
                 "complete plans; flat/low = shallow re-sampling)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_png


def _root_children(search):
    roots = [n for n in search.values() if n.parent_id is None]
    if not roots:
        return None, []
    root = roots[0]
    kids = [search[c] for c in root.children]
    return root, kids


def root_children(search, out_png):
    root, kids = _root_children(search)
    if not kids:
        return None
    kids = sorted(kids, key=lambda n: -n.visits)
    labels = [f"{{{n.left}}}⋈{{{n.right}}}" for n in kids]
    visits = [n.visits for n in kids]
    cost = [n.best_cost if n.best_cost is not None else np.nan for n in kids]
    chosen = [n.chosen for n in kids]

    fig, ax1 = plt.subplots(figsize=(max(8, len(kids) * 0.7), 6))
    x = np.arange(len(kids))
    bars = ax1.bar(x, visits, color=[GREEN_D if c else "#aeb6bf" for c in chosen],
                   zorder=3)
    ax1.set_ylabel("visits (UCT)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
    ax2 = ax1.twinx()
    ax2.plot(x, cost, "D", color=RED_D, ms=6, zorder=5, label="best cost")
    ax2.set_ylabel("best cost in subtree", color=RED_D)
    ax2.set_yscale("log")
    ax1.set_title("MCTS first join choices: visits (bars; green = chosen) vs "
                  "best cost found (red)\nIs attention concentrated on one branch, "
                  "and is it the cheap one?", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_png


def uct_decomposition(search, out_png, c_explore=1.4, gamma=0.5, agg="best"):
    root, kids = _root_children(search)
    if not kids or root.visits <= 0:
        return None
    rows = []
    for n in kids:
        if n.visits <= 0:
            continue
        q = (n.best_reward if agg == "best"
             else (n.sum_reward / n.visits if n.sum_reward is not None else 0.0))
        q = q if q is not None else 0.0
        explore = (c_explore * math.log(root.visits) / n.visits)
        explore = explore ** gamma if explore > 0 else 0.0
        rows.append((n, q, explore))
    if not rows:
        return None
    rows.sort(key=lambda t: -(t[1] + t[2]))
    labels = [f"{{{n.left}}}⋈{{{n.right}}}" for n, _, _ in rows]
    q = [r[1] for r in rows]
    e = [r[2] for r in rows]
    chosen = [n.chosen for n, _, _ in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(max(8, len(rows) * 0.7), 6))
    ax.bar(x, q, color=EXPLOIT_C, label="exploitation  q (value)", zorder=3)
    ax.bar(x, e, bottom=q, color=EXPLORE_C,
           label="exploration  (C·ln N/n)^γ", zorder=3)
    for xi, ch in zip(x, chosen):
        if ch:
            ax.text(xi, (q[xi] + e[xi]), "✓ chosen", ha="center", va="bottom",
                    fontsize=8, color=GREEN_D, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("UCT score = q + exploration")
    ax.legend(fontsize=8)
    ax.set_title(f"Why MCTS picked a branch: UCT decomposition (C={c_explore}, "
                 f"γ={gamma}, agg={agg})\nblue = value it had learned, orange = "
                 "exploration bonus", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_png



def overlap_with_dp(search, dp_spine, out_png):
    if not search or not dp_spine:
        return None
    explored = defaultdict(set)
    chosen = defaultdict(set)
    for n in search.values():
        rs = _relset(n)
        if len(rs) < 2:
            continue
        explored[len(rs)].add(rs)
        if n.chosen:
            chosen[len(rs)].add(rs)
    dp_by_size = defaultdict(set)
    for k in dp_spine:
        if len(k) >= 2:
            dp_by_size[len(k)].add(k)

    sizes = sorted(set(explored) | set(dp_by_size))
    dp_tot, dp_expl, dp_chosen = [], [], []
    for s in sizes:
        dp_tot.append(len(dp_by_size[s]))
        dp_expl.append(len(dp_by_size[s] & explored[s]))
        dp_chosen.append(len(dp_by_size[s] & chosen[s]))
    x = np.arange(len(sizes))
    w = 0.6
    fig, ax = plt.subplots(figsize=(max(8, len(sizes) * 1.1), 6))
    ax.bar(x, dp_tot, w, color="#d6dbdf", label="DP-optimal subplans", zorder=2)
    ax.bar(x, dp_expl, w, color=DP_C, label="...explored by MCTS", zorder=3)
    ax.bar(x, dp_chosen, w * 0.5, color=GREEN_D,
           label="...on MCTS winning spine", zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.set_xlabel("joinrel size")
    ax.set_ylabel("number of DP-optimal subplans")
    ax.legend(fontsize=8)
    ax.set_title("MCTS vs DP optimal subplans, by size\n"
                 "Did MCTS explore (and pick) the same building blocks DP proved "
                 "optimal?", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_png


def phase_reuse(phasesub, out_png):
    if not phasesub:
        return None
    phases = sorted({r["phase"] for r in phasesub})
    by_sub = defaultdict(set)
    size_of = {}
    for r in phasesub:
        by_sub[r["relids"]].add(r["phase"])
        size_of[r["relids"]] = r["size"]
    maxsize = max(size_of.values()) if size_of else 0
    subs = [s for s in by_sub if size_of[s] < maxsize]
    subs = sorted(subs, key=lambda s: (-len(by_sub[s]), size_of[s], s))
    subs = subs[:35]
    if not subs:
        return None
    M = np.zeros((len(subs), len(phases)))
    for i, s in enumerate(subs):
        for pj, ph in enumerate(phases):
            M[i, pj] = size_of[s] if ph in by_sub[s] else np.nan

    fig, ax = plt.subplots(figsize=(max(6, len(phases) * 1.1),
                                    0.34 * len(subs) + 1.5))
    im = ax.imshow(M, aspect="auto", cmap="viridis", interpolation="nearest",
                   vmin=2, vmax=max(2, maxsize))
    ax.set_xticks(range(len(phases)))
    ax.set_xticklabels([f"phase {p}" for p in phases])
    ax.set_yticks(range(len(subs)))
    ax.set_yticklabels(["{" + s + "}" for s in subs], fontsize=6)
    fig.colorbar(im, ax=ax, label="joinrel size", fraction=0.046, pad=0.04)
    ax.set_title("Subplan reuse across MCTS phases\n"
                 "(a filled row spanning many phases = a building block the "
                 "restarts keep rediscovering; gaps = phase found a different "
                 "clump)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_png


# ----------------------------------------------------------------------------
def fetch_dp_spine(psql, db, user, host, port, query_sql):
    """Run the query through standard DP and return the set of relid-sets on the
    DP optimal plan's spine, for the MCTS-vs-DP overlap graph.

    Self-contained: this is MCTS-side code, so it does its own psql call + parse
    and depends on nothing outside practical_mcts_qo.  The only requirement is
    that the standard-DP `plan_trace` extension is installed in the server; if it
    is not, this returns an empty set and the overlap graph is simply skipped."""
    script = "\n".join([
        "CREATE EXTENSION IF NOT EXISTS plan_trace;",
        "LOAD 'plan_trace';",
        "SET plan_trace.enabled = on;",
        "SET geqo = off;",
        r"\o /dev/null",
        f"EXPLAIN (COSTS on) {query_sql.strip().rstrip(';')};",
        r"\o",
        'COPY (SELECT relids,"left","right" FROM plan_trace_joins() '
        "ORDER BY id) TO STDOUT WITH (FORMAT csv, HEADER true);",
    ]) + "\n"
    proc = subprocess.run(
        [psql, "-h", host, "-p", str(port), "-d", db, "-U", user, "-X", "-q",
         "-v", "ON_ERROR_STOP=1", "-f", "-"],
        input=script, capture_output=True, text=True, check=False)
    out = (proc.stdout or "").strip()
    if not out:
        return set()

    # joinrels keyed by relid-set; walk the final plan tree (top = all rels)
    nodes = {}
    for r in csv.DictReader(io.StringIO(out)):
        if "relids" not in r or not r["relids"]:
            continue
        nodes[frozenset(r["relids"].split())] = (r.get("left") or "",
                                                 r.get("right") or "")
    if not nodes:
        return set()
    top = max(nodes, key=len)
    spine, stack = set(), [top]
    while stack:
        k = stack.pop()
        if k in spine:
            continue
        spine.add(k)
        left, right = nodes.get(k, ("", ""))
        for side in (left, right):
            if side:
                stack.append(frozenset(side.split()))
    return spine


def emit_all(search, iters, psql, conn, out):
    """conn = (db, user, host, port, query_sql).  Returns written paths."""
    written = []
    for fn in (best_vs_iter(iters, f"{out}_mcts_bestcost.png"),
               depth_per_iter(iters, f"{out}_mcts_depth.png"),
               root_children(search, f"{out}_mcts_rootkids.png"),
               uct_decomposition(search, f"{out}_mcts_uct.png")):
        if fn:
            written.append(fn)
    dp_spine = fetch_dp_spine(psql, *conn)
    fn = overlap_with_dp(search, dp_spine, f"{out}_mcts_dpoverlap.png")
    if fn:
        written.append(fn)
    return written
