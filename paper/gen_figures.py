#!/usr/bin/env python
"""Figures fig:outcomes and fig:results_detail of tfmctsqo.tex (black and white).

Draws MCTS-Extreme vs DPSize vs GEQO from the saved run data into
sigmod_industry/figures. Both figures are drawn at their printed size, so the
PDF is placed without scaling and font sizes below are the printed sizes:
  * outcomes_all.pdf   -> one column  (\\columnwidth, about 3.33 in)
  * results_detail.pdf -> two columns (\\textwidth,  about 7.0 in, figure*)

Metric definitions:
  * e2e = plan_time_ms + exec_time_ms ; exec = exec_time_ms
  * per-query value = MEAN over the 3 seeds (skipping timeouts/NaN)
  * outcome tie band = symmetric +/-5% on e2e
  * headline totals = per-seed workload sum, then MEAN over 3 seeds,
    over the apples-to-apples set (queries all three methods complete)
  * JOB-Complex query "12" excluded (MCTS times out 3/3, GEQO 1/3) -> n=29
"""
import json, collections
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
from matplotlib import font_manager
# Linux Libertine is not distributed with this artifact because the fonts carry
# their own licenses. If TrueType copies are placed in paper/fonts/, they are
# used and embedded as TrueType; otherwise matplotlib's default font is used.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
_FONTS = HERE / "fonts"
_FONT_FILES = ("LinLibertine_R.ttf", "LinLibertine_RB.ttf", "LinLibertine_I.ttf")
if all((_FONTS / _f).is_file() for _f in _FONT_FILES):
    for _f in _FONT_FILES:
        font_manager.fontManager.addfont(str(_FONTS / _f))
    matplotlib.rcParams["font.family"] = "Linux Libertine O"
matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["hatch.color"] = "#6E6E6E"
matplotlib.rcParams["hatch.linewidth"] = 0.8

DATA = ROOT / "runs"
OUT  = HERE / "figures"
MAIN = dict(config="single", reward="neg_log", agg="best")
MAIN_COMBO_BY_SOURCE = {
    "job": "d10_tk10_S_sb480",
    "jobcomplex": "d10_tk10_S_sb480",
    "imdb-ceb": "tk10",
}
TIE  = 0.05
COL  = {"dp": "#1A1A1A", "geqo": "#1A1A1A", "mcts": "#1A1A1A"}
KIND = {"Worse": "white", "Same": "#E6E6E6", "Better": "#1A1A1A"}
LABEL = {"dp": "DPSize", "geqo": "GEQO", "mcts": "MCTS-E"}

def load(p):
    return pd.DataFrame([json.loads(l) for l in Path(p).open() if l.strip()])

def prep():
    ceb = load(DATA / "CEB/results.jsonl")
    jj  = load(DATA / "JOB-JOBComplex/results.jsonl")
    df = pd.concat([ceb, jj], ignore_index=True)
    # tag method
    def method(r):
        if r["config"] in ("dp", "geqo"):
            return r["config"]
        if (all(r.get(k) == v for k, v in MAIN.items())
                and r.get("combo_id") == MAIN_COMBO_BY_SOURCE.get(r.get("source"))):
            return "mcts"
        return None
    df["method"] = df.apply(method, axis=1)
    df = df[df["method"].notna()].copy()
    missing = [
        (source, method_name)
        for source in MAIN_COMBO_BY_SOURCE
        for method_name in ("dp", "geqo", "mcts")
        if not ((df["source"] == source) & (df["method"] == method_name)).any()
    ]
    if missing:
        raise ValueError(f"missing primary benchmark series: {missing}")
    # e2e
    df["e2e_time_ms"] = df["plan_time_ms"] + df["exec_time_ms"]
    # exclude JOB-Complex query 12
    before = len(df)
    df = df[~((df["source"] == "jobcomplex") & (df["query"].astype(str) == "12"))].copy()
    print(f"[prep] dropped {before - len(df)} rows for jobcomplex q12")
    return df

def pq(df, src, col):
    """Per-query value, wide by method. Aggregates across the 3 seeds with
    MEAN (matching the dashboard's `Across seeds: mean` default).  Used for
    outcome counts, winner counts, and per-query primary-endpoint stats."""
    d = df[df.source == src]
    p = d.pivot_table(index="query", columns="method", values=col, aggfunc="mean")
    for m in ("dp", "geqo", "mcts"):
        if m not in p:
            p[m] = np.nan
    return p

def rels_of(df, src):
    d = df[df.source == src]
    return d.groupby("query")["rels"].median()

def fmt(x, nd=2):
    return f"{x:.{nd}f}"

# ----------------------------------------------------------------------------
df = prep()
report = {}
BENCH = [("job", "JOB"), ("jobcomplex", "JOB-Complex"), ("imdb-ceb", "IMDb-CEB")]

# ---- 1. per-query outcome counts (kind_stats), +/-5% on e2e --------------
print("\n===== OUTCOME COUNTS (e2e median, +/-5% tie) =====")
outcomes = {}
for src, name in BENCH:
    p = pq(df, src, "e2e_time_ms")
    res = {}
    for base in ("dp", "geqo"):
        a = p.dropna(subset=["mcts", base])
        r = a["mcts"] / a[base]
        better = int((r < 1 - TIE).sum())
        worse  = int((r > 1 + TIE).sum())
        same   = int(len(a) - better - worse)
        res[base] = dict(better=better, same=same, worse=worse, n=int(len(a)))
    outcomes[src] = res
    print(f"{name:12} vs DP  {res['dp']}   vs GEQO {res['geqo']}")
report["outcomes"] = outcomes

# ---- 2. headline table ----------------------------------------------------
def agg_ratio(d):
    """Aggregate totals over an apples-to-apples (source,query) set: per-seed
    workload sum, then MEAN over the 3 seeds (matching the dashboard's
    `Across seeds: mean` convention so the headline table, by-rel-count
    buckets, and combined slices use the same reduction as per-query stats)."""
    med = d.pivot_table(index=["source", "query"], columns="method", values="e2e_time_ms", aggfunc="mean")
    for m in ("dp", "geqo", "mcts"):
        if m not in med:
            med[m] = np.nan
    common = med.dropna(subset=["dp", "geqo", "mcts"]).index
    dd = d.set_index(["source", "query"])
    dd = dd[dd.index.isin(common)].reset_index()
    out = {"n": int(len(common))}
    for met, key in (("e2e_time_ms", "e2e"), ("exec_time_ms", "exec"), ("plan_cost", "cost")):
        seed_tot = dd.groupby(["method", "seed"])[met].sum().unstack("method")
        mean_tot = seed_tot.mean(axis=0)  # mean over seeds
        out[key] = {m: float(mean_tot.get(m, np.nan)) for m in ("dp", "geqo", "mcts")}
    plan = dd.groupby("method")["plan_time_ms"].mean()
    out["plan"] = {m: float(plan.get(m, np.nan)) for m in ("dp", "geqo", "mcts")}
    return out

def totals(df, src, rel_min=None):
    d = df[df.source == src].copy()
    if rel_min is not None:
        relmap = d.groupby("query")["rels"].max()  # rels only on MCTS rows
        keep = relmap[relmap >= rel_min].index
        d = d[d["query"].isin(keep)]
    return agg_ratio(d)

ROWS = [
    ("JOB", "job", None), ("JOB-Complex", "jobcomplex", None),
    ("JOB rels>=12", "job", 12), ("JOB-Complex rels>=12", "jobcomplex", 12),
    ("IMDb-CEB (200)", "imdb-ceb", None), ("IMDb-CEB rels>=12", "imdb-ceb", 12),
    ("IMDb-CEB rels>=14", "imdb-ceb", 14),
]
print("\n===== HEADLINE TABLE =====")
headline = {}
for label, src, rmin in ROWS:
    t = totals(df, src, rmin)
    h = {"n": t["n"]}
    for key, unit in (("e2e", 1000.0), ("exec", 1000.0)):  # ms->s
        dp, ge, mc = (t[key][m] for m in ("dp", "geqo", "mcts"))
        h[key] = dict(dp=dp/unit, geqo=ge/unit, mcts=mc/unit,
                      vs_geqo=ge/mc, vs_dp=dp/mc)
    dp, ge, mc = (t["plan"][m] for m in ("dp", "geqo", "mcts"))
    h["plan"] = dict(dp=dp, geqo=ge, mcts=mc, vs_geqo=ge/mc, vs_dp=dp/mc)
    headline[label] = h
    print(f"\n-- {label}  (n={t['n']}) --")
    print(f"   e2e (s)  DP={h['e2e']['dp']:.1f} GEQO={h['e2e']['geqo']:.1f} MCTS={h['e2e']['mcts']:.1f}"
          f"  vsGEQO={h['e2e']['vs_geqo']:.2f}x vsDP={h['e2e']['vs_dp']:.2f}x")
    print(f"   exec(s)  DP={h['exec']['dp']:.1f} GEQO={h['exec']['geqo']:.1f} MCTS={h['exec']['mcts']:.1f}"
          f"  vsGEQO={h['exec']['vs_geqo']:.2f}x vsDP={h['exec']['vs_dp']:.2f}x")
    print(f"   plan(ms) DP={h['plan']['dp']:.1f} GEQO={h['plan']['geqo']:.1f} MCTS={h['plan']['mcts']:.1f}"
          f"  vsGEQO={h['plan']['vs_geqo']:.2f}x vsDP={h['plan']['vs_dp']:.2f}x")
report["headline"] = headline

# ---- 3. primary endpoint: median per-query e2e speedup vs GEQO/DP ---------
print("\n===== PRIMARY ENDPOINT (median per-query e2e speedup) =====")
primary = {}
for src, name in BENCH:
    p = pq(df, src, "e2e_time_ms")
    row = {}
    for base in ("dp", "geqo"):
        a = p.dropna(subset=["mcts", base])
        sp = (a[base] / a["mcts"])  # speedup>1 = MCTS faster
        row[base] = dict(median=float(sp.median()), mean=float(sp.mean()), n=int(len(a)))
    primary[src] = row
    print(f"{name:12} vsGEQO median {row['geqo']['median']:.3f}x   vsDP median {row['dp']['median']:.3f}x")
report["primary"] = primary

# ---- 4. CEB per-rel-count buckets (aggregate speedup = ratio of sums) ------
print("\n===== IMDb-CEB rel-count buckets (ratio-of-sums e2e speedup) =====")
rels = rels_of(df, "imdb-ceb")
dist = rels.value_counts().sort_index().to_dict()
print("rel distribution:", {int(k): int(v) for k, v in dist.items()})
buckets = [("9-10", lambda r: r <= 10), ("11-12", lambda r: (r >= 11) & (r <= 12)),
           ("14-16", lambda r: r >= 14)]
ceb_buckets = {}
for bname, pred in buckets:
    keep = rels[pred(rels)].index
    t = agg_ratio(df[(df.source == "imdb-ceb") & (df["query"].isin(keep))])
    sp_dp = t["e2e"]["dp"] / t["e2e"]["mcts"]; sp_ge = t["e2e"]["geqo"] / t["e2e"]["mcts"]
    ceb_buckets[bname] = dict(n=t["n"], vs_dp=float(sp_dp), vs_geqo=float(sp_ge))
    print(f"   {bname:6} (n={t['n']})  vsDP={sp_dp:.2f}x  vsGEQO={sp_ge:.2f}x")
report["ceb_buckets"] = ceb_buckets
report["ceb_rel_dist"] = {int(k): int(v) for k, v in dist.items()}

# combined JOB U JOB-Complex, rels>=12 (intro headline)
keep_parts = []
for s in ("job", "jobcomplex"):
    rm = df[df.source == s].groupby("query")["rels"].max()
    keep_parts.append(df[(df.source == s) & (df["query"].isin(rm[rm >= 12].index))])
tc = agg_ratio(pd.concat(keep_parts))
report["combined_ge12"] = dict(n=tc["n"], vs_geqo=tc["e2e"]["geqo"]/tc["e2e"]["mcts"], vs_dp=tc["e2e"]["dp"]/tc["e2e"]["mcts"])
print(f"\nJOB+JOB-Complex rels>=12 (n={tc['n']}): vsGEQO={report['combined_ge12']['vs_geqo']:.2f}x vsDP={report['combined_ge12']['vs_dp']:.2f}x")

# CEB rels>=14 geomean per-query exec ratio (mcts/baseline) for intro
_p = pq(df, "imdb-ceb", "exec_time_ms").join(rels.rename("rels"))
_p = _p[_p["rels"] >= 14].dropna(subset=["dp", "geqo", "mcts"])
report["ceb_ge14_geomean_exec"] = dict(
    vs_dp=float(np.exp(np.log(_p["mcts"]/_p["dp"]).mean())),
    vs_geqo=float(np.exp(np.log(_p["mcts"]/_p["geqo"]).mean())))
print(f"CEB>=14 geomean per-query exec ratio mcts/DP={report['ceb_ge14_geomean_exec']['vs_dp']:.3f} mcts/GEQO={report['ceb_ge14_geomean_exec']['vs_geqo']:.3f}")

# ---- 5. planning fraction of e2e (MCTS) -----------------------------------
print("\n===== MCTS planning fraction of total e2e =====")
planfrac = {}
for src, name in BENCH:
    d = df[(df.source == src) & (df.method == "mcts")]
    frac = d["plan_time_ms"].sum() / d["e2e_time_ms"].sum()
    planfrac[src] = float(frac)
    print(f"{name:12} planning = {100*frac:.2f}% of e2e")
report["plan_fraction"] = planfrac

# ---- 6. winners per benchmark (min e2e, +/-5% co-winner) ------------------
print("\n===== WINNERS (min e2e median, +/-5% co-winner band) =====")
winners = {}
for src, name in BENCH:
    p = pq(df, src, "e2e_time_ms").dropna(subset=["dp", "geqo", "mcts"])
    cnt = collections.Counter()
    for _, row in p.iterrows():
        best = row[["dp", "geqo", "mcts"]].min()
        for m in ("dp", "geqo", "mcts"):
            if row[m] <= best * (1 + TIE):
                cnt[m] += 1
    winners[src] = {**{m: int(cnt[m]) for m in ("dp", "geqo", "mcts")}, "n": int(len(p))}
    print(f"{name:12} DP={cnt['dp']} GEQO={cnt['geqo']} MCTS={cnt['mcts']}  (n={len(p)}, co-wins allowed)")
report["winners"] = winners

# ---- 7. jobcomplex q12 timeout detail -------------------------------------
jc = load(DATA / "JOB-JOBComplex/results.jsonl")
jc = jc[jc.source == "jobcomplex"]
q12 = jc[jc["query"].astype(str) == "12"]
q12d = {}
for m, sel in (("mcts", lambda r: r["config"] == "single"), ("geqo", lambda r: r["config"] == "geqo"), ("dp", lambda r: r["config"] == "dp")):
    g = q12[q12.apply(sel, axis=1)]
    q12d[m] = dict(timeouts=int(g["timed_out"].fillna(False).sum()), n=int(len(g)))
report["q12"] = q12d
print("\n===== JOB-Complex q12 timeouts =====", q12d)

# ---- 8. configuration ablation (MCTS-Extreme variants only, IMDb-CEB) ------
# Workload TOTALS among the MCTS-Extreme configurations only (DPSize and GEQO
# excluded), on the IMDb-CEB rel9 run.  Each cell is the sum over queries of
# the per-query MEAN over the 3 seeds (timed-out seed-runs are NaN and skipped),
# mirroring the dashboard's "Absolute totals, mean" bars.  Totals are taken
# over the COMMON query set -- queries every configuration completes on >=1
# seed -- because K=3 times out on all seeds of a few hard queries; summing
# over each config's own set would unfairly drop those from K=3.  COST uses the
# search's own best-found plan cost (mcts_best_cost).
print("\n===== CONFIGURATION ABLATION (MCTS-E totals, IMDb-CEB) =====")
# Newer, more complete CEB merge: fresh neg_log d10 top-k sweep (k=8..12,
# replacing the older K=9/10/11 rows) plus the reward, aggregation, and
# plan-shape series.  Used only for the ablation tables.
abl = load(DATA / "CEB/results.jsonl")
abl = abl[~abl["config"].isin(["dp", "geqo"])].copy()
abl["e2e_time_ms"] = abl["plan_time_ms"] + abl["exec_time_ms"]
abl["depth"] = abl["gucs"].apply(lambda g: int(g["depth"]))
abl["top_k"] = abl["gucs"].apply(lambda g: int(g["top_k"]))
abl["kernels"] = abl["gucs"].apply(lambda g: int(g.get("kernels", 1)))  # internal K shape parameter; K=1 is linear/zig-zag
abl["cfg"] = abl.apply(lambda r: f"{r['reward']}|{r['agg']}|{r['kernels']}|{r['depth']}|{r['top_k']}", axis=1)
abl["timed_out_bool"] = abl["timed_out"].fillna(False).astype(bool)
abl_relmax = abl.groupby("query")["rels"].max()
ABL_METRICS = {"cost": "mcts_best_cost", "plan": "plan_time_ms",
               "runtime": "exec_time_ms", "e2e": "e2e_time_ms", "iters": "iters"}

def ablation_totals(sub):
    """Σ-of-per-query-mean totals per config over all queries that config
    completes (per-config union; NaN/timeout queries are skipped), plus
    timed-out seed-run counts.  A config that times out on every seed of a
    query simply omits it from its sum -- the TO column flags this."""
    piv = {m: sub.pivot_table(index="query", columns="cfg", values=col, aggfunc="mean")
           for m, col in ABL_METRICS.items()}
    to = sub.groupby("cfg")["timed_out_bool"].sum().astype(int)
    cfgs = sorted(piv["e2e"].columns,
                  key=lambda c: piv["e2e"][c].sum())  # by total e2e, lower=better
    out = []
    for c in cfgs:
        reward, agg, kernels, depth, top_k = c.split("|")
        out.append(dict(reward=reward, agg=agg, kernels=int(kernels),
                        depth=int(depth), top_k=int(top_k),
                        cost=float(piv["cost"][c].sum()),
                        plan_ms=float(piv["plan"][c].sum()),
                        exec_ms=float(piv["runtime"][c].sum()),
                        e2e_ms=float(piv["e2e"][c].sum()),
                        iters=float(piv["iters"][c].sum()),
                        n_queries=int(piv["e2e"][c].notna().sum()),
                        timeouts=int(to[c])))
    return dict(n_total=int(len(piv["e2e"].index)), rows=out)

abl_full = ablation_totals(abl)
abl_ge12 = ablation_totals(abl[abl["query"].isin(abl_relmax[abl_relmax >= 12].index)])
for label, blk in (("FULL", abl_full), ("rels>=12", abl_ge12)):
    print(f"-- {label} (n={blk['n_total']}) --")
    for r in blk["rows"]:
        print(f"   {r['reward']:13}{r['agg']:8} K={r['kernels']} d={r['depth']:<2} k={r['top_k']:<2}"
              f" cost={r['cost']/1e6:6.1f}M plan={r['plan_ms']/1e3:6.1f}s exec={r['exec_ms']/1e3:7.1f}s"
              f" e2e={r['e2e_ms']/1e3:7.1f}s iters={r['iters']/1e3:5.1f}k TO={r['timeouts']:>2} nq={r['n_queries']}")
report["ablation"] = dict(full=abl_full, rels_ge12=abl_ge12)

# (results_numbers.json is left untouched; this script only draws figures)

# ===========================================================================
#  FIGURES
# ===========================================================================
plt.rcParams.update({"font.size": 13, "font.weight": "normal",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.labelsize": 12, "axes.labelweight": "normal",
                     "xtick.labelsize": 11, "ytick.labelsize": 11,
                     "legend.fontsize": 11, "axes.titlesize": 12,
                     "axes.titleweight": "normal"})
OUTCOMES_SIZE = (3.40, 1.95)  # one column after the tight bounding box
DETAIL_SIZE = (7.02, 3.0)     # two columns (figure*) after the tight bounding box

def shares_to_100(counts):
    """Integer percentages that sum to 100: ordinary rounding, then, if the
    sum is off, the shares closest to the rounding boundary absorb the rest."""
    counts = np.asarray(counts, float)
    raw = 100 * counts / counts.sum()
    out = np.rint(raw).astype(int)
    diff = 100 - out.sum()
    if diff:
        resid = (raw - out) * np.sign(diff)  # largest = most under-/over-rounded
        for i in np.argsort(-resid, kind="stable")[: abs(diff)]:
            out[i] += int(np.sign(diff))
    return out

def _panels_center(axes):
    """Horizontal center of the panel frames in figure coordinates, so a
    shared legend sits over the plots rather than over the y-axis label."""
    return 0.5 * (axes[0].get_position().x0 + axes[-1].get_position().x1)

def save(fig, stem):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"   wrote {stem}.pdf/.png")

def fig_outcomes_combined(stem):
    """Three outcome-distribution panels (JOB / JOB-Complex / IMDb-CEB) in one
    column, with a shared y-axis and a single shared legend."""
    cats = ["Worse", "Same", "Better"]
    bench_info = [("job", "JOB"), ("jobcomplex", "JOB-Complex"), ("imdb-ceb", "IMDb-CEB")]
    fig, axes = plt.subplots(1, 3, figsize=OUTCOMES_SIZE, sharey=True)
    for ax, (src, label), letter in zip(axes, bench_info, "abc"):
        o = outcomes[src]
        x = np.arange(2); width = 0.62
        bottoms = np.zeros(2)
        counts = {b: [o[b][c.lower()] for c in cats] for b in ("dp", "geqo")}
        exact = {b: 100 * np.asarray(counts[b], float) / sum(counts[b]) for b in counts}
        shown = {b: shares_to_100(counts[b]) for b in counts}
        for k, c in enumerate(cats):
            share = np.array([exact["dp"][k], exact["geqo"][k]])
            ax.bar(x, share, width, bottom=bottoms, label=c, color=KIND[c],
                   hatch="////" if c == "Worse" else None,
                   edgecolor="#6E6E6E" if c == "Worse" else "none", linewidth=0.6)
            for i, b in enumerate(("dp", "geqo")):
                if share[i] > 6:
                    ax.text(x[i], bottoms[i] + share[i]/2, f"{shown[b][k]}%",
                            ha="center", va="center", fontsize=6.5,
                            color="white" if c == "Better" else "#1A1A1A",
                            bbox=dict(facecolor="white", edgecolor="none",
                                      boxstyle="round,pad=0.12") if c == "Worse" else None)
            bottoms += share
        ax.set_xticks(x); ax.set_xticklabels(["DPSize", "GEQO"], fontsize=6.5)
        ax.set_xlim(-0.55, 1.55)
        ax.set_title(f"({letter}) {label}\n(n={o['dp']['n']})",
                     fontsize=7.5, linespacing=1.05, fontweight="normal", pad=3)
        ax.tick_params(axis="x", pad=2, length=2)
        ax.tick_params(axis="y", labelsize=6.5, pad=1.5, length=2)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.3, lw=0.5); ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_linewidth(0.6)
    axes[0].set_ylabel("Share of queries (%)", fontsize=7.5, fontweight="normal", labelpad=2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=0.6)
    fig.legend(handles, labels, ncol=3, loc="upper center",
               bbox_to_anchor=(_panels_center(axes), 1.0), frameon=False,
               handlelength=1.6, columnspacing=1.6,
               prop={"size": 7.5, "weight": "normal"})
    save(fig, stem)

def _query_label(q):
    if isinstance(q, float) and q.is_integer():
        return str(int(q))
    return str(q)

def _paired_query_axis(ax, src, rel_min, title, x_fontsize):
    p = pq(df, src, "e2e_time_ms").join(rels_of(df, src).rename("rels"))
    p = p[p["rels"] >= rel_min].dropna(subset=["dp", "geqo", "mcts"])
    sp_dp = p["dp"] / p["mcts"]
    sp_ge = p["geqo"] / p["mcts"]
    order = sp_ge.sort_values(ascending=False).index
    sp_dp, sp_ge = sp_dp[order], sp_ge[order]

    x = np.arange(len(order))
    ax.vlines(x, sp_ge, sp_dp, color="#D0D0D0", lw=0.8, zorder=1)
    ax.scatter(x, sp_dp, s=16, color=COL["dp"], label="vs DPSize", zorder=3)
    ax.scatter(x, sp_ge, s=18, facecolor="white", edgecolor=COL["geqo"], linewidth=1.0,
               marker="s", label="vs GEQO", zorder=3)
    ax.axhline(1.0, color="black", lw=0.8)
    vals = np.r_[sp_dp.to_numpy(), sp_ge.to_numpy()]
    lo, hi = max(0.35, vals.min() * 0.85), min(3.4, vals.max() * 1.15)
    ax.set_ylim(lo, hi)
    ticks = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    ax.set_yticks([t for t in ticks if lo <= t <= hi])
    ax.set_yticklabels([f"{t:.1f}" for t in ax.get_yticks()])
    ax.set_xticks(x)
    ax.set_xticklabels([("\n" if i % 2 else "") + _query_label(q) for i, q in enumerate(order)],
                       rotation=0, ha="center", fontsize=x_fontsize,
                       fontweight="normal")
    ax.set_title(title, loc="center", fontsize=10, fontweight="normal", pad=3)
    ax.set_xlabel("Query", fontsize=9, labelpad=2)
    ax.tick_params(axis="x", pad=1, length=2)
    for i, t in enumerate(ax.get_xticklines()[::2]):
        t.set_markersize(2 if i % 2 == 0 else 9)
    ax.tick_params(axis="y", labelsize=8, pad=1)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

def _paired_ceb_axis(ax):
    names = list(ceb_buckets)
    dp = np.array([ceb_buckets[b]["vs_dp"] for b in names])
    ge = np.array([ceb_buckets[b]["vs_geqo"] for b in names])
    x = np.arange(len(names))

    ax.vlines(x, ge, dp, color="#D0D0D0", lw=0.8, zorder=1)
    ax.scatter(x, dp, s=20, color=COL["dp"], label="vs DPSize", zorder=3)
    ax.scatter(x, ge, s=22, facecolor="white", edgecolor=COL["geqo"], linewidth=1.0,
               marker="s", label="vs GEQO", zorder=3)
    ax.axhline(1.0, color="black", lw=0.8)
    for i, (d, g) in enumerate(zip(dp, ge)):
        ax.annotate(f"{d:.2f}", (i, d), xytext=(4, 3), textcoords="offset points",
                    color=COL["dp"], fontsize=7.5, va="center")
        ax.annotate(f"{g:.2f}", (i, g), xytext=(4, -5), textcoords="offset points",
                    color=COL["geqo"], fontsize=7.5, va="center")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b.replace('-', '\u2013')}\n(n={ceb_buckets[b]['n']})" for b in names],
                       fontsize=8, fontweight="normal")
    ax.set_ylim(0.9, 1.55)
    ax.set_xlabel("Relation count", fontsize=9, labelpad=2)
    ax.set_yticks([1.0, 1.25, 1.5])
    ax.set_yticklabels(["1.00", "1.25", "1.50"])
    ax.set_title("(c) IMDb-CEB subset", loc="center", fontsize=10,
                 fontweight="normal", pad=3)
    ax.tick_params(axis="x", pad=1, length=2)
    ax.tick_params(axis="y", labelsize=8, pad=1)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

def fig_results_detail(stem):
    fig, axes = plt.subplots(
        1, 3, figsize=DETAIL_SIZE,
        gridspec_kw={"width_ratios": [2.7, 2.1, 1.8]}
    )
    _paired_query_axis(axes[0], "job", 12, "(a) JOB, ≥12 relations", 8)
    _paired_query_axis(axes[1], "jobcomplex", 12, "(b) JOB-Complex, ≥12 relations", 8)
    _paired_ceb_axis(axes[2])
    # Align the gridlines of all panels: (a) and (b) share one y range, and
    # (c) maps it around parity at a quarter of the step, so its 1.00, 1.25
    # and 1.50 lie level with 1.0, 2.0 and 3.0 of (a) and (b).
    lo = min(ax.get_ylim()[0] for ax in axes[:2])
    hi = max(ax.get_ylim()[1] for ax in axes[:2])
    for ax in axes[:2]:
        ax.set_ylim(lo, hi)
    axes[2].set_ylim(1.0 + (lo - 1.0) / 4, 1.0 + (hi - 1.0) / 4)
    axes[0].set_ylabel("Speedup (baseline / MCTS-E)", fontsize=10, fontweight="normal")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.align_xlabels(axes)
    fig.tight_layout(rect=(0, 0, 1, 0.95), w_pad=1.1)
    fig.legend(handles, labels, ncol=2, loc="upper center",
               bbox_to_anchor=(_panels_center(axes), 1.0), frameon=False,
               prop={"size": 10, "weight": "normal"})
    save(fig, stem)

print("\n===== FIGURES =====")
fig_outcomes_combined("outcomes_all")
fig_results_detail("results_detail")
print("\nDONE.")
