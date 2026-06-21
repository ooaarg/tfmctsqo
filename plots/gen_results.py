#!/usr/bin/env python
"""Recompute all results-section numbers.
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

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "runs"
OUT  = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)
MAIN = dict(config="single", reward="neg_log", agg="best", combo_id="d10_tk10_S_sb480")
TIE  = 0.05
COL  = {"dp": "#1f77b4", "geqo": "#ff7f0e", "mcts": "#2ca02c"}
KIND = {"Worse": "#d62728", "Same": "#bdbdbd", "Better": "#2ca02c"}
LABEL = {"dp": "DPSize", "geqo": "GEQO", "mcts": "MCTS-E"}

def load(p):
    return pd.DataFrame([json.loads(l) for l in Path(p).open() if l.strip()])

def prep():
    ceb = load(DATA / "CEB/results.jsonl")
    jj  = load(DATA / "JOB-JOBComplex/results.jsonl")
    df = pd.concat([ceb, jj], ignore_index=True)
    def method(r):
        if r["config"] in ("dp", "geqo"):
            return r["config"]
        if all(r.get(k) == v for k, v in MAIN.items()):
            return "mcts"
        return None
    df["method"] = df.apply(method, axis=1)
    df = df[df["method"].notna()].copy()
    df["e2e_time_ms"] = df["plan_time_ms"] + df["exec_time_ms"]
    before = len(df)
    df = df[~((df["source"] == "jobcomplex") & (df["query"].astype(str) == "12"))].copy()
    print(f"[prep] dropped {before - len(df)} rows for jobcomplex q12")
    return df

def pq(df, src, col):
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

df = prep()
report = {}
BENCH = [("job", "JOB"), ("jobcomplex", "JOB-Complex"), ("imdb-ceb", "IMDb-CEB")]

# --- per-query outcome counts (kind_stats), +/-5% on e2e ---
print("\n===== OUTCOME COUNTS (e2e seed-mean, +/-5% tie) =====")
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

# --- headline table ---
def agg_ratio(d):
    qmean = d.pivot_table(index=["source", "query"], columns="method", values="e2e_time_ms", aggfunc="mean")
    for m in ("dp", "geqo", "mcts"):
        if m not in qmean:
            qmean[m] = np.nan
    common = qmean.dropna(subset=["dp", "geqo", "mcts"]).index
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

# --- primary endpoint: median per-query e2e speedup vs GEQO/DP ---
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

# --- CEB per-rel-count buckets (aggregate speedup = ratio of sums) ---
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

# --- planning fraction of e2e (MCTS) ---
print("\n===== MCTS planning fraction of total e2e =====")
planfrac = {}
for src, name in BENCH:
    d = df[(df.source == src) & (df.method == "mcts")]
    frac = d["plan_time_ms"].sum() / d["e2e_time_ms"].sum()
    planfrac[src] = float(frac)
    print(f"{name:12} planning = {100*frac:.2f}% of e2e")
report["plan_fraction"] = planfrac

# --- winners per benchmark (min e2e, +/-5% co-winner) ---
print("\n===== WINNERS (min e2e seed-mean, +/-5% co-winner band) =====")
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

# --- jobcomplex q12 timeout detail ---
jc = load(DATA / "JOB-JOBComplex/results.jsonl")
jc = jc[jc.source == "jobcomplex"]
q12 = jc[jc["query"].astype(str) == "12"]
q12d = {}
for m, sel in (("mcts", lambda r: r["config"] == "single"), ("geqo", lambda r: r["config"] == "geqo"), ("dp", lambda r: r["config"] == "dp")):
    g = q12[q12.apply(sel, axis=1)]
    q12d[m] = dict(timeouts=int(g["timed_out"].fillna(False).sum()), n=int(len(g)))
report["q12"] = q12d
print("\n===== JOB-Complex q12 timeouts =====", q12d)

# --- configuration ablation (MCTS-Extreme variants only, IMDb-CEB) ---
print("\n===== CONFIGURATION ABLATION (MCTS-E totals, IMDb-CEB) =====")
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

(OUT / "results_numbers.json").write_text(json.dumps(report, indent=2))
print("\n[wrote results_numbers.json]")

# --- figures ---
plt.rcParams.update({"font.size": 13, "font.weight": "normal",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.labelsize": 12, "axes.labelweight": "normal",
                     "xtick.labelsize": 11, "ytick.labelsize": 11,
                     "legend.fontsize": 11, "axes.titlesize": 12,
                     "axes.titleweight": "normal"})
PANEL_SIZE = (2.6, 2.0)  # tuned for legacy 3-per-row subfigures
DETAIL_SIZE = (7.2, 3.0)

def save(fig, stem):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"   wrote {stem}.pdf/.png")

def fig_kind(src, stem):
    o = outcomes[src]
    cats = ["Worse", "Same", "Better"]
    fig, ax = plt.subplots(figsize=PANEL_SIZE)
    x = np.arange(2); width = 0.6
    bottoms = np.zeros(2)
    vals = {c: np.array([o["dp"][c.lower()], o["geqo"][c.lower()]], float) for c in cats}
    denom = sum(vals.values())
    for c in cats:
        share = 100 * vals[c] / denom
        ax.bar(x, share, width, bottom=bottoms, label=c, color=KIND[c])
        for i in range(2):
            if share[i] > 4:
                ax.text(x[i], bottoms[i] + share[i] / 2, f"{share[i]:.0f}%", ha="center", va="center", fontsize=9)
        bottoms += share
    ax.set_xticks(x); ax.set_xticklabels(["vs DPSize", "vs GEQO"])
    ax.set_ylabel("Share of queries (%)"); ax.set_ylim(0, 100)
    ax.legend(ncol=3, fontsize=10, loc="upper center", bbox_to_anchor=(0.5, 1.10), frameon=False)
    ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
    save(fig, stem)

def fig_outcomes_combined(stem):
    cats = ["Worse", "Same", "Better"]
    bench_info = [("job", "JOB"), ("jobcomplex", "JOB-Complex"), ("imdb-ceb", "IMDb-CEB")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
    for ax, (src, label), letter in zip(axes, bench_info, "abc"):
        o = outcomes[src]
        x = np.arange(2); width = 0.6
        bottoms = np.zeros(2)
        vals = {c: np.array([o["dp"][c.lower()], o["geqo"][c.lower()]], float) for c in cats}
        denom = sum(vals.values())
        for c in cats:
            share = 100 * vals[c] / denom
            ax.bar(x, share, width, bottom=bottoms, label=c, color=KIND[c])
            for i in range(2):
                if share[i] > 4:
                    ax.text(x[i], bottoms[i] + share[i]/2, f"{share[i]:.0f}%",
                            ha="center", va="center", fontsize=9)
            bottoms += share
        ax.set_xticks(x); ax.set_xticklabels(["vs DPSize", "vs GEQO"], fontsize=10)
        ax.set_title(f"({letter}) {label} (n={o['dp']['n']})",
                     fontsize=10, fontweight="normal")
        ax.tick_params(axis="y", labelsize=10)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
    axes[0].set_ylabel("Share of queries (%)", fontsize=10, fontweight="normal")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="upper center",
               bbox_to_anchor=(0.5, 1.04), frameon=False,
               prop={"size": 10, "weight": "normal"})
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, stem)

def fig_perquery(src, rel_min, stem):
    p = pq(df, src, "e2e_time_ms").join(rels_of(df, src).rename("rels"))
    p = p[p["rels"] >= rel_min].dropna(subset=["dp", "geqo", "mcts"])
    sp_dp = p["dp"] / p["mcts"]; sp_ge = p["geqo"] / p["mcts"]
    order = sp_ge.sort_values(ascending=False).index
    sp_dp, sp_ge = sp_dp[order], sp_ge[order]
    x = np.arange(len(order)); w = 0.4
    fig, ax = plt.subplots(figsize=PANEL_SIZE)
    ax.bar(x - w/2, sp_dp, w, label="vs DPSize", color=COL["dp"])
    ax.bar(x + w/2, sp_ge, w, label="vs GEQO", color=COL["geqo"])
    ax.axhline(1.0, color="black", lw=0.8)
    ax.set_yscale("log"); ax.set_ylabel("Speedup (baseline / MCTS-E)")
    ax.set_xticks(x); ax.set_xticklabels([str(q) for q in order], rotation=90, fontsize=8)
    ax.tick_params(axis="x", pad=1)
    ax.set_xlabel("Query (sorted by speedup vs GEQO)")
    ax.legend(frameon=False, loc="upper right"); ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
    save(fig, stem)

def fig_ceb_by_rels(stem):
    names = list(ceb_buckets); x = np.arange(len(names)); w = 0.38
    dp = [ceb_buckets[b]["vs_dp"] for b in names]
    ge = [ceb_buckets[b]["vs_geqo"] for b in names]
    fig, ax = plt.subplots(figsize=PANEL_SIZE)
    b1 = ax.bar(x - w/2, dp, w, label="vs DPSize", color=COL["dp"])
    b2 = ax.bar(x + w/2, ge, w, label="vs GEQO", color=COL["geqo"])
    ax.axhline(1.0, color="black", lw=0.8)
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width()/2, r.get_height() + 0.01,
                    f"{r.get_height():.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([f"{b}\n(n={ceb_buckets[b]['n']})" for b in names])
    ax.set_ylabel("Aggregate e2e speedup (baseline / MCTS-E)")
    ax.set_xlabel("Joined relations"); ax.set_ylim(0.9, max(dp + ge) * 1.15)
    ax.legend(frameon=False, loc="upper left"); ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
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
    ax.vlines(x, sp_ge, sp_dp, color="#bdbdbd", lw=0.8, zorder=1)
    ax.scatter(x, sp_dp, s=16, color=COL["dp"], label="vs DPSize", zorder=3)
    ax.scatter(x, sp_ge, s=16, color=COL["geqo"], label="vs GEQO", zorder=3)
    ax.axhline(1.0, color="black", lw=0.8)
    vals = np.r_[sp_dp.to_numpy(), sp_ge.to_numpy()]
    lo, hi = max(0.45, vals.min() * 0.85), min(3.4, vals.max() * 1.15)
    ax.set_ylim(lo, hi)
    ticks = [0.5, 1.0, 1.5, 2.0, 3.0]
    ax.set_yticks([t for t in ticks if lo <= t <= hi])
    ax.set_yticklabels([f"{t:g}" for t in ax.get_yticks()])
    ax.set_xticks(x)
    ax.set_xticklabels([_query_label(q) for q in order],
                       rotation=60, ha="right",
                       rotation_mode="anchor", fontsize=x_fontsize,
                       fontweight="normal")
    ax.set_title(title, loc="center", fontsize=10, fontweight="normal", pad=3)
    ax.tick_params(axis="x", pad=1, length=2)
    ax.tick_params(axis="y", labelsize=8, pad=1)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

def _paired_ceb_axis(ax):
    names = list(ceb_buckets)
    dp = np.array([ceb_buckets[b]["vs_dp"] for b in names])
    ge = np.array([ceb_buckets[b]["vs_geqo"] for b in names])
    x = np.arange(len(names))

    ax.vlines(x, ge, dp, color="#bdbdbd", lw=0.8, zorder=1)
    ax.scatter(x, dp, s=20, color=COL["dp"], label="vs DPSize", zorder=3)
    ax.scatter(x, ge, s=20, color=COL["geqo"], label="vs GEQO", zorder=3)
    ax.axhline(1.0, color="black", lw=0.8)
    for i, (d, g) in enumerate(zip(dp, ge)):
        ax.annotate(f"{d:.2f}", (i, d), xytext=(4, 3), textcoords="offset points",
                    color=COL["dp"], fontsize=7.5, va="center")
        ax.annotate(f"{g:.2f}", (i, g), xytext=(4, -5), textcoords="offset points",
                    color=COL["geqo"], fontsize=7.5, va="center")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}\n(n={ceb_buckets[b]['n']})" for b in names],
                       fontsize=8, fontweight="normal")
    ax.set_ylim(0.9, 1.55)
    ax.set_yticks([1.0, 1.2, 1.4])
    ax.set_title("(c) IMDb-CEB (rels≥9, 200)", loc="center", fontsize=10,
                 fontweight="normal", pad=3)
    ax.tick_params(axis="x", pad=1)
    ax.tick_params(axis="y", labelsize=8, pad=1)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

def fig_results_detail(stem):
    fig, axes = plt.subplots(
        1, 3, figsize=DETAIL_SIZE,
        gridspec_kw={"width_ratios": [2.7, 2.1, 1.8]}
    )
    _paired_query_axis(axes[0], "job", 12, "(a) JOB, rels≥12", 7)
    _paired_query_axis(axes[1], "jobcomplex", 12, "(b) JOB-Complex, rels≥12", 8)
    _paired_ceb_axis(axes[2])
    axes[0].set_ylabel("Speedup", fontsize=10, fontweight="normal")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="upper center",
               bbox_to_anchor=(0.5, 1.02), frameon=False,
               prop={"size": 10, "weight": "normal"})
    fig.supxlabel("Queries", fontsize=10, fontweight="normal", y=0.16)
    fig.tight_layout(rect=(0, 0.10, 1, 0.92), w_pad=1.1)
    save(fig, stem)

print("\n===== FIGURES =====")
fig_kind("job", "job_kind_stats")
fig_perquery("job", 12, "job_12_rels")
fig_kind("jobcomplex", "job_complex_kind_stats")
fig_perquery("jobcomplex", 12, "job_complex_stats")
fig_kind("imdb-ceb", "ceb_kind_stats")
fig_ceb_by_rels("ceb_by_rels")
fig_outcomes_combined("outcomes_all")
fig_results_detail("results_detail")
print("\nDONE.")
