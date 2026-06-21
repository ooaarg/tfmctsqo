#!/usr/bin/env python
"""Statistical-significance companion to gen_results.py."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

DATA = Path(__file__).resolve().parent.parent / "runs"
MAIN = dict(config="single", reward="neg_log", agg="best", combo_id="d10_tk10_S_sb480")
TIE = 0.05
B = 10000
RNG = np.random.default_rng(42)


def load(p):
    return pd.DataFrame([json.loads(l) for l in Path(p).open() if l.strip()])


def prep():
    ceb = load(DATA / "CEB/results.jsonl")
    jj = load(DATA / "JOB-JOBComplex/results.jsonl")
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
    df = df[~((df["source"] == "jobcomplex") & (df["query"].astype(str) == "12"))].copy()
    return df


def slice_df(df, sources, rel_min=None):
    d = df[df["source"].isin(sources)].copy()
    if rel_min is not None:
        # rels carried on MCTS rows; max over rows per (source,query)
        relmap = d.groupby(["source", "query"])["rels"].max()
        keep = relmap[relmap >= rel_min].index
        d = d.set_index(["source", "query"])
        d = d[d.index.isin(keep)].reset_index()
    return d


def per_query_seed_mean(d):
    """Wide table indexed by (source,query): seed-mean e2e per method, on the
    apples-to-apples set where all three methods completed."""
    p = d.pivot_table(index=["source", "query"], columns="method",
                       values="e2e_time_ms", aggfunc="mean")
    for m in ("dp", "geqo", "mcts"):
        if m not in p:
            p[m] = np.nan
    return p.dropna(subset=["dp", "geqo", "mcts"])


def per_seed_totals(d, common_index):
    """3 per-seed workload sums per method, restricted to the common set."""
    dd = d.set_index(["source", "query"])
    dd = dd[dd.index.isin(common_index)].reset_index()
    return dd.groupby(["method", "seed"])["e2e_time_ms"].sum().unstack("method")


def analyse(df, name, sources, rel_min, base):
    d = slice_df(df, sources, rel_min)
    pq = per_query_seed_mean(d)
    n = len(pq)
    if n < 3:
        return None
    mcts = pq["mcts"].to_numpy()
    b = pq[base].to_numpy()

    # point estimate: ratio of sums (== paper headline)
    point = b.sum() / mcts.sum()

    # --- seed-level: 3 per-seed aggregate speedups ---
    seed_tot = per_seed_totals(d, pq.index)
    seed_sp = (seed_tot[base] / seed_tot["mcts"]).to_numpy()  # one ratio per seed
    seed_mean, seed_sd = float(seed_sp.mean()), float(seed_sp.std(ddof=1))

    # --- bootstrap CI on aggregate (resample queries, paired) ---
    idx = RNG.integers(0, n, size=(B, n))
    boot_agg = b[idx].sum(axis=1) / mcts[idx].sum(axis=1)
    ci_agg = np.percentile(boot_agg, [2.5, 97.5])

    # --- geometric mean of per-query speedups + bootstrap CI ---
    ratio = b / mcts                      # >1 == MCTS faster
    log_r = np.log(ratio)
    gm = float(np.exp(log_r.mean()))
    boot_gm = np.exp(log_r[idx].mean(axis=1))
    ci_gm = np.percentile(boot_gm, [2.5, 97.5])

    # Wilcoxon signed-rank on paired per-query e2e ---
    try:
        w_stat, w_p = stats.wilcoxon(b, mcts, alternative="two-sided")
        w_p = float(w_p)
    except ValueError:
        w_p = float("nan")

    # win/tie/loss for MCTS at +/-5%
    win = int((ratio > 1 + TIE).sum())   # MCTS faster by >5%
    loss = int((ratio < 1 - TIE).sum())
    tie = n - win - loss

    return dict(name=name, base=base, n=n, point=point,
                seed_mean=seed_mean, seed_sd=seed_sd, seed_sp=list(map(float, seed_sp)),
                ci_agg=(float(ci_agg[0]), float(ci_agg[1])),
                gm=gm, ci_gm=(float(ci_gm[0]), float(ci_gm[1])),
                wilcoxon_p=w_p, win=win, tie=tie, loss=loss)


def stars(p):
    if np.isnan(p):
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


SLICES = [
    ("JOB (full)",            ["job"],               None),
    ("JOB rels>=12",          ["job"],               12),
    ("JOB-Complex (n=29)",    ["jobcomplex"],        None),
    ("JOB-Complex rels>=12",  ["jobcomplex"],        12),
    ("JOB+JOBc rels>=12",     ["job", "jobcomplex"], 12),
    ("IMDb-CEB (200)",        ["imdb-ceb"],          None),
    ("IMDb-CEB rels>=12",     ["imdb-ceb"],          12),
    ("IMDb-CEB rels>=14",     ["imdb-ceb"],          14),
]

def analyse_uct(rel_min, col):
    """UCT-Extreme (agg=best) vs classical mean-UCT (agg=average), both at
    neg_log / K=1 / d10 / k10 on IMDb-CEB. Read from the ABLATION run
    (CEB), the source of Table 6. `col` is
    'e2e' or 'mcts_best_cost'. Ratio>1 means UCT-Extreme is faster / lower-
    cost. Same bootstrap + Wilcoxon protocol as analyse()."""
    rows = [json.loads(l) for l in
            (DATA / "CEB/results.jsonl").open() if l.strip()]
    def sel(agg):
        d = pd.DataFrame([r for r in rows if r["config"] not in ("dp", "geqo")
                          and r.get("reward") == "neg_log" and r.get("agg") == agg
                          and int(r["gucs"].get("kernels", 1)) == 1
                          and int(r["gucs"]["depth"]) == 10 and int(r["gucs"]["top_k"]) == 10])
        d["e2e"] = d["plan_time_ms"] + d["exec_time_ms"]
        return d
    best, avg = sel("best"), sel("average")
    if rel_min is not None:
        keep = best.groupby("query")["rels"].max()
        keep = keep[keep >= rel_min].index
        best, avg = best[best["query"].isin(keep)], avg[avg["query"].isin(keep)]
    pb = best.pivot_table(index="query", values=col, aggfunc="mean")[col]
    pa = avg.pivot_table(index="query", values=col, aggfunc="mean")[col]
    common = pb.index.intersection(pa.index)
    pb, pa = pb.loc[common].to_numpy(), pa.loc[common].to_numpy()
    n = len(common)
    point = pa.sum() / pb.sum()                       # mean-UCT total / UCT-Extreme total
    idx = RNG.integers(0, n, size=(B, n))
    boot = pa[idx].sum(axis=1) / pb[idx].sum(axis=1)
    ci = np.percentile(boot, [2.5, 97.5])
    try:
        _, w_p = stats.wilcoxon(pa, pb, alternative="two-sided"); w_p = float(w_p)
    except ValueError:
        w_p = float("nan")
    ratio = pa / pb
    win = int((ratio > 1 + TIE).sum()); loss = int((ratio < 1 - TIE).sum())
    return dict(n=n, point=float(point), uct_e_total=float(pb.sum()),
                muct_total=float(pa.sum()), ci_agg=(float(ci[0]), float(ci[1])),
                wilcoxon_p=w_p, win=win, tie=n - win - loss, loss=loss)


if __name__ == "__main__":
    df = prep()
    print(f"B={B} bootstrap resamples (query-level, paired), seed=42\n")
    report = {}
    for base in ("geqo", "dp"):
        print(f"################  MCTS-Extreme  vs  {base.upper()}  "
              f"(speedup>1 = MCTS faster)  ################")
        hdr = (f"{'slice':22} {'n':>3} {'point':>6} "
               f"{'boot95%CI(agg)':>16} {'seed mean+/-sd':>16} "
               f"{'geomean':>7} {'geo95%CI':>14} {'Wilcoxon':>10} {'W/T/L':>9}")
        print(hdr)
        print("-" * len(hdr))
        for name, sources, rel_min in SLICES:
            r = analyse(df, name, sources, rel_min, base)
            if r is None:
                continue
            report[f"{base}::{name}"] = r
            print(f"{name:22} {r['n']:3d} {r['point']:5.2f}x "
                  f"[{r['ci_agg'][0]:.2f},{r['ci_agg'][1]:.2f}] "
                  f"{r['seed_mean']:5.2f}+/-{r['seed_sd']:.3f}  "
                  f"{r['gm']:5.2f}x [{r['ci_gm'][0]:.2f},{r['ci_gm'][1]:.2f}] "
                  f"p={r['wilcoxon_p']:.1e} {stars(r['wilcoxon_p']):>3} "
                  f"{r['win']:>3}/{r['tie']}/{r['loss']}")
        print()

    print("########  UCT-Extreme (best) vs mean-UCT (average), IMDb-CEB ablation run, "
          "neg_log/K=1/d10/k10  ########")
    print("(ratio>1 = UCT-Extreme faster[e2e] / lower-cost; totals match Table 6)")
    for metric, col, unit, uname in (("e2e", "e2e", 1e3, "s"),
                                     ("best-cost", "mcts_best_cost", 1e6, "M")):
        for label, rm in ((f"CEB full (200)", None), (f"CEB rels>=12 (57)", 12)):
            u = analyse_uct(rm, col)
            report[f"uct::{metric}::{label}"] = u
            print(f"{metric:9} {label:18} n={u['n']:3d} UCT-E={u['uct_e_total']/unit:.1f}{uname} "
                  f"mean-UCT={u['muct_total']/unit:.1f}{uname}  ratio={u['point']:.3f} "
                  f"CI[{u['ci_agg'][0]:.3f},{u['ci_agg'][1]:.3f}] "
                  f"p={u['wilcoxon_p']:.2e} {stars(u['wilcoxon_p']):>3} "
                  f"W/T/L={u['win']}/{u['tie']}/{u['loss']}")
    print()

    out = Path(__file__).with_name("stats.json")
    out.write_text(json.dumps(report, indent=2))
    print(f"[wrote] {out}")
