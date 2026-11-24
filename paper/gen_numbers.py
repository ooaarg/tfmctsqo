#!/usr/bin/env python3
"""Generate every number, table body and figure data file of tfmctsqo.tex
from the saved experiment results. Nothing is re-run; the script only reads.

Outputs (all under paper/build/generated/):
  numbers.tex          \\newcommand macros used in the prose
  tab_*.tex            complete tabular environments
  fig_*.dat            data for the pgfplots figures
and tfmctsqo_numbers.json next to this script (audit trail, not uploaded).

Conventions (methodology section of the paper):
  * end-to-end time = planning time + execution time of one seed-run;
  * a seed-run is completed if it did not time out and has both times;
  * per-query value = mean over the seeds of the query;
  * common query set = queries that every compared method completes on every seed;
  * speedup = baseline total / MCTS-Extreme total over the common query set;
  * 95% CI = paired bootstrap over queries (10^4 resamples, seed 42) of that ratio;
  * p = two-sided Wilcoxon signed-rank test on paired per-query values
    (exact null distribution for n <= 50 without ties, normal approximation otherwise);
  * tie band = +-5%.

Stdlib only.  Usage: python3 gen_tfmctsqo_numbers.py
"""
import csv
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "build" / "generated"
# The artifact stores the two IMDb result archives under runs/CEB and
# runs/JOB-JOBComplex (originally MERGED_ceb-rel9-200-topk8-12-new and
# vs-pg-job-jobcomplex in the experiment workspace).
IMDB_CEB_RUN = ROOT / "runs" / "CEB"
IMDB_JOBC_RUN = ROOT / "runs" / "JOB-JOBComplex"
LEARNED = ROOT / "results"
SAIO_RUN = ROOT / "runs" / "CEB-SAIOvsII"
REB_RUNS = ROOT / "runs"
JOBC_SQL = ROOT / "source" / "resource" / "job_complex"

B = 10_000
TIE = 0.05
LARGE = 12

MACROS = {}
AUDIT = {}


# ---------------------------------------------------------------- utilities
def load_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs)


def completed(r):
    return (not r.get("timed_out")) and r.get("exec_time_ms") is not None \
        and r.get("plan_time_ms") is not None


def e2e(r):
    if r.get("e2e_time_ms") is not None and r.get("plan_time_ms") is None:
        return r["e2e_time_ms"]
    return r["plan_time_ms"] + r["exec_time_ms"]


def bootstrap_ratio(base, mcts, seed=42):
    """Paired bootstrap over queries of sum(base)/sum(mcts)."""
    n = len(base)
    rng = random.Random(seed)
    idx = range(n)
    vals = []
    for _ in range(B):
        s = rng.choices(idx, k=n)
        vals.append(sum(base[i] for i in s) / sum(mcts[i] for i in s))
    vals.sort()
    return vals[int(0.025 * B)], vals[int(0.975 * B) - 1]


def wilcoxon(x, y):
    """Two-sided Wilcoxon signed-rank test on paired samples (zeros dropped)."""
    d = [a - b for a, b in zip(x, y) if a != b]
    n = len(d)
    if n == 0:
        return 1.0
    absd = sorted((abs(v), i) for i, v in enumerate(d))
    ranks = [0.0] * n
    i = 0
    ties = []
    while i < n:
        j = i
        while j + 1 < n and absd[j + 1][0] == absd[i][0]:
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[absd[k][1]] = r
        if j > i:
            ties.append(j - i + 1)
        i = j + 1
    w_plus = sum(r for r, v in zip(ranks, d) if v > 0)
    if n <= 50 and not ties:
        maxs = n * (n + 1) // 2
        cnt = [0] * (maxs + 1)
        cnt[0] = 1
        for k in range(1, n + 1):
            for s in range(maxs, k - 1, -1):
                cnt[s] += cnt[s - k]
        total = 2 ** n
        w = int(round(w_plus))
        lo = sum(cnt[: w + 1]) / total
        hi = sum(cnt[w:]) / total
        return min(1.0, 2 * min(lo, hi))
    mu = n * (n + 1) / 4
    var = n * (n + 1) * (2 * n + 1) / 24 - sum(t ** 3 - t for t in ties) / 48
    z = (w_plus - mu) / math.sqrt(var)
    return math.erfc(abs(z) / math.sqrt(2))


def macro(name, value):
    assert re.fullmatch(r"[A-Za-z]+", name), name
    assert name not in MACROS, name
    MACROS[name] = value


def f2(x):
    return f"{x:.2f}"


def f1(x):
    return f"{x:.1f}"


def f0(x):
    return f"{x:,.0f}".replace(",", "{,}")


def pfmt(p):
    if p < 1e-4:
        k = int(math.floor(-math.log10(p)))
        return f"$<10^{{-{k}}}$"
    if p < 0.001:
        return f"{p:.1g}"
    if p < 0.01:
        return f"{p:.3f}"
    return f"{p:.2f}"


def cifmt(ci):
    return f"[{ci[0]:.2f}, {ci[1]:.2f}]"


def pct(x):
    return f"{100 * x:.1f}"


WORD = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
        7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve"}


def sql_relation_count(path):
    sql = Path(path).read_text()
    m = re.search(r"\bFROM\b(.*?)\bWHERE\b", sql, re.S | re.I)
    return len([t for t in m.group(1).split(",") if t.strip()])


# ------------------------------------------------------- comparison helpers
class Series:
    """Per-(query, method) completed values and timeout counts."""

    def __init__(self):
        self.vals = defaultdict(lambda: defaultdict(list))   # q -> m -> [row]
        self.runs = defaultdict(lambda: defaultdict(int))     # q -> m -> seed-runs
        self.tos = defaultdict(lambda: defaultdict(int))      # q -> m -> timeouts
        self.rels = {}

    def add(self, q, m, r):
        self.runs[q][m] += 1
        if completed(r):
            self.vals[q][m].append(r)
        else:
            self.tos[q][m] += 1
        if r.get("rels") is not None:
            self.rels[q] = max(self.rels.get(q, 0), int(r["rels"]))

    def queries(self, rel_min=None, rel_eq=None, qs=None):
        out = sorted(self.runs)
        if qs is not None:
            out = [q for q in out if q in qs]
        if rel_min is not None:
            out = [q for q in out if self.rels.get(q, 0) >= rel_min]
        if rel_eq is not None:
            out = [q for q in out if self.rels.get(q, 0) in rel_eq]
        return out

    def common(self, methods, qs):
        return [q for q in qs if all(self.tos[q][m] == 0 and self.runs[q][m] > 0
                                     and len(self.vals[q][m]) == self.runs[q][m]
                                     for m in methods)]

    def pq(self, q, m, f=e2e):
        return mean(f(r) for r in self.vals[q][m])

    def timeouts(self, m, qs):
        return sum(self.tos[q][m] for q in qs), sum(self.runs[q][m] for q in qs)

    def compare(self, base, mcts, qs, f=e2e):
        b = [self.pq(q, base, f) for q in qs]
        x = [self.pq(q, mcts, f) for q in qs]
        sp = sum(b) / sum(x)
        ci = bootstrap_ratio(b, x)
        p = wilcoxon(b, x)
        return dict(n=len(qs), speedup=sp, ci=ci, p=p, base_total=sum(b), mcts_total=sum(x))


# ======================================================== IMDb (PostgreSQL)
def imdb():
    ceb = load_jsonl(IMDB_CEB_RUN / "results.jsonl")
    jj = load_jsonl(IMDB_JOBC_RUN / "results.jsonl")
    main_combo = {"job": "d10_tk10_S_sb480", "jobcomplex": "d10_tk10_S_sb480", "imdb-ceb": "tk10"}
    S = {s: Series() for s in main_combo}
    for r in ceb + jj:
        src = r["source"]
        if r["config"] in ("dp", "geqo"):
            m = r["config"]
        elif (r["config"] == "single" and r.get("reward") == "neg_log" and r.get("agg") == "best"
              and r.get("combo_id") == main_combo[src]):
            m = "mcts"
        else:
            continue
        S[src].add(str(r["query"]), m, r)
    # JOB-Complex query 12 never completes under MCTS-Extreme, so its relation
    # count is taken from the SQL text.
    q12_rels = sql_relation_count(JOBC_SQL / "12.sql")
    S["jobcomplex"].rels.setdefault("12", q12_rels)
    return S, ceb


def imdb_tables(S):
    M3 = ("dp", "geqo", "mcts")
    rows = []
    spec = [("JOB", "job", None, "all"), ("JOB", "job", LARGE, "large"),
            ("JOB-Complex", "jobcomplex", None, "all"), ("JOB-Complex", "jobcomplex", LARGE, "large"),
            ("CEB", "imdb-ceb", None, "all"), ("CEB", "imdb-ceb", LARGE, "large"),
            ("CEB", "imdb-ceb", 14, "fourteen")]
    res = {}
    for name, src, rmin, tag in spec:
        s = S[src]
        qs_all = s.queries(rel_min=rmin)
        qs = s.common(M3, qs_all)
        tot = {m: sum(s.pq(q, m) for q in qs) / 1000 for m in M3}
        execs = {m: sum(s.pq(q, m, lambda r: r["exec_time_ms"]) for q in qs) / 1000 for m in M3}
        cost = {m: sum(s.pq(q, m, lambda r: r["plan_cost"]) for q in qs) / 1000 for m in M3}
        plan = {m: mean(r["plan_time_ms"] for q in qs for r in s.vals[q][m]) for m in M3}
        vs_g = s.compare("geqo", "mcts", qs)
        vs_d = s.compare("dp", "mcts", qs)
        tos = {m: s.timeouts(m, qs_all) for m in M3}
        # per-seed speedups (seed-level spread)
        seeds = sorted({r["seed"] for q in qs for r in s.vals[q]["mcts"]})
        seed_sp = {}
        for base in ("geqo", "dp"):
            sp = []
            for sd in seeds:
                bt = sum(e2e(r) for q in qs for r in s.vals[q][base] if r["seed"] == sd)
                mt = sum(e2e(r) for q in qs for r in s.vals[q]["mcts"] if r["seed"] == sd)
                sp.append(bt / mt)
            seed_sp[base] = sp
        planfrac = sum(r["plan_time_ms"] for q in qs for r in s.vals[q]["mcts"]) / \
            sum(e2e(r) for q in qs for r in s.vals[q]["mcts"])
        res[(src, tag)] = dict(name=name, n=len(qs), n_all=len(qs_all), tot=tot, exec=execs,
                               cost=cost, plan=plan, vs_geqo=vs_g, vs_dp=vs_d, tos=tos,
                               seed_sp=seed_sp, planfrac=planfrac)
    # Combined JOB + JOB-Complex large queries.
    b_g, b_d, x = [], [], []
    for src in ("job", "jobcomplex"):
        s = S[src]
        qs = s.common(M3, s.queries(rel_min=LARGE))
        b_g += [s.pq(q, "geqo") for q in qs]
        b_d += [s.pq(q, "dp") for q in qs]
        x += [s.pq(q, "mcts") for q in qs]
    comb = {}
    for base, b in (("geqo", b_g), ("dp", b_d)):
        comb[base] = dict(n=len(x), speedup=sum(b) / sum(x), ci=bootstrap_ratio(b, x), p=wilcoxon(b, x))
    res[("combined", "large")] = comb
    return res


def write_imdb(S, R):
    label = {"all": "all", "large": r"$n \geq 12$", "fourteen": r"$n \geq 14$"}
    lines = [r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llrrrrrcrrcr@{}}",
             r"\toprule",
             r" & & & \multicolumn{3}{c}{Total end-to-end time (s)} & \multicolumn{3}{c}{MCTS-Extreme vs.\ GEQO}"
             r" & \multicolumn{3}{c}{MCTS-Extreme vs.\ DPSize} \\",
             r"\cmidrule(lr){4-6} \cmidrule(lr){7-9} \cmidrule(l){10-12}",
             r"Workload & Subset & Queries & DPSize & GEQO & MCTS-Extreme & Speedup & 95\% CI & $p$"
             r" & Speedup & 95\% CI & $p$ \\",
             r"\midrule"]
    prev = None
    for key in [("job", "all"), ("job", "large"), ("jobcomplex", "all"), ("jobcomplex", "large"),
                ("imdb-ceb", "all"), ("imdb-ceb", "large"), ("imdb-ceb", "fourteen")]:
        r = R[key]
        if prev is not None and prev != key[0]:
            lines.append(r"\addlinespace")
        prev = key[0]
        g, d = r["vs_geqo"], r["vs_dp"]
        lines.append(
            f"{r['name'] if key[1] == 'all' else ''} & {label[key[1]]} & {r['n']} & "
            f"{f1(r['tot']['dp'])} & {f1(r['tot']['geqo'])} & {f1(r['tot']['mcts'])} & "
            f"{f2(g['speedup'])} & {cifmt(g['ci'])} & {pfmt(g['p'])} & "
            f"{f2(d['speedup'])} & {cifmt(d['ci'])} & {pfmt(d['p'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular*}"]
    (OUT / "tab_imdb_e2e.tex").write_text("\n".join(lines) + "\n")

    # Planning time and estimated cost.
    lines = [r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llrrrrrrrrr@{}}",
             r"\toprule",
             r" & & \multicolumn{3}{c}{Mean planning time (ms)} & \multicolumn{3}{c}{Total execution time (s)}"
             r" & \multicolumn{3}{c}{Total estimated cost ($10^3$)} \\",
             r"\cmidrule(lr){3-5} \cmidrule(lr){6-8} \cmidrule(l){9-11}",
             r"Workload & Subset & DPSize & GEQO & MCTS-Extreme & DPSize & GEQO & MCTS-Extreme & DPSize & GEQO & MCTS-Extreme \\",
             r"\midrule"]
    prev = None
    for key in [("job", "all"), ("job", "large"), ("jobcomplex", "all"), ("jobcomplex", "large"),
                ("imdb-ceb", "all"), ("imdb-ceb", "large")]:
        r = R[key]
        if prev is not None and prev != key[0]:
            lines.append(r"\addlinespace")
        prev = key[0]
        lines.append(
            f"{r['name'] if key[1] == 'all' else ''} & {label[key[1]]} & "
            + " & ".join(f1(r["plan"][m]) for m in ("dp", "geqo", "mcts")) + " & "
            + " & ".join(f1(r["exec"][m]) for m in ("dp", "geqo", "mcts")) + " & "
            + " & ".join(f0(r["cost"][m]) for m in ("dp", "geqo", "mcts")) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular*}"]
    (OUT / "tab_imdb_plan_cost.tex").write_text("\n".join(lines) + "\n")

    # Macros for prose.
    nm = {"job": "Job", "jobcomplex": "Jobc", "imdb-ceb": "Ceb"}
    tg = {"all": "All", "large": "Large", "fourteen": "Fourteen"}
    for (src, tag), r in R.items():
        if src == "combined":
            for base, c in r.items():
                b = "Geqo" if base == "geqo" else "Dp"
                macro(f"CombLargeSp{b}", f2(c["speedup"]))
                macro(f"CombLargeCi{b}", cifmt(c["ci"]))
                macro(f"CombLargeP{b}", pfmt(c["p"]))
                macro(f"CombLargeN{b}", str(c["n"]))
            continue
        k = nm[src] + tg[tag]
        macro(f"{k}N", str(r["n"]))
        for base in ("geqo", "dp"):
            b = "Geqo" if base == "geqo" else "Dp"
            c = r["vs_" + base]
            macro(f"{k}Sp{b}", f2(c["speedup"]))
            macro(f"{k}Ci{b}", cifmt(c["ci"]))
            macro(f"{k}P{b}", pfmt(c["p"]))
            macro(f"{k}SeedMin{b}", f2(min(r["seed_sp"][base])))
        for m, mm in (("dp", "Dp"), ("geqo", "Geqo"), ("mcts", "Mcts")):
            macro(f"{k}Tot{mm}", f1(r["tot"][m]))
            macro(f"{k}Exec{mm}", f1(r["exec"][m]))
            macro(f"{k}Plan{mm}", f1(r["plan"][m]))
            macro(f"{k}Cost{mm}", f0(r["cost"][m]))
        macro(f"{k}PlanFrac", pct(r["planfrac"]))
    # Per-query spread on the large JOB queries (vs GEQO).
    s = S["job"]
    qs = s.common(("dp", "geqo", "mcts"), s.queries(rel_min=LARGE))
    sp = [s.pq(q, "geqo") / s.pq(q, "mcts") for q in qs]
    macro("JobLargeFasterGeqo", str(sum(1 for x in sp if x > 1 + TIE)))
    macro("JobLargeSlowerGeqo", str(sum(1 for x in sp if x < 1 - TIE)))
    macro("JobLargeMinSpGeqo", f2(min(sp)))
    macro("JobLargeMaxSpGeqo", f2(max(sp)))
    # Deployment mode: DPSize below 12 relations, MCTS-Extreme at or above,
    # against default PostgreSQL (DPSize below, GEQO at or above).
    for src, nm in (("job", "Job"), ("jobcomplex", "Jobc"), ("imdb-ceb", "Ceb")):
        s = S[src]
        qs = s.common(("dp", "geqo", "mcts"), s.queries())
        dflt = [s.pq(q, "geqo" if s.rels[q] >= LARGE else "dp") for q in qs]
        hyb = [s.pq(q, "mcts" if s.rels[q] >= LARGE else "dp") for q in qs]
        macro(f"Hyb{nm}Sp", f2(sum(dflt) / sum(hyb)))
        macro(f"Hyb{nm}Ci", cifmt(bootstrap_ratio(dflt, hyb)))
        macro(f"Hyb{nm}Red", pct(1 - sum(hyb) / sum(dflt)))
        AUDIT[f"hybrid_{src}"] = dict(n=len(qs), default=sum(dflt), hybrid=sum(hyb))
    # Worked example for the introduction: JOB query 29c.
    s = S["job"]
    macro("ExQuery", "29c")
    macro("ExRels", str(s.rels["29c"]))
    for m, mm in (("dp", "Dp"), ("geqo", "Geqo"), ("mcts", "Mcts")):
        macro(f"ExPlan{mm}", f"{s.pq('29c', m, lambda r: r['plan_time_ms']) / 1000:.2f}")
        macro(f"ExEtoe{mm}", f1(s.pq("29c", m) / 1000))
    diffs = [R[(src, "large")]["plan"]["mcts"] / R[(src, "large")]["plan"]["geqo"] - 1
             for src in ("job", "jobcomplex", "imdb-ceb")]
    macro("PlanMaxAboveGeqo", f"{100 * max(diffs):.1f}")
    jc = S["jobcomplex"]
    macro("JobcQTwelveRels", str(jc.rels["12"]))
    macro("JobcQTwelveToMcts", str(jc.tos["12"]["mcts"]))
    macro("JobcQTwelveToGeqo", str(jc.tos["12"]["geqo"]))
    macro("JobcQTwelveToDp", str(jc.tos["12"]["dp"]))
    macro("JobcQueries", str(len(jc.runs)))
    macro("JobQueries", str(len(S["job"].runs)))
    macro("CebQueries", str(len(S["imdb-ceb"].runs)))
    # CEB relation-count distribution.
    dist = defaultdict(int)
    for q in S["imdb-ceb"].runs:
        dist[S["imdb-ceb"].rels[q]] += 1
    AUDIT["ceb_rel_dist"] = dict(sorted(dist.items()))
    macro("CebRelMin", str(min(dist)))
    macro("CebRelMax", str(max(dist)))
    AUDIT["imdb"] = {f"{k[0]}:{k[1]}": v for k, v in R.items()}


def outcomes_and_winners(S):
    M3 = ("dp", "geqo", "mcts")
    out, win = {}, {}
    for src in ("job", "jobcomplex", "imdb-ceb"):
        s = S[src]
        qs = s.common(M3, s.queries())
        o = {}
        for base in ("dp", "geqo"):
            better = same = worse = 0
            for q in qs:
                ratio = s.pq(q, "mcts") / s.pq(q, base)
                if ratio < 1 - TIE:
                    better += 1
                elif ratio > 1 + TIE:
                    worse += 1
                else:
                    same += 1
            o[base] = dict(better=better, same=same, worse=worse, n=len(qs))
        out[src] = o
        cnt = {m: 0 for m in M3}
        for q in qs:
            v = {m: s.pq(q, m) for m in M3}
            best = min(v.values())
            for m in M3:
                if v[m] <= best * (1 + TIE):
                    cnt[m] += 1
        cnt["n"] = len(qs)
        win[src] = cnt
    # Figure data: shares in percent.
    lines = ["label,worse,same,better,n"]
    bn = {"dp": "DPSize", "geqo": "GEQO"}
    for src, nm in (("job", "JOB"), ("jobcomplex", "JOB-Complex"), ("imdb-ceb", "CEB")):
        for base in ("dp", "geqo"):
            o = out[src][base]
            lines.append(f"{{{nm} vs.\\ {bn[base]}}},{100*o['worse']/o['n']:.2f},{100*o['same']/o['n']:.2f},"
                         f"{100*o['better']/o['n']:.2f},{o['n']}")
    (OUT / "fig_outcomes.dat").write_text("\n".join(lines) + "\n")
    # Winner table.
    lines = [r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lrrrr@{}}",
             r"\toprule",
             r"Workload & Queries & DPSize & GEQO & MCTS-Extreme \\",
             r"\midrule"]
    tot = {m: 0 for m in ("dp", "geqo", "mcts", "n")}
    for src, nm in (("job", "JOB"), ("jobcomplex", "JOB-Complex"), ("imdb-ceb", "CEB")):
        w = win[src]
        for k in tot:
            tot[k] += w[k]
        lines.append(f"{nm} & {w['n']} & {w['dp']} & {w['geqo']} & {w['mcts']} \\\\")
    lines += [r"\midrule", f"Total & {tot['n']} & {tot['dp']} & {tot['geqo']} & {tot['mcts']} \\\\",
              r"\bottomrule", r"\end{tabular*}"]
    (OUT / "tab_winners.tex").write_text("\n".join(lines) + "\n")
    nm = {"job": "Job", "jobcomplex": "Jobc", "imdb-ceb": "Ceb"}
    for src, o in out.items():
        for base in ("dp", "geqo"):
            b = "Geqo" if base == "geqo" else "Dp"
            for k in ("better", "same", "worse"):
                macro(f"{nm[src]}Out{b}{k.capitalize()}", str(o[base][k]))
    for src, w in win.items():
        for m, mm in (("dp", "Dp"), ("geqo", "Geqo"), ("mcts", "Mcts")):
            macro(f"{nm[src]}Win{mm}", str(w[m]))
    macro("WinTotalMcts", str(tot["mcts"]))
    macro("WinTotalDp", str(tot["dp"]))
    macro("WinTotalGeqo", str(tot["geqo"]))
    macro("WinTotalN", str(tot["n"]))
    AUDIT["outcomes"] = out
    AUDIT["winners"] = win


def perquery_figures(S):
    M3 = ("dp", "geqo", "mcts")
    for src, fn in (("job", "fig_perquery_job.dat"), ("jobcomplex", "fig_perquery_jobc.dat")):
        s = S[src]
        qs = s.common(M3, s.queries(rel_min=LARGE))
        rows = [(q, s.pq(q, "dp") / s.pq(q, "mcts"), s.pq(q, "geqo") / s.pq(q, "mcts")) for q in qs]
        rows.sort(key=lambda t: -t[2])
        lines = ["x query vsdp vsgeqo"] + [f"{i} {q} {a:.4f} {b:.4f}" for i, (q, a, b) in enumerate(rows)]
        (OUT / fn).write_text("\n".join(lines) + "\n")
    s = S["imdb-ceb"]
    buckets = [("9--10", lambda n: n <= 10), ("11--12", lambda n: 11 <= n <= 12), ("14--16", lambda n: n >= 14)]
    lines = ["x bucket n vsdp vsgeqo"]
    bres = {}
    for i, (bn, pred) in enumerate(buckets):
        qs = s.common(M3, [q for q in s.queries() if pred(s.rels[q])])
        dp = sum(s.pq(q, "dp") for q in qs) / sum(s.pq(q, "mcts") for q in qs)
        ge = sum(s.pq(q, "geqo") for q in qs) / sum(s.pq(q, "mcts") for q in qs)
        bres[bn] = dict(n=len(qs), vs_dp=dp, vs_geqo=ge)
        lines.append(f"{i} {{{bn}}} {len(qs)} {dp:.4f} {ge:.4f}")
    (OUT / "fig_ceb_buckets.dat").write_text("\n".join(lines) + "\n")
    AUDIT["ceb_buckets"] = bres
    macro("CebBucketLowGeqo", f2(bres["9--10"]["vs_geqo"]))
    macro("CebBucketHighGeqo", f2(bres["14--16"]["vs_geqo"]))
    macro("CebBucketLowN", str(bres["9--10"]["n"]))
    macro("CebBucketHighN", str(bres["14--16"]["n"]))


# ============================================== learned MCTS optimizers
def learned(S):
    rels_job = S["job"].rels
    # Default PostgreSQL in the main runs: DPSize below 12 relations, GEQO at or above.
    def default_m(s, q):
        return "geqo" if s.rels[q] >= LARGE else "dp"

    def summarize(pairs, large_q):
        """pairs: q -> (baseline_e2e, method_e2e)."""
        n = len(pairs)
        worse = sum(1 for b, x in pairs.values() if x / b > 1 + TIE)
        better = sum(1 for b, x in pairs.values() if x / b < 1 - TIE)
        same = n - worse - better
        sp_all = sum(b for b, _ in pairs.values()) / sum(x for _, x in pairs.values())
        lp = {q: v for q, v in pairs.items() if q in large_q}
        sp_large = sum(b for b, _ in lp.values()) / sum(x for _, x in lp.values())
        return dict(n=n, worse=100 * worse / n, same=100 * same / n, better=100 * better / n,
                    sp_all=sp_all, sp_large=sp_large, n_large=len(lp))

    res = {}
    # JOB
    large_job = {q for q, n in rels_job.items() if n >= LARGE}
    default = {r["query"].replace(".sql", ""): float(r["postgres_e2e_ms"])
               for r in csv.DictReader(open(LEARNED / "job/pg19/default/perquery.csv"))}
    aj = {r["query"]: float(r["python_ms"]) + float(r["postgres_ms"])
          for r in csv.DictReader(open(LEARNED / "job/pg19/AlphaJoin/perquery.csv")) if r["iter"] == "1"}
    res[("job", "aj")] = summarize({q: (default[q], aj[q]) for q in aj if q in default}, large_job)
    hq_rows = list(csv.DictReader(open(LEARNED / "job/pg19/HyperQO/perquery.csv")))
    hq = {r["qname"].replace(".sql", ""): (float(r["pg_e2e_ms"]), float(r["hq_e2e_ms"])) for r in hq_rows}
    res[("job", "hq")] = summarize(hq, large_job)
    s = S["job"]
    qs = s.common(("dp", "geqo", "mcts"), s.queries())
    res[("job", "mcts")] = summarize({q: (s.pq(q, default_m(s, q)), s.pq(q, "mcts")) for q in qs}, large_job)
    # JOB-Complex (the common query set of the main runs, i.e. without query 12)
    s = S["jobcomplex"]
    qs = s.common(("dp", "geqo", "mcts"), s.queries())
    large_jc = {q for q in qs if s.rels[q] >= LARGE}
    hqc_rows = list(csv.DictReader(open(LEARNED / "jobcomplex/pg19/HyperQO/perquery.csv")))
    hqc = {Path(r["sql_file"]).stem: (float(r["pg_e2e_ms"]), float(r["hq_e2e_ms"])) for r in hqc_rows}
    res[("jobc", "hq")] = summarize({q: hqc[q] for q in qs if q in hqc}, large_jc)
    res[("jobc", "mcts")] = summarize({q: (s.pq(q, default_m(s, q)), s.pq(q, "mcts")) for q in qs}, large_jc)

    def row(nm, r):
        return (f"{nm} & {r['worse']:.1f} & {r['same']:.1f} & {r['better']:.1f} & "
                f"{r['sp_all']:.2f} & {r['sp_large']:.2f} \\\\")
    lines = [r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lrrrrr@{}}",
             r"\toprule",
             r" & \multicolumn{3}{c}{Queries (\%)} & \multicolumn{2}{c}{Speedup} \\",
             r"\cmidrule(lr){2-4} \cmidrule(l){5-6}",
             r"Optimizer & Worse & Tie & Better & All & $n \geq 12$ \\",
             r"\midrule",
             rf"\multicolumn{{6}}{{l}}{{JOB ({res[('job','mcts')]['n']} queries)}} \\",
             row("AlphaJoin", res[("job", "aj")]),
             row("HyperQO", res[("job", "hq")]),
             row("MCTS-Extreme", res[("job", "mcts")]),
             r"\addlinespace",
             rf"\multicolumn{{6}}{{l}}{{JOB-Complex ({res[('jobc','mcts')]['n']} queries)}} \\",
             r"AlphaJoin & \multicolumn{5}{c}{did not run} \\",
             row("HyperQO", res[("jobc", "hq")]),
             row("MCTS-Extreme", res[("jobc", "mcts")]),
             r"\bottomrule", r"\end{tabular*}"]
    (OUT / "tab_learned.tex").write_text("\n".join(lines) + "\n")
    # Prose numbers.
    fallback = sum(1 for r in hq_rows if r["hint"].strip() == "PG") / len(hq_rows)
    fallback_c = sum(1 for r in hqc_rows if r["hint"].strip() == "PG") / len(hqc_rows)
    exec_impr = mean((float(r["pg_exec_ms"]) - float(r["hq_exec_ms"])) / float(r["pg_exec_ms"]) for r in hq_rows)
    aj_ratio = sorted(aj[q] / default[q] for q in aj)
    aj_median = aj_ratio[len(aj_ratio) // 2] if len(aj_ratio) % 2 else \
        (aj_ratio[len(aj_ratio) // 2 - 1] + aj_ratio[len(aj_ratio) // 2]) / 2
    macro("HqFallbackJob", pct(fallback))
    macro("HqFallbackJobc", pct(fallback_c))
    macro("HqExecImprJob", pct(exec_impr))
    macro("HqJobQueries", str(len(hq_rows)))
    macro("HqJobcQueries", str(len(hqc_rows)))
    macro("AjMedianSlowdown", f"{aj_median:.0f}")
    macro("AjSpAll", f2(res[("job", "aj")]["sp_all"]))
    macro("HqSpAllJob", f2(res[("job", "hq")]["sp_all"]))
    macro("HqSpAllJobc", f2(res[("jobc", "hq")]["sp_all"]))
    macro("MctsDefaultSpJob", f2(res[("job", "mcts")]["sp_all"]))
    macro("MctsDefaultSpJobLarge", f2(res[("job", "mcts")]["sp_large"]))
    macro("MctsDefaultSpJobc", f2(res[("jobc", "mcts")]["sp_all"]))
    macro("MctsDefaultSpJobcLarge", f2(res[("jobc", "mcts")]["sp_large"]))
    macro("MctsDefaultRedJob", pct(1 - 1 / res[("job", "mcts")]["sp_all"]))
    macro("MctsDefaultRedJobc", pct(1 - 1 / res[("jobc", "mcts")]["sp_all"]))
    AUDIT["learned"] = {f"{k[0]}:{k[1]}": v for k, v in res.items()}
    AUDIT["hq_fallback"] = dict(job=fallback, jobc=fallback_c, exec_impr=exec_impr, aj_median=aj_median)


# ================================================== ablation (IMDb CEB)
def ablation(ceb_rows, S):
    rows = [r for r in ceb_rows if r["config"] == "single"]
    cfgs = defaultdict(Series)
    for r in rows:
        g = r["gucs"]
        key = (r["reward"], r["agg"], int(g.get("kernels", 1)), int(g["depth"]), int(g["top_k"]))
        cfgs[key].add(r["query"], "x", r)
    keys = list(cfgs)
    rels = S["imdb-ceb"].rels
    all_q = sorted({q for k in keys for q in cfgs[k].runs})
    common = [q for q in all_q if all(cfgs[k].tos[q]["x"] == 0 and cfgs[k].runs[q]["x"] > 0 for k in keys)]
    large = [q for q in common if rels[q] >= LARGE]
    default = ("neg_log", "best", 1, 10, 10)

    def tot(k, qs, f):
        return sum(mean(f(r) for r in cfgs[k].vals[q]["x"]) for q in qs)

    res = {}
    for k in keys:
        res[k] = dict(
            e2e_all=tot(k, common, e2e), e2e_large=tot(k, large, e2e),
            cost_large=tot(k, large, lambda r: r["mcts_best_cost"]),
            cost_all=tot(k, common, lambda r: r["mcts_best_cost"]),
            plan=mean(r["plan_time_ms"] for q in all_q for r in cfgs[k].vals[q]["x"]),
            to=sum(cfgs[k].tos[q]["x"] for q in all_q),
            to_large=sum(cfgs[k].tos[q]["x"] for q in all_q if rels.get(q, 0) >= LARGE))
    d = res[default]
    groups = [
        ("Default", [default]),
        ("Reward map", [("neg_log", "best", 1, 4, 5), ("norm_neg_log", "best", 1, 4, 5),
                        ("neg_cost", "best", 1, 4, 5)]),
        ("Selection rule", [("neg_log", "average", 1, 10, 10)]),
        ("Shape parameter", [("neg_log", "best", 2, 10, 10), ("neg_log", "best", 3, 10, 10)]),
        ("Top-k filter", [("neg_log", "best", 1, 10, k) for k in (8, 9, 11, 12)]),
    ]
    rname = {"neg_log": r"$-\log \widehat{C}$", "norm_neg_log": r"normalized $-\log \widehat{C}$",
             "neg_cost": r"$-\widehat{C}$"}
    sname = {"best": "UCT-Extreme", "average": "UCT"}
    lines = [r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lllrrrrrrrr@{}}",
             r"\toprule",
             r" & & & & & & \multicolumn{2}{c}{End-to-end time} & Estimated cost & Mean & \\",
             r" & & & & & & \multicolumn{2}{c}{relative to default} & relative to default & planning & \\",
             r"\cmidrule(lr){7-8} \cmidrule(lr){9-9}",
             r"Varied knob & Reward map & Selection & $K$ & $d$ & $k$ & All & $n \geq 12$ & $n \geq 12$ & time (ms) & Timeouts \\",
             r"\midrule"]
    for gname, ks in groups:
        if gname != "Default":
            lines.append(r"\addlinespace")
        for i, k in enumerate(ks):
            r = res[k]
            lines.append(
                f"{gname if i == 0 else ''} & {rname[k[0]]} & {sname[k[1]]} & {k[2]} & {k[3]} & {k[4]} & "
                f"{r['e2e_all']/d['e2e_all']:.3f} & {r['e2e_large']/d['e2e_large']:.3f} & "
                f"{r['cost_large']/d['cost_large']:.3f} & {r['plan']:.1f} & {r['to']} \\\\")
    lines += [r"\bottomrule", r"\end{tabular*}"]
    (OUT / "tab_ablation.tex").write_text("\n".join(lines) + "\n")
    macro("AblCommonN", str(len(common)))
    macro("AblCommonLargeN", str(len(large)))
    macro("AblAllN", str(len(all_q)))
    macro("AblToKTwo", str(res[("neg_log", "best", 2, 10, 10)]["to"]))
    macro("AblToKThree", str(res[("neg_log", "best", 3, 10, 10)]["to"]))
    k2, k3 = res[("neg_log", "best", 2, 10, 10)], res[("neg_log", "best", 3, 10, 10)]
    macro("AblKTwoRel", f"{k2['e2e_all']/d['e2e_all']:.2f}")
    macro("AblKThreeRel", f"{k3['e2e_all']/d['e2e_all']:.2f}")
    nc = res[("neg_cost", "best", 1, 4, 5)]
    nl = res[("neg_log", "best", 1, 4, 5)]
    macro("AblNegCostVsNegLog", f"{nc['e2e_all']/nl['e2e_all']:.2f}")
    topk = [res[("neg_log", "best", 1, 10, k)]["e2e_all"] / d["e2e_all"] for k in (8, 9, 11, 12)]
    macro("AblTopkMin", f"{min(topk):.3f}")
    macro("AblTopkMax", f"{max(topk):.3f}")

    # UCT-Extreme vs UCT (paired, both complete).
    a = cfgs[("neg_log", "average", 1, 10, 10)]
    b = cfgs[default]
    uct = {}
    for tag, qsel in (("All", all_q), ("Large", [q for q in all_q if rels.get(q, 0) >= LARGE])):
        qs = [q for q in qsel if a.tos[q]["x"] == 0 and b.tos[q]["x"] == 0
              and a.vals[q]["x"] and b.vals[q]["x"]]
        for mname, f in (("Etoe", e2e), ("Cost", lambda r: r["mcts_best_cost"])):
            xa = [mean(f(r) for r in a.vals[q]["x"]) for q in qs]
            xb = [mean(f(r) for r in b.vals[q]["x"]) for q in qs]
            ratio = sum(xa) / sum(xb)
            ci = bootstrap_ratio(xa, xb)
            p = wilcoxon(xa, xb)
            uct[(tag, mname)] = dict(n=len(qs), ratio=ratio, ci=ci, p=p)
            macro(f"Uct{tag}{mname}Ratio", f"{ratio:.3f}")
            macro(f"Uct{tag}{mname}Ci", f"[{ci[0]:.3f}, {ci[1]:.3f}]")
            macro(f"Uct{tag}{mname}P", pfmt(p))
            macro(f"Uct{tag}{mname}Red", pct(1 - 1 / ratio))
    AUDIT["ablation"] = {"|".join(map(str, k)): v for k, v in res.items()}
    AUDIT["ablation_common"] = dict(n=len(common), large=len(large), all=len(all_q))
    AUDIT["uct"] = {f"{k[0]}:{k[1]}": v for k, v in uct.items()}


# ========================================= linear-tree search controls
def search_controls(S):
    rows = load_jsonl(SAIO_RUN / "results.jsonl")
    s = Series()
    for r in rows:
        m = {"saio_single_kernel_480": "saio", "iter_improve_single_kernel_480": "ii"}[r["combo_id"]]
        s.add(r["query"], m, r)
    ceb = S["imdb-ceb"]
    for q in list(s.runs):
        if q in ceb.vals:
            for r in ceb.vals[q]["mcts"]:
                s.add(q, "mcts", r)
            s.tos[q]["mcts"] += ceb.tos[q]["mcts"]
            s.runs[q]["mcts"] += ceb.tos[q]["mcts"]
            s.rels[q] = max(s.rels.get(q, 0), ceb.rels.get(q, 0))
    large = [q for q in sorted(s.runs) if s.rels.get(q, 0) >= LARGE]
    common = s.common(("mcts", "saio", "ii"), large)
    res = {}
    for m in ("mcts", "saio", "ii"):
        res[m] = dict(
            cost=sum(s.pq(q, m, lambda r: r["mcts_best_cost"]) for q in common),
            plan=sum(s.pq(q, m, lambda r: r["plan_time_ms"]) for q in common),
            exec=sum(s.pq(q, m, lambda r: r["exec_time_ms"]) for q in common),
            e2e=sum(s.pq(q, m) for q in common),
            iters=sum(s.pq(q, m, lambda r: r["iters"]) for q in common),
            to=s.timeouts(m, large)[0], runs=s.timeouts(m, large)[1])
    nm = {"mcts": "MCTS-Extreme", "saio": "SAIO", "ii": "Iterated improvement"}
    lines = [r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrrrr@{}}",
             r"\toprule",
             r"Search & Estimated cost ($10^6$) & Planning (s) & Execution (s) & End-to-end (s)"
             r" & Iterations ($10^3$) & Timeouts \\",
             r"\midrule"]
    for m in ("mcts", "saio", "ii"):
        r = res[m]
        lines.append(f"{nm[m]} & {r['cost']/1e6:.1f} & {r['plan']/1e3:.1f} & {r['exec']/1e3:.1f} & "
                     f"{r['e2e']/1e3:.1f} & {r['iters']/1e3:.1f} & {r['to']} of {r['runs']} \\\\")
    lines += [r"\bottomrule", r"\end{tabular*}"]
    (OUT / "tab_controls.tex").write_text("\n".join(lines) + "\n")
    macro("CtlCommonN", str(len(common)))
    macro("CtlLargeN", str(len(large)))
    macro("CtlRuns", str(res["mcts"]["runs"]))
    for m, mm in (("mcts", "Mcts"), ("saio", "Saio"), ("ii", "Ii")):
        macro(f"CtlTo{mm}", str(res[m]["to"]))
        macro(f"CtlEtoe{mm}", f1(res[m]["e2e"] / 1e3))
    for m, mm in (("mcts", "Mcts"), ("saio", "Saio"), ("ii", "Ii")):
        macro(f"CtlPlan{mm}", f1(res[m]["plan"] / 1e3))
        macro(f"CtlExec{mm}", f1(res[m]["exec"] / 1e3))
    macro("CtlSpSaio", f2(res["saio"]["e2e"] / res["mcts"]["e2e"]))
    macro("CtlSpIi", f2(res["ii"]["e2e"] / res["mcts"]["e2e"]))
    AUDIT["controls"] = dict(common=len(common), large=len(large), res=res)


# ====================================================== Stack (PostgreSQL)
def stack():
    s = Series()
    for r in load_jsonl(REB_RUNS / "stack-mcts-vs-geqo-200" / "results.jsonl"):
        s.add(r["query"], "mcts" if r["config"] == "single" else "geqo", r)
    for run in ("dp-stack-12plus", "dp-stack-12minus"):
        for r in load_jsonl(REB_RUNS / run / "results.jsonl"):
            s.add(r["query"], "dp", r)
    res = {}
    for tag, qs in (("Twelve", s.queries(rel_eq={12})), ("All", s.queries())):
        out = {"n_all": len(qs)}
        for base in ("geqo", "dp"):
            c = s.common(("mcts", base), qs)
            out[base] = s.compare(base, "mcts", c)
        for m in ("mcts", "geqo", "dp"):
            out["to_" + m] = s.timeouts(m, qs)
        out["plan"] = {m: mean(r["plan_time_ms"] for q in qs for r in s.vals[q][m]) for m in ("mcts", "geqo", "dp")}
        res[tag] = out
    dist = defaultdict(int)
    for q in s.runs:
        dist[s.rels.get(q)] += 1
    lines = [r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}llrrrr@{}}",
             r"\toprule",
             r"Subset & Optimizer & Timeouts & Speedup & 95\% CI & Queries \\",
             r"\midrule"]
    for tag, lab in (("Twelve", r"$n = 12$"), ("All", "all")):
        r = res[tag]
        lines.append(f"{lab} & MCTS-Extreme & {r['to_mcts'][0]} of {r['to_mcts'][1]} & & & \\\\")
        for base, bn in (("geqo", "GEQO"), ("dp", "DPSize")):
            c = r[base]
            lines.append(f" & {bn} & {r['to_' + base][0]} of {r['to_' + base][1]} & {f2(c['speedup'])} & "
                         f"{cifmt(c['ci'])} & {c['n']} \\\\")
        if tag == "Twelve":
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular*}"]
    (OUT / "tab_stack.tex").write_text("\n".join(lines) + "\n")
    for tag, r in res.items():
        for base, b in (("geqo", "Geqo"), ("dp", "Dp")):
            c = r[base]
            macro(f"Stack{tag}Sp{b}", f2(c["speedup"]))
            macro(f"Stack{tag}Ci{b}", cifmt(c["ci"]))
            macro(f"Stack{tag}P{b}", pfmt(c["p"]))
            macro(f"Stack{tag}N{b}", str(c["n"]))
        for m, mm in (("mcts", "Mcts"), ("geqo", "Geqo"), ("dp", "Dp")):
            macro(f"Stack{tag}To{mm}", str(r["to_" + m][0]))
            macro(f"Stack{tag}Runs{mm}", str(r["to_" + m][1]))
            macro(f"Stack{tag}Plan{mm}", f1(r["plan"][m]))
        macro(f"Stack{tag}Queries", str(r["n_all"]))
    macro("StackDistTwelve", str(dist[12]))
    macro("StackDistEleven", str(dist[11]))
    macro("StackDistNine", str(dist[9]))
    tw = res["Twelve"]
    macro("StackTwelveDoneMcts", str(tw["to_mcts"][1] - tw["to_mcts"][0]))
    macro("StackTwelveDoneGeqo", str(tw["to_geqo"][1] - tw["to_geqo"][0]))
    macro("StackTwelveToGeqoPct", f"{100 * tw['to_geqo'][0] / tw['to_geqo'][1]:.0f}")
    AUDIT["stack"] = res
    AUDIT["stack_dist"] = dict(dist)


# ============================================================== DuckDB port
def duckdb():
    runs = (("Job", "duckdb-job-12plus", "JOB"), ("Jobc", "duckdb-jobcomplex-12plus", "JOB-Complex"),
            ("Ceb", "duckdb-ceb-12plus", "CEB"))
    cells = {"": "native", "retTrue_ret0.5": "sel", "retTrue_ret1.0": "nogroup",
             "retFalse_ret0.5": "noret_a", "retFalse_ret1.0": "noret_b"}
    res = {}
    for tag, run, nm in runs:
        s = Series()
        nseeds = set()
        for r in load_jsonl(REB_RUNS / run / "results.jsonl"):
            m = cells[r["combo_id"]]
            if r.get("status") != "ok":
                r = dict(r, timed_out=True)
            s.add(r["query"], m, r)
            nseeds.add(r["seed"])
        qs = s.queries()
        out = {"nm": nm, "n_all": len(qs), "seeds": len(nseeds)}
        for m in ("sel", "nogroup", "noret_a", "noret_b"):
            c = s.common(("native", m), qs)
            out[m] = s.compare("native", m, c, f=lambda r: r["e2e_time_ms"])
            out["to_" + m] = s.timeouts(m, qs)
        out["to_native"] = s.timeouts("native", qs)
        out["plan"] = {m: mean(r["plan_time_ms"] for q in qs for r in s.vals[q][m]) for m in ("native", "sel")}
        out["mismatch"] = sum(1 for q in qs for m in s.vals[q] for r in s.vals[q][m]
                              if r.get("correctness") not in (None, "matched"))
        out["fallbacks"] = sum(int(r.get("fallback_count") or 0) for q in qs for m in s.vals[q] for r in s.vals[q][m])
        res[tag] = out
    lines = [r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lrrrrr@{}}",
             r"\toprule",
             r" & & & \multicolumn{2}{c}{With retention} & Without \\",
             r"\cmidrule(lr){4-5} \cmidrule(l){6-6}",
             r"Workload & Queries & Seeds & Speedup & 95\% CI & Speedup \\",
             r"\midrule"]
    for tag in ("Job", "Jobc", "Ceb"):
        r = res[tag]
        c = r["sel"]
        lines.append(f"{r['nm']} & {c['n']} & {r['seeds']} & {f2(c['speedup'])} & {cifmt(c['ci'])} & "
                     f"{f2(r['noret_a']['speedup'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular*}"]
    (OUT / "tab_duckdb.tex").write_text("\n".join(lines) + "\n")
    for tag, r in res.items():
        c = r["sel"]
        macro(f"Duck{tag}Sp", f2(c["speedup"]))
        macro(f"Duck{tag}Ci", cifmt(c["ci"]))
        macro(f"Duck{tag}P", pfmt(c["p"]))
        macro(f"Duck{tag}N", str(c["n"]))
        macro(f"Duck{tag}Seeds", str(r["seeds"]))
        macro(f"Duck{tag}SpNoRet", f2(r["noret_a"]["speedup"]))
        macro(f"Duck{tag}SpNoGroup", f2(r["nogroup"]["speedup"]))
        macro(f"Duck{tag}PlanNative", f1(r["plan"]["native"]))
        macro(f"Duck{tag}PlanSel", f1(r["plan"]["sel"]))
    macro("DuckTimeouts", str(sum(res[t]["to_sel"][0] + res[t]["to_native"][0] for t in res)))
    macro("DuckMismatches", str(sum(res[t]["mismatch"] for t in res)))
    AUDIT["duckdb"] = res


# =================================================================== main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    S, ceb_rows = imdb()
    R = imdb_tables(S)
    write_imdb(S, R)
    outcomes_and_winners(S)
    perquery_figures(S)
    learned(S)
    ablation(ceb_rows, S)
    search_controls(S)
    stack()
    duckdb()
    with open(OUT / "numbers.tex", "w") as f:
        f.write("% Generated by gen_tfmctsqo_numbers.py from saved results. Do not edit.\n")
        for k in sorted(MACROS):
            f.write(f"\\newcommand*{{\\N{k}}}{{{MACROS[k]}}}\n")
    def default(o):
        if isinstance(o, tuple):
            return list(o)
        return str(o)
    (HERE / "tfmctsqo_numbers.json").write_text(json.dumps(AUDIT, indent=1, default=default))
    print(f"wrote {len(MACROS)} macros and {len(list(OUT.glob('*')))} files to {OUT}")


if __name__ == "__main__":
    main()
