import argparse
import csv
import json
import os
import random
import sys

from ImportantConfig import Config
from sql2fea import TreeBuilder, value_extractor, Sql2Vec
from NET import TreeNet
from TreeLSTM import SPINN
from Hinter import Hinter
from mcts import MCTSHinterSearch
import mcts as mcts_mod


def load_sql_dir(sql_dir):
    files = []
    for name in os.listdir(sql_dir):
        if name.lower().endswith(".sql"):
            files.append(os.path.join(sql_dir, name))
    files.sort()
    queries = []
    for path in files:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            sql = f.read().strip()
            if sql:
                queries.append((path, sql))
    return queries


def percentile(data, p):
    if not data:
        return None
    data = sorted(data)
    k = (len(data) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(data) - 1)
    if f == c:
        return data[f]
    d0 = data[f] * (c - k)
    d1 = data[c] * (k - f)
    return d0 + d1


def summarize(label, arr):
    total = sum(arr)
    return {
        "label": label,
        "count": len(arr),
        "total_ms": total,
        "mean_ms": total / len(arr) if arr else 0.0,
        "p50_ms": percentile(arr, 50),
        "p75_ms": percentile(arr, 75),
        "p99_ms": percentile(arr, 99),
        "p995_ms": percentile(arr, 99.5),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run HyperQO on a directory of .sql files and compute E2E stats."
    )
    parser.add_argument("--sql-dir", required=True, help="Directory with .sql files")
    parser.add_argument(
        "--log-file", default="log_c3_h64_s4_t3.txt", help="Log file path"
    )
    parser.add_argument(
        "--max-queries", type=int, default=None, help="Limit number of queries"
    )
    parser.add_argument(
        "--mcts-model", default=None, help="Path to a trained MCTS model .pth"
    )
    parser.add_argument(
        "--auto-load-mcts",
        action="store_true",
        help="Auto-load MCTS model from model/<log_basename>.pth when available",
    )
    parser.add_argument(
        "--stats-json", default=None, help="Write summary + per-query to JSON"
    )
    parser.add_argument("--stats-csv", default=None, help="Write per-query rows to CSV")
    return parser.parse_args()


def main():
    args = parse_args()
    queries = load_sql_dir(args.sql_dir)
    if args.max_queries is not None:
        queries = queries[: args.max_queries]
    if not queries:
        print("No .sql files found.", file=sys.stderr)
        return 1

    config = Config()
    config.log_file = args.log_file
    mcts_mod.config.log_file = args.log_file

    random.seed(113)
    try:
        import torch

        torch.manual_seed(113)
    except Exception:
        pass

    tree_builder = TreeBuilder()
    sql2vec = Sql2Vec()
    value_network = SPINN(
        head_num=config.head_num,
        input_size=7 + 2,
        hidden_size=config.hidden_size,
        table_num=50,
        sql_size=40 * 40 + config.max_column,
    ).to(config.device)

    for name, param in value_network.named_parameters():
        from torch.nn import init

        if len(param.shape) == 2:
            init.xavier_normal(param)
        else:
            init.uniform(param)

    net = TreeNet(tree_builder=tree_builder, value_network=value_network)
    mcts_searcher = MCTSHinterSearch()
    hinter = Hinter(
        model=net,
        sql2vec=sql2vec,
        value_extractor=value_extractor,
        mcts_searcher=mcts_searcher,
    )

    # Load trained MCTS model if provided or auto-detected.
    if args.mcts_model:
        mcts_mod.predictionNet.load_state_dict(
            torch.load(args.mcts_model, map_location=config.cpudevice)
        )
    if args.auto_load_mcts:
        base = os.path.splitext(os.path.basename(args.log_file))[0]
        auto_path = os.path.join(config.modelpath, f"{base}.pth")
        if os.path.exists(auto_path):
            mcts_searcher.loadmodel()

    log_f = open(args.log_file, "w", encoding="utf-8")

    per_query = []
    pg_exec = []
    hq_exec = []
    pg_e2e = []
    hq_e2e = []

    print(len(queries), file=log_f)

    for idx, (path, sql) in enumerate(queries):
        print("----", idx, "-----", file=log_f)
        (
            pg_plan_time,
            pg_latency,
            mcts_time,
            hinter_plan_time,
            mphe_time,
            hinter_latency,
            actual_plans,
            actual_time,
        ) = hinter.hinterRun(sql)

        # Convert seconds-based timers to ms so all metrics share one unit.
        mcts_time *= 1000.0
        mphe_time *= 1000.0

        print("pg plan:", pg_plan_time, "pg run:", pg_latency, file=log_f)
        print(
            "mcts:",
            mcts_time,
            "plan gen:",
            hinter_plan_time,
            "MPHE:",
            mphe_time,
            "hinter latency:",
            hinter_latency,
            file=log_f,
        )
        print([actual_plans, actual_time], file=log_f)
        log_f.flush()

        hint = "PG"
        if isinstance(actual_plans, list) and actual_plans:
            hint = actual_plans[0]

        e2e_pg = pg_plan_time + pg_latency
        e2e_hq = mcts_time + hinter_plan_time + mphe_time + hinter_latency

        per_query.append(
            {
                "index": idx,
                "sql_file": path,
                "pg_exec_ms": pg_latency,
                "hq_exec_ms": hinter_latency,
                "pg_e2e_ms": e2e_pg,
                "hq_e2e_ms": e2e_hq,
                "exec_delta_ms": hinter_latency - pg_latency,
                "e2e_delta_ms": e2e_hq - e2e_pg,
                "hint": hint,
            }
        )

        pg_exec.append(pg_latency)
        hq_exec.append(hinter_latency)
        pg_e2e.append(e2e_pg)
        hq_e2e.append(e2e_hq)

    log_f.close()

    summary = {
        "pg_exec": summarize("pg_exec", pg_exec),
        "hq_exec": summarize("hq_exec", hq_exec),
        "pg_e2e": summarize("pg_e2e", pg_e2e),
        "hq_e2e": summarize("hq_e2e", hq_e2e),
        "queries": len(per_query),
    }

    print(json.dumps(summary, indent=2))

    if args.stats_json:
        with open(args.stats_json, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "per_query": per_query}, f, indent=2)

    if args.stats_csv:
        with open(args.stats_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "index",
                    "sql_file",
                    "pg_exec_ms",
                    "hq_exec_ms",
                    "pg_e2e_ms",
                    "hq_e2e_ms",
                    "exec_delta_ms",
                    "e2e_delta_ms",
                    "hint",
                ],
            )
            writer.writeheader()
            for row in per_query:
                writer.writerow(row)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
