"""Shared-core sweep execution using the existing cell and run formats."""

import datetime as dt
import hashlib
import json
import math
import os
import signal
from dataclasses import asdict

import lib
from adapters import get_adapter
from adapters.cache import cold_file


def resolve_cells(cfg, cells):
    import sweep

    adapter = get_adapter(cfg.implementation)
    resolved = []
    for mode, reward, agg, cid, combo in cells:
        if mode in adapter.BASELINES:
            settings = {}
        elif mode in {"single", "luby"}:
            merged = sweep.apply_gucs(cfg, combo)[mode == "single"]
            settings = adapter.settings_for(mode, merged, reward, agg, 1)
        else:
            raise ValueError(f"{cfg.implementation} does not support config {mode!r}")
        resolved.append((mode, reward, agg, cid, settings))
    if not resolved or not cfg.queries or cfg.n_seeds < 1:
        raise ValueError("Select at least one query, cell, and seed")
    seeds = cfg.seed_values or list(range(1, cfg.n_seeds + 1))
    if len(set(seeds)) != len(seeds) or any(type(s) is not int or s <= 0 for s in seeds):
        raise ValueError("Seeds must be distinct positive integers")
    if cfg.runtime.get("cache", "cold") not in {"cold", "warm"}:
        raise ValueError("runtime.cache must be cold or warm")
    timeout = cfg.runtime.get("timeout_seconds", 30)
    if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("runtime.timeout_seconds must be positive")
    return adapter, resolved, seeds


def answer_key(answer):
    # Bag semantics: preserve duplicate rows, ignore unspecified result order.
    rows = sorted(json.dumps(row, sort_keys=True, ensure_ascii=True) for row in answer)
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()


def run(cfg, args, cli_gucs, cells, query_paths):
    import sweep

    adapter, resolved, seeds = resolve_cells(cfg, cells)
    if args.parallel != 1:
        raise ValueError("Shared-core latency measurements require --parallel 1")
    if args.prewarm:
        raise ValueError("Use runtime.cache='warm' for shared-core warm-cache measurements")
    if args.dry_run:
        print(f"Implementation: {cfg.implementation} ({adapter.ENGINE})")
        print(f"Queries: {len(query_paths)}, cells: {len(resolved)}, seeds: {seeds}")
        print(f"Total runs: {len(query_paths) * len(resolved) * len(seeds)}")
        print(
            f"Cache: {cfg.runtime.get('cache', 'cold')}, timeout: {cfg.runtime.get('timeout_seconds', 30)}s"
        )
        for mode, reward, agg, cid, settings in resolved:
            print(
                lib.short_label(f"{cfg.implementation}:{mode}", reward, agg, cid),
                json.dumps(settings, sort_keys=True),
            )
        return 0
    adapter.validate_runtime(cfg)
    timestamp = dt.datetime.now().strftime("%Y-%m-%dT%H-%M-%S-%f")
    run_dir = args.runs_root / f"{timestamp}-{cfg.name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    cfg.n_seeds = len(seeds)
    dump = asdict(cfg)
    dump.update(cli_gucs=cli_gucs, n_combos=len({c[3] for c in resolved}), resolved_cells=resolved)
    (run_dir / "config.json").write_text(json.dumps(dump, indent=2) + "\n")
    (run_dir / "README.md").write_text(
        f"# {cfg.name}\n\n{cfg.implementation}, {len(query_paths)} queries, {len(resolved)} cells, {len(seeds)} seeds.\n"
    )
    (run_dir / "pid").write_text(str(os.getpid()))
    lib.write_status(run_dir, lib.STATUS_RUNNING)
    print(f"Run dir: {run_dir}", flush=True)
    failures = 0
    answers = {}
    total = len(query_paths) * len(resolved) * len(seeds)
    completed = 0

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        for qi, (query, path) in enumerate(query_paths.items()):
            for si, seed in enumerate(seeds):
                offset = (qi + si) % len(resolved)
                ordered = resolved[offset:] + resolved[:offset]
                for mode, reward, agg, cid, settings in ordered:
                    config = f"{cfg.implementation}:{mode}"
                    directory = (
                        config
                        if mode in adapter.BASELINES
                        else f"{config}__{reward}__{agg}" + (f"__{cid}" if cid else "")
                    )
                    folder = run_dir / directory / query
                    folder.mkdir(parents=True, exist_ok=True)
                    output = folder / f"seed-{seed}"
                    actual = dict(settings)
                    actual[
                        "seed"
                        if adapter.ENGINE == "duckdb" or mode in adapter.BASELINES
                        else "random_seed"
                    ] = seed
                    row = {
                        "query": query,
                        "source": str(path.parent),
                        "config": config,
                        "implementation": cfg.implementation,
                        "engine": adapter.ENGINE,
                        "mode": mode,
                        "reward": reward,
                        "agg": agg,
                        "combo_id": cid,
                        "gucs": actual,
                        "seed": seed,
                        "measurement": cfg.measurement,
                        "cache": cfg.runtime.get("cache", "cold"),
                        "status": "ok",
                        "timed_out": False,
                        "plan_path": str(output.with_suffix(".plan").relative_to(run_dir)),
                    }
                    try:
                        if row["cache"] == "cold":
                            if adapter.ENGINE == "duckdb":
                                cold_file(cfg.runtime["database"])
                            else:
                                sweep.restart_cluster(cfg)
                                sweep.drop_os_caches(args.drop_cache_command)
                        result = adapter.measure(cfg, mode, actual, path.read_text(), output)
                        answer = result.pop("answer", None)
                        if answer is not None:
                            output.with_suffix(".answer.json").write_text(json.dumps(answer))
                            row["answer_hash"] = answer_key(answer)
                            key = (query, seed)
                            answers.setdefault(key, []).append((row, mode in adapter.BASELINES))
                            row["correctness"] = "pending"
                        else:
                            row["correctness"] = "not_checked"
                        row.update(result)
                    except (TimeoutError, RuntimeError, ValueError, OSError) as exc:
                        failures += 1
                        row.update(
                            status="timeout" if isinstance(exc, TimeoutError) else "error",
                            timed_out=isinstance(exc, TimeoutError),
                            error=str(exc),
                            correctness="not_checked",
                        )
                        # Failed measurements stay visible in workload totals as censored penalties.
                        if cfg.measurement != "cost_only":
                            row.update(
                                e2e_time_ms=cfg.runtime.get("timeout_seconds", 30) * 1000,
                                exec_time_ms=cfg.runtime.get("timeout_seconds", 30) * 1000,
                                plan_time_ms=0,
                                penalized=True,
                            )
                    sweep.append_jsonl(run_dir / "results.jsonl", row)
                    completed += 1
                    print(
                        f"[{completed}/{total}] {query} {config} {cid} seed={seed}: {row['status']}",
                        flush=True,
                    )
        # Compare against a successful native cell after balanced scheduling has finished.
        rows = [json.loads(line) for line in (run_dir / "results.jsonl").read_text().splitlines()]
        for row in rows:
            if row.get("correctness") != "pending":
                continue
            references = {
                r["answer_hash"]
                for r, baseline in answers.get((row["query"], row["seed"]), [])
                if baseline
            }
            if not references:
                row["correctness"] = "no_baseline"
            elif references == {row["answer_hash"]}:
                row["correctness"] = "matched"
            else:
                row.update(
                    correctness="mismatch",
                    status="error",
                    error="Answer differs from native baseline",
                    penalized=True,
                )
                row.update(
                    e2e_time_ms=cfg.runtime.get("timeout_seconds", 30) * 1000,
                    exec_time_ms=cfg.runtime.get("timeout_seconds", 30) * 1000,
                    plan_time_ms=0,
                )
                failures += 1
        temporary = run_dir / "results.pending"
        temporary.write_text("".join(json.dumps(row) + "\n" for row in rows))
        temporary.replace(run_dir / "results.jsonl")
        lib.write_status(run_dir, lib.STATUS_FAILED if failures else lib.STATUS_DONE)
        return 1 if failures else 0
    except KeyboardInterrupt:
        lib.write_status(run_dir, lib.STATUS_STOPPED)
        return 130
    except Exception:
        lib.write_status(run_dir, lib.STATUS_FAILED)
        raise
    finally:
        (run_dir / "pid").unlink(missing_ok=True)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
