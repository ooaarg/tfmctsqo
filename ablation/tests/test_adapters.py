"""Adapter semantics, failure accounting, and result compatibility."""

import argparse
import contextlib
import io
from dataclasses import asdict
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import adapter_sweep
import lib
import sharing
import sweep
from adapters import common, duckdb_mctsopt, pg_mctsopt

ABLATION = Path(__file__).resolve().parents[1]


class AdapterTests(unittest.TestCase):
    def config(self, implementation="duckdb_mctsopt"):
        name = (
            "duckdb_shared_job.toml"
            if implementation == "duckdb_mctsopt"
            else "pg_shared_stack.toml"
        )
        with patch.dict(os.environ, {}, clear=True):
            return sweep.load_config(ABLATION / "configs" / name)

    def test_legacy_config_defaults_to_original_adapter(self):
        cfg = sweep.load_config(ABLATION / "configs/stack_mcts_vs_default_pg.toml")
        self.assertEqual(cfg.implementation, "mcts_extreme")
        self.assertEqual(lib.short_label("single", "neg_log", "best"), "sing/nl/best")

    def test_budget_mapping_and_single_semantics(self):
        values = {"start_budget": 125, "phases": 1}
        duck = duckdb_mctsopt.settings_for("single", values, "neg_log", "best", 42)
        pg = pg_mctsopt.settings_for("single", values, "neg_log", "best", 42)
        self.assertEqual(duck["iterations"], pg["start_budget"])
        self.assertEqual(duck["seed"], pg["random_seed"])
        self.assertFalse(duck["greedy_init"])
        self.assertFalse(duck["luby_enabled"])
        self.assertNotIn("start_budget", duck)
        for adapter in (duckdb_mctsopt, pg_mctsopt):
            with self.assertRaisesRegex(ValueError, "single requires"):
                adapter.settings_for("single", {"phases": 2}, "neg_log", "best", 1)

    def test_unsupported_parameters_are_not_silently_ignored(self):
        for adapter, settings in [
            (pg_mctsopt, {"greedy_init": True}),
            (duckdb_mctsopt, {"search_algorithm": "saio"}),
        ]:
            with self.assertRaisesRegex(ValueError, "Unsupported settings"):
                adapter.settings_for("single", settings, "neg_log", "best", 1)
        with self.assertRaises(ValueError):
            duckdb_mctsopt.settings_for(
                "single", {"iterations": 10, "start_budget": 20}, "neg_log", "best", 1
            )

    def test_luby_is_phase_budgeting_and_patience_is_preserved(self):
        settings = duckdb_mctsopt.settings_for(
            "luby", {"phases": 5, "patience": 2}, "neg_log", "best", 1
        )
        self.assertTrue(settings["luby_enabled"])
        self.assertEqual(settings["patience"], 2)
        self.assertEqual(settings["rollout"], "random")

    def test_seed_validation(self):
        cfg = self.config()
        cfg.seed_values = [101, 709]
        cells = [("single", "neg_log", "best", "", {})]
        self.assertEqual(adapter_sweep.resolve_cells(cfg, cells)[2], [101, 709])
        cfg.seed_values = [0]
        with self.assertRaisesRegex(ValueError, "positive"):
            adapter_sweep.resolve_cells(cfg, cells)

    def test_reject_cross_engine_baselines(self):
        with self.assertRaisesRegex(ValueError, "does not support"):
            adapter_sweep.resolve_cells(self.config(), [("pg_default", "n/a", "n/a", "", {})])

    def test_job_alias_rewrite_preserves_literals_and_comments(self):
        self.assertEqual(
            duckdb_mctsopt.adapt_sql("SELECT at.id, 'at' FROM aka_title at -- at\n"),
            'SELECT "at".id, \'at\' FROM aka_title "at" -- at\n',
        )

    def test_profile_ownership_and_timing(self):
        profile = {
            "query": {"sql": "SELECT 1;", "total_time": 0.01},
            **{
                k: {"total_time": 0.001}
                for k in ("parser", "planner", "optimizer", "physical_planner")
            },
        }
        self.assertEqual(
            duckdb_mctsopt.profile_metrics(profile, "SELECT 1;"),
            {"e2e_time_ms": 10, "plan_time_ms": 4, "exec_time_ms": 6},
        )
        with self.assertRaisesRegex(ValueError, "another statement"):
            duckdb_mctsopt.profile_metrics(profile, "SELECT 2;")

    def test_duckdb_diagnostics_are_read_after_disabling_search(self):
        cfg = self.config()
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "seed-1"
            profile = {
                "query": {"sql": "SELECT 1;", "total_time": 0.01},
                **{
                    k: {"total_time": 0.001}
                    for k in ("parser", "planner", "optimizer", "physical_planner")
                },
            }
            out.with_suffix(".profile.json").write_text(json.dumps(profile))
            stdout = '[{"1":1}]\n[]\n' + json.dumps(
                [
                    {
                        "regions": '[{"selected":true,"iterations":7,"search_ms":1.5}]',
                        "cost": 4,
                        "fallbacks": 0,
                    }
                ]
            )
            with patch.object(duckdb_mctsopt, "execute", return_value=stdout) as run:
                result = duckdb_mctsopt.measure(cfg, "single", {"enabled": True}, "SELECT 1;", out)
            script = run.call_args.args[1]
            self.assertLess(
                script.index("PRAGMA disable_profiling"),
                script.index("SET duckdb_mctsopt_enabled=false"),
            )
            self.assertLess(
                script.index("SET duckdb_mctsopt_enabled=false"),
                script.index("SELECT duckdb_mctsopt_diagnostics"),
            )
            self.assertEqual(result["iters"], 7)
            self.assertEqual(result["search_time_ms"], 1.5)

    def test_duckdb_result_documents_preserve_empty_answers(self):
        cfg = self.config()
        stats = {"regions": "[]", "cost": None, "fallbacks": 0}
        cases = [
            ("duckdb_default", "[][]"),
            ("single", "[][]" + json.dumps([stats])),
        ]
        with tempfile.TemporaryDirectory() as directory:
            for mode, stdout in cases:
                with (
                    self.subTest(mode=mode),
                    patch.object(duckdb_mctsopt, "execute", return_value=stdout),
                ):
                    output = Path(directory) / mode
                    profile = {
                        "query": {"sql": "SELECT 1 WHERE false;", "total_time": 0.01},
                        **{
                            k: {"total_time": 0.001}
                            for k in ("parser", "planner", "optimizer", "physical_planner")
                        },
                    }
                    output.with_suffix(".profile.json").write_text(json.dumps(profile))
                    result = duckdb_mctsopt.measure(cfg, mode, {}, "SELECT 1 WHERE false;", output)
                    self.assertEqual(result["answer"], [])
                    self.assertEqual(result["e2e_time_ms"], 10)

    def test_duckdb_unexpected_results_are_rejected(self):
        cfg = self.config()
        cfg.measurement = "cost_only"
        stats = json.dumps([{"regions": "[]", "cost": None, "fallbacks": 0}])
        cases = [
            ("duckdb_default", '[][ {"extra":1}]'),
            ("duckdb_default", "[][][]"),
            ("single", "[][]"),
            ("single", '[][ {"extra":1}]' + stats),
            ("single", "[][]" + stats + "[]"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            for mode, stdout in cases:
                with (
                    self.subTest(mode=mode, stdout=stdout),
                    patch.object(duckdb_mctsopt, "execute", return_value=stdout),
                ):
                    with self.assertRaisesRegex(ValueError, "Unexpected DuckDB statement results"):
                        duckdb_mctsopt.measure(
                            cfg, mode, {}, "SELECT 1;", Path(directory) / "measurement"
                        )

    def test_failed_process_preserves_error_and_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "measurement"
            command = [
                sys.executable,
                "-c",
                "import sys; print('partial result'); "
                "print('planner failed', file=sys.stderr); sys.exit(1)",
            ]
            with self.assertRaisesRegex(RuntimeError, "planner failed"):
                common.execute(command, "SELECT 1;", 5, output)
            self.assertEqual(output.with_suffix(".sql").read_text(), "SELECT 1;")
            self.assertEqual(output.with_suffix(".stdout").read_text(), "partial result\n")
            self.assertEqual(output.with_suffix(".stderr").read_text(), "planner failed\n")

    def test_duckdb_plan_nodes_preserve_cardinality_and_operator_time(self):
        tree = {
            "operator": [
                {
                    "type": "HASH_JOIN",
                    "timing": 0.001,
                    "intermediate_rows": 4,
                    "extra_info": {"Estimated Cardinality": "5"},
                    "children": [{"type": "TABLE_SCAN", "timing": 0.002, "intermediate_rows": 10}],
                }
            ]
        }
        nodes = lib.parse_plan_tree(json.dumps(tree))
        self.assertEqual([n["depth"] for n in nodes], [0, 1])
        self.assertEqual(nodes[0]["self_ms"], 1)
        self.assertEqual(nodes[0]["total_ms"], 3)
        self.assertEqual(nodes[0]["act_rows"], 4)

    def test_failures_remain_in_totals(self):
        cfg = self.config()
        cfg.queries = ["q"]
        cfg.n_seeds = 1
        cfg.runtime["cache"] = "warm"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            query = root / "q.sql"
            query.write_text("SELECT 1;")
            args = argparse.Namespace(
                parallel=1, prewarm=False, dry_run=False, runs_root=root, drop_cache_command=None
            )
            with (
                patch.object(duckdb_mctsopt, "validate_runtime"),
                patch.object(duckdb_mctsopt, "measure", side_effect=TimeoutError("timeout")),
            ):
                code = adapter_sweep.run(
                    cfg, args, {}, [("single", "neg_log", "best", "", {})], {"q": query}
                )
            result = next(root.glob("*/results.jsonl"))
            row = json.loads(result.read_text())
            self.assertEqual(code, 1)
            self.assertTrue(row["timed_out"])
            self.assertTrue(row["penalized"])
            totals = lib.aggregate_workload(lib.load_jsonl(result))
            self.assertEqual(totals.iloc[0]["e2e_time_ms__mean"], 30000)

    def test_answer_comparison_uses_bag_semantics(self):
        key = adapter_sweep.answer_key
        self.assertEqual(key([{"x": 1}, {"x": 2}]), key([{"x": 2}, {"x": 1}]))
        self.assertNotEqual(key([{"x": 1}, {"x": 1}]), key([{"x": 1}]))

    def test_grid_override_is_expanded_once(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), patch.dict(os.environ, {}, clear=True):
            result = sweep.main(
                [
                    "-c",
                    str(ABLATION / "configs/duckdb_shared_job.toml"),
                    "--gucs",
                    '{"start_budget":[125,250],"top_k":[10,20]}',
                    "--dry-run",
                ]
            )
        self.assertEqual(result, 0)
        self.assertIn("cells: 5", output.getvalue())
        self.assertIn("Total runs: 45", output.getvalue())

    def test_clearing_selected_cells_restores_grid_overrides(self):
        cfg = self.config()
        cfg.mcts_cells = [{"config": "single", "reward": "neg_log", "agg": "best"}]
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(output):
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(asdict(cfg)))
            result = sweep.main(
                [
                    "-c",
                    str(path),
                    "--mcts-cells",
                    "[]",
                    "--gucs",
                    '{"shared":{"start_budget":[125,250],"top_k":7}}',
                    "--dry-run",
                ]
            )
        self.assertEqual(result, 0)
        self.assertIn("cells: 3", output.getvalue())
        self.assertIn('"top_k": 7', output.getvalue())
        self.assertIn('"iterations": 125', output.getvalue())

    def test_saved_config_restores_pg_runtime_and_explicit_seeds(self):
        cfg = self.config("pg_mctsopt")
        cfg.pguser = "researcher"
        cfg.pgdata = "/data/test"
        cfg.seed_values = [101, 709]
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(asdict(cfg)))
            restored = sweep.load_config(path)
        self.assertEqual(restored.pguser, "researcher")
        self.assertEqual(restored.pgdata, "/data/test")
        self.assertEqual(restored.seed_values, [101, 709])

    def test_workbook_roundtrip_preserves_implementation_and_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run"
            run.mkdir()
            cfg = {"implementation": "duckdb_mctsopt", "queries": ["q"], "n_seeds": 1, "gucs": {}}
            (run / "config.json").write_text(json.dumps(cfg))
            row = dict(
                query="q",
                config="duckdb_mctsopt:single",
                implementation="duckdb_mctsopt",
                reward="neg_log",
                agg="best",
                combo_id="b250",
                gucs={"iterations": 250},
                seed=709,
                exec_time_ms=2,
                plan_time_ms=1,
                e2e_time_ms=3,
                status="ok",
                regions="x" * 40000,
            )
            (run / "results.jsonl").write_text(json.dumps(row) + "\n")
            book = sharing.export_workbook(run, root / "test.xlsx")
            imported = sharing.import_workbook(book, root / "imported")
            self.assertEqual(json.loads((imported / "results.jsonl").read_text()), row)
            self.assertEqual(json.loads((imported / "config.json").read_text()), cfg)


if __name__ == "__main__":
    unittest.main()
