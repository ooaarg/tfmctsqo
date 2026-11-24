"""Portable Stack workload identity and launch-plan regression checks."""

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

ABLATION = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ABLATION))
import sweep


class StackWorkloadTests(unittest.TestCase):
    def test_manifest_and_config_resolve_exact_reference_queries(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = sweep.load_config(ABLATION / "configs/stack_geqo_balanced_200.toml")
        self.assertEqual(cfg.db, "stack")
        directory = sweep.resolve_source_dir(cfg.sources[0])
        manifest = json.loads((directory.parent / "selection-manifest.json").read_text())
        expected = {q["family"] + "_" + Path(q["path"]).stem: q for q in manifest["queries"]}
        self.assertEqual(set(cfg.queries), set(expected))
        self.assertEqual(len(cfg.queries), 200)
        counts = Counter()
        for query in cfg.queries:
            path = sweep.find_query_source(query, cfg.sources)
            raw = path.read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), expected[query]["sha256"])
            match = re.search(r"\bFROM\b(.*?)\bWHERE\b", raw.decode(), re.I | re.S)
            relations = match.group(1).count(",") + 1
            self.assertEqual(relations, expected[query]["relations"])
            counts[relations] += 1
        self.assertEqual(counts, {12: 100, 11: 50, 9: 50})
        _, single = sweep.apply_gucs(cfg, {})
        sql = sweep.render_sql_template(sweep.repo_path(cfg.template_single), single)
        self.assertRegex(sql, r"mcts_extreme.phases\s*=\s*1;")
        self.assertRegex(sql, r"mcts_extreme.start_budget\s*=\s*480;")
        self.assertNotIn("luby", cfg.configs)

    def test_default_pg_preset_materializes_baseline_and_counts_once(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = sweep.load_config(ABLATION / "configs/stack_mcts_vs_default_pg.toml")
        self.assertEqual(cfg.configs, ["single", "pg_default"])
        self.assertEqual(sweep.cell_dirname("pg_default", "neg_log", "best", "grid"), "pg_default")
        with tempfile.TemporaryDirectory() as root:
            run_dir, _, _ = sweep.init_run_dir(cfg, {}, Path(root))
            sql = (run_dir / "pg_default.sql").read_text()
            self.assertRegex(sql, r"mcts_extreme.enabled\s*=\s*off;")
            self.assertRegex(sql, r"SET geqo\s*=\s*on;")
            self.assertRegex(sql, r"geqo_threshold\s*=\s*12;")
            self.assertRegex(sql, r"geqo_seed\s*=\s*0;")
        result = subprocess.run(
            [sys.executable, str(ABLATION / "sweep.py"), "-c",
             str(ABLATION / "configs/stack_mcts_vs_default_pg.toml"), "--dry-run",
             "--rewards", "neg_log norm_neg_log", "--aggs", "best average"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Cells (5)", result.stdout)
        self.assertIn("Total runs:    3000", result.stdout)

    def test_local_connections_respect_socket_environment(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = sweep.load_config(ABLATION / "configs/stack_geqo_balanced_200.toml")
        cfg.docker_container = ""
        for host in ["/path/to/postgres/socket", ""]:
            with patch.dict(os.environ, {"PGHOST": host}), patch.object(sweep.subprocess, "run") as run:
                run.return_value = subprocess.CompletedProcess([], 0, stdout="", stderr="")
                self.assertTrue(sweep.ensure_pg_running(cfg))
                sweep.run_prewarm(cfg, Path("query.sql"))
                sweep.run_psql(cfg, Path("single.sql"), reward="neg_log", agg="best",
                               seed=1, query_sql=Path("query.sql"))
                self.assertEqual(run.call_count, 3)
                for call in run.call_args_list:
                    argv = call.args[0]
                    self.assertEqual(argv[argv.index("-h") + 1], host or "/tmp")
                    self.assertEqual(argv[argv.index("-d") + 1], "stack")

    def test_cli_dry_run_from_outside_repository(self):
        result = subprocess.run(
            [sys.executable, str(ABLATION / "sweep.py"), "-c",
             str(ABLATION / "configs/stack_geqo_balanced_200.toml"), "--dry-run"],
            cwd="/tmp", capture_output=True, text=True,
            env={k: v for k, v in os.environ.items() if k not in {"DB", "JOB_DIR", "PGDATA"}},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Queries (200)", result.stdout)
        self.assertIn("Total runs:    1800", result.stdout)


if __name__ == "__main__":
    unittest.main()
