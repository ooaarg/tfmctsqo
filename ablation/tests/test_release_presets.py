"""Ensure portable DuckDB launch presets preserve the archived experiment design."""

import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

ABLATION = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ABLATION))
import sweep


class ReleasePresetTests(unittest.TestCase):
    def test_duckdb_presets_match_archived_cells_seeds_and_queries(self):
        for name in ("duckdb-job-12plus", "duckdb-jobcomplex-12plus", "duckdb-ceb-12plus"):
            with self.subTest(run=name):
                archive = json.loads((ABLATION.parent / "runs" / name / "config.json").read_text())
                preset = ABLATION / "configs" / (name.replace("-", "_") + ".toml")
                cfg = sweep.load_config(preset)
                self.assertEqual(cfg.queries, archive["queries"])
                self.assertEqual(cfg.seed_values, archive["seed_values"])
                for query in cfg.queries:
                    self.assertTrue(sweep.find_query_source(query, cfg.sources).is_file())
                result = subprocess.run(
                    [sys.executable, str(ABLATION / "sweep.py"), "-c", str(preset), "--dry-run"],
                    cwd="/tmp", capture_output=True, text=True,
                    env={k: v for k, v in os.environ.items() if k not in {"DB", "JOB_DIR"}},
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                actual = [json.loads(line[line.index("{"):]) for line in result.stdout.splitlines() if "{" in line]
                expected = [cell[4] for cell in archive["resolved_cells"]]
                self.assertCountEqual(actual, expected)
                self.assertIn(f"Total runs: {len(cfg.queries) * 5 * len(cfg.seed_values)}", result.stdout)
