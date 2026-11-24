#!/usr/bin/env python3
"""Select a reproducible subset of CEB IMDb queries by relation count."""

from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
DEFAULT_INPUT = BASE_DIR / "queries" / "ceb-imdb-3k"
DEFAULT_OUTPUT = BASE_DIR / "queries" / "rel9_seed42_200"


def strip_comments(sql: str) -> str:
    sql = re.sub(r"--.*?$", "", sql, flags=re.MULTILINE)
    return re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)


def from_clause(sql: str) -> str:
    cleaned = strip_comments(sql)
    match = re.search(
        r"\bfrom\b(.*?)(?:\bwhere\b|\bgroup\s+by\b|\border\s+by\b|\blimit\b|;|$)",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return match.group(1) if match else ""


def split_top_level_commas(text: str) -> list[str]:
    chunks: list[str] = []
    start = 0
    depth = 0

    for idx, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(0, depth - 1)
        elif char == "," and depth == 0:
            chunks.append(text[start:idx].strip())
            start = idx + 1

    tail = text[start:].strip()
    if tail:
        chunks.append(tail)
    return chunks


def relation_count(sql: str) -> int:
    clause = from_clause(sql)
    if not clause:
        return 0

    # CEB/JOB-style queries use comma joins: "table AS alias, table2 AS alias2".
    # Count table references in the FROM clause. Explicit JOINs are counted too.
    refs: list[str] = []
    for comma_chunk in split_top_level_commas(clause):
        refs.extend(
            chunk.strip()
            for chunk in re.split(r"\bjoin\b", comma_chunk, flags=re.IGNORECASE)
            if chunk.strip()
        )

    count = 0
    for ref in refs:
        if ref.startswith("("):
            continue
        if re.match(r"[A-Za-z_][\w.]*\s*(?:as\s+)?(?:[A-Za-z_][\w]*)?", ref, re.I):
            count += 1
    return count


def sql_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.sql") if path.is_file())


def sample_even_by_template(
    eligible: list[tuple[Path, int]],
    limit: int,
    rng: random.Random,
) -> list[tuple[Path, int]]:
    by_template: dict[str, list[tuple[Path, int]]] = defaultdict(list)
    for item in eligible:
        path, _ = item
        by_template[path.parent.name].append(item)

    templates = sorted(by_template)
    selected: list[tuple[Path, int]] = []
    base_quota = limit // len(templates)
    remainder = limit % len(templates)

    for index, template in enumerate(templates):
        candidates = by_template[template]
        quota = base_quota + (1 if index < remainder else 0)
        selected.extend(rng.sample(candidates, min(quota, len(candidates))))

    if len(selected) < min(limit, len(eligible)):
        already = {path for path, _ in selected}
        remaining = [item for item in eligible if item[0] not in already]
        selected.extend(rng.sample(remaining, min(limit - len(selected), len(remaining))))

    return selected


def write_manifest(
    selected: list[tuple[Path, int]],
    input_root: Path,
    output_root: Path,
    manifest_path: Path,
) -> None:
    with manifest_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["source", "output", "template", "relation_count"])
        for source, rel_count in sorted(selected):
            relative = source.relative_to(input_root)
            writer.writerow(
                [
                    str(relative),
                    str((output_root / relative).relative_to(output_root)),
                    source.parent.name,
                    rel_count,
                ]
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy a reproducible subset of CEB IMDb SQL files filtered by relation count.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-relations", type=int, default=9)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--strategy",
        choices=("random", "template-even"),
        default="template-even",
        help="random samples from all eligible queries; template-even spreads the sample across templates.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace the output directory if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = args.input.resolve()
    output_root = args.output.resolve()

    if not input_root.exists():
        raise SystemExit(f"input directory does not exist: {input_root}")
    if output_root.exists():
        if not args.force:
            raise SystemExit(f"output directory already exists; use --force: {output_root}")
        shutil.rmtree(output_root)

    eligible: list[tuple[Path, int]] = []
    for path in sql_files(input_root):
        rel_count = relation_count(path.read_text(errors="replace"))
        if rel_count >= args.min_relations:
            eligible.append((path, rel_count))

    if not eligible:
        raise SystemExit(f"no queries found with relation count >= {args.min_relations}")

    rng = random.Random(args.seed)
    limit = min(args.limit, len(eligible))
    if args.strategy == "random":
        selected = rng.sample(eligible, limit)
    else:
        selected = sample_even_by_template(eligible, limit, rng)

    output_root.mkdir(parents=True)
    for source, _rel_count in selected:
        target = output_root / source.relative_to(input_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    manifest_path = output_root / "selected_queries.csv"
    write_manifest(selected, input_root, output_root, manifest_path)

    print(f"input_sql_files={len(sql_files(input_root))}")
    print(f"eligible_min_relations_{args.min_relations}={len(eligible)}")
    print(f"selected={len(selected)}")
    print(f"seed={args.seed}")
    print(f"strategy={args.strategy}")
    print(f"output={output_root}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
