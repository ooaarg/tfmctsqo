#!/usr/bin/env python3
"""
Convert AlphaJoin tree hint to pg_hint_plan linear format.
Tree: ( ( cn ( ( ci rt ) ( t mc ) ) ) ct ) chn
Linear: cn ci rt t mc ct chn

This script was used to fix the "Zero-length delimited string" error when feeding
LEADING hints to pg_hint_plan; rewriting to linear form raised training accuracy to ~0.9.

Usage:
  # Convert a single hint (print linear form to stdout):
  python3 tree_to_linear_hint.py --hint "( ( cn ( ( ci rt ) ( t mc ) ) ) ct ) chn"

  # Rewrite a CSV (e.g. t6.sql): replace tree hints in columns 2 and 4 with linear. Overwrites in place.
  python3 tree_to_linear_hint.py --csv [path_to_csv]
  # Default path if omitted: tools/t6_extended.sql

  # Rename .txt files in ../data: "queryid ( tree ).txt" -> "queryid linear.txt"
  python3 tree_to_linear_hint.py
"""

import re
import os
import sys


def tree_to_linear(tree: str) -> str:
    """Extract table names in left-to-right order from tree hint."""
    parts = tree.split()
    tables = [p for p in parts if p not in ("(", ")")]
    return " ".join(tables)


def process_data_dir(data_dir: str) -> int:
    """Rename .txt files: replace tree hint in filename with linear hint."""
    count = 0
    for fname in os.listdir(data_dir):
        if not fname.endswith(".txt"):
            continue
        # Format: "queryid ( tree ).txt"
        m = re.match(r"^(.+?)\s+\((.+)\)\.txt$", fname)
        if not m:
            continue
        queryid = m.group(1).strip()
        tree = m.group(2).strip()
        linear = tree_to_linear(tree)
        new_fname = f"{queryid} {linear}.txt"
        if new_fname == fname:
            continue
        old_path = os.path.join(data_dir, fname)
        new_path = os.path.join(data_dir, new_fname)
        if os.path.exists(new_path) and old_path != new_path:
            print(f"skip (exists): {new_fname}", file=sys.stderr)
            continue
        os.rename(old_path, new_path)
        print(f"{fname} -> {new_fname}")
        count += 1
    return count


def process_csv(csv_path: str, out_path: str | None = None) -> int:
    """Rewrite CSV: replace tree hints in column 2 (and optionally 4) with linear."""
    out_path = out_path or csv_path
    count = 0
    lines_out = []
    with open(csv_path, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            parts = line.split(",", 4)  # queryName, hint, runtime, [realOrder], [rest]
            if len(parts) < 2:
                lines_out.append(line)
                continue
            hint = parts[1].strip()
            if "(" in hint or ")" in hint:
                linear = tree_to_linear(hint)
                parts[1] = linear
                count += 1
            if len(parts) >= 4:
                real = parts[3].strip()
                if "(" in real or ")" in real:
                    parts[3] = tree_to_linear(real)
            lines_out.append(",".join(parts))
    with open(out_path, "w") as f:
        f.write("\n".join(lines_out) + "\n")
    return count


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--hint" and len(sys.argv) > 2:
        print(tree_to_linear(sys.argv[2]))
        sys.exit(0)
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    if len(sys.argv) > 1 and sys.argv[1] == "--csv":
        csv_path = (
            sys.argv[2]
            if len(sys.argv) > 2
            else os.path.join(os.path.dirname(__file__), "t6_extended.sql")
        )
        n = process_csv(csv_path)
        print(f"Converted {n} hints in {csv_path}")
    else:
        n = process_data_dir(data_dir)
        print(f"Renamed {n} files in {data_dir}")
