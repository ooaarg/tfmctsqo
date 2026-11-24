#!/usr/bin/env python3
"""
Re-run all (query, Leading) pairs from t6.sql via psql on imdb2013 and
update the runtime column with fresh Execution Time.

Input:
  - AlphaJoin1.0/t6.sql : lines like
      queryId,leadingExpr,runtime,realOrder,matchFlag,timestamp
  - resource/jobquery/<queryId>.sql : text of the query

Output:
  - AlphaJoin1.0/t6_new.sql : same format as t6.sql but with updated runtime and timestamp
"""

import subprocess
import datetime as _dt
from pathlib import Path
import re

# This file lives at source/MyAlphaJoin/rerun_t6_with_psql.py, so its parent
# is MyAlphaJoin/. t6.sql / t6_new.sql sit inside AlphaJoin1.0/, and the JOB
# queries inside resource/jobquery/.
MYALPHA_DIR = Path(__file__).resolve().parent
T6_PATH = MYALPHA_DIR / "AlphaJoin1.0" / "t6.sql"
OUT_PATH = MYALPHA_DIR / "AlphaJoin1.0" / "t6_new.sql"
JOBQUERY_DIR = MYALPHA_DIR / "resource" / "jobquery"

PSQL = Path("psql")
DBNAME = "imdb2013"
DBUSER = "postgres"

TIMEOUT_SECONDS = 90  # 1.5 minutes; on timeout we record 90000 ms


def _rewrite_sql_with_leading(sql_text: str, leading_expr: str) -> str:
    """
    Best-effort rewrite of an implicit-join query into an explicit JOIN chain
    that follows the alias order implied by leading_expr.
    If anything looks unexpected, the original sql_text is returned unchanged.
    """
    try:
        # Very simple single-statement assumption.
        # Find FROM and WHERE (case-insensitive).
        m_from = re.search(r"\bFROM\b", sql_text, flags=re.IGNORECASE)
        if not m_from:
            return sql_text
        m_where = re.search(r"\bWHERE\b", sql_text, flags=re.IGNORECASE)

        from_start = m_from.end()
        from_end = m_where.start() if m_where else len(sql_text)

        from_clause = sql_text[from_start:from_end]
        before_from = sql_text[: m_from.start()]
        after_where = sql_text[from_end:]

        # Extract table aliases from FROM clause split by commas.
        alias_to_table: dict[str, str] = {}
        segments = [seg.strip() for seg in from_clause.split(",") if seg.strip()]
        for seg in segments:
            # Patterns: "table AS alias", "table alias", or just "table"
            m = re.match(
                r"(?i)^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s+(?:AS\s+)?([a-zA-Z_][a-zA-Z0-9_]*)\s*$",
                seg,
            )
            if m:
                table, alias = m.group(1), m.group(2)
            else:
                parts = seg.split()
                if len(parts) == 1:
                    table = alias = parts[0]
                else:
                    # Unexpected pattern – bail out.
                    return sql_text
            alias_to_table[alias] = table

        # If we have no aliases, nothing to do.
        if not alias_to_table:
            return sql_text

        # Extract conditions (everything after WHERE, if present).
        where_conds: list[str] = []
        if m_where:
            where_text = sql_text[m_where.end() :].strip().rstrip(";")
            if where_text:
                # Very naive split on AND; assumes no AND inside parentheses/string.
                for part in re.split(r"(?i)\bAND\b", where_text):
                    cond = part.strip()
                    if cond:
                        where_conds.append(cond)

        # Classify join vs filter predicates by counting aliases mentioned.
        def cond_aliases(cond: str) -> set[str]:
            present = set()
            for alias in alias_to_table:
                if re.search(rf"\b{re.escape(alias)}\.", cond):
                    present.add(alias)
            return present

        join_conds: list[tuple[set[str], str]] = []
        filter_conds: list[str] = []
        for cond in where_conds:
            aliases = cond_aliases(cond)
            if len(aliases) >= 2:
                join_conds.append((aliases, cond))
            else:
                filter_conds.append(cond)

        # Derive ordered aliases from leading_expr, keep only known aliases.
        tokens = [tok for tok in re.split(r"[()\s,]+", leading_expr) if tok]
        ordered_aliases: list[str] = []
        for tok in tokens:
            if tok in alias_to_table and tok not in ordered_aliases:
                ordered_aliases.append(tok)

        # Fallback: if nothing usable from leading_expr, keep original query.
        if not ordered_aliases:
            return sql_text

        # Ensure we cover all aliases from original FROM; append missing ones.
        for alias in alias_to_table:
            if alias not in ordered_aliases:
                ordered_aliases.append(alias)

        used_aliases: list[str] = []
        remaining_join_conds = join_conds.copy()

        # Start building new FROM ... JOIN ... chain.
        first_alias = ordered_aliases[0]
        used_aliases.append(first_alias)
        from_parts: list[str] = [f"{alias_to_table[first_alias]} AS {first_alias}"]

        for alias in ordered_aliases[1:]:
            used_aliases.append(alias)
            # Pick all join conditions that connect this alias with any used alias.
            on_conds: list[str] = []
            still_remaining: list[tuple[set[str], str]] = []
            for aliases, cond in remaining_join_conds:
                if alias in aliases and any(
                    a in aliases for a in used_aliases if a != alias
                ):
                    on_conds.append(cond)
                else:
                    still_remaining.append((aliases, cond))
            remaining_join_conds = still_remaining

            on_text = " AND ".join(on_conds) if on_conds else "TRUE"
            from_parts.append(f"JOIN {alias_to_table[alias]} AS {alias} ON {on_text}")

        new_from_clause = "FROM " + " ".join(from_parts)

        # Rebuild WHERE from remaining join conds + filters, if any.
        remaining_conds = [
            cond for _aliases, cond in remaining_join_conds
        ] + filter_conds
        if remaining_conds:
            new_where = "WHERE " + " AND ".join(remaining_conds)
        else:
            new_where = ""

        rebuilt = before_from + new_from_clause
        if new_where:
            rebuilt += "\n" + new_where

        # Preserve trailing semicolon if present in original.
        if sql_text.strip().endswith(";"):
            rebuilt = rebuilt.rstrip(";") + ";"

        return rebuilt
    except Exception:
        # On any unexpected pattern, keep original SQL.
        return sql_text


def parse_execution_time(psql_output: str) -> float | None:
    """
    Parse 'Execution Time: xxx ms' from EXPLAIN ANALYZE output.
    """
    for line in psql_output.splitlines():
        line = line.strip()
        if "Execution Time" in line and "ms" in line:
            # e.g. "Execution Time: 547.987 ms"
            parts = line.split()
            try:
                idx = parts.index("Time:") + 1
                return float(parts[idx])
            except Exception:
                # fallback: take token before "ms"
                for i, token in enumerate(parts):
                    if token == "ms" and i > 0:
                        try:
                            return float(parts[i - 1])
                        except ValueError:
                            continue
    return None


def run_one(query_id: str, leading_expr: str, sql_text: str) -> float | None:
    """
    Run EXPLAIN ANALYZE with pg_hint_plan hints for given query and return Execution Time in ms.

    Steps:
      - Rewrite the query to use an explicit JOIN chain that follows leading_expr
        as much as possible.
      - Set planner GUCs to reduce join reordering.
      - Apply a Leading(...) hint and a simple NestLoop(a b) chain.

    We always apply a Leading(...) hint based on leading_expr and, in addition,
    derive a simple chain of NestLoop(a b) hints from the aliases that appear
    in leading_expr in left-to-right order. For example, for:

        leading_expr = "( ( ( t k ) mc ) cn )"

    we will emit:

        /*+ Leading(( ( ( t k ) mc ) cn ))
              NestLoop(t k)
              NestLoop(k mc)
              NestLoop(mc cn) */
    """
    # Extract alias tokens from the leading expression in a very simple way:
    # split on punctuation and whitespace and keep non-empty pieces.
    tokens = [tok for tok in re.split(r"[()\s,]+", leading_expr) if tok]

    # Build Leading(...) plus a chain of NestLoop(a b) for adjacent aliases.
    hint_parts: list[str] = [f"Leading({leading_expr})"]
    for a, b in zip(tokens, tokens[1:]):
        hint_parts.append(f"NestLoop({a} {b})")

    hint = "/*+ " + " ".join(hint_parts) + " */"

    # Rewrite SQL text to explicit JOINs guided by leading_expr.
    rewritten_sql = _rewrite_sql_with_leading(sql_text, leading_expr)

    # Apply GUCs to discourage join collapsing, then run EXPLAIN ANALYZE.
    wrapped = (
        "SET from_collapse_limit = 1;\n"
        "SET join_collapse_limit = 1;\n"
        f"{hint}\n"
        "EXPLAIN ANALYZE\n" + rewritten_sql
    )
    print(f"Wrapped: {wrapped}")

    try:
        proc = subprocess.run(
            [
                str(PSQL),
                "-U",
                DBUSER,
                "-d",
                DBNAME,
            ],
            input=wrapped.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=TIMEOUT_SECONDS,
        )
        out = proc.stdout.decode("utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        # Query took longer than 5 minutes — record the timeout value (ms)
        return float(TIMEOUT_SECONDS * 1000)
    # You can uncomment below for debugging if needed:
    # print(f"--- {query_id} ---\n{out}\n", flush=True)
    return parse_execution_time(out)


def main() -> None:
    if not T6_PATH.exists():
        raise SystemExit(f"Input file not found: {T6_PATH}")

    lines_out: list[str] = []
    now = _dt.datetime.now()
    ts_prefix = now.strftime("%m-%d %H:%M")

    with T6_PATH.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip():
                continue

            parts = line.split(",", 5)
            if len(parts) < 3:
                # malformed, keep as-is
                lines_out.append(line)
                continue

            qid = parts[0].strip()
            leading_expr = parts[1].strip()
            old_runtime = parts[2].strip()

            # Remaining columns (realOrder, matchFlag, timestamp or whatever is there)
            tail = ""
            if len(parts) > 3:
                tail = ",".join(parts[3:])

            query_path = JOBQUERY_DIR / f"{qid}.sql"
            if not query_path.exists():
                # No query text found – keep old runtime
                lines_out.append(line)
                continue

            sql_text = query_path.read_text(encoding="utf-8", errors="replace")
            print(f"Running {qid} with leading {leading_expr}")
            runtime = run_one(qid, leading_expr, sql_text)
            print(f"Runtime: {runtime}")
            if runtime is None:
                # failed to parse, keep old runtime
                new_line = f"{qid},{leading_expr},{old_runtime}"
            else:
                new_line = f"{qid},{leading_expr},{runtime:.3f}"

            if tail:
                # overwrite timestamp at the very end with current ts_prefix if it looks like a timestamp
                tail_parts = tail.split(",")
                if tail_parts and len(tail_parts[-1].strip()) >= 5:
                    tail_parts[-1] = ts_prefix
                    tail = ",".join(tail_parts)
                new_line = f"{new_line},{tail}"

            lines_out.append(new_line)

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for l in lines_out:
            f.write(l + "\n")


if __name__ == "__main__":
    main()
