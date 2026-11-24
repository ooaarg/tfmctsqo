"""
Generate auxiliary resources used by AlphaJoin training:

- `../resource/jobtablename/<queryName>`: list of table aliases used by the query
- `../resource/shorttolong`: mapping alias -> table name

This implementation parses SQL under `../resource/jobquery/` and does not require
any database connection.
"""

import os
import re
from typing import Dict, List, Tuple

querydir = "../resource/jobquery"  # JOB queries (*.sql)
tablenamedir = "../resource/jobtablename"  # aliases involved in each query (one file per queryName)
shorttolongpath = "../resource/shorttolong"  # mapping alias -> table name

_FROM_RE = re.compile(r"\\bFROM\\b", flags=re.IGNORECASE)
_WHERE_RE = re.compile(r"\\bWHERE\\b", flags=re.IGNORECASE)


def _extract_from_section(sql: str) -> str:
    """Return the substring between FROM and WHERE (or end-of-string)."""
    m_from = _FROM_RE.search(sql)
    if not m_from:
        return ""
    m_where = _WHERE_RE.search(sql, m_from.end())
    if m_where:
        return sql[m_from.end() : m_where.start()]
    return sql[m_from.end() :]


def _parse_table_aliases(from_section: str) -> Tuple[List[str], Dict[str, str]]:
    """
    Returns (aliases_in_order, alias_to_table).

    Handles common JOB formats:
      - FROM table AS t, ...
      - FROM table t, ...
      - JOIN chains with optional join keywords.
    """
    alias_to_table: Dict[str, str] = {}
    aliases: List[str] = []

    # Collapse whitespace and split on commas to get individual table/join chunks.
    chunks = [
        c.strip() for c in re.sub(r"\\s+", " ", from_section).split(",") if c.strip()
    ]
    for chunk in chunks:
        tokens = chunk.split()
        if not tokens:
            continue
        if tokens[0].upper() == "FROM":
            tokens = tokens[1:]
        if not tokens:
            continue

        # Skip leading join keywords if present.
        while tokens and tokens[0].upper() in {
            "INNER",
            "LEFT",
            "RIGHT",
            "FULL",
            "CROSS",
            "JOIN",
        }:
            tokens = tokens[1:]
        if not tokens:
            continue

        table = tokens[0].strip().strip(";").strip(",")
        alias = None
        if len(tokens) >= 3 and tokens[1].upper() == "AS":
            alias = tokens[2]
        elif len(tokens) >= 2:
            alias = tokens[1]
        else:
            alias = table

        table = table.lower()
        alias = alias.strip().strip(";").strip(",").strip(")").strip("(").lower()
        if not alias:
            continue

        alias_to_table[alias] = table
        if alias not in aliases:
            aliases.append(alias)

    return aliases, alias_to_table


def _legacy_parse_table_aliases(sql: str) -> Tuple[List[str], Dict[str, str]]:
    """
    Fallback parser that mimics the original AlphaJoin logic:
    - scan line-by-line to find FROM/WHERE
    - assume 'table AS alias' patterns
    """
    lines = sql.splitlines()
    j = 0
    k = 0
    # find first line containing FROM
    for i, line in enumerate(lines):
        j = i
        if "FROM" in line.upper():
            break
    # find first line containing WHERE
    for i, line in enumerate(lines):
        k = i
        if "WHERE" in line.upper():
            break

    aliases: List[str] = []
    alias_to_table: Dict[str, str] = {}

    # Lines between FROM and WHERE (inclusive of FROM block, exclusive of WHERE)
    for i in range(j, max(j, k) - 1):
        parts = lines[i].split()
        if "AS" in [p.upper() for p in parts]:
            # normalize positions independently of case
            upper = [p.upper() for p in parts]
            try:
                idx = upper.index("AS")
            except ValueError:
                continue
            if idx == 0 or idx + 1 >= len(parts):
                continue
            table = parts[idx - 1].strip().strip(",").lower()
            alias = parts[idx + 1].strip().strip(",").lower()
            aliases.append(alias)
            alias_to_table[alias] = table

    # Last table line right before WHERE (may not end with a comma)
    if k - 1 >= 0 and k - 1 < len(lines):
        parts = lines[k - 1].split()
        upper = [p.upper() for p in parts]
        if "AS" in upper:
            try:
                idx = upper.index("AS")
            except ValueError:
                idx = -1
            if idx > 0 and idx + 1 < len(parts):
                table = parts[idx - 1].strip().strip(",").lower()
                alias = parts[idx + 1].strip().strip(",").lower()
                if alias not in aliases:
                    aliases.append(alias)
                alias_to_table[alias] = table

    return aliases, alias_to_table


def getResource() -> None:
    """Parse all JOB queries and materialize table-alias resources."""
    os.makedirs(tablenamedir, exist_ok=True)

    short_to_long: Dict[str, str] = {}
    fileList = sorted([f for f in os.listdir(querydir) if f.endswith(".sql")])
    for queryName in fileList:
        querypath = os.path.join(querydir, queryName)
        with open(querypath, "r", encoding="utf-8") as f:
            sql = f.read()

        from_section = _extract_from_section(sql)
        aliases, alias_to_table = _parse_table_aliases(from_section)
        # Fallback to legacy line-based parser if the robust parser failed.
        if not aliases:
            aliases, alias_to_table = _legacy_parse_table_aliases(sql)
        if not aliases:
            raise RuntimeError(f"Failed to parse table aliases from: {querypath}")

        # Write `jobtablename/<queryNameWithoutExt>`
        out_path = os.path.join(tablenamedir, queryName[:-4])
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(str(aliases))

        # Merge into global mapping.
        short_to_long.update(alias_to_table)

    # Dump mapping (keys are what matters for encoding/training).
    with open(shorttolongpath, "w", encoding="utf-8") as f:
        f.write(str(short_to_long))

    print("Parsed queries:", len(fileList), "\\tunique aliases:", len(short_to_long))


if __name__ == "__main__":
    getResource()
