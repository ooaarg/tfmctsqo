"""Adapters for shared-core experiments. Legacy configs retain their SQL templates."""

from adapters import duckdb_mctsopt, pg_mctsopt

IMPLEMENTATIONS = ("mcts_extreme", "pg_mctsopt", "duckdb_mctsopt")


def get_adapter(name):
    if name == "pg_mctsopt":
        return pg_mctsopt
    if name == "duckdb_mctsopt":
        return duckdb_mctsopt
    raise ValueError(f"No shared-core adapter for {name!r}")
