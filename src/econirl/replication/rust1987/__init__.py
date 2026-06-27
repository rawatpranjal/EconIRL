"""Rust (1987) replication package."""

from econirl.replication.rust1987.tables import (
    table_ii_descriptives,
    table_iv_transitions,
    table_v_structural,
)
from econirl.replication.rust1987.monte_carlo import (
    run_monte_carlo,
    summarize_monte_carlo,
)
from econirl.replication.rust1987.export import (
    table_to_latex,
    save_all_tables,
)

__all__ = [
    "table_ii_descriptives",
    "table_iv_transitions",
    "load_stordat_group4_panel",
    "table_ix_group4",
    "table_v_structural",
    "run_monte_carlo",
    "summarize_monte_carlo",
    "table_to_latex",
    "save_all_tables",
]


def __getattr__(name: str):
    """Lazily expose the exact Table IX profile without breaking ``python -m``."""
    if name in {"load_stordat_group4_panel", "table_ix_group4"}:
        from econirl.replication.rust1987.table_ix import (
            load_stordat_group4_panel,
            table_ix_group4,
        )

        return {
            "load_stordat_group4_panel": load_stordat_group4_panel,
            "table_ix_group4": table_ix_group4,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
