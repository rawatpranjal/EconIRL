"""
Built-in datasets for econirl.

This module provides access to classic datasets used in the dynamic discrete
choice literature, primarily for replication and teaching purposes.
"""

from econirl.datasets.rust_bus import load_rust_bus
from econirl.datasets.occupational_choice import load_occupational_choice
from econirl.datasets.keane_wolpin import load_keane_wolpin, get_keane_wolpin_info
from econirl.datasets.robinson_crusoe import load_robinson_crusoe, get_robinson_crusoe_info
from econirl.datasets.equipment_replacement import load_equipment_replacement

__all__ = [
    "load_rust_bus",
    "load_occupational_choice",
    "load_keane_wolpin",
    "get_keane_wolpin_info",
    "load_robinson_crusoe",
    "get_robinson_crusoe_info",
    "load_equipment_replacement",
]
