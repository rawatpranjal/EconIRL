"""
Built-in datasets for econirl.

This module provides access to classic datasets used in the dynamic discrete
choice literature, primarily for replication and teaching purposes.
"""

from econirl.datasets.rust_bus import load_rust_bus

__all__ = ["load_rust_bus"]
