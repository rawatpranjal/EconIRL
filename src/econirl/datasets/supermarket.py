"""Aguirregabiria (1999) supermarket pricing and inventory dataset.

This module loads the real supermarket data from Aguirregabiria (1999)
"The Dynamics of Markups and Inventories in Retailing Firms" (Review
of Economic Studies). The dataset tracks 534 products in a single
Spanish supermarket over 29 months. Each product-month observation
includes inventory levels, sales, orders, retail and wholesale prices,
and promotion indicators.

The data is preprocessed into a DDC panel with discretized states
(inventory bin x lagged promotion status) and actions (promotion
decision x order decision).

State space:
    10 states = 5 inventory quintile bins x 2 lagged promotion status.

Action space:
    4 actions: (no promo, no order), (no promo, order),
    (promo, no order), (promo, order).

Reference:
    Aguirregabiria, V. (1999). "The Dynamics of Markups and Inventories
    in Retailing Firms." Review of Economic Studies, 66(2), 275-308.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd

from econirl.core.types import Panel, Trajectory
from econirl.environments.supermarket import (
    N_ACTIONS,
    N_FEATURES,
    N_STATES,
)


def load_supermarket(
    as_panel: bool = False,
    data_path: str | Path | None = None,
) -> Union[pd.DataFrame, Panel]:
    """Load the Aguirregabiria (1999) supermarket dataset.

    Returns the real supermarket data with 534 products tracked over
    26 usable periods (29 months minus lagged variable construction).
    Each observation is a product-month with discretized state and
    action variables suitable for DDC estimation.

    Args:
        as_panel: If True, return Panel object for econirl estimators.
        data_path: Path to the supermarket_data.csv file. If None,
            uses the bundled dataset.

    Returns:
        DataFrame with columns: product_id, period, state, action,
        next_state, inventory_bin, lagged_promotion, promotion,
        ordered, sales, inventory, orders, wholesale_price,
        retail_price, markup_pct, stockout.

        If as_panel=True, returns Panel object.
    """
    if data_path is None:
        data_path = Path(__file__).parent / "supermarket_data.csv"

    df = pd.read_csv(data_path)

    if as_panel:
        import numpy as np

        trajectories = []
        for pid, group in df.groupby("product_id"):
            group = group.sort_values("period")
            trajectories.append(
                Trajectory(
                    individual_id=int(pid),
                    states=np.array(group["state"].values, dtype=np.int32),
                    actions=np.array(group["action"].values, dtype=np.int32),
                    next_states=np.array(group["next_state"].values, dtype=np.int32),
                )
            )
        return Panel(trajectories=trajectories)

    return df


def get_supermarket_info() -> dict:
    """Return metadata about the supermarket dataset."""
    return {
        "name": "Aguirregabiria (1999) Supermarket Pricing/Inventory",
        "description": (
            "Real supermarket data from Aguirregabiria (1999 REStud). "
            "534 products over 29 months in a Spanish supermarket. "
            "10 states (inventory bin x lagged promotion), "
            "4 actions (promotion x order decision)."
        ),
        "source": "http://individual.utoronto.ca/vaguirre/data/data.html",
        "license": "Academic use",
        "n_states": N_STATES,
        "n_actions": N_ACTIONS,
        "n_features": N_FEATURES,
        "state_description": "Inventory quintile bin x lagged promotion status",
        "action_description": (
            "No promo + no order (0), No promo + order (1), "
            "Promo + no order (2), Promo + order (3)"
        ),
        "n_observations": 13884,
        "n_products": 534,
        "n_periods": 26,
        "ground_truth": False,
        "use_case": "Retail IO, pricing dynamics, inventory management, promotions",
    }
