"""Dixit entry/exit dataset for firm dynamics DDC estimation.

This module generates synthetic firm entry and exit data from the
Dixit (1989) model as implemented in the Abbring and Klein DDC
teaching package. A firm observes market profitability and decides
each period whether to be active or inactive. Entry and exit have
sunk costs that create hysteresis in the firm's behavior.

State space:
    n_profit_bins x 2 incumbent status discrete states.
    Default: 10 profit bins x 2 incumbent status = 20 states.

Action space:
    2 actions: inactive (0) and active (1).

Reference:
    Dixit, A.K. (1989). "Entry and Exit Decisions under Uncertainty."
    Journal of Political Economy, 97(3), 620-638.

    Abbring, J.H. & Klein, T.J. (2020). "Dynamic Discrete Choice."
    https://github.com/jabbring/dynamic-discrete-choice
"""

from __future__ import annotations

from typing import Union

import pandas as pd

from econirl.core.types import Panel
from econirl.environments.entry_exit import (
    EntryExitEnvironment,
    N_ACTIONS,
    N_FEATURES,
    N_STATES,
    PROFIT_LABELS,
    STATUS_LABELS,
    state_to_components,
)
from econirl.simulation.synthetic import simulate_panel


def load_entry_exit(
    n_individuals: int = 1000,
    n_periods: int = 100,
    as_panel: bool = False,
    seed: int = 42,
    profit_slope: float = 1.0,
    entry_cost: float = -2.0,
    exit_cost: float = -0.5,
    operating_cost: float = -0.5,
    persistence: float = 0.7,
    discount_factor: float = 0.95,
) -> Union[pd.DataFrame, Panel]:
    """Generate firm entry/exit trajectory data.

    Creates synthetic firm trajectories from the Dixit entry/exit
    model. Each period the firm observes market profitability and
    decides whether to be active or inactive. Entering an inactive
    market costs a sunk entry fee. Exiting costs a sunk exit fee.

    Args:
        n_individuals: Number of firms to simulate.
        n_periods: Number of time periods per firm.
        as_panel: If True, return Panel object for econirl estimators.
        seed: Random seed for reproducibility.
        profit_slope: Sensitivity of profit to market state.
        entry_cost: Sunk cost of entering (negative).
        exit_cost: Sunk cost of exiting (negative).
        operating_cost: Per-period fixed cost of being active (negative).
        persistence: AR(1) persistence of profit process.
        discount_factor: Time discount factor.

    Returns:
        DataFrame with columns: firm_id, period, state, action,
        next_state, profit_bin, incumbent_status, profit_label,
        status_label, is_active, entered, exited.

        If as_panel=True, returns Panel object.
    """
    env = EntryExitEnvironment(
        profit_slope=profit_slope,
        entry_cost=entry_cost,
        exit_cost=exit_cost,
        operating_cost=operating_cost,
        persistence=persistence,
        discount_factor=discount_factor,
        seed=seed,
    )

    panel = simulate_panel(
        env,
        n_individuals=n_individuals,
        n_periods=n_periods,
        seed=seed,
    )

    if as_panel:
        return panel

    records = []
    for traj in panel.trajectories:
        tid = traj.individual_id
        for t in range(len(traj.states)):
            s = int(traj.states[t])
            a = int(traj.actions[t])
            ns = int(traj.next_states[t])
            pb, status = state_to_components(s)
            records.append({
                "firm_id": tid,
                "period": t,
                "state": s,
                "action": a,
                "next_state": ns,
                "profit_bin": pb,
                "incumbent_status": status,
                "profit_label": PROFIT_LABELS[pb],
                "status_label": STATUS_LABELS[status],
                "is_active": a == 1,
                "entered": status == 0 and a == 1,
                "exited": status == 1 and a == 0,
            })

    return pd.DataFrame(records)


def get_entry_exit_info() -> dict:
    """Return metadata about the entry/exit dataset."""
    return {
        "name": "Dixit Entry/Exit (Synthetic)",
        "description": (
            "Synthetic firm entry/exit DDC from the Dixit (1989) model. "
            "20 states (profit bin x incumbent status), 2 actions "
            "(inactive/active). Sunk entry and exit costs create hysteresis."
        ),
        "source": "Simulated from EntryExitEnvironment (Abbring-Klein teaching package)",
        "n_states": N_STATES,
        "n_actions": N_ACTIONS,
        "n_features": N_FEATURES,
        "state_description": "Profit bin x incumbent status (active/inactive last period)",
        "action_description": "Inactive (0) / Active (1)",
        "true_parameters": {
            "profit_slope": 1.0,
            "entry_cost": -2.0,
            "exit_cost": -0.5,
            "operating_cost": -0.5,
        },
        "ground_truth": True,
        "use_case": "Firm dynamics, entry/exit hysteresis, industrial organization",
    }
