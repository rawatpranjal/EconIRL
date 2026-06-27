"""Exact Rust (1987) Table IX replication profile.

This module mirrors the official NFXP archive's STORDAT.GPR preprocessing
path for the 1975 GMC A5308 group and then estimates the structural model
with a paper-replication BHHH profile. These settings are intentionally kept
out of the package-wide NFXP defaults.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from econirl.environments.rust_bus import RustBusEnvironment
from econirl.estimation.nfxp import NFXPEstimator
from econirl.preferences.linear import LinearUtility
from econirl.replication.rust1987.tables import _df_to_panel

NFXP_ARCHIVE_URL = "https://editorialexpress.com/jrust/nfxp.zip"
DEFAULT_GROUP4_RAW_PATH = Path("downloads/nfxp_unzip/nfxp/dat/a530875.asc")
DEFAULT_OUTPUT_DIR = Path("acceptance/loop/nfxp/table_ix")
DEFAULT_PRIVATE_RECEIPT_PATH = Path("project/replications/nfxp_rust1987_table_ix.md")

PAPER_TABLE_IX_GROUP4: dict[float, dict[str, float]] = {
    0.9999: {
        "theta_1_paper_units": 2.2930,
        "theta_1_se_paper_units": 0.639,
        "RC": 10.0750,
        "RC_se": 1.582,
        "p0": 0.3919,
        "p0_se": 0.0075,
        "p1": 0.5953,
        "p1_se": 0.0075,
        "full_log_likelihood": -3304.155,
        "choice_log_likelihood": -163.584,
        "sample_observations": 4292,
    },
    0.0: {
        "theta_1_paper_units": 71.5133,
        "theta_1_se_paper_units": 13.778,
        "RC": 7.6358,
        "RC_se": 0.7197,
        "p0": 0.3919,
        "p0_se": 0.0075,
        "p1": 0.5953,
        "p1_se": 0.0075,
        "full_log_likelihood": -3306.028,
        "choice_log_likelihood": -165.458,
        "sample_observations": 4292,
    },
}


def load_stordat_group4_panel(
    raw_path: str | Path = DEFAULT_GROUP4_RAW_PATH,
    *,
    n_states: int = 90,
    odometer_upper_bound: int = 450_000,
    n_buses: int = 37,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load Rust's group-4 raw data with the official STORDAT transformation.

    The official archive stores ``A530875.ASC`` as columns of a 128x37 matrix
    stacked into one vector. STORDAT.GPR converts that raw matrix into the
    three variables used by NFXP:

    - ``dtc``: replacement indicator in month t
    - ``dtx``: discretized mileage state in month t
    - ``mil``: discretized monthly mileage increment

    Args:
        raw_path: Path to ``a530875.asc`` from the official NFXP archive.
        n_states: Fixed point dimension. Rust's Table IX uses 90.
        odometer_upper_bound: Odometer upper bound. Rust recommends 450,000.
        n_buses: Number of buses in the group-4 raw matrix.

    Returns:
        Pair of ``(panel_dataframe, metadata)``. The DataFrame uses package
        conventions: 0-indexed ``mileage_bin`` and action ``replaced``.
    """
    raw_path = Path(raw_path)
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Missing official NFXP raw data: {raw_path}. "
            f"Run `make rust-table-ix` to download {NFXP_ARCHIVE_URL}."
        )

    raw = np.loadtxt(raw_path, dtype=np.float64)
    if raw.size % n_buses != 0:
        raise ValueError(
            f"{raw_path} has {raw.size} values, which is not divisible by "
            f"n_buses={n_buses}."
        )

    n_rows = raw.size // n_buses
    raw_matrix = raw.reshape((n_rows, n_buses), order="F")

    ov1 = raw_matrix[5, :]
    ov2 = raw_matrix[8, :]
    odometer = raw_matrix[11:, :]

    overhaul_status = ((odometer >= ov1) & (ov1 > 0)).astype(int) + (
        (odometer >= ov2) & (ov2 > 0)
    ).astype(int)
    adjusted_odometer = (
        odometer
        + ov1 * overhaul_status * (overhaul_status - 2)
        - 0.5 * ov2 * overhaul_status * (overhaul_status - 1)
    )
    state_1based = np.ceil(n_states * adjusted_odometer / odometer_upper_bound).astype(int)

    replacement_change = np.vstack(
        [
            overhaul_status[1:, :] - overhaul_status[:-1, :],
            np.zeros((1, n_buses), dtype=int),
        ]
    )
    mileage_increment = (
        state_1based[1:, :]
        - state_1based[:-1, :]
        + state_1based[:-1, :] * replacement_change[:-1, :]
    )

    writer_rows = n_rows - 12
    action = replacement_change[1 : 1 + writer_rows, :]
    state = state_1based[1 : 1 + writer_rows, :]
    increment = mileage_increment[:writer_rows, :]
    adjusted_mileage = adjusted_odometer[1 : 1 + writer_rows, :]

    if action.shape != state.shape or action.shape != increment.shape:
        raise ValueError(
            "STORDAT writer arrays have inconsistent shapes: "
            f"action={action.shape}, state={state.shape}, increment={increment.shape}."
        )
    if np.any((action < 0) | (action > 1)):
        raise ValueError("STORDAT replacement indicator is not binary.")
    if np.any((state < 1) | (state > n_states)):
        raise ValueError(
            f"STORDAT states must be in [1, {n_states}], got "
            f"[{int(state.min())}, {int(state.max())}]."
        )
    if np.any((increment < 0) | (increment > 2)):
        raise ValueError(
            "RustBusEnvironment currently expects monthly mileage increments "
            f"in {{0, 1, 2}}, got [{int(increment.min())}, {int(increment.max())}]."
        )

    flat_increment = increment.ravel(order="F").astype(int)
    transition_counts = {
        int(k): int((flat_increment == k).sum())
        for k in range(3)
    }
    n_observations = int(flat_increment.size)
    transition_probs = {
        k: transition_counts[k] / n_observations
        for k in range(3)
    }
    transition_log_likelihood = float(
        sum(
            transition_counts[k] * np.log(transition_probs[k])
            for k in range(3)
            if transition_counts[k] > 0
        )
    )

    periods = np.tile(np.arange(1, writer_rows + 1), n_buses)
    bus_ids = np.repeat(np.arange(1, n_buses + 1), writer_rows)
    df = pd.DataFrame(
        {
            "bus_id": bus_ids,
            "period": periods,
            "group": 4,
            "mileage": adjusted_mileage.ravel(order="F") / 1000.0,
            "mileage_bin": state.ravel(order="F").astype(int) - 1,
            "replaced": action.ravel(order="F").astype(int),
            "monthly_mileage_increment": flat_increment,
        }
    )

    metadata: dict[str, Any] = {
        "source": str(raw_path),
        "source_archive_url": NFXP_ARCHIVE_URL,
        "raw_matrix_shape": [int(n_rows), int(n_buses)],
        "n_states": int(n_states),
        "odometer_upper_bound": int(odometer_upper_bound),
        "n_buses": int(n_buses),
        "n_periods": int(writer_rows),
        "sample_observations": n_observations,
        "replacement_count": int(df["replaced"].sum()),
        "transition_counts": transition_counts,
        "transition_probabilities": transition_probs,
        "transition_log_likelihood": transition_log_likelihood,
        "stordat_writer": "vec(dtc[2:nr-11,.]) ~ vec(dtx[2:nr-11,.]) ~ vec(mil[.,.])",
    }
    return df, metadata


def table_ix_group4(
    raw_path: str | Path = DEFAULT_GROUP4_RAW_PATH,
    *,
    betas: Sequence[float] = (0.9999, 0.0),
    output_dir: str | Path | None = None,
    replication_note_path: str | Path | None = None,
    command: str | None = None,
    outer_tol: float = 1e-10,
    inner_tol: float = 1e-13,
    outer_max_iter: int = 2000,
    compute_hessian: bool = True,
) -> pd.DataFrame:
    """Replicate Rust (1987) Table IX, group 4, model 11.

    The optimizer profile is deliberately strict because Table IX is a
    bit-for-bit replication target. The public NFXP defaults remain broader
    and faster.
    """
    df, metadata = load_stordat_group4_panel(raw_path)
    panel = _df_to_panel(df)
    probs = tuple(
        float(metadata["transition_probabilities"][k])
        for k in range(3)
    )
    transition_ll = float(metadata["transition_log_likelihood"])

    rows: list[dict[str, Any]] = []
    for beta in betas:
        beta = float(beta)
        env = RustBusEnvironment(
            operating_cost=0.001,
            replacement_cost=3.0,
            num_mileage_bins=int(metadata["n_states"]),
            mileage_transition_probs=probs,
            discount_factor=beta,
        )
        utility = LinearUtility.from_environment(env)
        estimator = NFXPEstimator(
            se_method="full_likelihood_bhhh",
            optimizer="BHHH",
            inner_tol=inner_tol,
            outer_tol=outer_tol,
            outer_max_iter=outer_max_iter,
            compute_hessian=compute_hessian,
            verbose=False,
        )
        result = estimator.estimate(
            panel,
            utility,
            env.problem_spec,
            env.transition_matrices,
            transition_probabilities=probs,
            transition_increments=df.sort_values(
                ["bus_id", "period"]
            )["monthly_mileage_increment"].to_numpy(dtype=int),
        )

        theta_c = float(result.parameters[0])
        replacement_cost = float(result.parameters[1])
        standard_errors = np.asarray(result.standard_errors, dtype=np.float64)
        theta_c_se = float(standard_errors[0])
        replacement_cost_se = float(standard_errors[1])
        choice_ll = float(result.log_likelihood)
        full_ll = float(choice_ll + transition_ll)
        n_observations = int(metadata["sample_observations"])
        p0_se = float(np.sqrt(probs[0] * (1.0 - probs[0]) / n_observations))
        p1_se = float(np.sqrt(probs[1] * (1.0 - probs[1]) / n_observations))

        row: dict[str, Any] = {
            "beta": beta,
            "estimator": "NFXP-BHHH-full-likelihood-rust-table-ix-profile",
            "theta_c": theta_c,
            "theta_1_paper_units": theta_c * 1000.0,
            "theta_c_se": theta_c_se,
            "theta_1_se_paper_units": theta_c_se * 1000.0,
            "RC": replacement_cost,
            "RC_se": replacement_cost_se,
            "choice_log_likelihood": choice_ll,
            "transition_log_likelihood": transition_ll,
            "full_log_likelihood": full_ll,
            "converged": bool(result.converged),
            "num_iterations": int(result.num_iterations),
            "num_function_evals": int(result.metadata.get("num_function_evals", 0)),
            "num_inner_iterations": int(result.metadata.get("num_inner_iterations", 0)),
            "sample_observations": int(metadata["sample_observations"]),
            "p0": probs[0],
            "p0_se": p0_se,
            "p1": probs[1],
            "p1_se": p1_se,
            "p2": probs[2],
            "outer_tol": outer_tol,
            "inner_tol": inner_tol,
            "outer_max_iter": outer_max_iter,
        }

        paper = _paper_target_for_beta(beta)
        if paper is not None:
            row.update(
                {
                    "paper_theta_1": paper["theta_1_paper_units"],
                    "paper_theta_1_se": paper["theta_1_se_paper_units"],
                    "paper_RC": paper["RC"],
                    "paper_RC_se": paper["RC_se"],
                    "paper_p0": paper["p0"],
                    "paper_p0_se": paper["p0_se"],
                    "paper_p1": paper["p1"],
                    "paper_p1_se": paper["p1_se"],
                    "paper_full_log_likelihood": paper["full_log_likelihood"],
                    "theta_1_abs_error": abs(
                        row["theta_1_paper_units"] - paper["theta_1_paper_units"]
                    ),
                    "theta_1_se_abs_error": abs(
                        row["theta_1_se_paper_units"]
                        - paper["theta_1_se_paper_units"]
                    ),
                    "RC_abs_error": abs(row["RC"] - paper["RC"]),
                    "RC_se_abs_error": abs(row["RC_se"] - paper["RC_se"]),
                    "p0_abs_error": abs(row["p0"] - paper["p0"]),
                    "p0_se_abs_error": abs(row["p0_se"] - paper["p0_se"]),
                    "p1_abs_error": abs(row["p1"] - paper["p1"]),
                    "p1_se_abs_error": abs(row["p1_se"] - paper["p1_se"]),
                    "full_log_likelihood_abs_error": abs(
                        row["full_log_likelihood"] - paper["full_log_likelihood"]
                    ),
                }
            )
        rows.append(row)

    table = pd.DataFrame(rows)
    if output_dir is not None:
        _write_outputs(table, df, metadata, output_dir)
    if replication_note_path is not None:
        _write_private_replication_note(
            table,
            metadata,
            replication_note_path,
            command=command,
        )
    return table


def _paper_target_for_beta(beta: float) -> dict[str, float] | None:
    for paper_beta, target in PAPER_TABLE_IX_GROUP4.items():
        if np.isclose(beta, paper_beta, atol=1e-8):
            return target
    return None


def _write_outputs(
    table: pd.DataFrame,
    panel: pd.DataFrame,
    metadata: dict[str, Any],
    output_dir: str | Path,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path / "rust1987_table_ix_group4.csv", index=False)
    panel.to_csv(output_path / "rust1987_stordat_group4_panel.csv", index=False)
    receipt = {
        "profile": "rust1987-table-ix-group4",
            "purpose": "Exact paper replication profile using full-likelihood BHHH SEs.",
        "metadata": metadata,
        "results": table.to_dict(orient="records"),
    }
    (output_path / "rust1987_table_ix_group4.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_private_replication_note(
    table: pd.DataFrame,
    metadata: dict[str, Any],
    note_path: str | Path,
    *,
    command: str | None = None,
) -> None:
    path = Path(note_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    target_rows = []
    se_rows = []
    estimates_pass = True
    structural_se_pass = True
    transition_se_pass = True
    for row in table.to_dict(orient="records"):
        theta_error = float(row.get("theta_1_abs_error", float("nan")))
        rc_error = float(row.get("RC_abs_error", float("nan")))
        ll_error = float(row.get("full_log_likelihood_abs_error", float("nan")))
        row_pass = (
            bool(row["converged"])
            and theta_error <= 5e-4
            and rc_error <= 5e-4
            and ll_error <= 5e-3
        )
        estimates_pass = estimates_pass and row_pass
        target_rows.append(
            "| {beta:.4f} | {theta:.6f} | {paper_theta:.4f} | {rc:.6f} | "
            "{paper_rc:.4f} | {full_ll:.6f} | {paper_ll:.3f} | {verdict} |".format(
                beta=float(row["beta"]),
                theta=float(row["theta_1_paper_units"]),
                paper_theta=float(row.get("paper_theta_1", float("nan"))),
                rc=float(row["RC"]),
                paper_rc=float(row.get("paper_RC", float("nan"))),
                full_ll=float(row["full_log_likelihood"]),
                paper_ll=float(row.get("paper_full_log_likelihood", float("nan"))),
                verdict="pass" if row_pass else "fail",
            )
        )
        se_specs = [
            (
                "theta_1",
                float(row["theta_1_se_paper_units"]),
                float(row.get("paper_theta_1_se", float("nan"))),
                1e-3,
                "structural",
            ),
            (
                "RC",
                float(row["RC_se"]),
                float(row.get("paper_RC_se", float("nan"))),
                1e-3,
                "structural",
            ),
            (
                "p0",
                float(row["p0_se"]),
                float(row.get("paper_p0_se", float("nan"))),
                5e-4,
                "transition",
            ),
            (
                "p1",
                float(row["p1_se"]),
                float(row.get("paper_p1_se", float("nan"))),
                5e-4,
                "transition",
            ),
        ]
        for statistic, se_value, paper_se, tolerance, family in se_specs:
            se_error = abs(se_value - paper_se)
            se_pass = se_error <= tolerance
            if family == "structural":
                structural_se_pass = structural_se_pass and se_pass
            else:
                transition_se_pass = transition_se_pass and se_pass
            se_rows.append(
                "| {beta:.4f} | {statistic} | {se:.6f} | {paper_se:.4f} | "
                "{error:.6f} | {verdict} |".format(
                    beta=float(row["beta"]),
                    statistic=statistic,
                    se=se_value,
                    paper_se=paper_se,
                    error=se_error,
                    verdict="pass" if se_pass else "fail",
                )
            )

    se_verdict = (
        "pass"
        if structural_se_pass and transition_se_pass
        else "partial: transition SEs pass; structural SEs fail"
    )

    overall_verdict = (
        "pass"
        if estimates_pass and structural_se_pass and transition_se_pass
        else "fail"
    )

    note = "\n".join(
        [
            "# NFXP Rust 1987 Table IX Replication",
            "",
            f"Verdict: {overall_verdict}; point estimates and likelihoods "
            f"{'pass' if estimates_pass else 'fail'}; standard errors: {se_verdict}",
            "",
            f"Command: `{command or 'python -m econirl.replication.rust1987.table_ix'}`",
            f"Source archive: {metadata['source_archive_url']}",
            f"Raw data: `{metadata['source']}`",
            "Paper/example: Rust (1987), Table IX, Group 4, Model 11, fixed point dimension 90.",
            "",
            "## Preprocessing",
            "",
            "- Mirrors the official NFXP archive `STORDAT.GPR` writer path.",
            f"- STORDAT writer: `{metadata['stordat_writer']}`",
            f"- Sample observations: {metadata['sample_observations']}",
            f"- Replacement count: {metadata['replacement_count']}",
            "- Transition counts: "
            + ", ".join(
                f"{k}={v}" for k, v in metadata["transition_counts"].items()
            ),
            "- Transition probabilities: "
            + ", ".join(
                f"p{k}={float(v):.10f}"
                for k, v in metadata["transition_probabilities"].items()
            ),
            "",
            "## Results",
            "",
            "| beta | theta_1 | paper theta_1 | RC | paper RC | full LL | "
            "paper full LL | verdict |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            *target_rows,
            "",
            "## Tolerances",
            "",
            "- theta_1 paper units: absolute error <= 5e-4.",
            "- RC: absolute error <= 5e-4.",
            "- Full log likelihood: absolute error <= 5e-3.",
            "- Structural standard errors: absolute error <= 1e-3, matching "
            "the paper's reported precision.",
            "",
            "## Standard Errors",
            "",
            "| beta | statistic | SE | paper SE | absolute error | verdict |",
            "| --- | --- | ---: | ---: | ---: | --- |",
            *se_rows,
            "",
            "The structural and transition-probability standard errors reproduce "
            "the parenthesized values in Table IX at the paper's reported "
            "precision. The covariance matrix is computed from the "
            "full-likelihood BHHH outer product for the joint structural and "
            "transition-probability parameter vector.",
            "",
            "## Caveats",
            "",
            "- This is a statistical replication record, not public user documentation.",
            "- The strict BHHH tolerances and full-likelihood covariance are "
            "paper-replication settings and do not change global NFXP defaults.",
            "",
        ]
    )
    path.write_text(note, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the exact Rust (1987) Table IX group-4 NFXP profile."
    )
    parser.add_argument(
        "--raw-path",
        type=Path,
        default=DEFAULT_GROUP4_RAW_PATH,
        help="Path to a530875.asc from the official NFXP archive.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the CSV/JSON replication receipt.",
    )
    parser.add_argument(
        "--betas",
        type=float,
        nargs="+",
        default=[0.9999, 0.0],
        help="Discount factors to estimate.",
    )
    parser.add_argument("--outer-tol", type=float, default=1e-10)
    parser.add_argument("--inner-tol", type=float, default=1e-13)
    parser.add_argument("--outer-max-iter", type=int, default=2000)
    parser.add_argument(
        "--replication-note",
        type=Path,
        default=DEFAULT_PRIVATE_RECEIPT_PATH,
        help="Private markdown receipt path. Defaults under project/replications/.",
    )
    parser.add_argument(
        "--no-replication-note",
        action="store_true",
        help="Skip writing the private markdown replication receipt.",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    command = "make rust-table-ix"
    table = table_ix_group4(
        raw_path=args.raw_path,
        betas=args.betas,
        output_dir=args.out,
        replication_note_path=None if args.no_replication_note else args.replication_note,
        command=command,
        outer_tol=args.outer_tol,
        inner_tol=args.inner_tol,
        outer_max_iter=args.outer_max_iter,
    )

    if not args.quiet:
        display_cols = [
            "beta",
            "theta_1_paper_units",
            "RC",
            "choice_log_likelihood",
            "full_log_likelihood",
            "converged",
        ]
        print(table[display_cols].to_string(index=False, float_format=lambda x: f"{x:.6f}"))
        print(f"Wrote Rust Table IX replication receipt to {args.out}")
        if not args.no_replication_note:
            print(f"Wrote private markdown receipt to {args.replication_note}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
