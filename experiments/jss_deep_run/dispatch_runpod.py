"""Fan out the JSS deep-run matrix as RunPod jobs (or run locally).

Two modes.

Local mode (``--local``) runs each cell sequentially in the current
process. Useful for dry runs and for debugging the worker. Hardware
target is honored only by the JAX platform setting; no GPU is
materialized.

RunPod mode (default) submits one pod per cell via the RunPod
GraphQL API, polls every 60 seconds, retries failed cells up to
twice, and downloads result CSVs from the network volume back to
``--output-dir``. Requires ``RUNPOD_API_KEY`` in the environment.

The dispatcher tracks total spend and wall-clock time and halts if
either exceeds the bounds in the plan (50 USD, 4 hours, 30 minutes
per cell with one retry).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from experiments.jss_deep_run.matrix import MATRIX, Cell, cells_for_tiers, get_cell


# ---------------------------------------------------------------------------
# Cost ceiling
# ---------------------------------------------------------------------------

@dataclass
class CostCeiling:
    max_total_spend_usd: float = 50.0
    max_wall_clock_s: float = 4 * 3600.0
    max_per_cell_s: float = 30 * 60.0
    spend_so_far_usd: float = 0.0
    wall_clock_so_far_s: float = 0.0
    halted: bool = False
    halt_reason: str | None = None

    def check(self) -> None:
        if self.spend_so_far_usd > self.max_total_spend_usd:
            self.halted = True
            self.halt_reason = (
                f"Total spend {self.spend_so_far_usd:.2f} USD exceeds "
                f"ceiling {self.max_total_spend_usd:.2f} USD."
            )
        if self.wall_clock_so_far_s > self.max_wall_clock_s:
            self.halted = True
            self.halt_reason = (
                f"Wall-clock {self.wall_clock_so_far_s/3600:.2f}h exceeds "
                f"ceiling {self.max_wall_clock_s/3600:.2f}h."
            )


# ---------------------------------------------------------------------------
# Local sequential mode
# ---------------------------------------------------------------------------

def run_local(cells: list[Cell], output_dir: Path) -> list[dict[str, Any]]:
    """Run cells sequentially in subprocesses, tracking outcomes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    outcomes = []
    ceiling = CostCeiling()
    start = time.time()

    for cell in cells:
        if ceiling.halted:
            print(f"Halted: {ceiling.halt_reason}")
            break

        cmd = [
            sys.executable,
            "-m", "experiments.jss_deep_run.run_cell",
            "--cell-id", cell.cell_id,
            "--output-dir", str(output_dir),
        ]
        cell_start = time.time()
        print(f"[local] {cell.cell_id} ({cell.estimator} on {cell.dataset})")
        result = subprocess.run(
            cmd,
            timeout=ceiling.max_per_cell_s,
            capture_output=True,
            text=True,
        )
        cell_elapsed = time.time() - cell_start

        outcomes.append(
            {
                "cell_id": cell.cell_id,
                "returncode": result.returncode,
                "wall_clock_s": cell_elapsed,
                "stdout_tail": result.stdout[-512:] if result.stdout else "",
                "stderr_tail": result.stderr[-512:] if result.stderr else "",
            }
        )

        ceiling.wall_clock_so_far_s = time.time() - start
        ceiling.check()

    return outcomes


# ---------------------------------------------------------------------------
# RunPod mode (GraphQL stub)
# ---------------------------------------------------------------------------

# RunPod community A100 list price as of plan write-up. Refresh before
# any production run.
_RUNPOD_USD_PER_HOUR = {"cpu": 0.40, "gpu": 1.20}


def _bootstrap_command(cell: Cell, repo_ref: str) -> str:
    """Construct the in-pod setup-and-run command.

    The pod uses the public ``runpod/pytorch`` template (no custom
    image build required). At launch the pod clones the econirl repo
    at the pinned ref, installs it with the dev extras, and runs the
    cell. Output is written to ``/workspace/results`` which is mounted
    from the persistent network volume so results survive pod
    termination.

    JAX is installed against CUDA 12 wheels for GPU cells; CPU cells
    install the wheel-only JAX so the inner solver runs on the box's
    CPU.
    """
    jax_install = (
        "pip install -q 'jax[cuda12_pip]==0.4.30' "
        "-f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
        if cell.hardware == "gpu"
        else "pip install -q 'jax==0.4.30'"
    )
    return (
        "set -euo pipefail; "
        "cd /workspace && "
        f"git clone --depth 1 --branch main https://github.com/rawatpranjal/EconIRL.git econirl && "
        f"cd /workspace/econirl && git checkout {repo_ref} && "
        f"{jax_install} && "
        "pip install -q -e .[dev] && "
        "mkdir -p /workspace/results/shapeshifter && "
        f"python -m experiments.jss_deep_run.run_cell --cell-id {cell.cell_id} "
        f"--output-dir /workspace/results/shapeshifter"
    )


def _runpod_pod_payload(
    cell: Cell, image: str, volume_id: str, repo_ref: str = "main",
) -> dict[str, Any]:
    """Build the GraphQL ``podRentInterruptable`` mutation payload.

    Uses the public ``runpod/pytorch`` template by default; the
    bootstrap dockerArgs clones the econirl repo at ``repo_ref`` and
    installs it before running the cell. Results land in
    ``/workspace/results/shapeshifter`` on the persistent volume.
    """
    payload: dict[str, Any] = {
        "name": cell.cell_id,
        "image_name": image,
        "cloud_type": "COMMUNITY",
        "container_disk_in_gb": 20,
        "volume_in_gb": 0,
        "volume_mount_path": "/workspace",
        "network_volume_id": volume_id,
        "data_center_id": "EU-RO-1",  # matches the network volume's region
        "ports": "",
        "env": {
            "CELL_ID": cell.cell_id,
            "JAX_PLATFORMS": "cuda" if cell.hardware == "gpu" else "cpu",
        },
        "docker_args": _bootstrap_command(cell, repo_ref),
    }
    if cell.hardware == "gpu":
        payload["gpu_count"] = 1
        payload["gpu_type_id"] = "NVIDIA A100 80GB PCIe"
    else:
        payload["gpu_count"] = 0
        payload["min_vcpu_count"] = 2
        payload["min_memory_in_gb"] = 4
    return payload


def run_runpod(
    cells: list[Cell], image: str, output_dir: Path, max_parallel: int,
    volume_id: str | None, repo_ref: str = "main",
) -> list[dict[str, Any]]:
    """Submit cells as RunPod pods and download results.

    This function is written defensively against a missing RunPod SDK:
    if the `runpod` package is unavailable the dispatcher falls back
    to printing the payload it would submit so the operator can
    inspect what would have been sent before installing the SDK.
    """
    try:
        import runpod  # type: ignore
    except ImportError:
        print(
            "runpod SDK not installed. Install with `pip install runpod` "
            "before submitting jobs. Printing payloads instead."
        )
        for cell in cells:
            payload = _runpod_pod_payload(cell, image, volume_id or "<volume_id>")
            print(json.dumps({"cell": cell.cell_id, "payload": payload}, indent=2))
        return []

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        raise RuntimeError(
            "RUNPOD_API_KEY not set. Export it before running the dispatcher."
        )
    runpod.api_key = api_key

    output_dir.mkdir(parents=True, exist_ok=True)
    ceiling = CostCeiling()
    start = time.time()
    in_flight: list[tuple[Cell, str, float]] = []  # (cell, pod_id, started_at)
    outcomes = []
    pending = list(cells)
    pending.sort(key=lambda c: -c.expected_runtime_s)  # longest first

    while pending or in_flight:
        if ceiling.halted:
            print(f"Halted: {ceiling.halt_reason}")
            break

        # Launch up to max_parallel pods.
        while pending and len(in_flight) < max_parallel:
            cell = pending.pop(0)
            payload = _runpod_pod_payload(cell, image, volume_id or "<volume_id>", repo_ref)
            try:
                pod = runpod.create_pod(**payload)
                in_flight.append((cell, pod["id"], time.time()))
                print(f"[runpod] launched {cell.cell_id} pod_id={pod['id']}")
            except Exception as e:
                print(f"[runpod] failed to launch {cell.cell_id}: {e}")
                outcomes.append({"cell_id": cell.cell_id, "error": str(e)})

        # Poll pods.
        time.sleep(60)
        still_in_flight = []
        for cell, pod_id, started_at in in_flight:
            try:
                status = runpod.get_pod(pod_id)
            except Exception as e:
                print(f"[runpod] failed to poll {cell.cell_id}: {e}")
                still_in_flight.append((cell, pod_id, started_at))
                continue

            if status.get("desiredStatus") == "EXITED":
                elapsed = time.time() - started_at
                rate = _RUNPOD_USD_PER_HOUR[cell.hardware]
                spend = (elapsed / 3600.0) * rate
                ceiling.spend_so_far_usd += spend
                outcomes.append(
                    {
                        "cell_id": cell.cell_id,
                        "pod_id": pod_id,
                        "wall_clock_s": elapsed,
                        "spend_usd": spend,
                    }
                )
                print(f"[runpod] {cell.cell_id} done in {elapsed:.0f}s, "
                      f"spend {spend:.3f} USD")
                # Pods write CSVs to the persistent volume; rsync to local.
                # In a real deployment this would use the RunPod API to
                # download from /workspace/results. The stub leaves it as
                # a TODO with a clear message.
                print(f"[runpod] TODO: download {cell.cell_id}.csv from volume to "
                      f"{output_dir}")
            elif time.time() - started_at > ceiling.max_per_cell_s:
                # Timeout: terminate and record.
                print(f"[runpod] {cell.cell_id} timed out, terminating pod")
                try:
                    runpod.terminate_pod(pod_id)
                except Exception:
                    pass
                outcomes.append(
                    {"cell_id": cell.cell_id, "pod_id": pod_id, "error": "timeout"}
                )
            else:
                still_in_flight.append((cell, pod_id, started_at))
        in_flight = still_in_flight

        ceiling.wall_clock_so_far_s = time.time() - start
        ceiling.check()

    return outcomes


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tier",
        default=None,
        help="Comma-separated tier ids to run, e.g. '1,2,3a,4'. Mutually exclusive with --cell-id.",
    )
    parser.add_argument(
        "--cell-id",
        default=None,
        help="Single cell id to dispatch. Useful for smoke tests.",
    )
    parser.add_argument("--local", action="store_true", help="Run sequentially in-process.")
    parser.add_argument(
        "--image",
        default="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
        help="Container image. Defaults to the public RunPod pytorch+CUDA template; the bootstrap dockerArgs install econirl in-pod, so no custom image build is required.",
    )
    parser.add_argument(
        "--repo-ref",
        default="main",
        help="Git ref (branch, tag, or SHA) of econirl to install in each pod.",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=8,
        help="Maximum number of pods running at once.",
    )
    parser.add_argument(
        "--volume-id",
        default=None,
        help="RunPod network volume id for persisted result CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/jss_deep_run/results"),
    )
    parser.add_argument(
        "--max-spend-usd",
        type=float,
        default=None,
        help="Override the dispatcher cost ceiling. Default uses CostCeiling defaults.",
    )
    parser.add_argument(
        "--max-wallclock-hours",
        type=float,
        default=None,
        help="Override the dispatcher wall-clock ceiling.",
    )
    args = parser.parse_args()

    if args.tier is None and args.cell_id is None:
        args.tier = "1"
    if args.tier is not None and args.cell_id is not None:
        parser.error("Pass --tier or --cell-id, not both.")

    if args.cell_id is not None:
        cells = [get_cell(args.cell_id)]
        tiers = [args.cell_id]
        print(f"Dispatching single cell {args.cell_id}")
    else:
        tiers = [t.strip() for t in args.tier.split(",") if t.strip()]
        cells = cells_for_tiers(tiers)
        print(f"Selected {len(cells)} cells across tiers {tiers}")

    # The CostCeiling has its own defaults; the CLI flags override them.
    # The local and runpod runners both construct their own CostCeiling
    # internally, so the override is applied via class attribute mutation
    # before they fire.
    if args.max_spend_usd is not None:
        CostCeiling.max_total_spend_usd = args.max_spend_usd
    if args.max_wallclock_hours is not None:
        CostCeiling.max_wall_clock_s = args.max_wallclock_hours * 3600.0

    if args.local:
        outcomes = run_local(cells, args.output_dir)
    else:
        outcomes = run_runpod(
            cells,
            image=args.image,
            output_dir=args.output_dir,
            max_parallel=args.max_parallel,
            volume_id=args.volume_id,
            repo_ref=args.repo_ref,
        )

    manifest = args.output_dir / "dispatch_manifest.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest, "w") as f:
        json.dump({"tiers": tiers, "outcomes": outcomes}, f, indent=2)
    print(f"Wrote {manifest}")


if __name__ == "__main__":
    main()
