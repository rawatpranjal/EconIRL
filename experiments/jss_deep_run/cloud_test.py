"""Cloud-side runner for ad-hoc pytest invocations and shell commands.

The matrix is dispatched by ``dispatch_runpod.py``. For non-matrix
work — running unit tests after a code change, building the paper
PDF, smoke-testing scaffolding, dispatching a single ss-* cell —
submit a one-off RunPod pod via this script. Same image, same
persistent volume, same cost accounting model.

The image's entrypoint is ``bash -lc``, so any of the three modes
below resolve to a single shell command string passed as dockerArgs.

Usage examples::

    # Run a pytest selector on a CPU pod:
    python -m experiments.jss_deep_run.cloud_test \\
        --pytest "tests/test_shapeshifter_env.py -v"

    # Build the JSS paper PDF on a CPU pod (TeX Live is in the image):
    python -m experiments.jss_deep_run.cloud_test \\
        --shell "cd papers/econirl_package && latexmk -pdf main.tex"

    # Run a Python module:
    python -m experiments.jss_deep_run.cloud_test \\
        --module experiments.jss_deep_run.smoke_test

Hardware defaults to CPU. Pass ``--gpu`` for GPU. Default cost
ceiling is 5 USD; override with ``--max-spend-usd``. Logs stream to
stdout via the RunPod log API; any artifacts written to
``/workspace/results`` land in the persistent network volume and can
be pulled back via ``runpodctl receive``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

_RUNPOD_USD_PER_HOUR = {"cpu": 0.40, "gpu": 1.20}
_DEFAULT_IMAGE = "econirl-deep-run:v1"
_REPO_ROOT_IN_POD = "/workspace/econirl"


def _build_command(args: argparse.Namespace) -> str:
    """Construct the bash command string from one of the mode flags.

    The command is wrapped in a ``cd`` to the repo root so relative
    paths in the user's selector resolve as they would locally.
    """
    if args.pytest:
        return f"cd {_REPO_ROOT_IN_POD} && python -m pytest {args.pytest}"
    if args.shell:
        return f"cd {_REPO_ROOT_IN_POD} && {args.shell}"
    if args.module:
        return f"cd {_REPO_ROOT_IN_POD} && python -m {args.module}"
    raise ValueError("Pass exactly one of --pytest, --shell, --module.")


def _runpod_payload(
    command: str,
    image: str,
    hardware: str,
    volume_id: str | None,
    name: str,
) -> dict[str, Any]:
    return {
        "cloudType": "COMMUNITY",
        "gpuCount": 1 if hardware == "gpu" else 0,
        "gpuTypeId": "NVIDIA A100 80GB" if hardware == "gpu" else None,
        "name": name,
        "imageName": image,
        "containerDiskInGb": 20,
        "volumeInGb": 0,
        "volumeMountPath": "/workspace/results",
        "networkVolumeId": volume_id,
        "ports": "",
        "env": [
            {"key": "JAX_PLATFORMS", "value": "cuda" if hardware == "gpu" else "cpu"},
        ],
        "dockerArgs": command,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pytest", help="Pytest selector run from the repo root.")
    mode.add_argument("--shell", help="Raw shell command run via bash -lc.")
    mode.add_argument("--module", help="Python module run with `python -m`.")
    parser.add_argument("--gpu", action="store_true", help="Use a GPU pod.")
    parser.add_argument("--image", default=_DEFAULT_IMAGE)
    parser.add_argument(
        "--volume-id",
        default=os.environ.get("RUNPOD_VOLUME_ID"),
        help="RunPod network volume id; defaults to $RUNPOD_VOLUME_ID.",
    )
    parser.add_argument("--max-spend-usd", type=float, default=5.0)
    parser.add_argument(
        "--max-wallclock-s",
        type=float,
        default=30 * 60.0,
        help="Hard wall-clock ceiling in seconds (default 30 min).",
    )
    parser.add_argument("--name", default=None, help="Pod name (default: timestamp).")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the payload that would be submitted and exit.",
    )
    args = parser.parse_args()

    command = _build_command(args)
    hardware = "gpu" if args.gpu else "cpu"
    pod_name = args.name or f"cloud-test-{int(time.time())}"
    print(f"[cloud_test] hardware={hardware} max_spend={args.max_spend_usd} USD")
    print(f"[cloud_test] command: {command}")

    payload = _runpod_payload(command, args.image, hardware, args.volume_id, pod_name)

    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return

    try:
        import runpod  # type: ignore
    except ImportError:
        print(
            "[cloud_test] runpod SDK not installed. Install with `pip install "
            "runpod` first. Printing payload that would have been submitted:"
        )
        print(json.dumps(payload, indent=2))
        sys.exit(2)

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("[cloud_test] RUNPOD_API_KEY not set; export it first.", file=sys.stderr)
        sys.exit(2)
    runpod.api_key = api_key

    pod = runpod.create_pod(**payload)
    pod_id = pod["id"]
    print(f"[cloud_test] launched pod_id={pod_id}")

    started_at = time.time()
    rate = _RUNPOD_USD_PER_HOUR[hardware]
    poll_interval_s = 30.0

    while True:
        elapsed = time.time() - started_at
        spend = (elapsed / 3600.0) * rate
        if spend > args.max_spend_usd:
            print(
                f"[cloud_test] spend {spend:.3f} USD exceeded ceiling "
                f"{args.max_spend_usd}; terminating pod"
            )
            try:
                runpod.terminate_pod(pod_id)
            except Exception:
                pass
            sys.exit(3)
        if elapsed > args.max_wallclock_s:
            print(
                f"[cloud_test] wall-clock {elapsed:.0f}s exceeded ceiling "
                f"{args.max_wallclock_s:.0f}s; terminating pod"
            )
            try:
                runpod.terminate_pod(pod_id)
            except Exception:
                pass
            sys.exit(3)

        try:
            status = runpod.get_pod(pod_id)
        except Exception as e:
            print(f"[cloud_test] poll failed ({e}); retrying in {poll_interval_s:.0f}s")
            time.sleep(poll_interval_s)
            continue

        desired = status.get("desiredStatus")
        if desired == "EXITED":
            print(
                f"[cloud_test] pod exited after {elapsed:.0f}s, "
                f"spend {spend:.3f} USD"
            )
            break
        time.sleep(poll_interval_s)


if __name__ == "__main__":
    main()
