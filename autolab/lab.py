#!/usr/bin/env python3
"""AutoLab — Autonomous overnight experimentation for econirl.

Orchestrator loop: reads program.md → asks Claude for next experiment →
runs it via subprocess → logs results → repeats.

Usage:
    python autolab/lab.py                    # Run the loop
    python autolab/lab.py --dry-run          # Propose 3 experiments without running
    python autolab/lab.py --resume           # Continue from existing log
    python autolab/lab.py --max-experiments 5
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG_PATH = ROOT / "experiment_log.jsonl"
FINDINGS_PATH = ROOT / "FINDINGS.md"
PROGRAM_PATH = ROOT / "program.md"
RESULTS_DIR = ROOT / "results"
RUN_SCRIPT = ROOT / "run_experiment.py"

DEFAULT_EXPERIMENT_TIMEOUT = 120  # seconds per experiment (toy problems finish in <30s)
CLAUDE_TIMEOUT = 120  # seconds for Claude proposal


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def read_program() -> str:
    """Read the human-written research program."""
    return PROGRAM_PATH.read_text()


def read_log() -> list[dict]:
    """Read JSONL log, skipping malformed lines."""
    if not LOG_PATH.exists():
        return []
    entries = []
    for line in LOG_PATH.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def best_per_estimator(log: list[dict]) -> dict[str, dict]:
    """Return best result per estimator (by pct_optimal)."""
    best = {}
    for entry in log:
        if entry.get("status") != "success":
            continue
        name = entry.get("estimator", "")
        pct = entry.get("pct_optimal")
        if pct is None:
            continue
        if name not in best or pct > best[name].get("pct_optimal", -999):
            best[name] = entry
    return best


def append_log(entry: dict) -> None:
    """Append one JSON line to the experiment log."""
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Claude proposal
# ---------------------------------------------------------------------------

def _build_prompt(program: str, log: list[dict], exp_num: int) -> str:
    """Build the prompt for Claude to propose the next experiment."""
    # Last 10 experiments summary
    recent = log[-10:] if log else []
    recent_text = ""
    for e in recent:
        status = e.get("status", "?")
        est = e.get("estimator", "?")
        pct = e.get("pct_optimal")
        pct_str = f"{pct:.1f}%" if pct is not None else "N/A"
        eid = e.get("experiment_id", "?")
        recent_text += f"  {eid}: {est} → {pct_str} ({status})\n"
    if not recent_text:
        recent_text = "  (no experiments yet)\n"

    # Best per estimator
    best = best_per_estimator(log)
    best_text = ""
    for name in sorted(best.keys()):
        b = best[name]
        pct = b.get("pct_optimal", 0)
        best_text += f"  {name}: {pct:.1f}%\n"
    if not best_text:
        best_text = "  (no results yet)\n"

    return f"""You are an ML researcher running autonomous hyperparameter experiments on a tiny 5-state MDP.

## Research Program
{program}

## Recent Experiments (last 10)
{recent_text}
## Best Results Per Estimator
{best_text}
## Task
Propose experiment #{exp_num}. Pick from the bag of ideas in the research program based on results so far.

All experiments MUST use: n_states=5, discount_factor=0.95, n_agents=100, n_periods=50, seed=42.

If an estimator has no baseline yet, run defaults first (Idea #1). Focus effort where the gap to target is largest.

Return ONLY a JSON object (no markdown fences) with these fields:
- "reasoning": string explaining why this experiment (1-2 sentences)
- "idea_number": int (1-25) from the bag of ideas
- "experiment_id": "exp_{exp_num:03d}"
- "estimator": one of ["NFXP", "CCP", "MCE IRL", "MaxEnt IRL", "Max Margin", "TD-CCP", "GLADIUS", "GAIL", "AIRL", "GCL"]
- "dgp": {{"n_states": 5, "discount_factor": 0.95}}
- "estimator_kwargs": object with estimator-specific hyperparameters
- "n_agents": 100
- "n_periods": 50
- "seed": 42

For TD-CCP, put TDCCPConfig fields inside "estimator_kwargs"."config".
"""


def ask_claude(program: str, log: list[dict], exp_num: int) -> dict | None:
    """Ask Claude to propose the next experiment. Returns parsed config or None."""
    prompt = _build_prompt(program, log, exp_num)

    try:
        result = subprocess.run(
            ["claude", "-p", prompt, "--output-format", "json", "--max-turns", "1"],
            capture_output=True,
            text=True,
            timeout=CLAUDE_TIMEOUT,
        )
    except FileNotFoundError:
        print("ERROR: 'claude' CLI not found. Install Claude Code first.")
        return None
    except subprocess.TimeoutExpired:
        print(f"WARNING: Claude proposal timed out after {CLAUDE_TIMEOUT}s")
        return None

    if result.returncode != 0:
        print(f"WARNING: Claude exited with code {result.returncode}")
        print(f"  stderr: {result.stderr[:500]}")
        return None

    raw = result.stdout.strip()

    # With --output-format json, Claude returns a JSON wrapper with a "result" field
    try:
        wrapper = json.loads(raw)
        if isinstance(wrapper, dict) and "result" in wrapper:
            text = wrapper["result"]
        else:
            text = raw
    except json.JSONDecodeError:
        text = raw

    # Strip markdown fences if present
    if isinstance(text, str):
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            # Remove first and last fence lines
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)

        try:
            config = json.loads(text)
        except json.JSONDecodeError:
            print(f"WARNING: Could not parse Claude response as JSON:\n{text[:500]}")
            return None
    elif isinstance(text, dict):
        config = text
    else:
        print(f"WARNING: Unexpected Claude response type: {type(text)}")
        return None

    # Validate required fields
    if "estimator" not in config:
        print("WARNING: Claude proposal missing 'estimator' field")
        return None

    # Ensure experiment_id
    config.setdefault("experiment_id", f"exp_{exp_num:03d}")

    return config


# ---------------------------------------------------------------------------
# Experiment execution
# ---------------------------------------------------------------------------

def run_experiment(config: dict, timeout: int = DEFAULT_EXPERIMENT_TIMEOUT) -> dict:
    """Run one experiment via subprocess, return result dict."""
    exp_id = config.get("experiment_id", "unknown")
    exp_dir = RESULTS_DIR / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    config_path = exp_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2))

    print(f"  Running {config['estimator']} ({exp_id})...")
    t0 = time.time()

    try:
        result = subprocess.run(
            [sys.executable, str(RUN_SCRIPT), str(config_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {
            "experiment_id": exp_id,
            "status": "error",
            "error": f"Experiment timed out after {timeout}s",
            "estimator": config.get("estimator", "unknown"),
        }

    wall_time = time.time() - t0

    # Save raw output
    (exp_dir / "stdout.txt").write_text(result.stdout)
    if result.stderr:
        (exp_dir / "stderr.txt").write_text(result.stderr)

    if result.returncode != 0:
        return {
            "experiment_id": exp_id,
            "status": "error",
            "error": f"Process exited with code {result.returncode}: {result.stderr[:500]}",
            "estimator": config.get("estimator", "unknown"),
        }

    try:
        out = json.loads(result.stdout.strip())
    except json.JSONDecodeError:
        return {
            "experiment_id": exp_id,
            "status": "error",
            "error": f"Could not parse output as JSON: {result.stdout[:500]}",
            "estimator": config.get("estimator", "unknown"),
        }

    # Attach config and timing metadata
    out["config"] = config.get("estimator_kwargs", {})
    out["dgp"] = config.get("dgp", {})
    out["wall_time"] = round(wall_time, 2)
    out["timestamp"] = datetime.now(timezone.utc).isoformat()

    # Save result
    (exp_dir / "result.json").write_text(json.dumps(out, indent=2))

    return out


# ---------------------------------------------------------------------------
# Findings generation (pure Python, no Claude)
# ---------------------------------------------------------------------------

BASELINES = {
    "NFXP": 100.0, "CCP": 100.0, "MCE IRL": 100.0,
    "TD-CCP": 0.0, "GLADIUS": 0.0, "Max Margin": 0.0,
    "AIRL": 0.0, "MaxEnt IRL": 0.0, "GCL": 0.0, "GAIL": 0.0,
}


def generate_findings(log: list[dict]) -> None:
    """Write FINDINGS.md from experiment log (no Claude call)."""
    best = best_per_estimator(log)
    total = len(log)
    successes = sum(1 for e in log if e.get("status") == "success")
    errors = total - successes

    lines = [
        "# AutoLab Findings",
        "",
        f"*Auto-generated at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*",
        f"*{total} experiments ({successes} success, {errors} errors)*",
        "",
        "## Leaderboard (best pct_optimal per estimator)",
        "",
        "| Estimator | Best | Baseline | Delta |",
        "|-----------|------|----------|-------|",
    ]

    for name in sorted(BASELINES.keys(), key=lambda n: -(best.get(n, {}).get("pct_optimal", 0))):
        baseline = BASELINES[name]
        if name in best:
            pct = best[name]["pct_optimal"]
            delta = pct - baseline
            sign = "+" if delta >= 0 else ""
            lines.append(f"| {name} | {pct:.1f}% | {baseline:.1f}% | {sign}{delta:.1f}% |")
        else:
            lines.append(f"| {name} | — | {baseline:.1f}% | — |")

    # --- Progress Trajectory ---
    # Track first result, best result, improvement, and run count per estimator
    TARGETS = {
        "NFXP": 99.5, "CCP": 99.5, "MCE IRL": 99.0,
        "TD-CCP": 98.0, "GLADIUS": 98.0, "Max Margin": 95.0,
        "AIRL": 90.0, "MaxEnt IRL": 90.0, "GCL": 90.0, "GAIL": 90.0,
    }
    first_per_est: dict[str, float] = {}
    best_per_est: dict[str, float] = {}
    count_per_est: dict[str, int] = {}
    for entry in log:
        if entry.get("status") != "success":
            continue
        est = entry.get("estimator", "")
        pct = entry.get("pct_optimal")
        if pct is None:
            continue
        count_per_est[est] = count_per_est.get(est, 0) + 1
        if est not in first_per_est:
            first_per_est[est] = pct
        if est not in best_per_est or pct > best_per_est[est]:
            best_per_est[est] = pct

    if first_per_est:
        lines += [
            "",
            "## Progress Trajectory",
            "",
            "| Estimator | First | Best | Improvement | # Runs | Target |",
            "|-----------|-------|------|-------------|--------|--------|",
        ]
        for name in sorted(TARGETS.keys(), key=lambda n: -(best_per_est.get(n, 0))):
            if name in first_per_est:
                first = first_per_est[name]
                bst = best_per_est[name]
                delta = bst - first
                sign = "+" if delta >= 0 else ""
                target = TARGETS.get(name, "—")
                runs = count_per_est.get(name, 0)
                lines.append(f"| {name} | {first:.1f}% | {bst:.1f}% | {sign}{delta:.1f}% | {runs} | {target}% |")

    # --- Ideas Explored ---
    idea_results: dict[int, list[float]] = {}
    for entry in log:
        idea = entry.get("idea_number")
        if idea is None:
            # Check inside config
            idea = entry.get("config", {}).get("idea_number")
        if idea is not None:
            pct = entry.get("pct_optimal")
            if idea not in idea_results:
                idea_results[idea] = []
            if pct is not None:
                idea_results[idea].append(pct)

    if idea_results:
        lines += [
            "",
            "## Ideas Explored",
            "",
            "| Idea # | Times Tried | Best pct_optimal |",
            "|--------|-------------|------------------|",
        ]
        for idea_num in sorted(idea_results.keys()):
            results = idea_results[idea_num]
            tries = len(results)
            best_pct = max(results) if results else None
            best_str = f"{best_pct:.1f}%" if best_pct is not None else "N/A"
            lines.append(f"| {idea_num} | {tries} | {best_str} |")

    # --- Ground Truth Report ---
    gt_entries = [e for e in log if e.get("status") == "success" and e.get("ground_truth")]
    if gt_entries:
        lines += [
            "",
            "## Ground Truth Report",
            "",
            "| # | ID | Estimator | Difficulty | Criteria | Passed |",
            "|---|-----|-----------|------------|----------|--------|",
        ]
        for i, entry in enumerate(gt_entries, 1):
            eid = entry.get("experiment_id", "?")
            est = entry.get("estimator", "?")
            diff = entry.get("difficulty", "?")
            gt = entry.get("ground_truth", {})
            all_ok = entry.get("all_criteria_passed", False)
            # Format each criterion
            details = []
            for cname, cinfo in gt.items():
                mark = "v" if cinfo.get("passed") else "X"
                actual = cinfo.get("actual")
                thresh = cinfo.get("threshold")
                if isinstance(actual, bool):
                    details.append(f"{mark} {cname}={actual}")
                elif actual is not None:
                    details.append(f"{mark} {cname}={actual} (need {thresh})")
                else:
                    details.append(f"X {cname}=N/A")
            criteria_str = "; ".join(details) if details else "—"
            all_str = "ALL PASS" if all_ok else "FAIL"
            lines.append(f"| {i} | {eid} | {est} | {diff} | {criteria_str} | {all_str} |")

    lines += [
        "",
        "## Experiment History",
        "",
        "| # | ID | Estimator | pct_optimal | Time | Status |",
        "|---|-----|-----------|-------------|------|--------|",
    ]

    for i, entry in enumerate(log, 1):
        eid = entry.get("experiment_id", "?")
        est = entry.get("estimator", "?")
        status = entry.get("status", "?")
        pct = entry.get("pct_optimal")
        pct_str = f"{pct:.1f}%" if pct is not None else "N/A"
        t = entry.get("wall_time", entry.get("time_seconds", "?"))
        t_str = f"{t:.0f}s" if isinstance(t, (int, float)) else str(t)
        lines.append(f"| {i} | {eid} | {est} | {pct_str} | {t_str} | {status} |")

    FINDINGS_PATH.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def should_stop(program: str, exp_num: int, max_experiments: int,
                start_time: float, wall_budget: float) -> str | None:
    """Return stop reason or None to continue."""
    if "## Stop" in program:
        return "## Stop found in program.md"
    if exp_num > max_experiments:
        return f"Max experiments reached ({max_experiments})"
    elapsed = time.time() - start_time
    if elapsed > wall_budget:
        return f"Wall-clock budget exceeded ({wall_budget/3600:.1f}h)"
    return None


def main():
    parser = argparse.ArgumentParser(description="AutoLab — autonomous experimentation for econirl")
    parser.add_argument("--dry-run", action="store_true", help="Propose 3 experiments without running")
    parser.add_argument("--resume", action="store_true", help="Continue from existing log")
    parser.add_argument("--max-experiments", type=int, default=50, help="Max experiments to run")
    parser.add_argument("--wall-budget", type=float, default=4.0, help="Wall-clock budget in hours")
    parser.add_argument("--max-memory-gb", type=int, default=4, help="Memory limit per experiment in GB")
    parser.add_argument("--single-thread", action=argparse.BooleanOptionalAction, default=True,
                        help="Pin experiments to single thread (default: True)")
    parser.add_argument("--timeout", type=int, default=DEFAULT_EXPERIMENT_TIMEOUT,
                        help=f"Per-experiment timeout in seconds (default: {DEFAULT_EXPERIMENT_TIMEOUT})")
    args = parser.parse_args()

    wall_budget = args.wall_budget * 3600  # convert to seconds

    if not PROGRAM_PATH.exists():
        print(f"ERROR: {PROGRAM_PATH} not found")
        sys.exit(1)

    log = read_log() if args.resume else []
    start_exp = len(log) + 1
    start_time = time.time()

    if args.dry_run:
        print("=== DRY RUN: proposing 3 experiments ===\n")
        program = read_program()
        for i in range(3):
            exp_num = start_exp + i
            print(f"--- Proposal #{exp_num} ---")
            config = ask_claude(program, log, exp_num)
            if config:
                reasoning = config.pop("reasoning", "")
                print(f"  Reasoning: {reasoning}")
                print(f"  Config: {json.dumps(config, indent=2)}")
                # Add a fake entry so Claude sees it in context for next proposal
                log.append({"experiment_id": config["experiment_id"],
                            "estimator": config["estimator"],
                            "status": "proposed", "pct_optimal": None})
            else:
                print("  (Claude did not return a valid proposal)")
            print()
        return

    print(f"=== AutoLab starting ({args.max_experiments} max, {args.wall_budget}h budget) ===")
    if log:
        print(f"  Resuming from {len(log)} existing experiments")
    print()

    exp_num = start_exp
    while True:
        # Re-read program each iteration (allows mid-run edits)
        program = read_program()

        reason = should_stop(program, exp_num, args.max_experiments, start_time, wall_budget)
        if reason:
            print(f"\n=== Stopping: {reason} ===")
            break

        print(f"--- Experiment #{exp_num} ---")

        # 1. Ask Claude for next experiment
        config = ask_claude(program, log, exp_num)
        if config is None:
            print("  Skipping: Claude did not return a valid proposal")
            exp_num += 1
            continue

        reasoning = config.pop("reasoning", "")
        print(f"  Reasoning: {reasoning}")
        print(f"  Estimator: {config['estimator']}")

        # DGP guard: cap n_states at 10 for toy experiments
        dgp = config.get("dgp", {})
        if dgp.get("n_states", 5) > 10:
            print(f"  WARNING: Capping n_states from {dgp['n_states']} to 10")
            dgp["n_states"] = 10
            config["dgp"] = dgp

        # Inject resource guardrails
        config["max_memory_gb"] = args.max_memory_gb
        config["single_thread"] = args.single_thread

        # 2. Run the experiment
        result = run_experiment(config, timeout=args.timeout)

        # 3. Log
        append_log(result)
        log.append(result)

        # 4. Report
        status = result.get("status", "?")
        pct = result.get("pct_optimal")
        pct_str = f"{pct:.1f}%" if pct is not None else "N/A"
        wall = result.get("wall_time", "?")
        diff = result.get("difficulty", "?")
        gt_pass = result.get("all_criteria_passed")
        gt_str = f" | ground_truth={'PASS' if gt_pass else 'FAIL'}" if gt_pass is not None else ""
        print(f"  Result: {status} | pct_optimal={pct_str} | {diff}{gt_str} | time={wall}s")

        # 5. Update findings
        generate_findings(log)

        exp_num += 1
        print()

    # Final findings
    generate_findings(log)
    elapsed = time.time() - start_time
    print(f"\n=== Done: {len(log)} experiments in {elapsed/60:.1f} minutes ===")
    print(f"  Log: {LOG_PATH}")
    print(f"  Findings: {FINDINGS_PATH}")


if __name__ == "__main__":
    main()
