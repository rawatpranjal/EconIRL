"""Shared harness for the per-experiment simulation-study pages.

Generalizes ``scripts/quick_all_estimators.py`` (the Abstract MDP 1 exemplar)
over experiment cells so each ``docs/simulation_studies/`` sub-page is a thin
script: cells + roster + narrative. The honesty contract is inherited verbatim:

1. Raw facts only, written by the code. The per-experiment JSON stores, per
   estimator per cell per replication, the actual recovered parameters,
   standard errors, policy total-variation distance vs the oracle, value RMSE,
   Type A/B/C counterfactual regret, runtime, the converged flag, and the
   verbatim exception on failure. Nothing is hand-entered.
2. Every printed number is recomputed from that JSON at render time. The page
   table is a pure function of the raw records; rendering twice from the same
   JSON yields byte-identical output.
3. Crashes are first class: shown with the verbatim message, never dropped.
4. No fabricated metrics. Parameter recovery is reported only for the
   structural family on cells with a finite true theta in the same gauge;
   coverage only where standard errors are finite, with the SE-availability
   rate always visible so missing inference cannot hide behind an "n/a".
5. Stamped and reproducible: seeds, date, package version, and every cell's
   environment config live in the JSON.
6. ``--verify`` re-renders the console table purely from the saved JSON.
7. Honest scope: excluded estimators are listed with reasons; slow estimators
   may run fewer replications (``max_reps``) but are labeled, never hidden.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import inspect
import json
import os
import signal
import sys
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from econirl.simulation.synthetic import simulate_panel
from validation.benchmark import metrics as M
from validation.benchmark.regret import estimator_regret
from validation.benchmark.runner import _oracle


@dataclass(frozen=True)
class RosterEntry:
    """A benchmarkable estimator on the uniform ``.estimate`` path.

    Attributes:
        name: Display name.
        family: ``"structural"`` (gets parameter metrics on finite-theta cells)
            or ``"behavioral"`` (policy/value/regret only).
        run: ``run(env, panel)`` returning an object exposing ``parameters``,
            ``standard_errors``, ``policy``, ``value_function``, ``converged``.
        max_reps: Optional cap on replications for slow estimators. The table's
            "Ran" column shows the reduced denominator, so the cap is visible.
    """

    name: str
    family: str
    run: Callable[[object, object], object]
    max_reps: int | None = None
    timeout: int | None = None  # per-fit budget in seconds (overrides the cell's)


@dataclass(frozen=True)
class Cell:
    """One experiment cell: an environment plus panel and replication config."""

    cell_id: str
    label: str
    description: str
    env_factory: Callable[[], object]
    roster: tuple[RosterEntry, ...]
    n_individuals: int = 500
    n_periods: int = 50
    seed: int = 42
    n_replications: int = 3
    param_block: bool = False  # render the bias/SE/RMSE/coverage table
    figure: str | None = None  # absolute PNG path for the 1x2 DGP figure
    fit_timeout: int | None = None  # default per-fit budget in seconds


# ---------------------------------------------------------------------------
# Run: collect raw facts
# ---------------------------------------------------------------------------


def _to_list(x):
    if x is None:
        return None
    return np.asarray(x, dtype=np.float64).reshape(-1).tolist()


def _package_version() -> str:
    try:
        import econirl

        return getattr(econirl, "__version__", "unknown")
    except Exception:  # noqa: BLE001
        return "unknown"


class _FitTimeout(BaseException):
    """Raised by the SIGALRM handler when a fit exceeds its budget.

    Derives from BaseException so an estimator's internal ``except Exception``
    cannot swallow it (MMP's optimizer loop did exactly that and ran 32 minutes
    past a 15-minute budget). Still best-effort: a fit stuck inside one long
    native call is only interrupted when control returns to Python.
    """


def _run_one(env, panel, entry: RosterEntry, cell_id: str, rep: int,
             oracle_policy, oracle_value, timeout: int | None = None) -> dict:
    """Fit one estimator on one panel and record the raw facts.

    A ``timeout`` (seconds) is enforced with SIGALRM; a fit that exceeds it is
    recorded as a TimeoutError with the budget in the message — the timeout is
    data, not a silent drop.
    """
    t0 = time.time()
    rec = {"estimator": entry.name, "family": entry.family, "cell": cell_id,
           "rep": rep, "params": None, "standard_errors": None,
           "policy_tv": None, "value_rmse": None, "regret": None,
           "runtime": None, "converged": None, "error": None}
    try:
        if timeout:
            def _on_alarm(signum, frame):  # noqa: ARG001
                raise _FitTimeout(f"fit exceeded the {timeout}s budget")

            old_handler = signal.signal(signal.SIGALRM, _on_alarm)
            signal.alarm(int(timeout))
        try:
            res = entry.run(env, panel)
        finally:
            if timeout:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        rec["runtime"] = time.time() - t0
        rec["converged"] = bool(getattr(res, "converged", True))
        pol = getattr(res, "policy", None)
        if pol is not None and np.asarray(pol).shape == oracle_policy.shape:
            rec["policy_tv"] = M.policy_tv(np.asarray(pol), oracle_policy)
        rec["value_rmse"] = M.value_rmse(
            None if getattr(res, "value_function", None) is None
            else np.asarray(res.value_function), oracle_value)
        rec["params"] = _to_list(getattr(res, "parameters", None))
        rec["standard_errors"] = _to_list(getattr(res, "standard_errors", None))
        try:
            rr = estimator_regret(env, getattr(res, "parameters", None), pol)
            rec["regret"] = {"baseline": rr.baseline, "type_a": rr.type_a,
                             "type_b": rr.type_b, "type_c": rr.type_c,
                             "transferred": rr.transferred}
        except Exception:  # noqa: BLE001 - regret is best-effort, never fabricated
            rec["regret"] = None
    except _FitTimeout as exc:
        rec["runtime"] = time.time() - t0
        rec["error"] = f"TimeoutError: {exc}"
    except Exception as exc:  # noqa: BLE001 - the failure IS the result
        rec["runtime"] = time.time() - t0
        rec["error"] = f"{type(exc).__name__}: {exc}"
    return rec


def _cell_meta(cell: Cell, env) -> dict:
    names = list(env.parameter_names)
    true_theta = (np.asarray(env.get_true_parameter_vector(), dtype=np.float64).tolist()
                  if names else None)
    return {
        "cell_id": cell.cell_id,
        "label": cell.label,
        "description": cell.description,
        "num_states": int(env.num_states),
        "num_actions": int(env.num_actions),
        "discount_factor": float(env.problem_spec.discount_factor),
        "n_individuals": cell.n_individuals,
        "n_periods": cell.n_periods,
        "seed": cell.seed,
        "n_replications": cell.n_replications,
        "param_block": cell.param_block,
        "figure": os.path.basename(cell.figure) if cell.figure else None,
        "parameter_names": names,
        "true_theta": true_theta,
        "diagnostics": M.feature_diagnostics(np.asarray(env.feature_matrix)),
        "roster": [{"name": e.name, "family": e.family, "max_reps": e.max_reps}
                   for e in cell.roster],
    }


def run_experiment(cells: tuple[Cell, ...], *, title: str, diagnoses: dict,
                   excluded: list[dict], extra_meta: dict | None = None,
                   only_estimator: str | None = None, verbose: bool = False,
                   checkpoint: str | None = None) -> dict:
    """Run every cell serially and return the raw-facts dict.

    When ``checkpoint`` is set, every completed record is appended to that
    JSONL file as it finishes, and records already present there (matched on
    estimator/cell/rep) are reused instead of re-run — so a killed run loses
    at most the fit in flight.
    """
    done: dict[tuple, dict] = {}
    if checkpoint and os.path.exists(checkpoint):
        with open(checkpoint) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done[(r["estimator"], r["cell"], r["rep"])] = r

    records, cell_metas, snippets = [], [], {}
    for cell in cells:
        env = cell.env_factory()
        cell_metas.append(_cell_meta(cell, env))
        oracle_policy, oracle_value = _oracle(env)
        if cell.figure is not None:
            from validation.benchmark.figures import dgp_figure

            fig_panel = simulate_panel(env, n_individuals=cell.n_individuals,
                                       n_periods=cell.n_periods,
                                       seed=cell.seed + 1000)
            os.makedirs(os.path.dirname(cell.figure), exist_ok=True)
            dgp_figure(env, fig_panel, oracle_value, cell.figure)
        for entry in cell.roster:
            snippets.setdefault(entry.name, inspect.getsource(entry.run))
        for rep in range(cell.n_replications):
            seed = cell.seed + 1000 + rep
            panel = simulate_panel(env, n_individuals=cell.n_individuals,
                                   n_periods=cell.n_periods, seed=seed)
            for entry in cell.roster:
                if only_estimator is not None and entry.name != only_estimator:
                    continue
                if entry.max_reps is not None and rep >= entry.max_reps:
                    continue
                key = (entry.name, cell.cell_id, rep)
                if key in done:
                    records.append(done[key])
                    if verbose:
                        print(f"  [{cell.cell_id}] rep {rep} {entry.name:14s} "
                              f"(from checkpoint)", flush=True)
                    continue
                rec = _run_one(env, panel, entry, cell.cell_id, rep,
                               oracle_policy, oracle_value,
                               timeout=entry.timeout or cell.fit_timeout)
                records.append(rec)
                if checkpoint:
                    with open(checkpoint, "a") as f:
                        f.write(json.dumps(rec) + "\n")
                if verbose:
                    status = rec["error"] or f"tv={rec['policy_tv']}"
                    print(f"  [{cell.cell_id}] rep {rep} {entry.name:14s} "
                          f"{rec['runtime']:.1f}s {status}", flush=True)

    meta = {
        "title": title,
        "date": _dt.date.today().isoformat(),
        "package_version": _package_version(),
        "cells": cell_metas,
        "oracle": "true-parameter policy/value via SoftBellmanOperator + value_iteration",
        "determinism": (
            "structural estimators are deterministic given the seeds; neural/IRL "
            "estimators may have nondeterministic training internals, so their "
            "behavioral numbers can vary slightly across runs"
        ),
        "excluded": excluded,
        "regret": (
            "Counterfactual regret follows the package Type A (payoff shift), "
            "Type B (transition change), Type C (action penalty) taxonomy; "
            "regret = initial_distribution . (oracle_value - estimated_value), "
            "lower is better. Estimators with a recovered reward re-solve it "
            "under each intervention (transfer); estimators without one keep "
            "their fixed policy (cannot adapt)."
        ),
        "snippets": snippets,
        "diagnoses": diagnoses,
        "honesty": (
            "Every number in the rendered table is recomputed from the records "
            "below. Crashes carry the verbatim exception. true_theta per cell is "
            "included so bias can be recomputed independently."
        ),
    }
    if extra_meta:
        meta.update(extra_meta)
    return {"meta": meta, "records": records}


def merge_estimator(old: dict, new: dict, estimator: str) -> dict:
    """Replace one estimator's records (and snippet/diagnosis) in an existing run.

    Used by ``--only-estimator``: the diagnose -> fix -> retry loop re-runs a
    single estimator without repeating the whole experiment. Cell metas and the
    other estimators' records are kept verbatim from the old JSON.
    """
    fresh = [r for r in new["records"] if r["estimator"] == estimator]
    if not fresh:
        raise ValueError(f"--only-estimator produced no records for '{estimator}'")
    fresh_cells = {r["cell"] for r in fresh}
    kept = [r for r in old["records"]
            if not (r["estimator"] == estimator and r["cell"] in fresh_cells)]
    merged = dict(old)
    merged["records"] = kept + fresh
    merged["meta"] = dict(old["meta"])
    merged["meta"]["snippets"] = {**old["meta"]["snippets"],
                                  estimator: new["meta"]["snippets"][estimator]}
    merged["meta"]["diagnoses"] = {**old["meta"]["diagnoses"],
                                   estimator: new["meta"]["diagnoses"].get(estimator, "")}
    merged["meta"]["date"] = new["meta"]["date"]
    return merged


# ---------------------------------------------------------------------------
# Render: pure functions of the raw records
# ---------------------------------------------------------------------------


def _fmt(x, nd=4):
    return "-" if x is None else f"{x:.{nd}f}"


def _cell_records(data: dict, cell_id: str) -> dict[str, list]:
    by_est: dict[str, list] = {}
    for r in data["records"]:
        if r["cell"] == cell_id:
            by_est.setdefault(r["estimator"], []).append(r)
    return by_est


def _agg_regret(recs, key):
    vals = [r["regret"][key] for r in recs
            if r["error"] is None and r.get("regret") is not None]
    return float(np.mean(vals)) if vals else None


def _results_table(cell_meta: dict, by_est: dict[str, list]) -> list[str]:
    """The per-cell results table (one row per roster estimator)."""
    true_theta = (np.asarray(cell_meta["true_theta"], dtype=np.float64)
                  if cell_meta["true_theta"] is not None else None)
    L = []
    L.append("| Estimator | Family | Ran | Conv | Recovered params | Param RMSE | "
             "Policy TV | Regret base | Regret A | Regret B | Regret C | Time (s) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for spec in cell_meta["roster"]:
        name, family = spec["name"], spec["family"]
        recs = by_est.get(name, [])
        if not recs:
            continue
        ok = [r for r in recs if r["error"] is None]
        ran = f"{len(ok)}/{len(recs)}"
        conv = f"{sum(1 for r in ok if r['converged'])}/{len(ok)}" if ok else "-"
        plists = [r["params"] for r in ok if r["params"] is not None]
        if plists and len({len(p) for p in plists}) == 1:
            k = len(plists[0])
            if true_theta is not None and k != true_theta.shape[0]:
                # Not parameters in the data-generating gauge (e.g. a tabular
                # reward or a choice-probability table); printing the raw
                # vector would invite a meaningless comparison to theta.
                params_s = f"not in theta gauge ({k} values)"
            else:
                mean_p = np.mean(np.asarray(plists, dtype=np.float64), axis=0)
                params_s = "[" + ", ".join(f"{v:.3f}" for v in mean_p) + "]"
        else:
            params_s = "-"
        prmse = "-"
        if family == "structural" and true_theta is not None:
            rmses = [float(np.sqrt(np.mean((np.asarray(r["params"]) - true_theta) ** 2)))
                     for r in ok
                     if r["params"] is not None and len(r["params"]) == true_theta.shape[0]]
            if rmses:
                prmse = f"{np.mean(rmses):.4f}"
        tvs = [r["policy_tv"] for r in ok if r["policy_tv"] is not None]
        tv = _fmt(float(np.mean(tvs)) if tvs else None)
        rts = [r["runtime"] for r in ok if r["runtime"] is not None]
        rt = "-" if not rts else f"{np.mean(rts):.1f}"
        crashed = [r for r in recs if r["error"] is not None]
        note = f" (crashed {len(crashed)}/{len(recs)})" if crashed else ""
        L.append(f"| {name}{note} | {family} | {ran} | {conv} | {params_s} | {prmse} | "
                 f"{tv} | {_fmt(_agg_regret(recs, 'baseline'))} | "
                 f"{_fmt(_agg_regret(recs, 'type_a'))} | {_fmt(_agg_regret(recs, 'type_b'))} | "
                 f"{_fmt(_agg_regret(recs, 'type_c'))} | {rt} |")
    L.append("")
    return L


def _param_block(cell_meta: dict, by_est: dict[str, list]) -> list[str]:
    """Bias / empirical SE / RMSE / coverage / SE-availability per parameter.

    Structural family only, recomputed from raw records via
    ``metrics.parameter_summary``. The SE-availability rate is always shown so
    an estimator that rarely delivers usable standard errors is visibly flagged
    rather than hiding behind a blank coverage column.
    """
    true_theta = np.asarray(cell_meta["true_theta"], dtype=np.float64)
    names = cell_meta["parameter_names"]
    L = []
    L.append("| Estimator | Parameter | True | Mean est | Bias | Emp. SE | RMSE | "
             "95% coverage | SE avail |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for spec in cell_meta["roster"]:
        if spec["family"] != "structural":
            continue
        recs = by_est.get(spec["name"], [])
        ok = [r for r in recs if r["error"] is None and r["params"] is not None
              and len(r["params"]) == true_theta.shape[0]]
        if len(ok) < 2:
            continue
        est = np.asarray([r["params"] for r in ok], dtype=np.float64)
        ses = np.asarray([r["standard_errors"]
                          if (r["standard_errors"] is not None
                              and len(r["standard_errors"]) == true_theta.shape[0])
                          else [np.nan] * true_theta.shape[0]
                          for r in ok], dtype=np.float64)
        s = M.parameter_summary(est, ses, true_theta, names)
        se_rate = f"{s['se_available_rate']:.0%} ({len(ok)} reps)"
        for k, pname in enumerate(names):
            cov = s["coverage_95"][k]
            cov_s = "-" if cov is None else f"{cov:.2f} +/- {s['coverage_95_mc_se'][k]:.2f}"
            L.append(f"| {spec['name']} | {pname} | {true_theta[k]:.3f} | "
                     f"{s['mean_estimate'][k]:.3f} | {s['bias'][k]:+.3f} | "
                     f"{s['empirical_se'][k]:.3f} | {s['rmse'][k]:.3f} | {cov_s} | {se_rate} |")
    L.append("")
    L.append("Coverage is the share of replications whose normal-approximation 95% "
             "interval contains the truth, with its Monte Carlo standard error; it is "
             "computed only where every replication produced a finite standard error. "
             "SE avail is the share of replications with finite standard errors for "
             "all parameters.")
    L.append("")
    return L


_TABLE_NOTE = (
    "Param RMSE is the structural family only (recovered theta vs true, same "
    "gauge). Recovered params are shown only when the estimator's parameter "
    "vector lives in the data-generating gauge; a tabular reward or a "
    "choice-probability table is labeled rather than printed, because "
    "comparing it to theta entry by entry would be meaningless. Policy TV is "
    "total-variation distance from the true-parameter policy. Conv is the "
    "converged flag reported by the estimator itself; a conservative flag can "
    "read False while the recovered policy is accurate, so read it next to "
    "Policy TV, not alone. Regret is welfare loss (lower is better): `base` "
    "is the observed world; `A` payoff shift, `B` transition change, `C` "
    "action penalty. Estimators that recovered a reward in the linear feature "
    "gauge re-solve it under each intervention and adapt. Large Type C regret "
    "has two distinct routes: estimators with no reward in that gauge keep "
    "their frozen policy and cannot adapt, and an estimator that transfers a "
    "badly scaled reward adapts to the wrong world."
)


def render_page(data: dict, narrative: dict) -> str:
    """Render a docs sub-page from the raw records. Pure function of its inputs."""
    meta = data["meta"]
    single = len(meta["cells"]) == 1
    L = [f"# {narrative['title']}\n"]
    L.append(narrative["intro"].strip() + "\n")

    for cm in meta["cells"]:
        by_est = _cell_records(data, cm["cell_id"])
        if not single:
            L.append(f"## {cm['label']}\n")
        tt = ("" if cm["true_theta"] is None
              else f"True theta `{[round(x, 4) for x in cm['true_theta']]}`. ")
        d = cm["diagnostics"]
        L.append(
            f"{cm['description']} {cm['n_individuals']} x {cm['n_periods']} "
            f"observations, {cm['n_replications']} replications, seed {cm['seed']}. "
            f"{tt}Design rank {d['feature_rank']}/{d['num_features']}, condition "
            f"number {d['condition_number']:.2e}. Generated {meta['date']} with "
            f"econirl {meta['package_version']}.\n"
        )
        before = narrative.get("cells", {}).get(cm["cell_id"], {}).get("before")
        if before:
            L.append(before.strip() + "\n")
        if cm.get("figure"):
            L.append(f"![Simulated trajectories and the optimal value function "
                     f"for {cm['label']}](../_static/simulation_studies/{cm['figure']})\n")
        L.append("### Results\n" if not single else "## Results\n")
        L.extend(_results_table(cm, by_est))
        L.append(_TABLE_NOTE + "\n")
        if cm["param_block"] and cm["true_theta"] is not None:
            L.append("### Parameter recovery\n" if not single
                     else "## Parameter recovery\n")
            L.extend(_param_block(cm, by_est))
        crashed = [(r["estimator"], r["error"]) for r in data["records"]
                   if r["cell"] == cm["cell_id"] and r["error"] is not None]
        if crashed:
            L.append("### Failures\n" if not single else "## Failures\n")
            seen = set()
            for est, err in crashed:
                if (est, err) in seen:
                    continue
                seen.add((est, err))
                L.append(f"- {est}: `{err}`")
            L.append("")
        after = narrative.get("cells", {}).get(cm["cell_id"], {}).get("after")
        if after:
            L.append(after.strip() + "\n")

    probes = meta.get("feasibility_probes")
    if probes:
        L.append("## Feasibility probes\n")
        L.append(probes["description"].strip() + "\n")
        L.append("| Estimator | States | Outcome | Time (s) | Detail |")
        L.append("|---|---|---|---|---|")
        for p in probes["probes"]:
            sec = "-" if p.get("seconds") is None else f"{p['seconds']:.1f}"
            detail = p.get("error") or ""
            L.append(f"| {p['estimator']} | {p['n_states']} | {p['outcome']} | "
                     f"{sec} | {detail} |")
        L.append("")
        L.append(f"Each probe is a single fit in its own subprocess with a hard "
                 f"{probes['timeout_seconds']:.0f}-second budget; `timeout` means the "
                 "fit was killed at the budget, with no number invented for it.\n")

    extra = narrative.get("extra_sections")
    if extra:
        L.append(extra.strip() + "\n")

    L.append("## Code used\n")
    L.append("The exact construction for each estimator (configs are modest "
             "defaults with documented fixes, not tuned per cell):\n")
    seen = set()
    diagnoses = meta.get("diagnoses", {})
    for cm in meta["cells"]:
        for spec in cm["roster"]:
            name = spec["name"]
            if name in seen or name not in meta["snippets"]:
                continue
            seen.add(name)
            L.append(f"### {name}\n")
            if diagnoses.get(name):
                L.append(f"{diagnoses[name]}\n")
            L.append("```python")
            L.append(meta["snippets"][name].rstrip())
            L.append("```\n")

    L.append("## Reproduce\n")
    L.append("```bash")
    script = narrative["script"]
    L.append(f"python {script}                 # run + write JSON")
    L.append(f"python {script} --page          # regenerate this page")
    L.append(f"python {script} --verify        # re-derive the table from JSON")
    L.append("```\n")
    L.append(f"Raw facts: `{narrative['results_rel']}`. {meta['regret']}\n")
    if meta["excluded"]:
        L.append("Excluded from this run: " +
                 "; ".join(f"{e['name']} ({e['reason']})" for e in meta["excluded"]) + ".")
    return "\n".join(L)


def render_console(data: dict) -> str:
    """Compact console table per cell, recomputed from the raw records."""
    meta = data["meta"]
    lines = [f"{meta['title']}  ({meta['date']}, econirl {meta['package_version']})"]
    for cm in meta["cells"]:
        by_est = _cell_records(data, cm["cell_id"])
        true_theta = (np.asarray(cm["true_theta"], dtype=np.float64)
                      if cm["true_theta"] is not None else None)
        lines.append("")
        lines.append(f"[{cm['cell_id']}] {cm['label']}: {cm['num_states']} states, "
                     f"{cm['num_actions']} actions, beta {cm['discount_factor']}, "
                     f"{cm['n_replications']} reps")
        header = (f"{'Estimator':<14} {'Family':<11} {'Ran':>5} {'ParamRMSE':>10} "
                  f"{'PolicyTV':>9} {'Time(s)':>8}  Notes")
        lines.append(header)
        lines.append("-" * len(header))
        for spec in cm["roster"]:
            recs = by_est.get(spec["name"], [])
            if not recs:
                continue
            ok = [r for r in recs if r["error"] is None]
            crashed = [r for r in recs if r["error"] is not None]
            ran = f"{len(ok)}/{len(recs)}"
            prmse = "n/a"
            if spec["family"] == "structural" and ok and true_theta is not None:
                rmses = [float(np.sqrt(np.mean((np.asarray(r["params"]) - true_theta) ** 2)))
                         for r in ok
                         if r["params"] is not None and len(r["params"]) == true_theta.shape[0]]
                if rmses:
                    prmse = f"{np.mean(rmses):.4f}"
            tvs = [r["policy_tv"] for r in ok if r["policy_tv"] is not None]
            tv = f"{np.mean(tvs):.4f}" if tvs else "n/a"
            rts = [r["runtime"] for r in ok if r["runtime"] is not None]
            rt = f"{np.mean(rts):.1f}" if rts else "n/a"
            note = ""
            if crashed:
                note = f"CRASHED {len(crashed)}/{len(recs)}: {crashed[0]['error'][:60]}"
            lines.append(f"{spec['name']:<14} {spec['family']:<11} {ran:>5} "
                         f"{prmse:>10} {tv:>9} {rt:>8}  {note}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI shared by the per-page scripts
# ---------------------------------------------------------------------------


def main_cli(*, cells: tuple[Cell, ...], title: str, narrative: dict,
             diagnoses: dict, excluded: list[dict], results_json: str,
             page_path: str, extra_meta: dict | None = None) -> None:
    parser = argparse.ArgumentParser(description=title)
    parser.add_argument("--replications", type=int, default=None,
                        help="Override every cell's replication count.")
    parser.add_argument("--only-estimator", type=str, default=None,
                        help="Re-run a single estimator and merge into the saved JSON.")
    parser.add_argument("--only-cell", type=str, default=None,
                        help="Restrict the run to one cell id.")
    parser.add_argument("--verify", action="store_true",
                        help="Re-render the console table from the saved JSON only.")
    parser.add_argument("--page", action="store_true",
                        help="Write the docs sub-page from the saved JSON only.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verify or args.page:
        if not os.path.exists(results_json):
            sys.exit(f"No JSON at {results_json}. Run without --verify/--page first.")
        data = json.load(open(results_json))
        if args.page:
            # Diagnoses and the excluded list are interpretive prose, not
            # measured facts: render them from the current script so a
            # rewritten diagnosis reaches the page without re-running. Code
            # snippets are NOT refreshed - they document the exact code that
            # produced the stored records and only change through a re-run or
            # an --only-estimator merge.
            data["meta"]["diagnoses"] = diagnoses
            data["meta"]["excluded"] = excluded
            # DGP figures are presentation artifacts that regenerate
            # deterministically from the seeds, so a missing figure never
            # requires re-running the benchmark, and a figure added to the
            # cell config after a run still renders on the page.
            fig_by_cell = {c.cell_id: c.figure for c in cells if c.figure}
            for cm in data["meta"]["cells"]:
                if cm["cell_id"] in fig_by_cell:
                    cm["figure"] = os.path.basename(fig_by_cell[cm["cell_id"]])
            for c in cells:
                if c.figure is not None and not os.path.exists(c.figure):
                    from validation.benchmark.figures import dgp_figure

                    env = c.env_factory()
                    _, oracle_value = _oracle(env)
                    fig_panel = simulate_panel(env, n_individuals=c.n_individuals,
                                               n_periods=c.n_periods,
                                               seed=c.seed + 1000)
                    os.makedirs(os.path.dirname(c.figure), exist_ok=True)
                    dgp_figure(env, fig_panel, oracle_value, c.figure)
                    print(f"Wrote {c.figure}")
            with open(page_path, "w") as f:
                f.write(render_page(data, narrative))
            print(f"Wrote {page_path}")
        else:
            print(render_console(data))
            print(f"\n(verified: table re-derived purely from {results_json})")
        return

    run_cells = cells
    if args.only_cell is not None:
        run_cells = tuple(c for c in cells if c.cell_id == args.only_cell)
        if not run_cells:
            sys.exit(f"Unknown cell '{args.only_cell}'. "
                     f"Valid: {', '.join(c.cell_id for c in cells)}")
    if args.replications is not None:
        run_cells = tuple(dataclasses.replace(c, n_replications=args.replications)
                          for c in run_cells)

    ckpt = results_json + ".checkpoint.jsonl"
    os.makedirs(os.path.dirname(results_json), exist_ok=True)
    data = run_experiment(run_cells, title=title, diagnoses=diagnoses,
                          excluded=excluded, extra_meta=extra_meta,
                          only_estimator=args.only_estimator, verbose=args.verbose,
                          checkpoint=ckpt)

    if args.only_estimator is not None and os.path.exists(results_json):
        data = merge_estimator(json.load(open(results_json)), data, args.only_estimator)

    with open(results_json, "w") as f:
        json.dump(data, f, indent=2)
    if os.path.exists(ckpt):
        os.remove(ckpt)  # the run completed; the JSON is the record

    print(render_console(data))
    print(f"\nRaw facts: {results_json}")
    print(f"Page:      python {narrative['script']} --page")
    print(f"Verify:    python {narrative['script']} --verify")
