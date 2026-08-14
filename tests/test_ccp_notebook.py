"""Static contract checks for the dedicated CCP workflow notebook."""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK = Path("examples/ccp/ccp_applied_workflow.ipynb")


def _source() -> str:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join("".join(cell["source"]) for cell in payload["cells"])


def test_ccp_notebook_has_stable_schema_and_unique_cell_ids() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    ids = [cell["id"] for cell in payload["cells"]]

    assert payload["nbformat"] == 4
    assert len(ids) == len(set(ids))
    assert all(ids)


def test_ccp_notebook_covers_complete_public_workflow() -> None:
    source = _source()

    for required in (
        "from econirl import CCP",
        "RewardSpec",
        "simulate_panel",
        "num_policy_iterations=3",
        "model.fit(",
        "model.summary()",
        "model.diagnostics_",
        "model.npl_converged_",
        "Held-out negative log likelihood",
        "model.counterfactual(",
        "pickle.dumps(model)",
        "restored.predict_proba",
    ):
        assert required in source


def test_ccp_notebook_has_no_checkout_import_override() -> None:
    source = _source()

    assert "sys.path" not in source
    assert "PYTHONPATH" not in source
