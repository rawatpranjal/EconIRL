"""Static contract checks for the dedicated MCE-IRL workflow notebook."""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK = Path("examples/mce-irl/mce_irl_applied_workflow.ipynb")


def _source() -> str:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join("".join(cell["source"]) for cell in payload["cells"])


def test_mce_irl_notebook_has_stable_schema_and_unique_cell_ids() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    ids = [cell["id"] for cell in payload["cells"]]
    assert payload["nbformat"] == 4
    assert len(ids) == len(set(ids))
    assert all(ids)


def test_mce_irl_notebook_covers_complete_public_workflow() -> None:
    source = _source()
    for required in (
        "from econirl import DeterministicTransitions, MCEIRL",
        "module_outside_checkout",
        "Installed package import",
        "MCEIRLTask",
        "model.fit(",
        "model.summary()",
        "model.diagnostics_",
        "Held-out negative log likelihood",
        "model.counterfactual(",
        "pickle.dumps(model)",
        "restored.predict_proba",
    ):
        assert required in source


def test_mce_irl_notebook_has_no_checkout_import_override() -> None:
    source = _source()
    assert "sys.path" not in source
    assert "PYTHONPATH" not in source


def test_mce_irl_notebook_records_installed_package_execution() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    outputs = "\n".join(
        "".join(output.get("text", []))
        for cell in payload["cells"]
        for output in cell.get("outputs", [])
    )
    assert "Installed package import: True" in outputs
    assert all(
        cell["execution_count"] is not None
        for cell in payload["cells"]
        if cell["cell_type"] == "code"
    )
