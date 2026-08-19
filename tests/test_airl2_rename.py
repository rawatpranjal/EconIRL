"""Load-bearing compatibility tests for the AIRL-Het to AIRL2 rename."""

from __future__ import annotations

import base64
import importlib
import pickle

import pytest

_LEGACY_CONFIG_PICKLE = base64.b64decode(
    "gASVXgMAAAAAAACMJ2Vjb25pcmwuZXN0aW1hdGlvbi5hZHZlcnNhcmlhbC5haXJsX2hldJSM"
    "DUFJUkxIZXRDb25maWeUk5QpgZR9lCiMDG51bV9zZWdtZW50c5RLAowLZXhpdF9hY3Rpb26U"
    "SwKMD2Fic29yYmluZ19zdGF0ZZRLBYwLcmV3YXJkX3R5cGWUjAd0YWJ1bGFylIwJcmV3YXJk"
    "X2xylEc/hHrhR64Ue4wTZGlzY3JpbWluYXRvcl9zdGVwc5RLBYwQZ2VuZXJhdG9yX3NvbHZl"
    "cpSMBmh5YnJpZJSMDWdlbmVyYXRvcl90b2yURz5FeY7iMIw6jBJnZW5lcmF0b3JfbWF4X2l0"
    "ZXKUTYgTjA9tYXhfYWlybF9yb3VuZHOUS2SMFGFpcmxfY29udmVyZ2VuY2VfdG9slEc/Gjbi"
    "6xxDLYwRbWF4X2VtX2l0ZXJhdGlvbnOUSzKMEmVtX2NvbnZlcmdlbmNlX3RvbJRHP1BiTdLx"
    "qfyMEmNvbnNpc3RlbmN5X3dlaWdodJRHP7mZmZmZmZqMD3ByaW9yX3Ntb290aGluZ5RHP4R6"
    "4UeuFHuMCXByaW9yX21pbpRHAAAAAAAAAACMDXByaW9yX2RhbXBpbmeURwAAAAAAAAAAjBNy"
    "ZXdhcmRfd2VpZ2h0X2RlY2F5lEcAAAAAAAAAAIwQbm9ybWFsaXplX3Jld2FyZJSJjBV1bml0"
    "X25vcm1hbGl6ZV9yZXdhcmSUiYwSZ3JhZGllbnRfY2xpcF9ub3JtlEcAAAAAAAAAAIwSYW50"
    "aXN5bW1ldHJpY19pbml0lImMDmluaXRpYWxpemF0aW9ulIwGcmFuZG9tlIwYaW5pdGlhbGl6"
    "YXRpb25fc21vb3RoaW5nlEc/8AAAAAAAAIwZaW5pdGlhbGl6YXRpb25fbDJfcGVuYWx0eZRH"
    "AAAAAAAAAACMC3VzZV9zaGFwaW5nlIiMDHNoYXBpbmdfY29lZpROjBJzaGFwaW5nX2wyX3Bl"
    "bmFsdHmURz5FeY7iMIw6jBBnZW5lcmF0b3JfcmV3YXJklIwJcmVjb3ZlcmVklIwQcG9saWN5"
    "X3N0ZXBfc2l6ZZRHP/AAAAAAAACMD21pbl9haXJsX3JvdW5kc5RLAYwHdmVyYm9zZZSJjARz"
    "ZWVklEsqdWIu"
)


def test_airl2_is_canonical_and_legacy_construction_warns() -> None:
    """Every canonical surface agrees; the old spelling remains loud."""
    canonical = importlib.import_module("econirl.estimation.adversarial.airl2")
    adversarial = importlib.import_module("econirl.estimation.adversarial")
    estimation = importlib.import_module("econirl.estimation")
    estimators = importlib.import_module("econirl.estimators")
    package = importlib.import_module("econirl")

    assert canonical.AIRL2Estimator.__name__ == "AIRL2Estimator"
    assert canonical.AIRL2Config.__name__ == "AIRL2Config"
    assert adversarial.AIRL2Estimator is canonical.AIRL2Estimator
    assert estimation.AIRL2 is canonical.AIRL2Estimator
    assert package.AIRL2 is estimators.AIRL2
    assert package.AIRL2.__name__ == "AIRL2"
    model = package.AIRL2(
        n_states=6,
        n_actions=3,
        exit_action=2,
        absorbing_state=5,
        compute_se=True,
        n_bootstrap=2,
    )
    assert model.num_segments == 2
    assert model.compute_se is True

    legacy = importlib.import_module("econirl.estimation.adversarial.airl_het")
    with pytest.warns(DeprecationWarning, match="AIRLHetConfig.*AIRL2Config"):
        config = legacy.AIRLHetConfig(exit_action=2, absorbing_state=5)
    with pytest.warns(DeprecationWarning, match="AIRLHetEstimator.*AIRL2Estimator"):
        estimator = legacy.AIRLHetEstimator(config)

    assert type(config) is canonical.AIRL2Config
    assert type(estimator) is canonical.AIRL2Estimator


def test_legacy_airl_het_pickle_loads_as_canonical_airl2() -> None:
    """A real pre-rename module path must deserialize into the new class."""
    canonical = importlib.import_module("econirl.estimation.adversarial.airl2")

    with pytest.warns(DeprecationWarning, match="AIRLHetConfig.*AIRL2Config"):
        restored = pickle.loads(_LEGACY_CONFIG_PICKLE)

    assert type(restored) is canonical.AIRL2Config
    assert restored.exit_action == 2
    assert restored.absorbing_state == 5
    rewritten = pickle.dumps(restored, protocol=4)
    assert b"econirl.estimation.adversarial.airl2" in rewritten
    assert b"AIRL2Config" in rewritten
