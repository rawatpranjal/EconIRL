# Other Estimators

## Important Links

- [Core Estimators](core.md)
- [Choosing and Comparing Estimators](../comparing_estimators.md)
- [API Reference](../api/index.rst)

This is the complete list of implemented estimators outside the Core roster.
A row means estimator code exists in this repository. It does not imply the same
validation or public-support level as a Core estimator.

Aliases are listed once. AIRL-Het resolves to AIRL2. NeuralGLADIUS resolves to
GLADIUS, which remains in Core.

| Estimator | Code home |
| --- | --- |
| [TD-CCP](tdccp.md) | `src/econirl/estimators/tdccp.py` |
| [AIRL2](airl2.md) | `src/econirl/estimators/airl2.py` |
| [NNES](nnes.md) | `src/econirl/estimators/nnes.py` |
| [MPEC](mpec.md) | `src/econirl/estimation/mpec.py` |
| Neural MPEC | `src/econirl/estimation/neural_mpec.py` |
| [UFXP](ufxp.md) | `src/econirl/estimators/ufxp.py` |
| Neural UFXP | `src/econirl/estimators/ufxp_neural.py` |
| SEES | `src/econirl/estimators/sees.py` |
| [RHIP](rhip.md) | `src/econirl/estimators/rhip.py` |
| [f-IRL](f_irl.md) | `src/econirl/estimation/f_irl.py` |
| [IQ-Learn](iq_learn.md) | `src/econirl/estimation/iq_learn.py` |
| MaxEnt IRL | `src/econirl/estimators/maxent_irl.py` |
| Deep MaxEnt IRL | `src/econirl/contrib/deep_maxent_irl.py` |
| Max Margin IRL | `src/econirl/estimators/max_margin_irl.py` |
| Max Margin Planning | `src/econirl/contrib/max_margin_planning.py` |
| GAIL | `src/econirl/contrib/gail.py` |
| GCL | `src/econirl/estimators/gcl.py` |
| Bayesian IRL | `src/econirl/contrib/bayesian_irl.py` |
| Behavioral Cloning | `src/econirl/estimation/behavioral_cloning.py` |

```{toctree}
:maxdepth: 1

tdccp
airl2
nnes
mpec
ufxp
rhip
f_irl
iq_learn
```
