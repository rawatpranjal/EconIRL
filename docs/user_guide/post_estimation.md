# Post-Estimation Diagnostics

After a fit converges, confirm the standard errors, inspect the implied behavior,
and run the counterfactuals the estimator supports.

## Standard errors

The structural estimators report asymptotic standard errors. The `se_method`
argument selects the covariance estimator.

| `se_method` | Covariance |
| --- | --- |
| `asymptotic` | Inverse observed-information matrix. |
| `robust` | Sandwich form, the default. |
| `clustered` | Sandwich clustered by individual for panel dependence. |
| `bootstrap` | Pairs-cluster resampling over individuals. |

```python
model = NFXP(n_states=90, discount=0.9999, se_method="clustered")
model.fit(df, state="mileage_bin", action="replaced", id="bus_id")
print(model.se_)
```

## Inspect the fit

Read the fitted policy, value function, and the implied choice probabilities, and
compare them against the observed action frequencies.

```python
print(model.summary())
proba = model.predict_proba(states)
```

## Counterfactuals

Change a structural primitive and read the new policy or value. Counterfactuals
are only as credible as the identification behind the parameters, so confirm the
pre-estimation checks first.

```python
cf = model.counterfactual(RC=4.0)
print(cf.policy[50, 1])
```

For the standard against which a replication is judged, see the
[replications ledger](../replications.md).
