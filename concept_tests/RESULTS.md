# Concept-test results

Run on 2026-07-16 with:

```bash
python athenak_dust_tests/concept_tests/test_local_be_pmbr_temporal_order.py

python athenak_dust_tests/concept_tests/test_yang_history_imex_temporal_order.py
```

Both programs completed successfully with exit code 0.  The tests freeze the
particle-mesh weights and densities, so the measured slopes isolate temporal
order at a fixed spatial discretization.

## Local-BE + gather + PMBR

The final entries in the refinement tables were:

| Measurement | Smallest `dt` | Error | Observed slope |
|---|---:|---:|---:|
| One-step error | 2.44140625e-4 | 8.88336340e-8 | 1.955656 |
| Fixed-time error, `t = 0.25` | 2.44140625e-4 | 6.12658531e-5 | 0.982745 |

The asymptotic behavior is therefore

```text
one-step error       ~ O(dt^2)
fixed-final-time     ~ O(dt^1)
```

This diagnoses a globally first-order method for overlapping particle clouds.

## Yang-history IMEX construction

### Scalar modal identity

Without the proposed implementation clamp, the history construction reproduces
the exact full-step scalar decay to floating-point accuracy.  The clamp changes
the finite-step result in the stiff regime:

| Stage decay `x` | Unclamped | Exact | Clamped |
|---:|---:|---:|---:|
| 1 | 5.56667905e-1 | 5.56667905e-1 | 5.56667905e-1 |
| 10 | 2.85733937e-3 | 2.85733937e-3 | 1.40090188e-4 |
| 100 | 3.62759050e-26 | 3.62759050e-26 | 1.14793600e-43 |

### Temporal-order checks

| Model | Expected diagnostic | Final observed slope |
|---|---|---:|
| True matrix exponential with noncommuting explicit operator | One-step slope approaches 3 | 2.995599 |
| One Yang sub-cloud map + parent collapse + PMBR | One-step slope approaches 2 | 1.998607 |
| Yang-history IMEX, pure drag, overlapping clouds | One-step slope approaches 2 | 1.996537 |
| Yang-history IMEX with noncommuting explicit operator | One-step slope approaches 2 | 1.995343 |
| Same overlapping-cloud model at fixed `t = 0.5` | Global slope approaches 1 | 0.996777 |

Thus the proposed coefficient algebra is second-order when its implicit map is a
true exponential semigroup.  After Yang sub-cloud evolution is collapsed to
parent particles and mirrored with PMBR, the one-step error is only
`O(dt^2)`, producing global first-order convergence at fixed spatial
discretization.

### Constant-force equilibrium

For

```text
y' = -y + 1,    y(0) = 1,
```

the exact solution remains at the equilibrium `y = 1`.  The proposed
exponential-drag/explicit-force construction produced:

| `dt` | Numerical `y` | Exact `y` |
|---:|---:|---:|
| 0.1 | 9.99689097e-1 | 1 |
| 1 | 8.67626948e-1 | 1 |
| 10 | 8.20152554e-4 | 1 |
| 100 | 6.38776768e-42 | 1 |

This shows that the tested construction is classically accurate as
`dt -> 0` for a true exponential operator, but it is not
asymptotic-preserving for a stiff drag/forcing equilibrium.
