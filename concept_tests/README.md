# Dust particle-mesh concept tests

These small, stand-alone Python programs isolate the time discretization of the
dust particle-mesh algorithms from the rest of AthenaK.  They are not AthenaK
regression tests: positions, interpolation weights, gas densities, and particle
masses are frozen so that only temporal order is measured.

The numerical results from the current versions are summarized in
[`RESULTS.md`](RESULTS.md).

The tests compare a numerical particle-mesh update with the matrix exponential
of the corresponding infinitesimal PMBR system.  For a method of global order
`p`, the expected errors are:

- one-step error: `O(dt**(p + 1))`;
- error at a fixed final time: `O(dt**p)`.

Consequently, a one-step slope approaching 2 and a fixed-time slope approaching
1 diagnose a first-order method.

## Requirements

The scripts require NumPy and SciPy. From the directory containing both repositories,
run them with a Python environment that provides those packages:

```bash
python athenak_dust_tests/concept_tests/test_local_be_pmbr_temporal_order.py
python athenak_dust_tests/concept_tests/test_yang_history_imex_temporal_order.py
```

Each script exits nonzero if its final observed slopes do not match the expected
behavior described below.

## `test_local_be_pmbr_temporal_order.py`

This reproduces the original local-BE design:

1. deposit `m W c` and `m W c v`, where `c = a alpha/(1 + a alpha)`;
2. solve for a provisional gas velocity independently in every cell;
3. gather that velocity to the parent particles;
4. apply the particle kicks and mirrored PMBR gas update;
5. use this map in the two implicit stages of `imex2+`.

For overlapping particle clouds, the local cell division omits the nonlocal
`G^T M C G` coupling of the exact backward-Euler stage.  Expected result:

- one-step slope approaches 2;
- fixed-time slope approaches 1.

## `test_yang_history_imex_temporal_order.py`

This checks the mode-dependent history weight proposed in
`IMEX_coeff_for_Yang16_single_species.md` at several levels:

1. A scalar decay mode verifies the unclamped modal identity exactly and shows
   how the proposed clamp changes the finite-step solution.
2. A genuine matrix exponential with a noncommuting explicit operator verifies
   that the coefficient algebra has one-step error `O(dt**3)`.
3. A two-cell, two-particle overlapping-cloud model implements the Yang
   single-stopping-time sub-cloud solve, parent-particle collapse, and PMBR.
   It shows that the same coefficients have one-step slope 2 and fixed-time
   slope 1 once TSC-style overlap is present.
4. A scalar constant-force equilibrium demonstrates that the exponential drag
   stages plus explicit forcing are not asymptotic-preserving at large `dt/ts`.

The distinction between items 2 and 3 is the central result: the history
coefficient is second-order for a true exponential semigroup, but the Yang
sub-cloud map followed by parent collapse and PMBR is not that semigroup.
