"""Measure the temporal order of the local-BE + gather + PMBR design.

This is a deliberately tiny linear model with two gas cells and two Lagrangian
particles.  Each row of ``weights`` contains the interpolation weights of one
particle over the gas cells.  Both rows have two nonzero entries, so the test
contains the cross-cell coupling that is absent for NGP particles.

The reference semi-discrete PMBR equations, with frozen weights, are

    dv_p/dt = alpha_p * (sum_k W[p,k] u_k - v_p),
    rho_k du_k/dt = -sum_p m_p W[p,k] dv_p/dt.

Writing y = [u_0, ..., u_Ncell-1, v_0, ..., v_Npart-1], these equations are the
constant linear ODE ``dy/dt = D y``.  ``scipy.linalg.expm(dt * D)`` therefore
provides an unambiguous temporal reference at fixed spatial discretization.

The numerical stage reproduces the original AthenaK design:

* solve each cell locally using deposits proportional to
  ``c_p = a alpha_p/(1 + a alpha_p)``;
* gather the provisional gas field to the parent particles;
* mirror the actual parent-particle kicks back to the gas with PMBR.

The two stages are then combined with the pure-drag ``imex2+`` coefficients.
For an exact global backward-Euler stage this composition would be second
order.  With overlapping clouds, however, the local solve omits cross-cell
terms.  The expected diagnostic is:

* one-step error proportional to dt**2;
* fixed-final-time error proportional to dt**1.

Run from ``athenak_cc/`` with

    python athenak_dust_tests/concept_tests/test_local_be_pmbr_temporal_order.py

The program exits nonzero if the final measured slopes are not close to 2 and
1, respectively.
"""

import numpy as np
from scipy.linalg import expm


GAMMA = 1.0 + 1.0 / np.sqrt(2.0)


def build_pmbr_drag_matrix(weights, particle_masses, gas_densities, drag_rates):
    """Return the exact infinitesimal PMBR operator D for frozen PM weights."""
    weights = np.asarray(weights, dtype=float)
    particle_masses = np.asarray(particle_masses, dtype=float)
    gas_densities = np.asarray(gas_densities, dtype=float)
    drag_rates = np.asarray(drag_rates, dtype=float)
    nparticle, ncell = weights.shape
    matrix = np.zeros((ncell + nparticle, ncell + nparticle))

    for p in range(nparticle):
        # Particle row: dv_p/dt = alpha_p * (G_p u - v_p).
        matrix[ncell + p, :ncell] += drag_rates[p] * weights[p]
        matrix[ncell + p, ncell + p] -= drag_rates[p]

        # Gas rows are the exact mirrored particle force.  Using the already
        # assembled particle row makes the action-reaction structure explicit.
        for k in range(ncell):
            factor = -particle_masses[p] * weights[p, k] / gas_densities[k]
            matrix[k] += factor * matrix[ncell + p]

    return matrix


def local_be_pmbr_stage(state, stage_dt, weights, particle_masses, gas_densities, drag_rates):
    """Apply one local cell solve, parent gather, and mirrored PMBR update."""
    weights = np.asarray(weights, dtype=float)
    particle_masses = np.asarray(particle_masses, dtype=float)
    gas_densities = np.asarray(gas_densities, dtype=float)
    drag_rates = np.asarray(drag_rates, dtype=float)
    nparticle, ncell = weights.shape
    gas_velocity = state[:ncell].copy()
    particle_velocity = state[ncell:].copy()

    # This is c_j in the AthenaK kernel.  The deposits below use one factor W,
    # whereas the exact eliminated global system contains W^T M C W.
    coupling = stage_dt * drag_rates / (1.0 + stage_dt * drag_rates)
    deposited_mass = np.sum(particle_masses[:, None] * weights * coupling[:, None], axis=0)
    deposited_momentum = np.sum(particle_masses[:, None] * weights * (coupling * particle_velocity)[:, None], axis=0)
    provisional_gas = (gas_densities * gas_velocity + deposited_momentum) / (gas_densities + deposited_mass)

    # The parent particle receives one gathered kick.  PMBR mirrors that same
    # kick, which preserves total momentum but does not make the local solve the
    # exact resolvent of the globally coupled PMBR equations.
    particle_kick = coupling * (weights @ provisional_gas - particle_velocity)
    particle_new = particle_velocity + particle_kick
    gas_new = gas_velocity - np.sum(particle_masses[:, None] * weights * particle_kick[:, None], axis=0) / gas_densities
    return np.concatenate([gas_new, particle_new])


def imex2plus_local_be_step(state, dt, weights, particle_masses, gas_densities, drag_rates):
    """Apply the two working pure-drag stages of imex2+."""
    stage_dt = GAMMA * dt
    stage1 = local_be_pmbr_stage(state, stage_dt, weights, particle_masses, gas_densities, drag_rates)
    stage1_rate = (stage1 - state) / stage_dt

    # The low-storage g0/a_twid formula reduces to this expression for a pure
    # drag problem: q2 = y_n + (1-gamma) dt R(stage1).
    stage2_rhs = state + (1.0 - GAMMA) * dt * stage1_rate
    return local_be_pmbr_stage(stage2_rhs, stage_dt, weights, particle_masses, gas_densities, drag_rates)


def observed_order(previous_error, error):
    """Return the slope for a factor-of-two reduction in dt."""
    return np.nan if previous_error is None else np.log2(previous_error / error)


def main():
    # Non-symmetric weights avoid cancellations associated with a uniform
    # particle lattice.  Different stopping times show that the failure is not
    # tied to a single decay eigenmode.
    weights = np.array([[0.75, 0.25], [0.20, 0.80]])
    particle_masses = np.array([0.7, 0.4])
    gas_densities = np.array([1.0, 1.3])
    drag_rates = np.array([4.0, 0.8])
    initial_state = np.array([0.4, -0.3, 1.2, -0.8])
    drag_matrix = build_pmbr_drag_matrix(weights, particle_masses, gas_densities, drag_rates)

    print("one-step error: a first-order method approaches slope 2")
    print("dt                 error              slope")
    previous_error = None
    final_one_step_order = np.nan
    for dt in [2.0**(-k) for k in range(4, 13)]:
        numerical = imex2plus_local_be_step(initial_state, dt, weights, particle_masses, gas_densities, drag_rates)
        exact = expm(dt * drag_matrix) @ initial_state
        error = np.linalg.norm(numerical - exact, ord=np.inf)
        final_one_step_order = observed_order(previous_error, error)
        print(f"{dt:18.9e} {error:18.9e} {final_one_step_order:10.6f}")
        previous_error = error

    print("\nfixed-final-time error: a first-order method approaches slope 1")
    print("dt                 error              slope")
    final_time = 0.25
    previous_error = None
    final_global_order = np.nan
    exact_final = expm(final_time * drag_matrix) @ initial_state
    for nstep in [8, 16, 32, 64, 128, 256, 512, 1024]:
        dt = final_time / nstep
        numerical = initial_state.copy()
        for _ in range(nstep):
            numerical = imex2plus_local_be_step(numerical, dt, weights, particle_masses, gas_densities, drag_rates)
        error = np.linalg.norm(numerical - exact_final, ord=np.inf)
        final_global_order = observed_order(previous_error, error)
        print(f"{dt:18.9e} {error:18.9e} {final_global_order:10.6f}")
        previous_error = error

    if not (1.9 < final_one_step_order < 2.1 and 0.9 < final_global_order < 1.1):
        raise SystemExit("unexpected slopes: the concept-test assumptions or implementation changed")


if __name__ == "__main__":
    main()
