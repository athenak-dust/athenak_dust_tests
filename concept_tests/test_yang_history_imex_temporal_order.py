"""Check the proposed Yang16 single-species history-weighted IMEX construction.

The proposal in ``IMEX_coeff_for_Yang16_single_species.md`` replaces the
backward-Euler drag stage by an exact single-stopping-time sub-cloud update and
records a mode-dependent history impulse.  This program separates two claims:

1. For a true exponential operator, the new history algebra is correct.  A
   scalar decay mode is reproduced exactly (without the implementation clamp),
   and an exact matrix exponential combined with a noncommuting explicit
   operator has one-step error O(dt**3).
2. The Yang particle-mesh stage is not a true exponential semigroup after its
   sub-cloud kicks are collapsed to parent particles and mirrored to the gas by
   PMBR.  With overlapping clouds, the same history construction has one-step
   error O(dt**2) and fixed-time error O(dt), i.e. global first order.

The frozen-weight reference equations are

    dv_p/dt = (sum_k W[p,k] u_k - v_p) / ts,
    Mgas_k du_k/dt = -sum_p m_p W[p,k] dv_p/dt.

The second equation is the infinitesimal form of the actual PMBR update.  Its
constant matrix is exponentiated with SciPy to isolate temporal error.

The script also demonstrates two independent finite-step issues:

* clamping ``HistWeight`` breaks the exact scalar modal identity;
* exponential drag stages plus an explicit constant force do not preserve the
  stiff terminal equilibrium.

Run from ``athenak_cc/`` with

    python athenak_dust_tests/concept_tests/test_yang_history_imex_temporal_order.py

The final assertions encode the expected slopes and make the script suitable
for repeatable design checks, while the printed tables retain the full evidence.
"""

import numpy as np
from scipy.linalg import expm


GAMMA = 1.0 + 1.0 / np.sqrt(2.0)
G0 = (2.0 * GAMMA - 1.0) / (2.0 * GAMMA * GAMMA)
EXPLICIT_STAGE2_WEIGHT = 1.0 / (2.0 * GAMMA)
BACKWARD_EXPONENT_WEIGHT = (GAMMA - 1.0) / GAMMA
HISTORY_COEFFICIENT = (1.0 - 2.0 * GAMMA * GAMMA) / (2.0 * GAMMA * GAMMA)


def history_weight(x, clamp=False):
    """Return the proposed modal weight; optionally reproduce its two clamps."""
    if abs(x) < 1.0e-12:
        return 1.0

    backward_argument = BACKWARD_EXPONENT_WEIGHT * x
    if clamp:
        backward_argument = min(backward_argument, 12.0)
    decay_increment = np.expm1(-x)
    weighted_coefficient = (np.expm1(backward_argument) - G0 * decay_increment) / decay_increment
    if clamp:
        weighted_coefficient = min(max(weighted_coefficient, -2.5), 2.5)
    return weighted_coefficient / HISTORY_COEFFICIENT


def scalar_pure_drag_amplification(stage_decay, clamp=False):
    """Return the full-step amplification for one local decay eigenmode."""
    stage_amplification = np.exp(-stage_decay)
    stage_increment = stage_amplification - 1.0
    recorded_impulse = history_weight(stage_decay, clamp=clamp) * stage_increment
    stage2_rhs = G0 * stage_amplification + (1.0 - G0) + HISTORY_COEFFICIENT * recorded_impulse
    return stage_amplification * stage2_rhs


def scalar_constant_force_equilibrium(z):
    """Advance y'=-y+1 from its exact equilibrium y=1; z=-dt."""
    dt = -z
    implicit_rate = -1.0
    state = 1.0
    force = 1.0
    forward_flow = np.exp(GAMMA * dt * implicit_rate)
    backward_flow = np.exp((1.0 - GAMMA) * dt * implicit_rate)
    stage1_rhs = state + GAMMA * dt * force
    stage1 = forward_flow * stage1_rhs
    history_impulse = (backward_flow - 1.0 - G0 * (forward_flow - 1.0)) * stage1_rhs
    stage2_rhs = G0 * stage1 + (1.0 - G0) * state + EXPLICIT_STAGE2_WEIGHT * dt * force + history_impulse
    return forward_flow * stage2_rhs


def exact_exponential_imex_step(state, dt, explicit_matrix, implicit_matrix):
    """Apply the proposed history construction when the implicit flow is exact."""
    identity = np.eye(len(state))
    forward_flow = expm(GAMMA * dt * implicit_matrix)
    backward_flow = expm((1.0 - GAMMA) * dt * implicit_matrix)
    stage1_rhs = state + GAMMA * dt * (explicit_matrix @ state)
    stage1 = forward_flow @ stage1_rhs

    # This is the matrix-function form of c_be*w(x)*(exp(-x)-1).
    history_impulse = (backward_flow - identity - G0 * (forward_flow - identity)) @ stage1_rhs
    stage2_rhs = G0 * stage1 + (1.0 - G0) * state
    stage2_rhs += EXPLICIT_STAGE2_WEIGHT * dt * (explicit_matrix @ stage1)
    stage2_rhs += history_impulse
    return forward_flow @ stage2_rhs


def build_pmbr_drag_matrix(weights, gas_masses, particle_masses, stopping_time):
    """Return the exact infinitesimal parent-particle + PMBR drag operator."""
    nparticle, ncell = weights.shape
    drag_rate = 1.0 / stopping_time
    matrix = np.zeros((ncell + nparticle, ncell + nparticle))
    matrix[ncell:, :ncell] = drag_rate * weights
    matrix[ncell:, ncell:] = -drag_rate * np.eye(nparticle)
    matrix[:ncell, :] = -np.diag(1.0 / gas_masses) @ weights.T @ np.diag(particle_masses) @ matrix[ncell:, :]
    return matrix


def yang_subcloud_stage(state, stage_dt, weights, gas_masses, particle_masses, stopping_time, record_history=True):
    """Apply one exact local sub-cloud solve, parent collapse, and PMBR update.

    ``mean_kick`` is the gas--mean-dust relative mode with decay rate
    ``(1+epsilon)/ts``.  ``particle_velocity-vbar`` is the dust-deviation mode
    with decay rate ``1/ts``.  The history impulse weights these modes
    separately, exactly as the proposed six-component F/K implementation does.
    """
    nparticle, ncell = weights.shape
    gas_velocity = state[:ncell]
    particle_velocity = state[ncell:]
    subcloud_kick = np.zeros((nparticle, ncell))
    subcloud_record = np.zeros((nparticle, ncell))
    deviation_decay = np.exp(-stage_dt / stopping_time)
    deviation_weight = history_weight(stage_dt / stopping_time)

    for k in range(ncell):
        subcloud_mass = particle_masses * weights[:, k]
        dust_mass = np.sum(subcloud_mass)
        epsilon = dust_mass / gas_masses[k]
        mean_dust_velocity = 0.0 if dust_mass == 0.0 else np.dot(subcloud_mass, particle_velocity) / dust_mass
        mean_decay = np.exp(-stage_dt * (1.0 + epsilon) / stopping_time)
        mean_kick = (1.0 - mean_decay) / (1.0 + epsilon) * (gas_velocity[k] - mean_dust_velocity)
        deviation_kick = (deviation_decay - 1.0) * (particle_velocity - mean_dust_velocity)
        subcloud_kick[:, k] = mean_kick + deviation_kick

        if record_history:
            mean_weight = history_weight(stage_dt * (1.0 + epsilon) / stopping_time)
            subcloud_record[:, k] = mean_weight * mean_kick + deviation_weight * deviation_kick

    # Collapse all sub-cloud increments to the one velocity carried by each
    # parent particle, then mirror that parent increment back to the gas.
    particle_kick = np.sum(weights * subcloud_kick, axis=1)
    particle_record = np.sum(weights * subcloud_record, axis=1)
    updated = state.copy()
    updated[ncell:] += particle_kick
    updated[:ncell] -= (weights.T @ (particle_masses * particle_kick)) / gas_masses

    history_impulse = np.zeros_like(state)
    history_impulse[ncell:] = particle_record
    history_impulse[:ncell] = -(weights.T @ (particle_masses * particle_record)) / gas_masses
    return updated, history_impulse


def yang_history_imex_step(state, dt, explicit_matrix, weights, gas_masses, particle_masses, stopping_time):
    """Apply both working stages of the proposed Yang-history IMEX method."""
    stage_dt = GAMMA * dt
    stage1_rhs = state + GAMMA * dt * (explicit_matrix @ state)
    stage1, history_impulse = yang_subcloud_stage(stage1_rhs, stage_dt, weights, gas_masses, particle_masses, stopping_time)
    stage2_rhs = G0 * stage1 + (1.0 - G0) * state
    stage2_rhs += EXPLICIT_STAGE2_WEIGHT * dt * (explicit_matrix @ stage1)
    stage2_rhs += HISTORY_COEFFICIENT * history_impulse
    stage2, _ = yang_subcloud_stage(stage2_rhs, stage_dt, weights, gas_masses, particle_masses, stopping_time)
    return stage2


def convergence(step, exact, time_steps):
    """Return errors and factor-of-two convergence slopes."""
    errors = [np.linalg.norm(step(dt) - exact(dt)) for dt in time_steps]
    orders = [np.log2(errors[i - 1] / errors[i]) for i in range(1, len(errors))]
    return errors, orders


def print_convergence(title, time_steps, errors, orders):
    print(f"\n{title}")
    print("dt                 error              slope")
    for dt, error, order in zip(time_steps, errors, [np.nan] + orders):
        print(f"{dt:18.9e} {error:18.9e} {order:10.6f}")


def main():
    print("IMEX coefficients")
    print(f"gamma={GAMMA:.15g} g0={G0:.15g} beta2={EXPLICIT_STAGE2_WEIGHT:.15g}")
    print(f"backward_weight={BACKWARD_EXPONENT_WEIGHT:.15g} history_coefficient={HISTORY_COEFFICIENT:.15g}")

    print("\nscalar pure-drag modal identity and clamp effect")
    print("stage x            proposed           exact              clamped")
    for stage_decay in (1.0e-4, 0.1, 1.0, 10.0, 100.0):
        proposed = scalar_pure_drag_amplification(stage_decay)
        exact = np.exp(-stage_decay / GAMMA)
        clamped = scalar_pure_drag_amplification(stage_decay, clamp=True)
        print(f"{stage_decay:10.4g} {proposed:18.9e} {exact:18.9e} {clamped:18.9e}")
        if not np.isclose(proposed, exact, rtol=2.0e-13, atol=1.0e-300):
            raise SystemExit("unclamped scalar modal identity failed")

    print("\nconstant-force equilibrium y'=-y+1, starting from y=1")
    print("dt                 numerical          exact")
    for dt in (0.1, 1.0, 10.0, 100.0):
        numerical = scalar_constant_force_equilibrium(-dt)
        print(f"{dt:18.9e} {numerical:18.9e} {1.0:18.9e}")

    time_steps = [2.0**(-k) for k in range(4, 11)]

    # These two matrices do not commute.  Reaching one-step slope 3 therefore
    # checks the mixed explicit/implicit order terms, not just pure drag.
    explicit_matrix2 = np.array([[0.2, 1.1], [-0.7, -0.1]])
    implicit_matrix2 = np.array([[-2.0, 0.4], [0.3, -1.2]])
    initial2 = np.array([0.7, -1.3])
    matrix_errors, matrix_orders = convergence(
        lambda dt: exact_exponential_imex_step(initial2, dt, explicit_matrix2, implicit_matrix2),
        lambda dt: expm(dt * (explicit_matrix2 + implicit_matrix2)) @ initial2,
        time_steps,
    )
    print_convergence("true exponential operator: one-step slope should approach 3", time_steps, matrix_errors, matrix_orders)

    # Both parent particles overlap both gas cells, which is the smallest model
    # exposing the difference between independent local sub-cloud evolution and
    # the global parent-particle PMBR operator.
    weights = np.array([[0.75, 0.25], [0.35, 0.65]])
    gas_masses = np.array([1.0, 0.8])
    particle_masses = np.array([0.6, 0.9])
    stopping_time = 0.7
    drag_matrix = build_pmbr_drag_matrix(weights, gas_masses, particle_masses, stopping_time)
    explicit_matrix4 = np.array([
        [0.0, 0.3, 0.2, 0.0],
        [-0.4, 0.0, 0.0, -0.1],
        [0.5, 0.0, 0.0, 0.2],
        [0.0, -0.3, -0.6, 0.0],
    ])
    initial4 = np.array([0.2, -0.4, 1.1, -0.7])

    single_map_errors, single_map_orders = convergence(
        lambda dt: yang_subcloud_stage(initial4, dt, weights, gas_masses, particle_masses, stopping_time, record_history=False)[0],
        lambda dt: expm(dt * drag_matrix) @ initial4,
        time_steps,
    )
    print_convergence("one Yang sub-cloud + PMBR map: one-step slope approaches 2", time_steps, single_map_errors, single_map_orders)

    zero_explicit = np.zeros_like(explicit_matrix4)
    pure_drag_errors, pure_drag_orders = convergence(
        lambda dt: yang_history_imex_step(initial4, dt, zero_explicit, weights, gas_masses, particle_masses, stopping_time),
        lambda dt: expm(dt * drag_matrix) @ initial4,
        time_steps,
    )
    print_convergence("Yang-history IMEX, pure drag: one-step slope approaches 2", time_steps, pure_drag_errors, pure_drag_orders)

    full_errors, full_orders = convergence(
        lambda dt: yang_history_imex_step(initial4, dt, explicit_matrix4, weights, gas_masses, particle_masses, stopping_time),
        lambda dt: expm(dt * (explicit_matrix4 + drag_matrix)) @ initial4,
        time_steps,
    )
    print_convergence("Yang-history IMEX with noncommuting explicit part: one-step slope approaches 2", time_steps, full_errors, full_orders)

    final_time = 0.5
    exact_final = expm(final_time * (explicit_matrix4 + drag_matrix)) @ initial4
    global_steps = [16, 32, 64, 128, 256, 512]
    global_dts = [final_time / nstep for nstep in global_steps]
    global_errors = []
    for nstep, dt in zip(global_steps, global_dts):
        numerical = initial4.copy()
        for _ in range(nstep):
            numerical = yang_history_imex_step(numerical, dt, explicit_matrix4, weights, gas_masses, particle_masses, stopping_time)
        global_errors.append(np.linalg.norm(numerical - exact_final))
    global_orders = [np.log2(global_errors[i - 1] / global_errors[i]) for i in range(1, len(global_errors))]
    print_convergence("Yang-history IMEX at fixed final time: global slope approaches 1", global_dts, global_errors, global_orders)

    if not (2.8 < matrix_orders[-1] < 3.2):
        raise SystemExit("exact exponential construction did not show the expected one-step third order")
    if not (1.9 < single_map_orders[-1] < 2.1):
        raise SystemExit("single Yang map did not show the expected one-step second order")
    if not (1.9 < pure_drag_orders[-1] < 2.1 and 1.9 < full_orders[-1] < 2.1):
        raise SystemExit("overlapping-cloud Yang-history tests did not show one-step second order")
    if not (0.9 < global_orders[-1] < 1.1):
        raise SystemExit("overlapping-cloud Yang-history test did not show global first order")


if __name__ == "__main__":
    main()
