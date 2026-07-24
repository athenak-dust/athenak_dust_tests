"""Compare candidate solvers for the coupled particle--mesh drag stage.

This is a deliberately small, stand-alone mathematical test.  It does not run
AthenaK.  Particle positions, interpolation weights, gas densities, particle
masses, and stopping times are frozen so that spatial, sampling, migration,
boundary, and task-graph errors cannot obscure the temporal behavior.

For one velocity component, the frozen-weight backward-Euler drag stage is

    A u_star = b,

    A = diag(rho) + W.T @ diag(m*c) @ W,
    b = rho*u + W.T @ (m*c*v),
    c_j = a/(t_s,j + a),

where ``a`` is the diagonal implicit stage length.  Once a trial gas field x
has been selected, the physical update is always committed with the same
particle kick and mirrored PMBR gas response:

    dv = c * (W@x - v),
    v_new = v + dv,
    rho*(u_new-u) = -W.T @ (m*dv).

The tested field solvers are:

* ``local``: x0 = P^{-1} b with row-sum P = rho + W.T@(m*c);
* ``dc1``: one preconditioned Richardson correction after x0;
* ``defect2``: two Richardson corrections after x0;
* ``cheb8``: eight nonstationary Chebyshev/Richardson corrections after x0;
* ``pcg``: strict matrix-free preconditioned conjugate gradients;
* ``dense``: numpy.linalg.solve, used as the backward-Euler stage oracle;
* ``adaptive``: try dc1, accept with the physical residual bound from method
  note 7, otherwise continue with strict PCG.

The script performs four tests:

1. fixed-weight temporal convergence of every solver inside pure-drag imex2+;
2. representative resolved, stiff, and stiff/high-loading stage comparisons;
3. randomized dense-versus-matrix-free and momentum-conservation checks;
4. an adaptive no-false-acceptance check over the randomized ensemble.

Run from ``athenak_cc/`` with

    python athenak_dust_tests/concept_tests/test_coupled_drag_solver_methods.py

The test requires NumPy and SciPy and exits nonzero if its required order,
reference-solve, conservation, or adaptive-acceptance checks fail.
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm


GAMMA = 1.0 + 1.0 / np.sqrt(2.0)
GAM0_STAGE2 = (2.0 * GAMMA - 1.0) / (2.0 * GAMMA * GAMMA)
GAM1_STAGE2 = 1.0 - GAM0_STAGE2
BETA_STAGE2 = 1.0 / (2.0 * GAMMA)
ATWID_STAGE2 = (1.0 - 2.0 * GAMMA * GAMMA) / (2.0 * GAMMA)
STRICT_TOL = 2.0e-13


@dataclass
class StageData:
    """Frozen coefficients and mesh system for one backward-Euler stage."""

    weights: np.ndarray
    particle_masses: np.ndarray
    gas_densities: np.ndarray
    stopping_times: np.ndarray
    gas_velocity: np.ndarray
    particle_velocity: np.ndarray
    coupling: np.ndarray
    matrix: np.ndarray
    rhs: np.ndarray
    preconditioner: np.ndarray
    epsilon_c_max: float


@dataclass
class StageDiagnostics:
    """Diagnostics returned by a candidate stage solver."""

    residual_norm: float
    iterations: int = 0
    fast_accepted: bool = False
    error_bound: float = np.nan
    acceptance_target: float = np.nan


def build_pmbr_drag_matrix(weights, particle_masses, gas_densities, stopping_times):
    """Return the exact infinitesimal frozen-weight PMBR operator D."""
    weights = np.asarray(weights, dtype=float)
    particle_masses = np.asarray(particle_masses, dtype=float)
    gas_densities = np.asarray(gas_densities, dtype=float)
    stopping_times = np.asarray(stopping_times, dtype=float)
    nparticle, ncell = weights.shape
    matrix = np.zeros((ncell + nparticle, ncell + nparticle))

    for p in range(nparticle):
        drag_rate = 1.0 / stopping_times[p]
        matrix[ncell + p, :ncell] += drag_rate * weights[p]
        matrix[ncell + p, ncell + p] -= drag_rate
        for k in range(ncell):
            factor = -particle_masses[p] * weights[p, k] / gas_densities[k]
            matrix[k] += factor * matrix[ncell + p]
    return matrix


def prepare_stage(state, stage_dt, weights, particle_masses, gas_densities, stopping_times):
    """Build A, b, and the row-sum preconditioner for one frozen stage."""
    weights = np.asarray(weights, dtype=float)
    particle_masses = np.asarray(particle_masses, dtype=float)
    gas_densities = np.asarray(gas_densities, dtype=float)
    stopping_times = np.asarray(stopping_times, dtype=float)
    state = np.asarray(state, dtype=float)
    nparticle, ncell = weights.shape
    gas_velocity = state[:ncell].copy()
    particle_velocity = state[ncell:].copy()
    coupling = stage_dt / (stopping_times + stage_dt)
    weighted_mass = particle_masses * coupling
    matrix = np.diag(gas_densities) + weights.T @ (weighted_mass[:, None] * weights)
    rhs = gas_densities * gas_velocity + weights.T @ (weighted_mass * particle_velocity)
    deposited_row_sum = weights.T @ weighted_mass
    preconditioner = gas_densities + deposited_row_sum
    epsilon_c_max = float(np.max(deposited_row_sum / gas_densities))
    return StageData(
        weights=weights,
        particle_masses=particle_masses,
        gas_densities=gas_densities,
        stopping_times=stopping_times,
        gas_velocity=gas_velocity,
        particle_velocity=particle_velocity,
        coupling=coupling,
        matrix=matrix,
        rhs=rhs,
        preconditioner=preconditioner,
        epsilon_c_max=epsilon_c_max,
    )


def apply_matrix(stage, field):
    """Apply A without using the explicitly assembled dense matrix."""
    gathered = stage.weights @ field
    scattered = stage.weights.T @ (stage.particle_masses * stage.coupling * gathered)
    return stage.gas_densities * field + scattered


def preconditioned_residual_norm(stage, residual):
    """Return ||r||_{P^{-1}} for unit cell volumes."""
    return float(np.sqrt(np.dot(residual, residual / stage.preconditioner)))


def local_guess(stage):
    """Return x0=P^{-1}b, the current cell-local field solve."""
    return stage.rhs / stage.preconditioner


def richardson_corrections(stage, field, count):
    """Apply ``count`` row-sum-preconditioned defect corrections."""
    field = field.copy()
    for _ in range(count):
        residual = stage.rhs - apply_matrix(stage, field)
        field += residual / stage.preconditioner
    return field


def chebyshev_corrections(stage, field, count=8):
    """Apply a fixed Chebyshev batch using the proven spectrum enclosure.

    Method note 7 gives eigenvalues of P^{-1}A in

        [1/(1+epsilon_c_max), 1].

    The product of nonstationary Richardson factors with reciprocals of the
    Chebyshev roots minimizes the worst-case error polynomial on that interval.
    Alternating low/high roots limits intermediate amplification.  This is an
    independently specified concept method; it is not copied from report 9.
    """
    lower = 1.0 / (1.0 + stage.epsilon_c_max)
    upper = 1.0
    center = 0.5 * (upper + lower)
    radius = 0.5 * (upper - lower)
    roots = np.array([
        center + radius * np.cos((2*k + 1) * np.pi / (2*count))
        for k in range(count)
    ])
    order = []
    for k in range((count + 1) // 2):
        order.append(k)
        if count - 1 - k != k:
            order.append(count - 1 - k)

    field = field.copy()
    for k in order:
        residual = stage.rhs - apply_matrix(stage, field)
        field += (residual / stage.preconditioner) / roots[k]
    return field


def pcg_solve(stage, initial, tolerance=STRICT_TOL):
    """Solve A x=b with matrix-free row-sum-preconditioned CG."""
    field = initial.copy()
    residual = stage.rhs - apply_matrix(stage, field)
    zvec = residual / stage.preconditioner
    direction = zvec.copy()
    rz = float(np.dot(residual, zvec))
    rhs_scale = max(preconditioned_residual_norm(stage, stage.rhs), 1.0)
    max_iterations = max(8, 4 * len(field))

    if np.sqrt(max(rz, 0.0)) <= tolerance * rhs_scale:
        return field, 0

    for iteration in range(1, max_iterations + 1):
        adir = apply_matrix(stage, direction)
        denominator = float(np.dot(direction, adir))
        if denominator <= 0.0:
            raise RuntimeError("PCG lost positive definiteness")
        alpha = rz / denominator
        field += alpha * direction
        residual -= alpha * adir
        znew = residual / stage.preconditioner
        rz_new = float(np.dot(residual, znew))
        if np.sqrt(max(rz_new, 0.0)) <= tolerance * rhs_scale:
            true_residual = stage.rhs - apply_matrix(stage, field)
            if preconditioned_residual_norm(stage, true_residual) <= 10.0 * tolerance * rhs_scale:
                return field, iteration
            residual = true_residual
            znew = residual / stage.preconditioner
            rz_new = float(np.dot(residual, znew))
        beta = rz_new / rz
        direction = znew + beta * direction
        rz = rz_new
    raise RuntimeError("PCG failed to converge")


def commit_trial(stage, field):
    """Commit one particle kick and its exactly mirrored PMBR gas update."""
    particle_kick = stage.coupling * (stage.weights @ field - stage.particle_velocity)
    particle_new = stage.particle_velocity + particle_kick
    gas_new = stage.gas_velocity - (
        stage.weights.T @ (stage.particle_masses * particle_kick)
    ) / stage.gas_densities
    return np.concatenate([gas_new, particle_new])


def state_norm(state, gas_densities, particle_masses):
    """Mass-weighted gas+particle velocity norm."""
    ncell = len(gas_densities)
    return float(np.sqrt(
        np.dot(gas_densities, state[:ncell] ** 2)
        + np.dot(particle_masses, state[ncell:] ** 2)
    ))


def total_momentum(state, gas_densities, particle_masses):
    """Return total gas+particle momentum for unit cell volumes."""
    ncell = len(gas_densities)
    return float(
        np.dot(gas_densities, state[:ncell])
        + np.dot(particle_masses, state[ncell:])
    )


def adaptive_error_bound(stage, residual):
    """Return the conservative physical error bound from method note 7."""
    epsilon = stage.epsilon_c_max
    theta = epsilon / (1.0 + epsilon)
    cmax = float(np.max(stage.coupling))
    response = (1.0 + epsilon) * np.sqrt(
        (1.0 + epsilon) * theta * theta + cmax * theta
    )
    return response * preconditioned_residual_norm(stage, residual)


def solve_stage_field(stage, method, outer_dt=None):
    """Return a trial gas field and diagnostics for the selected method."""
    field0 = local_guess(stage)
    iterations = 0
    fast_accepted = False
    error_bound = np.nan
    acceptance_target = np.nan

    if method == "local":
        field = field0
    elif method == "dc1":
        field = richardson_corrections(stage, field0, 1)
    elif method == "defect2":
        field = richardson_corrections(stage, field0, 2)
    elif method == "cheb8":
        field = chebyshev_corrections(stage, field0, 8)
        iterations = 8
    elif method == "pcg":
        field, iterations = pcg_solve(stage, field0)
    elif method == "dense":
        field = np.linalg.solve(stage.matrix, stage.rhs)
    elif method == "adaptive":
        if outer_dt is None:
            raise ValueError("adaptive solve requires the outer timestep")
        field = richardson_corrections(stage, field0, 1)
        residual = stage.rhs - apply_matrix(stage, field)
        error_bound = adaptive_error_bound(stage, residual)
        candidate = commit_trial(stage, field)
        state_scale = max(
            state_norm(candidate, stage.gas_densities, stage.particle_masses),
            1.0,
        )
        rtol_step = min(1.0e-3, 0.25 * outer_dt ** 3)
        acceptance_target = 2.0e-14 + rtol_step * state_scale
        if error_bound <= acceptance_target:
            fast_accepted = True
        else:
            field, iterations = pcg_solve(stage, field)
    else:
        raise ValueError(f"unknown solver method: {method}")

    residual = stage.rhs - apply_matrix(stage, field)
    diagnostics = StageDiagnostics(
        residual_norm=preconditioned_residual_norm(stage, residual),
        iterations=iterations,
        fast_accepted=fast_accepted,
        error_bound=error_bound,
        acceptance_target=acceptance_target,
    )
    return field, diagnostics


def drag_stage(state, stage_dt, method, weights, particle_masses, gas_densities, stopping_times, outer_dt=None):
    """Apply one selected frozen-weight backward-Euler/approximate stage."""
    stage = prepare_stage(
        state,
        stage_dt,
        weights,
        particle_masses,
        gas_densities,
        stopping_times,
    )
    field, diagnostics = solve_stage_field(stage, method, outer_dt=outer_dt)
    return commit_trial(stage, field), diagnostics


def imex2plus_step(state, dt, method, weights, particle_masses, gas_densities, stopping_times):
    """Apply the two working pure-drag stages of imex2+."""
    stage_dt = GAMMA * dt
    stage1, _ = drag_stage(
        state,
        stage_dt,
        method,
        weights,
        particle_masses,
        gas_densities,
        stopping_times,
        outer_dt=dt,
    )
    stage1_rate = (stage1 - state) / stage_dt
    stage2_rhs = state + (1.0 - GAMMA) * dt * stage1_rate
    stage2, _ = drag_stage(
        stage2_rhs,
        stage_dt,
        method,
        weights,
        particle_masses,
        gas_densities,
        stopping_times,
        outer_dt=dt,
    )
    return stage2


def imex2plus_mixed_step(state, dt, method, explicit_matrix, weights, particle_masses, gas_densities, stopping_times):
    """Apply imex2+ to noncommuting explicit and particle--mesh drag operators.

    This follows the AthenaK low-storage ordering.  The recorded implicit rate
    excludes the explicit increment: R1=(stage1-stage1_rhs)/(gamma*dt).
    """
    stage_dt = GAMMA * dt
    stage1_rhs = state + stage_dt * (explicit_matrix @ state)
    stage1, _ = drag_stage(
        stage1_rhs,
        stage_dt,
        method,
        weights,
        particle_masses,
        gas_densities,
        stopping_times,
        outer_dt=dt,
    )
    stage1_rate = (stage1 - stage1_rhs) / stage_dt
    stage2_rhs = (
        GAM0_STAGE2 * stage1
        + GAM1_STAGE2 * state
        + BETA_STAGE2 * dt * (explicit_matrix @ stage1)
        + ATWID_STAGE2 * dt * stage1_rate
    )
    stage2, _ = drag_stage(
        stage2_rhs,
        stage_dt,
        method,
        weights,
        particle_masses,
        gas_densities,
        stopping_times,
        outer_dt=dt,
    )
    return stage2


def fitted_order(step_sizes, errors, count=4):
    """Fit error=C*dt^p over the last ``count`` non-roundoff points."""
    step_sizes = np.asarray(step_sizes, dtype=float)
    errors = np.asarray(errors, dtype=float)
    usable = np.where(errors > 1.0e-13)[0]
    if len(usable) < count:
        usable = np.arange(len(errors))
    chosen = usable[-count:]
    slope, _ = np.polyfit(np.log(step_sizes[chosen]), np.log(errors[chosen]), 1)
    return float(slope)


def temporal_order_test():
    """Measure one-step and fixed-time orders for all candidate solvers."""
    weights = np.array([[0.75, 0.25], [0.20, 0.80]])
    particle_masses = np.array([0.7, 0.4])
    gas_densities = np.array([1.0, 1.3])
    stopping_times = np.array([0.25, 1.25])
    initial_state = np.array([0.4, -0.3, 1.2, -0.8])
    drag_matrix = build_pmbr_drag_matrix(
        weights,
        particle_masses,
        gas_densities,
        stopping_times,
    )
    methods = ["local", "dc1", "defect2", "cheb8", "pcg", "dense", "adaptive"]
    # The local method contains both the outer IMEX O(dt^3) truncation term and
    # its O(dt^2) particle--mesh stage defect.  The latter dominates only on a
    # sufficiently fine ladder, so retain the refinement depth of the original
    # local-BE concept test instead of fitting a misleading pre-asymptotic mix.
    one_step_dts = np.array([2.0 ** (-k) for k in range(4, 13)])
    final_time = 0.25
    step_counts = np.array([8, 16, 32, 64, 128, 256, 512, 1024])
    fixed_dts = final_time / step_counts
    exact_final = expm(final_time * drag_matrix) @ initial_state
    results = {}

    print("\nTEMPORAL ORDER: fixed weights, mixed stopping times")
    print("method       one-step p    fixed-time p    finest one-step    finest fixed-time")
    for method in methods:
        one_step_errors = []
        for dt in one_step_dts:
            numerical = imex2plus_step(
                initial_state,
                dt,
                method,
                weights,
                particle_masses,
                gas_densities,
                stopping_times,
            )
            exact = expm(dt * drag_matrix) @ initial_state
            one_step_errors.append(np.linalg.norm(numerical - exact, ord=np.inf))

        fixed_errors = []
        for nstep, dt in zip(step_counts, fixed_dts):
            numerical = initial_state.copy()
            for _ in range(nstep):
                numerical = imex2plus_step(
                    numerical,
                    dt,
                    method,
                    weights,
                    particle_masses,
                    gas_densities,
                    stopping_times,
                )
            fixed_errors.append(np.linalg.norm(numerical - exact_final, ord=np.inf))

        one_order = fitted_order(one_step_dts, one_step_errors)
        fixed_order = fitted_order(fixed_dts, fixed_errors)
        results[method] = {
            "one_order": one_order,
            "fixed_order": fixed_order,
            "one_error": one_step_errors[-1],
            "fixed_error": fixed_errors[-1],
        }
        print(
            f"{method:10s} {one_order:13.6f} {fixed_order:15.6f} "
            f"{one_step_errors[-1]:18.9e} {fixed_errors[-1]:18.9e}"
        )

    if not (1.8 < results["local"]["one_order"] < 2.2):
        raise SystemExit("local one-step order is not approximately two")
    if not (0.8 < results["local"]["fixed_order"] < 1.2):
        raise SystemExit("local fixed-time order is not approximately one")
    for method in methods[1:]:
        if not (2.7 < results[method]["one_order"] < 3.3):
            raise SystemExit(f"{method} one-step order is not approximately three")
        if not (1.7 < results[method]["fixed_order"] < 2.3):
            raise SystemExit(f"{method} fixed-time order is not approximately two")
    return results


def mixed_temporal_order_test():
    """Check the full IMEX order with a noncommuting explicit operator."""
    weights = np.array([[0.75, 0.25], [0.20, 0.80]])
    particle_masses = np.array([0.7, 0.4])
    gas_densities = np.array([1.0, 1.3])
    stopping_times = np.array([0.25, 1.25])
    initial_state = np.array([0.4, -0.3, 1.2, -0.8])
    drag_matrix = build_pmbr_drag_matrix(
        weights,
        particle_masses,
        gas_densities,
        stopping_times,
    )
    explicit_matrix = np.array([
        [0.0, 0.3, 0.2, 0.0],
        [-0.4, 0.0, 0.0, -0.1],
        [0.5, 0.0, 0.0, 0.2],
        [0.0, -0.3, -0.6, 0.0],
    ])
    if np.linalg.norm(explicit_matrix @ drag_matrix - drag_matrix @ explicit_matrix) < 1.0e-10:
        raise SystemExit("mixed-order test accidentally used commuting operators")

    methods = ["local", "dc1", "defect2", "cheb8", "pcg", "dense", "adaptive"]
    one_step_dts = np.array([2.0 ** (-k) for k in range(4, 13)])
    final_time = 0.25
    step_counts = np.array([8, 16, 32, 64, 128, 256, 512, 1024])
    fixed_dts = final_time / step_counts
    total_matrix = drag_matrix + explicit_matrix
    exact_final = expm(final_time * total_matrix) @ initial_state
    results = {}

    print("\nMIXED TEMPORAL ORDER: noncommuting explicit operator + drag")
    print("method       one-step p    fixed-time p    finest one-step    finest fixed-time")
    for method in methods:
        one_step_errors = []
        for dt in one_step_dts:
            numerical = imex2plus_mixed_step(
                initial_state,
                dt,
                method,
                explicit_matrix,
                weights,
                particle_masses,
                gas_densities,
                stopping_times,
            )
            exact = expm(dt * total_matrix) @ initial_state
            one_step_errors.append(np.linalg.norm(numerical - exact, ord=np.inf))

        fixed_errors = []
        for nstep, dt in zip(step_counts, fixed_dts):
            numerical = initial_state.copy()
            for _ in range(nstep):
                numerical = imex2plus_mixed_step(
                    numerical,
                    dt,
                    method,
                    explicit_matrix,
                    weights,
                    particle_masses,
                    gas_densities,
                    stopping_times,
                )
            fixed_errors.append(np.linalg.norm(numerical - exact_final, ord=np.inf))

        one_order = fitted_order(one_step_dts, one_step_errors)
        fixed_order = fitted_order(fixed_dts, fixed_errors)
        results[method] = {
            "one_order": one_order,
            "fixed_order": fixed_order,
            "one_error": one_step_errors[-1],
            "fixed_error": fixed_errors[-1],
        }
        print(
            f"{method:10s} {one_order:13.6f} {fixed_order:15.6f} "
            f"{one_step_errors[-1]:18.9e} {fixed_errors[-1]:18.9e}"
        )

    if not (1.8 < results["local"]["one_order"] < 2.2):
        raise SystemExit("mixed local one-step order is not approximately two")
    if not (0.8 < results["local"]["fixed_order"] < 1.2):
        raise SystemExit("mixed local fixed-time order is not approximately one")
    for method in methods[1:]:
        if not (2.7 < results[method]["one_order"] < 3.3):
            raise SystemExit(f"mixed {method} one-step order is not approximately three")
        if not (1.7 < results[method]["fixed_order"] < 2.3):
            raise SystemExit(f"mixed {method} fixed-time order is not approximately two")
    return results


def relative_stage_error(candidate, reference, gas_densities, particle_masses):
    """Return mass-weighted candidate error relative to the strict stage state."""
    denominator = max(state_norm(reference, gas_densities, particle_masses), 1.0e-30)
    return state_norm(candidate - reference, gas_densities, particle_masses) / denominator


def representative_stage_test():
    """Compare stage errors in resolved, stiff, and high-loading examples."""
    weights = np.array([
        [0.62, 0.28, 0.10],
        [0.08, 0.57, 0.35],
        [0.31, 0.18, 0.51],
        [0.47, 0.44, 0.09],
    ])
    gas_densities = np.array([1.0, 0.8, 1.4])
    initial_state = np.array([0.5, -0.2, 0.1, 1.1, -0.9, 0.7, -0.4])
    scenarios = [
        ("resolved/moderate", 1.0e-3, np.array([0.2, 0.7, 1.3, 0.4]), 1.0),
        ("stiff/moderate", 0.1, np.array([1.0e-4, 3.0e-4, 7.0e-5, 2.0e-4]), 1.0),
        ("stiff/high-load", 0.1, np.array([1.0e-4, 3.0e-4, 7.0e-5, 2.0e-4]), 100.0),
    ]
    base_masses = np.array([0.5, 0.8, 0.4, 0.6])
    methods = ["local", "dc1", "defect2", "cheb8", "pcg", "adaptive"]

    print("\nREPRESENTATIVE BACKWARD-EULER STAGES")
    print("scenario             eps_c_max method       relative state error   ||r||_P^-1  detail")
    rows = []
    for name, stage_dt, stopping_times, mass_scale in scenarios:
        particle_masses = mass_scale * base_masses
        stage = prepare_stage(
            initial_state,
            stage_dt,
            weights,
            particle_masses,
            gas_densities,
            stopping_times,
        )
        dense_field, _ = solve_stage_field(stage, "dense")
        reference = commit_trial(stage, dense_field)
        for method in methods:
            field, diagnostics = solve_stage_field(
                stage,
                method,
                outer_dt=stage_dt / GAMMA,
            )
            candidate = commit_trial(stage, field)
            error = relative_stage_error(
                candidate,
                reference,
                gas_densities,
                particle_masses,
            )
            if method == "adaptive":
                detail = "fast" if diagnostics.fast_accepted else f"pcg:{diagnostics.iterations}"
            elif method == "pcg":
                detail = f"iter:{diagnostics.iterations}"
            else:
                detail = "-"
            print(
                f"{name:20s} {stage.epsilon_c_max:9.3e} {method:10s} "
                f"{error:20.9e} {diagnostics.residual_norm:13.5e}  {detail}"
            )
            rows.append((name, method, error, diagnostics.residual_norm, detail))
    return rows


def random_stage(rng):
    """Generate a nonuniform overlapping-cloud stage for randomized checks."""
    ncell = int(rng.integers(2, 5))
    nparticle = int(rng.integers(ncell, 2 * ncell + 3))
    weights = rng.dirichlet(0.7 * np.ones(ncell), size=nparticle)
    gas_densities = 10.0 ** rng.uniform(-0.5, 0.5, size=ncell)
    base_masses = 10.0 ** rng.uniform(-0.5, 0.5, size=nparticle)
    target_loading = 10.0 ** rng.uniform(-3.0, 2.0)
    raw_loading = np.max((weights.T @ base_masses) / gas_densities)
    particle_masses = base_masses * target_loading / raw_loading
    stage_dt = 10.0 ** rng.uniform(-3.0, -0.1)
    stopping_times = stage_dt * 10.0 ** rng.uniform(-3.0, 3.0, size=nparticle)
    state = rng.normal(size=ncell + nparticle)
    return state, stage_dt, weights, particle_masses, gas_densities, stopping_times


def randomized_test(ncase=400, seed=731942):
    """Check matrix-free algebra, conservation, PCG, and adaptive acceptance."""
    rng = np.random.default_rng(seed)
    max_apply_error = 0.0
    max_pcg_error = 0.0
    max_momentum_error = 0.0
    fast_accepts = 0
    fallbacks = 0
    false_accepts = 0
    max_accepted_fraction = 0.0
    pcg_iterations = []

    for _ in range(ncase):
        state, stage_dt, weights, particle_masses, gas_densities, stopping_times = random_stage(rng)
        stage = prepare_stage(
            state,
            stage_dt,
            weights,
            particle_masses,
            gas_densities,
            stopping_times,
        )
        probe = rng.normal(size=len(gas_densities))
        dense_apply = stage.matrix @ probe
        matrix_free_apply = apply_matrix(stage, probe)
        apply_error = np.linalg.norm(matrix_free_apply - dense_apply) / max(
            np.linalg.norm(dense_apply),
            1.0,
        )
        max_apply_error = max(max_apply_error, apply_error)

        dense_field, _ = solve_stage_field(stage, "dense")
        dense_state = commit_trial(stage, dense_field)
        pcg_field, pcg_diag = solve_stage_field(stage, "pcg")
        pcg_state = commit_trial(stage, pcg_field)
        pcg_error = relative_stage_error(
            pcg_state,
            dense_state,
            gas_densities,
            particle_masses,
        )
        max_pcg_error = max(max_pcg_error, pcg_error)
        pcg_iterations.append(pcg_diag.iterations)

        initial_momentum = total_momentum(state, gas_densities, particle_masses)
        for method in ("local", "dc1", "defect2", "cheb8", "pcg", "dense"):
            field, _ = solve_stage_field(stage, method)
            candidate = commit_trial(stage, field)
            momentum_error = abs(
                total_momentum(candidate, gas_densities, particle_masses)
                - initial_momentum
            ) / max(abs(initial_momentum), state_norm(state, gas_densities, particle_masses), 1.0)
            max_momentum_error = max(max_momentum_error, momentum_error)

        outer_dt = stage_dt / GAMMA
        adaptive_field, adaptive_diag = solve_stage_field(stage, "adaptive", outer_dt=outer_dt)
        adaptive_state = commit_trial(stage, adaptive_field)
        actual_error = state_norm(
            adaptive_state - dense_state,
            gas_densities,
            particle_masses,
        )
        if adaptive_diag.fast_accepted:
            fast_accepts += 1
            fraction = actual_error / adaptive_diag.acceptance_target
            max_accepted_fraction = max(max_accepted_fraction, fraction)
            if actual_error > adaptive_diag.acceptance_target * (1.0 + 2.0e-12):
                false_accepts += 1
        else:
            fallbacks += 1

    print("\nRANDOMIZED FROZEN-WEIGHT ENSEMBLE")
    print(f"cases                              {ncase}")
    print(f"max matrix-free apply relative err {max_apply_error:.9e}")
    print(f"max strict-PCG stage relative err  {max_pcg_error:.9e}")
    print(f"max normalized momentum drift      {max_momentum_error:.9e}")
    print(f"adaptive fast accepts              {fast_accepts}")
    print(f"adaptive PCG fallbacks             {fallbacks}")
    print(f"adaptive false accepts             {false_accepts}")
    print(f"max actual/target among accepts    {max_accepted_fraction:.9e}")
    print(
        "strict-PCG iterations min/median/p95/max "
        f"{np.min(pcg_iterations)}/{np.median(pcg_iterations):.1f}/"
        f"{np.percentile(pcg_iterations, 95):.1f}/{np.max(pcg_iterations)}"
    )

    if max_apply_error > 5.0e-13:
        raise SystemExit("matrix-free A application disagrees with dense A")
    if max_pcg_error > 2.0e-11:
        raise SystemExit("strict PCG disagrees with the dense coupled-stage solve")
    if max_momentum_error > 5.0e-13:
        raise SystemExit("PMBR commit failed momentum conservation")
    if false_accepts != 0:
        raise SystemExit("adaptive solver falsely accepted an inaccurate dc1 state")
    if fast_accepts == 0 or fallbacks == 0:
        raise SystemExit("randomized ensemble did not exercise both adaptive branches")
    return {
        "cases": ncase,
        "max_apply_error": max_apply_error,
        "max_pcg_error": max_pcg_error,
        "max_momentum_error": max_momentum_error,
        "fast_accepts": fast_accepts,
        "fallbacks": fallbacks,
        "false_accepts": false_accepts,
        "max_accepted_fraction": max_accepted_fraction,
        "pcg_iterations": pcg_iterations,
    }


def main():
    temporal_order_test()
    mixed_temporal_order_test()
    representative_stage_test()
    randomized_test()
    print("\nPASS: all coupled-drag solver concept tests completed")


if __name__ == "__main__":
    main()
