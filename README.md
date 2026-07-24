# AthenaK Dust Tests

This repository collects validation and numerical-test material for the Lagrangian dust module developed in [athenak-dust/athenak](https://github.com/athenak-dust/athenak).

It includes:

- Jupyter notebooks documenting test setups, results, and analysis;
- AthenaK input files used by the tests;
- small stand-alone Python concept tests; and
- plots and other supporting test material when useful.

The notebooks cover basic dust–gas drag validation, comparisons with published test problems, shearing-box and streaming-instability tests, and CPU/GPU runs. Each notebook contains its own build, input, and execution notes.

For reproducibility, clone this repository alongside the AthenaK source repository so that the directories are arranged as:

```text
athenak/
athenak_dust_tests/
```

The code under test is maintained in the `dust` branch of [athenak-dust/athenak](https://github.com/athenak-dust/athenak/tree/dust).

## Using the dust module

### Build configuration

The dust sources are part of every AthenaK build. Most dust calculations use one of the built-in problem
generators (`dust_damping`, `dust_nsh`, `dusty_wave`, or `streaming_linear`):

```bash
cmake -S athenak -B athenak/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DPROBLEM=built_in_pgens
cmake --build athenak/build -j
```

Select the built-in generator at run time with `<problem>/pgen_name`. MPI, CUDA, and
other Kokkos options can be added to the same configure command in the usual AthenaK
way.

A custom problem generator must instead be selected when CMake configures the build.
For example, the Tier-S snapshot-seeded clump tests use:

```bash
cmake -S athenak -B athenak/build-snapshot \
  -DCMAKE_BUILD_TYPE=Release \
  -DPROBLEM=tests/dust_snapshot_clump
cmake --build athenak/build-snapshot -j
```

That executable always calls `dust_snapshot_clump`; `<problem>/pgen_name` is not used.
Its input also needs the source particle VTK snapshot named by
`<problem>/snapshot_file`. See
[`inputs/dust_snapshot_clump_tier_s.athinput`](inputs/dust_snapshot_clump_tier_s.athinput)
for the complete example. A custom pgen build should be reconfigured and rebuilt when
switching back to `built_in_pgens` or to another custom pgen.

### Minimal dust input

The dust module requires Hydro, Particles, and the `imex2+` integrator. Relevant part in an example single-species configuration is:

```text
<time>
integrator = imex2+

<hydro>
eos = isothermal

<particles>
particle_type = dust
pusher = imex_dust
ppc = 1.0

<dust>
nspecies = 1
taus_1 = 0.1
dust_to_gas = 0.01
back_reaction = true
deposit = tsc
dt_cfl = 0.5
drag_solver = local

<problem>
pgen_name = dusty_wave
```

The mesh must be at least two-dimensional and have at least two ghost zones. Ordinary
periodic boundaries are the supported default. With back-reaction enabled, the gas
must use an isothermal EOS because drag heating is not yet added to the gas energy
equation.


### Required supporting blocks

Only dust-specific or dust-constrained parameters are listed here; the remaining
Hydro, mesh, time, output, and problem-generator parameters retain their standard
AthenaK meanings.

| Block/parameter | Required/default | Meaning and accepted values |
|---|---|---|
| `<time>/integrator` | required: `imex2+` | The dust task graph is implemented for the IMEX2+ scheme. Other integrators are rejected. |
| `<hydro>/eos` | `isothermal` when `back_reaction = true` | `ideal` is allowed only without gas back-reaction; drag heating is not implemented. A `<mhd>` block is currently incompatible with dust drag. |
| `<mesh>/nghost` | at least `2` | Particle-mesh assignment needs two or more ghost cells. PPMX reconstruction independently requires three. |
| mesh boundary flags | periodic | Use ordinary `periodic` boundaries for supported calculations. See “Current limitations” for the incomplete 3-D shear-periodic path. |
| `<particles>/particle_type` | required: `dust` | Allocates the dust particle fields, including species, stopping time, mass, and IMEX registers. |
| `<particles>/pusher` | required: `imex_dust` | Advances the particles inside the coupled dust IMEX stages. |
| `<particles>/ppc` | `1.0` | Mean number of numerical particles per cell used to allocate and initialize particles. It may be non-integer, although individual pgens can impose stronger placement requirements. |
| `<particles>/assign_tag` | `index_order` | Unique particle-tag layout: `index_order` assigns contiguous rank-local ranges; `rank_order` interleaves tags by MPI rank. |
| `<problem>/rho0` | `1.0` | Reference gas density used with `dust_to_gas` when a pgen calls the default particle-mass initializer. A pgen may replace that mass assignment. |
| `<shearing_box>/qshear` | required if the block exists | Dimensionless shear parameter used in the dust Coriolis/tidal update and particle transport velocity. |
| `<shearing_box>/omega0` | required if the block exists | Rotation frequency used by the dust shearing-box source terms. |
| `<shearing_box>/stratified` | `false` | If `true`, add vertical gravity to dust particles in a 3-D shearing box. The 3-D shear-periodic coupling limitation still applies. |

### `<dust>` parameter reference

| Parameter | Default | Meaning and accepted values |
|---|---:|---|
| `back_reaction` | `true` | If `true`, deposit the drag reaction onto the gas; if `false`, particles feel the gas but do not change it. |
| `gamma_switch` | `false` | If enabled, use IMEX2+ $\gamma=1/2$ when $\Delta t$ exceeds the largest active stopping time and $\gamma=1+1/\sqrt{2}$ otherwise. The switch improves stiff-limit asymptotic accuracy. |
| `dt_cfl` | `0.5` | Multiplicative CFL factor for the particle transport timestep, $\min(\Delta x/|v_{\rm transport}|)$. |
| `dust_to_gas` | `0.01` | Total dust-to-gas mass ratio used by the default particle-mass initialization. A pgen may deliberately overwrite particle masses. |
| `nspecies` | `1` | Number of dust species; must be at least one. |
| `taus_1`, ..., `taus_N` | required | Positive stopping time for every species, where `N = nspecies`. Under `species_fixed`, these values are authoritative. |
| `stopping_time_mode` | `species_fixed` | `species_fixed`: every particle is validated against its species-table value; `particle_static`: accept arbitrary positive per-particle stopping times and find their global maximum once after initialization; `dynamic`: allow changing per-particle stopping times and refresh the maximum once per cycle. |
| `deposit` | `tsc` | Particle-mesh assignment and gather kernel: nearest-grid-point (`ngp`), cloud-in-cell (`cic`), or triangular-shaped-cloud (`tsc`). |
| `drag_solver` | `local` | Implicit gas-stage solver: `local`, `applya`, `dc1`, `dc2`, `pcg`, or `adaptive`. The choices are described below. |
| `drag_rtol` | `1.0e-11` | Positive relative residual tolerance for strict PCG solves. |
| `drag_atol` | `1.0e-14` | Non-negative absolute residual tolerance for strict PCG solves and the adaptive certificate. |
| `drag_iter_max` | `200` | Maximum PCG iterations; must be at least one. |
| `drag_fail_policy` | `abort` | Action after solver failure. `abort` is currently the only implemented value. |
| `drag_diagnostic_interval` | `0` | Print a `DUST_SOLVER_DIAG` line every this many cycles (`0` disables periodic diagnostics). A final `DUST_SOLVER_SUMMARY` is printed for coupled-solver runs. |
| `drag_adaptive_order_c` | `0.25` | Positive coefficient in the adaptive order target $C(\Delta t/t_{\rm ref})^3$. |
| `drag_adaptive_rtol_max` | `1.0e-3` | Positive ceiling on the adaptive relative acceptance target. |
| `drag_adaptive_tref` | `1.0` | Positive reference time $t_{\rm ref}$ used to nondimensionalize $\Delta t$ in that target. |
| `drag_adaptive_state_floor` | `1.0` | Positive floor for the state norm used by the adaptive absolute error certificate. |

The solver choices have different purposes:

| `drag_solver` | Behavior |
|---|---|
| `local` | Cell-local approximate implicit solve. This is the least expensive baseline, but its stage coupling is generally only first-order accurate in time. |
| `applya` | Diagnostic/timing path that applies the matrix-free coupled operator to the local guess without correcting it. Do not use it as a production solver. |
| `dc1` | Local guess plus one fixed defect-correction sweep. This is the current cost/accuracy sweet spot in resolved regimes. |
| `dc2` | Local guess plus two fixed defect-correction sweeps, accepted without a residual check or PCG fallback. |
| `pcg` | Strict matrix-free preconditioned conjugate-gradient solve to `drag_atol + drag_rtol ||b_d||` in each velocity component; useful as a validation/reference path. |
| `adaptive` | Try one defect correction, certify its error against the timestep-scaled target, and fall back to strict PCG if the certificate fails. |

The four adaptive parameters affect only `drag_solver = adaptive`; the PCG tolerances
and iteration limit affect `pcg` and an adaptive fallback.

### Problem-generator parameters

The `<problem>` block initializes the gas and particles and is therefore specific to
the selected pgen. With `-DPROBLEM=built_in_pgens`, set `pgen_name` to one of the
built-in dust generators listed above. Parameters such as `rho0`, perturbation
amplitudes, NSH mass fractions (`eps_1`, ...), particle placement, and random seeds are
pgen-specific. The tracked files in [`inputs/`](inputs/) are complete runnable
examples and should be treated as the reference configurations.

### Current limitations

- Dust particles require a 2-D or 3-D Hydro mesh; 1-D and MHD are not supported.
- SMR/AMR is not yet supported by the dust module.
- Gas back-reaction currently requires an isothermal EOS.
- Ordinary periodic boundaries are supported. The 3-D shear-periodic path is incomplete:
  the conservative remap of deposited fields is not implemented, and coupled solvers
  other than `local` explicitly reject shear-periodic radial boundaries.
- Particle data are not currently stored in AthenaK restart files, so long particle
  runs cannot yet be resumed from a standard restart.
