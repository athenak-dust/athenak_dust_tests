#!/usr/bin/env python3
"""Plot Bai & Stone (2010) particle-density convergence PDFs.

Particle snapshots contain positions rather than an assigned dust density.  For
each snapshot this script deposits equal particle masses onto the native periodic
mesh with TSC, gathers that density back to every particle with the same TSC
weights, and accumulates the particle-weighted cumulative distribution
P(rho_p at particle > threshold).  By default this reproduces the Run-BA
statistic in the lower-left panel of Bai & Stone (2010), Figure 6; command-line
options also support Run AB and other periodic square boxes.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np


TIME_RE = re.compile(rb"time=\s*([+\-0-9.eE]+)")


@dataclass(frozen=True)
class RunSpec:
    label: str
    nx: int
    run_dir: Path


def parse_run_spec(text: str) -> RunSpec:
    """Parse LABEL:NX:RUN_DIR while allowing colons inside RUN_DIR."""
    fields = text.split(":", 2)
    if len(fields) != 3:
        raise argparse.ArgumentTypeError("run must have the form LABEL:NX:RUN_DIR")
    label, nx_text, run_dir = fields
    try:
        nx = int(nx_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid grid size in {text!r}") from exc
    return RunSpec(label=label, nx=nx, run_dir=Path(run_dir).expanduser())


def read_particle_time(path: Path) -> float:
    """Read only the time metadata from an AthenaK legacy particle VTK file."""
    with path.open("rb") as stream:
        stream.readline()
        metadata = stream.readline()
    match = TIME_RE.search(metadata)
    if match is None:
        raise ValueError(f"Could not find time in {path}")
    return float(match.group(1))


def read_particle_points(path: Path) -> tuple[float, np.ndarray]:
    """Read time and point coordinates from an AthenaK legacy particle VTK file."""
    with path.open("rb") as stream:
        line = stream.readline()
        metadata = stream.readline()
        match = TIME_RE.search(metadata)
        if match is None:
            raise ValueError(f"Could not find time in {path}")
        time = float(match.group(1))

        while line and not line.startswith(b"POINTS "):
            line = stream.readline()
        if not line:
            raise ValueError(f"Could not find POINTS section in {path}")
        count = int(line.split()[1])
        points = np.fromfile(stream, dtype=">f4", count=3 * count)
        if points.size != 3 * count:
            raise ValueError(f"Truncated POINTS section in {path}")
    return time, points.reshape(count, 3)


def tsc_axis(
    coordinates: np.ndarray, minimum: float, maximum: float, nx: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return periodic cell indices and TSC weights for one coordinate axis."""
    length = maximum - minimum
    dx = length / nx
    wrapped = (coordinates.astype(np.float64) - minimum) % length + minimum
    index = np.floor((wrapped - minimum) / dx).astype(np.int64)
    delta = (wrapped - (minimum + (index + 0.5) * dx)) / dx
    weights = np.stack(
        (
            0.5 * (0.5 - delta) ** 2,
            0.75 - delta**2,
            0.5 * (0.5 + delta) ** 2,
        )
    )
    return index, weights


def ambient_density_at_particles(
    points: np.ndarray,
    nx: int,
    ppc: float,
    bounds: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Deposit normalized TSC density and gather it back to particle locations."""
    xmin, xmax, zmin, zmax = bounds
    ix, wx = tsc_axis(points[:, 0], xmin, xmax, nx)
    iz, wz = tsc_axis(points[:, 1], zmin, zmax, nx)

    cell_counts = np.zeros(nx * nx, dtype=np.float64)
    offsets = (-1, 0, 1)
    for az, oz in enumerate(offsets):
        target_z = (iz + oz) % nx
        for ax, ox in enumerate(offsets):
            target_x = (ix + ox) % nx
            flat_index = target_x + nx * target_z
            cell_counts += np.bincount(
                flat_index,
                weights=wz[az] * wx[ax],
                minlength=nx * nx,
            )

    # Equal particle masses give <rho_p> when the mean assigned count is ppc.
    normalized_grid = cell_counts.reshape(nx, nx) / ppc
    ambient = np.zeros(points.shape[0], dtype=np.float64)
    for az, oz in enumerate(offsets):
        target_z = (iz + oz) % nx
        for ax, ox in enumerate(offsets):
            target_x = (ix + ox) % nx
            ambient += wz[az] * wx[ax] * normalized_grid[target_z, target_x]
    return normalized_grid, ambient


def scheduled_snapshots(
    run_dir: Path, time_min: float, time_max: float, cadence: float
) -> dict[float, Path]:
    """Map nominal scheduled times to the nearest particle snapshot."""
    files = sorted((run_dir / "pvtk").glob("*.part.vtk"))
    if not files:
        raise FileNotFoundError(f"No particle VTK files in {run_dir / 'pvtk'}")

    selected: dict[float, tuple[float, Path]] = {}
    for path in files:
        time = read_particle_time(path)
        target = cadence * round(time / cadence)
        if target < time_min - 0.5 * cadence or target > time_max + 0.5 * cadence:
            continue
        if abs(time - target) > 0.25 * cadence:
            continue
        mismatch = abs(time - target)
        if target not in selected or mismatch < selected[target][0]:
            selected[target] = (mismatch, path)
    return {target: value[1] for target, value in selected.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        type=parse_run_spec,
        help="run specification LABEL:NX:RUN_DIR; repeat for every curve",
    )
    parser.add_argument("--output", type=Path, required=True, help="output PNG path")
    parser.add_argument("--data-output", type=Path, help="output NPZ path")
    parser.add_argument("--time-min", type=float, default=300.0)
    parser.add_argument("--time-max", type=float, default=1500.0)
    parser.add_argument("--cadence", type=float, default=10.0)
    parser.add_argument("--ppc", type=float, default=9.0)
    parser.add_argument("--eta-r", type=float, default=0.05)
    parser.add_argument(
        "--half-box",
        type=float,
        help="physical half-width of the square box (default: 20*eta-r for Run BA)",
    )
    parser.add_argument("--run-name", default="BA", help="run label shown in the plot")
    parser.add_argument("--log-density-min", type=float, default=-1.0)
    parser.add_argument("--log-density-max", type=float, default=4.0)
    parser.add_argument("--density-bins", type=int, default=440)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snapshots = {
        spec.label: scheduled_snapshots(
            spec.run_dir, args.time_min, args.time_max, args.cadence
        )
        for spec in args.run
    }
    common_times = sorted(set.intersection(*(set(item) for item in snapshots.values())))
    if not common_times:
        raise ValueError("The requested runs have no common scheduled snapshot times")

    expected_times = np.arange(
        max(args.time_min, common_times[0]),
        min(args.time_max, common_times[-1]) + 0.5 * args.cadence,
        args.cadence,
    )
    missing = [time for time in expected_times if time not in common_times]
    if missing:
        raise ValueError(f"Missing common scheduled snapshots: {missing}")
    common_times = [float(time) for time in expected_times]

    half_box = args.half_box if args.half_box is not None else 20.0 * args.eta_r
    bounds = (-half_box, half_box, -half_box, half_box)
    log_edges = np.linspace(
        args.log_density_min, args.log_density_max, args.density_bins + 1
    )
    density_threshold = 10.0 ** log_edges[:-1]

    cdfs: list[np.ndarray] = []
    maximum_density: list[float] = []
    particle_counts: list[int] = []
    for spec in args.run:
        histogram = np.zeros(args.density_bins, dtype=np.int64)
        samples = 0
        run_maximum = 0.0
        expected_particles = int(round(args.ppc * spec.nx * spec.nx))
        for target in common_times:
            path = snapshots[spec.label][target]
            actual_time, points = read_particle_points(path)
            if len(points) != expected_particles:
                raise ValueError(
                    f"{path}: found {len(points)} particles, expected {expected_particles}"
                )
            grid, ambient = ambient_density_at_particles(
                points, spec.nx, args.ppc, bounds
            )
            if not np.isclose(grid.mean(), 1.0, rtol=5.0e-12, atol=5.0e-12):
                raise ValueError(f"TSC mass check failed for {path}: mean={grid.mean()}")

            run_maximum = max(run_maximum, float(ambient.max()))
            log_ambient = np.log10(ambient)
            # Include rare out-of-range samples in the endpoint bins.
            log_ambient = np.clip(
                log_ambient,
                np.nextafter(log_edges[0], log_edges[-1]),
                np.nextafter(log_edges[-1], log_edges[0]),
            )
            histogram += np.histogram(log_ambient, bins=log_edges)[0]
            samples += len(ambient)
            print(
                f"{spec.label:>4s} target={target:6.1f} actual={actual_time:10.6f} "
                f"mean={grid.mean():.12f} max_at_particles={ambient.max():10.4f}",
                flush=True,
            )

        cumulative = np.cumsum(histogram[::-1], dtype=np.int64)[::-1] / samples
        cdfs.append(cumulative)
        maximum_density.append(run_maximum)
        particle_counts.append(expected_particles)
        print(
            f"{spec.label}: {len(common_times)} snapshots, {samples} particle samples, "
            f"maximum rho_p/<rho_p>={run_maximum:.6g}",
            flush=True,
        )

    colors = ("#00d923", "#e632e6", "#ed1c24", "#2450e6", "#111111")
    fig, axis = plt.subplots(figsize=(7.0, 5.7))
    for index, (spec, cdf) in enumerate(zip(args.run, cdfs)):
        color = colors[index % len(colors)]
        positive = cdf > 0.0
        axis.loglog(
            density_threshold[positive],
            cdf[positive],
            color=color,
            linewidth=2.0,
            label=rf"${spec.nx}^2$",
        )
    # Preserve the Bai (2010)-like range for ordinary cases, but do not clip
    # a higher-density tail when a run produces one.
    axis.set_xlim(1.0e-1, max(2.0e3, 1.25 * max(maximum_density)))
    axis.set_ylim(1.0e-5, 1.0)
    axis.set_xlabel(r"$\rho_p/\langle\rho_p\rangle$")
    axis.set_ylabel(r"$P(>\rho_p)$")
    axis.text(
        0.055,
        0.92,
        f"Run {args.run_name}\n"
        + rf"$N_{{\rm pc}}={args.ppc:g}$"
        + "\n"
        + rf"$t={common_times[0]:.0f}$--${common_times[-1]:.0f}\,\Omega^{{-1}}$",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=12,
    )
    axis.legend(loc="lower left", frameon=True, framealpha=1.0)
    axis.grid(which="major", color="0.88", linewidth=0.7)
    axis.tick_params(which="both", direction="in", top=True, right=True)
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        args.output,
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
        transparent=False,
    )
    print(f"wrote {args.output}")

    data_output = args.data_output or args.output.with_suffix(".npz")
    data_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        data_output,
        labels=np.array([spec.label for spec in args.run]),
        nx=np.array([spec.nx for spec in args.run]),
        ppc=float(args.ppc),
        half_box=float(half_box),
        run_name=np.array(args.run_name),
        times=np.array(common_times),
        density_threshold=density_threshold,
        cumulative_probability=np.stack(cdfs),
        maximum_density=np.array(maximum_density),
        particles_per_snapshot=np.array(particle_counts),
    )
    print(f"wrote {data_output}")


if __name__ == "__main__":
    main()
