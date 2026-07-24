#!/usr/bin/env python3
"""Reconstruct Johansen et al. (2007) Run BA particle-density snapshots.

The AthenaK particle VTK files contain positions but not a mesh-assigned dust
density. This script reads those positions and applies the same periodic TSC
kernel used by the dust drag module.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


TIME_RE = re.compile(rb"time=\s*([+\-0-9.eE]+)")


def read_particle_points(path: Path) -> tuple[float, np.ndarray]:
    """Read the time and point coordinates from an AthenaK legacy particle VTK file."""
    with path.open("rb") as stream:
        description = stream.readline()
        metadata = stream.readline()
        match = TIME_RE.search(metadata)
        if match is None:
            raise ValueError(f"Could not find time in {path}")
        time = float(match.group(1))

        line = description
        while line and not line.startswith(b"POINTS "):
            line = stream.readline()
        if not line:
            raise ValueError(f"Could not find POINTS section in {path}")
        count = int(line.split()[1])
        points = np.fromfile(stream, dtype=">f4", count=3 * count)
        if points.size != 3 * count:
            raise ValueError(f"Truncated POINTS section in {path}")
    return time, points.reshape(count, 3)


def tsc_density(
    points: np.ndarray,
    nx: int,
    bounds: tuple[float, float, float, float],
    epsilon: float,
    ppc: float,
    batch_size: int = 200_000,
) -> np.ndarray:
    """Assign equal-mass 2-D particles to a periodic cell-centered mesh with TSC."""
    xmin, xmax, zmin, zmax = bounds
    dx = (xmax - xmin) / nx
    dz = (zmax - zmin) / nx
    density = np.zeros(nx * nx, dtype=np.float64)

    for start in range(0, points.shape[0], batch_size):
        chunk = points[start : start + batch_size]
        x = (chunk[:, 0].astype(np.float64) - xmin) % (xmax - xmin) + xmin
        z = (chunk[:, 1].astype(np.float64) - zmin) % (zmax - zmin) + zmin

        ix = np.floor((x - xmin) / dx).astype(np.int64)
        iz = np.floor((z - zmin) / dz).astype(np.int64)
        delx = (x - (xmin + (ix + 0.5) * dx)) / dx
        delz = (z - (zmin + (iz + 0.5) * dz)) / dz
        wx = np.stack(
            (0.5 * (0.5 - delx) ** 2, 0.75 - delx**2, 0.5 * (0.5 + delx) ** 2)
        )
        wz = np.stack(
            (0.5 * (0.5 - delz) ** 2, 0.75 - delz**2, 0.5 * (0.5 + delz) ** 2)
        )

        for az, oz in enumerate((-1, 0, 1)):
            target_z = (iz + oz) % nx
            for ax, ox in enumerate((-1, 0, 1)):
                target_x = (ix + ox) % nx
                flat_index = target_x + nx * target_z
                density += np.bincount(
                    flat_index,
                    weights=wz[az] * wx[ax],
                    minlength=nx * nx,
                )

    # Each numerical particle has mass epsilon*rho_g*cell_volume/ppc.
    return (epsilon / ppc) * density.reshape(nx, nx)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="AthenaK run directory containing pvtk/")
    parser.add_argument("--output", type=Path, help="output PNG (default: RUN_DIR/run_BA_fig2.png)")
    parser.add_argument("--nx", type=int, default=256)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--ppc", type=float, default=25.0)
    parser.add_argument("--eta-r", type=float, default=0.05)
    parser.add_argument("--times", type=float, nargs="+", default=(40.0, 80.0, 120.0, 160.0))
    parser.add_argument("--save-fields", action="store_true", help="also save TSC fields as NPZ")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = sorted((args.run_dir / "pvtk").glob("*.part.vtk"))
    if not files:
        raise FileNotFoundError(f"No particle VTK files in {args.run_dir / 'pvtk'}")

    by_time: dict[float, tuple[Path, np.ndarray]] = {}
    for path in files:
        time, points = read_particle_points(path)
        by_time[time] = (path, points)

    selected: list[tuple[float, Path, np.ndarray]] = []
    for requested in args.times:
        time = min(by_time, key=lambda value: abs(value - requested))
        # AthenaK schedules an output after crossing its target time; it does not trim
        # the timestep to land exactly on intermediate output times.
        if abs(time - requested) > max(1.0e-2, 1.0e-5 * abs(requested)):
            raise ValueError(f"No snapshot at t={requested}; available times: {sorted(by_time)}")
        path, points = by_time[time]
        selected.append((time, path, points))

    half_box = 20.0 * args.eta_r
    bounds = (-half_box, half_box, -half_box, half_box)
    fields: list[np.ndarray] = []
    for time, path, points in selected:
        density = tsc_density(points, args.nx, bounds, args.epsilon, args.ppc)
        fields.append(density)
        rms = np.sqrt(np.mean((density - density.mean()) ** 2))
        print(
            f"t={time:6.1f}  file={path.name}  particles={len(points):d}  "
            f"mean={density.mean():.6g}  rms={rms:.6g}  max={density.max():.6g}"
        )

    colors = (
        (0.00, "#000000"),
        (0.04, "#000040"),
        (0.20, "#0000ff"),
        (0.43, "#ff0000"),
        (0.72, "#ffff00"),
        (1.00, "#ffffff"),
    )
    cmap = LinearSegmentedColormap.from_list("johansen2007", colors)
    extent = np.array(bounds) / args.eta_r
    ncols = 2
    nrows = (len(fields) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(9.0, 4.25 * nrows), squeeze=False)
    for axis, (time, _, _), density in zip(axes.flat, selected, fields):
        axis.imshow(
            density,
            origin="lower",
            extent=extent,
            interpolation="nearest",
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            aspect="equal",
        )
        axis.set_title(rf"$t={time:.1f}\,\Omega^{{-1}}$")
        axis.set_xlabel(r"$x/(\eta r)$")
        axis.set_ylabel(r"$z/(\eta r)$")
    for axis in axes.flat[len(fields) :]:
        axis.set_visible(False)
    fig.tight_layout()

    output = args.output or args.run_dir / "run_BA_fig2.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    print(f"wrote {output}")

    if args.save_fields:
        field_path = output.with_suffix(".npz")
        np.savez_compressed(
            field_path,
            times=np.array([item[0] for item in selected]),
            density=np.stack(fields),
            extent=extent,
        )
        print(f"wrote {field_path}")


if __name__ == "__main__":
    main()
