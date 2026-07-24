#!/usr/bin/env python3
"""Reconstruct the four Johansen et al. (2007) Run AB onset snapshots."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

from plot_run_BA import read_particle_points, tsc_density


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="AthenaK run directory containing pvtk/")
    parser.add_argument("--output", type=Path, help="output PNG")
    parser.add_argument("--nx", type=int, default=256)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--ppc", type=float, default=25.0)
    parser.add_argument("--eta-r", type=float, default=0.05)
    parser.add_argument("--times", type=float, nargs="+", default=(0.0, 6.0, 8.0, 12.0))
    parser.add_argument("--save-fields", action="store_true")
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
        if abs(time - requested) > max(1.0e-2, 1.0e-5 * abs(requested)):
            raise ValueError(f"No snapshot at t={requested}; available times: {sorted(by_time)}")
        path, points = by_time[time]
        selected.append((time, path, points))

    expected_particles = int(round(args.ppc * args.nx * args.nx))
    half_box = args.eta_r
    bounds = (-half_box, half_box, -half_box, half_box)
    fields: list[np.ndarray] = []
    for time, path, points in selected:
        if len(points) != expected_particles:
            raise ValueError(
                f"{path}: found {len(points)} particles, expected {expected_particles}"
            )
        density = tsc_density(points, args.nx, bounds, args.epsilon, args.ppc)
        fields.append(density)
        rms = np.sqrt(np.mean((density - density.mean()) ** 2))
        print(
            f"t={time:5.1f} file={path.name} particles={len(points):d} "
            f"mean={density.mean():.8g} rms={rms:.8g} max={density.max():.8g}"
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
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 8.5), squeeze=False)
    for axis, (time, _, _), density in zip(axes.flat, selected, fields):
        axis.imshow(
            density,
            origin="lower",
            extent=extent,
            interpolation="nearest",
            cmap=cmap,
            vmin=0.0,
            vmax=5.0,
            aspect="equal",
        )
        axis.set_title(rf"$t={time:.1f}\,\Omega^{{-1}}$")
        axis.set_xlabel(r"$x/(\eta r)$")
        axis.set_ylabel(r"$z/(\eta r)$")
    fig.tight_layout()

    output = args.output or args.run_dir / "run_AB_onset.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output,
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
        transparent=False,
    )
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
