#!/usr/bin/env python3
"""Measure the TSC particle-density evolution of Johansen Run AB."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = sorted((args.run_dir / "pvtk").glob("*.part.vtk"))
    if not files:
        raise FileNotFoundError(f"No particle VTK files in {args.run_dir / 'pvtk'}")

    expected_particles = int(round(args.ppc * args.nx * args.nx))
    bounds = (-args.eta_r, args.eta_r, -args.eta_r, args.eta_r)
    times: list[float] = []
    means: list[float] = []
    rms: list[float] = []
    maxima: list[float] = []
    for path in files:
        time, points = read_particle_points(path)
        if len(points) != expected_particles:
            raise ValueError(
                f"{path}: found {len(points)} particles, expected {expected_particles}"
            )
        density = tsc_density(points, args.nx, bounds, args.epsilon, args.ppc)
        mean = float(density.mean())
        fluctuation = float(np.sqrt(np.mean((density - mean) ** 2)))
        maximum = float(density.max())
        times.append(time)
        means.append(mean)
        rms.append(fluctuation)
        maxima.append(maximum)
        print(
            f"t={time:8.4f} mean={mean:.10g} rms={fluctuation:.10g} "
            f"max={maximum:.10g}",
            flush=True,
        )

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 6.5), sharex=True)
    axes[0].plot(times, maxima, color="#d62728", linewidth=2.0)
    axes[0].set_ylabel(r"$\max(\rho_p/\langle\rho_p\rangle)$")
    axes[1].plot(times, rms, color="#1f4ed8", linewidth=2.0)
    axes[1].set_ylabel(r"${\rm RMS}(\rho_p/\langle\rho_p\rangle)$")
    axes[1].set_xlabel(r"$t\,\Omega$")
    for axis in axes:
        axis.axvline(6.0, color="0.55", linestyle="--", linewidth=1.0)
        axis.axvline(12.0, color="0.55", linestyle="--", linewidth=1.0)
        axis.grid(alpha=0.25)
    fig.suptitle(r"Johansen et al. (2007) Run AB: AthenaK TSC density")
    fig.tight_layout()

    output = args.output or args.run_dir / "run_AB_density_evolution.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output,
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
        transparent=False,
    )
    print(f"wrote {output}")

    data_output = output.with_suffix(".npz")
    np.savez_compressed(
        data_output,
        times=np.array(times),
        mean_density=np.array(means),
        rms_density=np.array(rms),
        maximum_density=np.array(maxima),
    )
    print(f"wrote {data_output}")


if __name__ == "__main__":
    main()
