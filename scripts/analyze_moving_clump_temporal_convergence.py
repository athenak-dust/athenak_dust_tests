#!/usr/bin/env python3
"""Measure fixed-time convergence of moving-TSC clump history moments.

The run directory must contain a small-step strict-PCG reference named
``order_ref2_pcg`` and the three refinement rungs named
``order_METHOD_d000625``, ``..._d0003125``, and ``..._d00015625``.  The norm
combines gas x/y momentum and kinetic energy with dust x/y momentum.  It is a
compact end-to-end diagnostic; full particle/mesh state norms remain preferable
when restartable stage snapshots become available.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path


RUNS = (("d000625", 0.000625), ("d0003125", 0.0003125), ("d00015625", 0.00015625))
METHODS = ("local", "dc1", "pcg", "adaptive_fast")


def last_data(path: Path) -> list[float]:
    rows = [line for line in path.read_text().splitlines() if line and not line.startswith("#")]
    if not rows:
        raise RuntimeError(f"no history data in {path}")
    return [float(value) for value in rows[-1].split()]


def moments(root: Path, basename: str) -> list[float]:
    hydro = last_data(root / f"{basename}.hydro.hst")
    dust = last_data(root / f"{basename}.user.hst")
    # Both files begin with time, dt, and mass.  Hydro then stores momentum and
    # directional kinetic energy; the common dust columns always include momentum.
    return hydro[3:5] + hydro[6:8] + dust[3:5]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    reference = moments(args.root, "order_ref2_pcg")
    print("| method | error(dt) | error(dt/2) | error(dt/4) | p1 | p2 |")
    print("|---|---:|---:|---:|---:|---:|")
    for method in METHODS:
        errors = []
        for tag, _ in RUNS:
            state = moments(args.root, f"order_{method}_{tag}")
            errors.append(math.sqrt(sum((value - ref) ** 2 for value, ref in zip(state, reference))))
        orders = [math.log(errors[index] / errors[index + 1], 2.0) for index in range(2)]
        print(
            f"| {method} | {errors[0]:.9e} | {errors[1]:.9e} | {errors[2]:.9e} | "
            f"{orders[0]:.6f} | {orders[1]:.6f} |"
        )


if __name__ == "__main__":
    main()
