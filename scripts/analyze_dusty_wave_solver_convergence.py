#!/usr/bin/env python3
"""Report spatial convergence for the dusty-wave drag-solver ladder.

Expected files are named ``wave_MODE_nN-errs.dat`` beneath ``--root``.  The
error file is written by AthenaK's dusty-wave problem generator; column nine is
the total gas-density + gas-velocity + particle-velocity L1 error.  The tested
ladder uses N=32,64,128,256, four transverse cells, CFL-limited timesteps, and a
fixed final time.  Thus it measures the convergence of the complete
space-and-time discretization rather than a spatial operator in isolation.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path


MODES = ("local", "dc1", "pcg", "adaptive")
RESOLUTIONS = (32, 64, 128, 256)


def last_data_row(path: Path) -> list[float]:
    rows = [line for line in path.read_text().splitlines() if line and not line.startswith("#")]
    if not rows:
        raise RuntimeError(f"no data rows in {path}")
    return [float(value) for value in rows[-1].split()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    print("| mode | e32 | e64 | e128 | e256 | p32-64 | p64-128 | p128-256 |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for mode in MODES:
        errors = []
        for resolution in RESOLUTIONS:
            row = last_data_row(args.root / f"wave_{mode}_n{resolution}-errs.dat")
            if int(row[0]) != resolution:
                raise RuntimeError(f"resolution mismatch for mode={mode}, N={resolution}")
            errors.append(row[8])
        orders = [math.log(errors[index] / errors[index + 1], 2.0) for index in range(3)]
        values = " | ".join(f"{value:.8e}" for value in errors)
        slopes = " | ".join(f"{value:.6f}" for value in orders)
        print(f"| {mode} | {values} | {slopes} |")


if __name__ == "__main__":
    main()
