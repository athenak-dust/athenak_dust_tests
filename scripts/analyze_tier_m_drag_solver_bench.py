#!/usr/bin/env python3
"""Validate and summarize Tier-M drag-solver items 1 and 2.

The expected matrix is three repetitions of five solvers for each case:

* ``lina``: 128 cells per wavelength, 2,000 cycles (4,000 implicit stages);
* ``ab``: JY07 Run AB at 256^2 and 25 PPC, 100 cycles (200 stages).

The analyzer fails closed on missing logs, incomplete cycle counts, missing solver
summaries, wrong modes, or wrong stage counts. Ratios are formed against the median
same-case ``local`` time from the same executable and allocation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import statistics


CASES = {
    "lina": (2000, 4000),
    "ab": (100, 200),
}
MODES = ("local", "dc1", "dc2", "pcg", "adaptive")
REPETITIONS = (1, 2, 3)
TIME_RE = re.compile(r"cpu time used\s*=\s*([0-9.eE+-]+)")
CYCLE_RE = re.compile(r"^time=[0-9.eE+-]+ cycle=([0-9]+)\s*$", re.MULTILINE)
ZONE_RATE_RE = re.compile(r"zone-cycles/cpu_second\s*=\s*([0-9.eE+-]+)")
PARTICLE_RATE_RE = re.compile(r"particle-updates/cpu_second\s*=\s*([0-9.eE+-]+)")
SUMMARY_RE = re.compile(r"^# DUST_SOLVER_SUMMARY\s+(.+)$", re.MULTILINE)


@dataclass(frozen=True)
class Run:
    seconds: float
    zone_rate: float
    particle_rate: float
    summary: dict[str, str]


def parse_summary(line: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for token in line.split():
        key, value = token.split("=", 1)
        fields[key] = value
    return fields


def exactly_one(pattern: re.Pattern[str], text: str, label: str, path: Path) -> str:
    matches = pattern.findall(text)
    if len(matches) != 1:
        raise RuntimeError(f"expected one {label} in {path}, found {len(matches)}")
    return matches[0]


def read_run(path: Path, expected_cycle: int, expected_stages: int, mode: str) -> Run:
    if not path.is_file():
        raise RuntimeError(f"missing benchmark log: {path}")
    text = path.read_text()
    seconds = float(exactly_one(TIME_RE, text, "timing record", path))
    zone_rate = float(exactly_one(ZONE_RATE_RE, text, "zone rate", path))
    particle_rate = float(exactly_one(PARTICLE_RATE_RE, text, "particle rate", path))
    cycles = CYCLE_RE.findall(text)
    if not cycles or int(cycles[-1]) != expected_cycle:
        found = cycles[-1] if cycles else "none"
        raise RuntimeError(f"{path} ended at cycle {found}, expected {expected_cycle}")

    summaries = SUMMARY_RE.findall(text)
    if mode == "local":
        if summaries:
            raise RuntimeError(f"unexpected coupled-solver summary in local run {path}")
        summary: dict[str, str] = {}
    else:
        if len(summaries) != 1:
            raise RuntimeError(f"expected one solver summary in {path}, found {len(summaries)}")
        summary = parse_summary(summaries[0])
        if summary.get("mode") != mode:
            raise RuntimeError(f"{path} reports mode={summary.get('mode')}, expected {mode}")
        if int(summary["stages"]) != expected_stages:
            raise RuntimeError(
                f"{path} executed {summary['stages']} stages, expected {expected_stages}"
            )
    return Run(seconds, zone_rate, particle_rate, summary)


def median_summary_value(runs: list[Run], key: str) -> float | None:
    values = [float(run.summary[key]) for run in runs if key in run.summary]
    return statistics.median(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="directory containing lina/ and ab/")
    parser.add_argument("--platform", required=True, help="human-readable platform label")
    args = parser.parse_args()

    all_runs: dict[tuple[str, str], list[Run]] = {}
    for case, (cycles, stages) in CASES.items():
        for mode in MODES:
            all_runs[(case, mode)] = [
                read_run(
                    args.root / case / f"bench_{case}_{mode}_r{repetition}.log",
                    cycles,
                    stages,
                    mode,
                )
                for repetition in REPETITIONS
            ]

    print(f"### {args.platform}\n")
    print("| case | mode | median wall (s) | min--max (s) | / local | particle updates/s | fast accept | PCG iter median/p95/max |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for case in CASES:
        local_seconds = statistics.median(run.seconds for run in all_runs[(case, "local")])
        for mode in MODES:
            runs = all_runs[(case, mode)]
            values = [run.seconds for run in runs]
            median_seconds = statistics.median(values)
            fast = median_summary_value(runs, "fast_accept")
            stages = median_summary_value(runs, "stages")
            fast_fraction = fast / stages if mode == "adaptive" and fast is not None and stages else None
            pcg_stages = median_summary_value(runs, "pcg_stages")
            iter_med = median_summary_value(runs, "pcg_iter_median")
            iter_p95 = median_summary_value(runs, "pcg_iter_p95")
            iter_max = median_summary_value(runs, "pcg_iter_max")
            iteration_text = "--"
            if pcg_stages is not None and pcg_stages > 0:
                iteration_text = f"{iter_med:.0f}/{iter_p95:.0f}/{iter_max:.0f}"
            fast_text = "--" if fast_fraction is None else f"{fast_fraction:.3f}"
            particle_rate = statistics.median(run.particle_rate for run in runs)
            print(
                f"| {case} | {mode} | {median_seconds:.6f} | "
                f"{min(values):.6f}--{max(values):.6f} | {median_seconds/local_seconds:.3f} | "
                f"{particle_rate:.6e} | {fast_text} | {iteration_text} |"
            )


if __name__ == "__main__":
    main()
