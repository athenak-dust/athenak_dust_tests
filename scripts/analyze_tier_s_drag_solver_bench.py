#!/usr/bin/env python3
"""Validate and summarize Tier-S drag-solver benchmark logs.

The runner stores logs as ``general/bench_CASE_MODE_rN.log`` for the damping
and linA cases, and ``clump/bench_CASE_MODE_rN.log`` for the three snapshot
cases.  This analyzer deliberately fails if a run stopped before its frozen
cycle count.  That guard caught an early benchmark attempt in which the
damping problem's input-file ``tlim`` ended the run after 81 rather than 2000
cycles.

Example:

    ~/projects/venvs/asf/bin/python scripts/analyze_tier_s_drag_solver_bench.py \
      --root /tmp/athenak_drag_solver_bench_mac --platform "Mac MPI-8"

The output is a Markdown table.  ``applya`` is a diagnostic mode that executes
and discards one real matrix-free operator application; it is not a distinct
physical drag update.  Ratios use the same-build local mode for the same case,
so they isolate solver overhead from machine-to-machine differences.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import statistics


CASES = {
    "damping": ("general", 2000, 4000),
    "lina": ("general", 2000, 4000),
    "ordinary": ("clump", 500, 1000),
    "stiff": ("clump", 500, 1000),
    "migration": ("clump", 500, 1000),
}
MODES = ("local", "applya", "dc1", "pcg", "adaptive")
REPETITIONS = (1, 2, 3)
TIME_RE = re.compile(r"cpu time used\s*=\s*([0-9.eE+-]+)")
CYCLE_RE = re.compile(r"^time=[0-9.eE+-]+ cycle=([0-9]+)\s*$", re.MULTILINE)
SUMMARY_RE = re.compile(r"^# DUST_SOLVER_SUMMARY\s+(.+)$", re.MULTILINE)


@dataclass(frozen=True)
class Run:
    seconds: float
    summary: dict[str, str]


def parse_summary(line: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for token in line.split():
        key, value = token.split("=", 1)
        fields[key] = value
    return fields


def read_run(path: Path, expected_cycle: int, expected_stages: int, mode: str) -> Run:
    if not path.is_file():
        raise RuntimeError(f"missing benchmark log: {path}")
    text = path.read_text()
    times = TIME_RE.findall(text)
    cycles = CYCLE_RE.findall(text)
    summaries = SUMMARY_RE.findall(text)
    if len(times) != 1:
        raise RuntimeError(f"expected one timing record in {path}, found {len(times)}")
    if not cycles or int(cycles[-1]) != expected_cycle:
        found = cycles[-1] if cycles else "none"
        raise RuntimeError(f"{path} ended at cycle {found}, expected {expected_cycle}")
    if mode == "local":
        if summaries:
            raise RuntimeError(f"unexpected coupled-solver summary in local run {path}")
        summary: dict[str, str] = {}
    else:
        if len(summaries) != 1:
            raise RuntimeError(f"expected one solver summary in {path}, found {len(summaries)}")
        summary = parse_summary(summaries[0])
        if int(summary["stages"]) != expected_stages:
            raise RuntimeError(
                f"{path} executed {summary['stages']} stages, expected {expected_stages}"
            )
        if summary["mode"] != mode:
            raise RuntimeError(f"{path} reports mode={summary['mode']}, expected {mode}")
    return Run(float(times[0]), summary)


def median_summary_value(runs: list[Run], key: str) -> float | None:
    values = [float(run.summary[key]) for run in runs if key in run.summary]
    return statistics.median(values) if values else None


def fmt(value: float | None, digits: int = 3) -> str:
    return "--" if value is None else f"{value:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="directory containing general/ and clump/")
    parser.add_argument("--platform", required=True, help="human-readable table heading")
    args = parser.parse_args()

    all_runs: dict[tuple[str, str], list[Run]] = {}
    for case, (subdir, cycles, stages) in CASES.items():
        for mode in MODES:
            all_runs[(case, mode)] = [
                read_run(
                    args.root / subdir / f"bench_{case}_{mode}_r{rep}.log",
                    cycles,
                    stages,
                    mode,
                )
                for rep in REPETITIONS
            ]

    print(f"### {args.platform}\n")
    print("| case | mode | median wall (s) | min--max (s) | / local | fast accept | PCG iter median/p95/max |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for case in CASES:
        local_seconds = statistics.median(run.seconds for run in all_runs[(case, "local")])
        for mode in MODES:
            runs = all_runs[(case, mode)]
            values = [run.seconds for run in runs]
            median_seconds = statistics.median(values)
            fast = median_summary_value(runs, "fast_accept")
            stages = median_summary_value(runs, "stages")
            fast_fraction = None
            if mode == "adaptive" and fast is not None and stages != 0:
                fast_fraction = fast / stages
            pcg_stages = median_summary_value(runs, "pcg_stages")
            iter_med = median_summary_value(runs, "pcg_iter_median")
            iter_p95 = median_summary_value(runs, "pcg_iter_p95")
            iter_max = median_summary_value(runs, "pcg_iter_max")
            iteration_text = "--"
            if pcg_stages is not None and pcg_stages > 0:
                iteration_text = f"{iter_med:.0f}/{iter_p95:.0f}/{iter_max:.0f}"
            print(
                f"| {case} | {mode} | {median_seconds:.6f} | "
                f"{min(values):.6f}--{max(values):.6f} | {median_seconds/local_seconds:.3f} | "
                f"{fmt(fast_fraction)} | {iteration_text} |"
            )


if __name__ == "__main__":
    main()
