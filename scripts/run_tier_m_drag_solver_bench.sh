#!/usr/bin/env bash
# Run Tier-M items 1 and 2 on one GPU.
#
# Item 1: linA, 128 cells per wavelength, 2,000 cycles.
# Item 2: JY07 Run AB, 256^2 and 25 PPC, 100 cycles from the smooth t=0 state.
#
# Each solver/case pair is a fresh process and is repeated three times.  The
# script writes only under OUTPUT_DIR and verifies that every run reaches its
# requested cycle count.  History/particle file output is disabled by the
# benchmark inputs so I/O does not dominate the timing window.

set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 GENERAL_BIN INPUT_DIR OUTPUT_DIR" >&2
  exit 2
fi

general_bin=$(realpath "$1")
input_dir=$(realpath "$2")
output_dir=$3

for required in \
  "$general_bin" \
  "$input_dir/streaming_linA_128.athinput" \
  "$input_dir/streaming_AB_johansen2007_tier_m.athinput"; do
  if [[ ! -f "$required" ]]; then
    echo "required file does not exist: $required" >&2
    exit 2
  fi
done

mkdir -p "$output_dir/lina" "$output_dir/ab"
output_dir=$(realpath "$output_dir")

run_case() {
  local input=$1
  local case_name=$2
  local mode=$3
  local repetition=$4
  local cycles=$5
  shift 5

  local basename="bench_${case_name}_${mode}_r${repetition}"
  local log="$output_dir/$case_name/${basename}.log"
  (
    cd "$output_dir/$case_name"
    "$general_bin" -i "$input" time/nlim="$cycles" time/tlim=100000 \
      job/basename="$basename" dust/drag_solver="$mode" "$@"
  ) > "$log" 2>&1

  if ! grep -Eq "^time=[0-9.eE+-]+ cycle=${cycles}[[:space:]]*$" "$log"; then
    echo "run did not reach cycle ${cycles}: $log" >&2
    exit 1
  fi
}

for repetition in 1 2 3; do
  for mode in local dc1 dc2 pcg adaptive; do
    run_case "$input_dir/streaming_linA_128.athinput" lina "$mode" \
      "$repetition" 2000 output1/dcycle=-1 meshblock/nx1=16 meshblock/nx2=16

    run_case "$input_dir/streaming_AB_johansen2007_tier_m.athinput" ab "$mode" \
      "$repetition" 100
  done
done

echo "Tier-M items 1 and 2 complete: $output_dir"
