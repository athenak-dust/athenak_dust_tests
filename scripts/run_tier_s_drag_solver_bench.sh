#!/usr/bin/env bash
# Run the frozen Tier-S drag-solver performance matrix.
#
# This script writes only beneath an explicit output directory.  It never writes
# benchmark artifacts into the source repository.  The two AthenaK executables
# normally differ only in their compiled problem generator:
#
#   general binary: default/general test pgens (damping and streaming_linear)
#   clump binary:   PROBLEM=tests/dust_snapshot_clump
#
# Mac MPI-8 example:
#
#   scripts/run_tier_s_drag_solver_bench.sh mac-mpi8 \
#     /tmp/build-general/src/athena /tmp/build-clump/src/athena inputs \
#     /path/to/BA2.prtcl_all.00121.part.vtk /tmp/tier_s_results
#
# Single-GPU example (WSL2 or a cluster GPU node):
#
#   scripts/run_tier_s_drag_solver_bench.sh single-gpu \
#     /path/to/athena_gpu_general /path/to/athena_gpu_clump /path/to/inputs \
#     /path/to/BA2.prtcl_all.00121.part.vtk /path/to/results

set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: $0 mac-mpi8|single-gpu GENERAL_BIN CLUMP_BIN INPUT_DIR SNAPSHOT_VTK OUTPUT_DIR" >&2
  exit 2
fi

platform=$1
general_bin=$(realpath "$2")
clump_bin=$(realpath "$3")
input_dir=$(realpath "$4")
snapshot_vtk=$(realpath "$5")
output_dir=$6

case "$platform" in
  mac-mpi8)
    export PATH=/opt/local/bin:$PATH
    launch=(/opt/local/bin/mpiexec -n 8)
    ;;
  single-gpu|wsl-gpu)
    launch=()
    ;;
  *)
    echo "unknown platform '$platform'; expected mac-mpi8 or single-gpu" >&2
    exit 2
    ;;
esac

for required in "$general_bin" "$clump_bin" "$snapshot_vtk"; do
  if [[ ! -f "$required" ]]; then
    echo "required file does not exist: $required" >&2
    exit 2
  fi
done

mkdir -p "$output_dir/general" "$output_dir/clump"
output_dir=$(realpath "$output_dir")

run_case() {
  local binary=$1
  local input=$2
  local case_name=$3
  local mode=$4
  local repetition=$5
  local cycles=$6
  local subdir=$7
  shift 7

  local basename="bench_${case_name}_${mode}_r${repetition}"
  local log="$output_dir/$subdir/${basename}.log"
  (
    cd "$output_dir/$subdir"
    "${launch[@]}" "$binary" -i "$input" time/nlim="$cycles" \
      job/basename="$basename" dust/drag_solver="$mode" "$@"
  ) > "$log" 2>&1

  # AthenaK prints this exact final line after reaching nlim.  Checking it here
  # prevents an input-file tlim from silently shortening a timing sample.
  if ! grep -Eq "^time=[0-9.eE+-]+ cycle=${cycles}[[:space:]]*$" "$log"; then
    echo "run did not reach cycle ${cycles}: $log" >&2
    exit 1
  fi
}

for repetition in 1 2 3; do
  for mode in local applya dc1 dc2 pcg adaptive; do
    run_case "$general_bin" "$input_dir/dust_damping.athinput" damping "$mode" \
      "$repetition" 2000 general meshblock/nx1=8 meshblock/nx2=8 particles/ppc=3 \
      time/tlim=100000 output1/dt=100000

    run_case "$general_bin" "$input_dir/streaming_linA_128.athinput" lina "$mode" \
      "$repetition" 2000 general mesh/nx1=32 mesh/nx2=32 meshblock/nx1=8 \
      meshblock/nx2=8 output1/dcycle=100000

    run_case "$clump_bin" "$input_dir/dust_snapshot_clump_tier_s.athinput" ordinary \
      "$mode" "$repetition" 500 clump problem/snapshot_file="$snapshot_vtk" \
      output1/dcycle=100000

    run_case "$clump_bin" "$input_dir/dust_snapshot_clump_tier_s_stiff.athinput" stiff \
      "$mode" "$repetition" 500 clump problem/snapshot_file="$snapshot_vtk" \
      output1/dcycle=100000

    run_case "$clump_bin" "$input_dir/dust_snapshot_migration_tier_s.athinput" migration \
      "$mode" "$repetition" 500 clump problem/snapshot_file="$snapshot_vtk" \
      output1/dcycle=100000
  done
done

echo "Tier-S benchmark complete: $output_dir"
echo "Analyze with: python scripts/analyze_tier_s_drag_solver_bench.py --root '$output_dir' --platform '$platform'"
