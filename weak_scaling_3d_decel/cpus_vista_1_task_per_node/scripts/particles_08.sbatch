#!/bin/bash
#SBATCH --job-name=particle_scaling_08
#SBATCH --output=/work/10737/clolds/vista/athenak_dust/athenak_dust_tests/weak_scaling_3d_decel/cpus_vista_1_task_per_node/logs/n_08.%j.out
#SBATCH --error=/work/10737/clolds/vista/athenak_dust/athenak_dust_tests/weak_scaling_3d_decel/cpus_vista_1_task_per_node/logs/n_08.%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gg
#SBATCH --account=AST25015

cd /work/10737/clolds/vista/athenak_dust

ibrun ./athenak/build-mpi/src/athena \
    -i ./athenak_dust_tests/weak_scaling_3d_decel/uniform_decel.athinput \
    -d ./athenak_dust_tests/weak_scaling_3d_decel/cpus_vista_1_task_per_node/logs \
    mesh/nx1=32 \
    mesh/nx2=128 \
    mesh/nx3=64 \
