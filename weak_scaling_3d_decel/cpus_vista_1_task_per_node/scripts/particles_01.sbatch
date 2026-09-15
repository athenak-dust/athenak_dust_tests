#!/bin/bash
#SBATCH --job-name=particle_scaling_01
#SBATCH --output=/work/10737/clolds/vista/athenak_dust/athenak_dust_tests/weak_scaling_3d_decel/cpus_vista_1_task_per_node/logs/n_01.%j.out
#SBATCH --error=/work/10737/clolds/vista/athenak_dust/athenak_dust_tests/weak_scaling_3d_decel/cpus_vista_1_task_per_node/logs/n_01.%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gg
#SBATCH --account=AST25015

cd /work/10737/clolds/vista/athenak_dust

ibrun ./athenak/build-gpu-mpi/src/athena \
    -i ./athenak_dust_tests/weak_scaling_3d_decel/uniform_decel.athinput \
    -d ./athenak_dust_tests/weak_scaling_3d_decel/cpus_vista_1_task_per_node/logs \
    mesh/nx1=32 \
    mesh/nx2=32 \
    mesh/nx3=32 \
