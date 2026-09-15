#!/bin/bash
#SBATCH --job-name=particle_scaling_32
#SBATCH --output=/work/10737/clolds/vista/athenak_dust/athenak_dust_tests/weak_scaling_3d_decel/gpus_vista/logs/n_32.%j.out
#SBATCH --error=/work/10737/clolds/vista/athenak_dust/athenak_dust_tests/weak_scaling_3d_decel/gpus_vista/logs/n_32.%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gh
#SBATCH --account=AST25015

cd /work/10737/clolds/vista/athenak_dust

ibrun ./athenak/build-gpu-mpi/src/athena \
    -i ./athenak_dust_tests/weak_scaling_3d_decel/uniform_decel.athinput \
    -d ./athenak_dust_tests/weak_scaling_3d_decel/gpus_vista/logs \
    mesh/nx1=256 \
    mesh/nx2=2048 \
    mesh/nx3=1024 \
