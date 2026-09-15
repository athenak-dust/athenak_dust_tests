#!/bin/bash
#SBATCH --job-name=particle_scaling_16
#SBATCH --output=/work/10737/clolds/vista/athenak_dust/athenak_dust_tests/weak_scaling/gpus_vista/logs/n_16.%j.out
#SBATCH --error=/work/10737/clolds/vista/athenak_dust/athenak_dust_tests/weak_scaling/gpus_vista/logs/n_16.%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=16
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gh
#SBATCH --account=AST25015

cd /work/10737/clolds/vista/athenak_dust

ibrun -np 16 athenak/build-gpu-mpi/src/athena \
    -i athenak_dust_tests/weak_scaling/uniform_decel.athinput \
    -d athenak_dust_tests/weak_scaling/gpus_vista/logs \
    mesh/nx1=256 \
    mesh/nx2=1024 \
    mesh/nx3=1024 \
