#!/bin/bash
#SBATCH --job-name=particle_scaling_01
#SBATCH --output=/home1/10737/clolds/athenak_dust/athenak_dust_tests/weak_scaling_3d_decel/gpus_horizon/logs/n_02.%j.out
#SBATCH --error=/home1/10737/clolds/athenak_dust/athenak_dust_tests/weak_scaling_3d_decel/gpus_horizon/logs/n_02.%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --partition=debug
#SBATCH --account=AST25015

cd /home1/10737/clolds/athenak_dust/

ibrun ./athenak/build-gpu-mpi/src/athena \
    -i ./athenak_dust_tests/weak_scaling_3d_decel/uniform_decel.athinput \
    -d ./athenak_dust_tests/weak_scaling_3d_decel/gpus_horizon/logs \
    mesh/nx1=256 \
    mesh/nx2=512 \
    mesh/nx3=256 \
