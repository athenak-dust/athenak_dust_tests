#!/bin/bash
#SBATCH --job-name=particle_scaling_01
#SBATCH --output=/work/10737/clolds/vista/athenak_dust/athenak_dust_tests/weak_scaling_3d_decel/gpus_vista/logs/pc2_n_04.%j.out
#SBATCH --error=/work/10737/clolds/vista/athenak_dust/athenak_dust_tests/weak_scaling_3d_decel/gpus_vista/logs/pc2_n_04.%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gh
#SBATCH --account=AST25015

cd /work/10737/clolds/vista/athenak_dust

module purge
module load nvidia/24.7
module load openmpi/5.0.5

ibrun ./athenak/build-gpu-mpi/src/athena \
    -i ./athenak_dust_tests/weak_scaling_3d_decel/uniform_decel.athinput \
    -d ./athenak_dust_tests/weak_scaling_3d_decel/gpus_vista/logs \
    mesh/nx1=256 \
    mesh/nx2=512 \
    mesh/nx3=512 \
    time/integrator=rk2 \
    dust/coupling=pc2 \
