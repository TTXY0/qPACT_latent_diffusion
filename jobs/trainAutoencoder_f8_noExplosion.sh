#!/bin/bash
#----------------------------------------------------
# Sample Slurm job script
#   for TACC Lonestar6 AMD Milan nodes
#
#   *** Serial Job in Normal Queue***
# 
# Last revised: October 22, 2021
#
# Notes:
#
#  -- Copy/edit this script as desired.  Launch by executing
#     "sbatch milan.serial.slurm" on a Lonestar6 login node.
#
#  -- Serial codes run on a single node (upper case N = 1).
#       A serial code ignores the value of lower case n,
#       but slurm needs a plausible value to schedule the job.
#
#  -- Use TACC's launcher utility to run multiple serial 
#       executables at the same time, execute "module load launcher" 
#       followed by "module help launcher".
#----------------------------------------------------

#SBATCH -J kl-f8           # Job name
#SBATCH -o kl-f8_noExplosion.o%j       # Name of stdout output file
#SBATCH -e kl-f8_noExplosion.e%j       # Name of stderr error file
#SBATCH -p gpu-a100          # Queue (partition) name
#SBATCH -N 1               # Total # of nodes (must be 1 for serial)
#SBATCH -n 1               # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 15:00:00        # Run time (hh:mm:ss)
#SBATCH --mail-type=all    # Send email at begin and end of job
#SBATCH -A FTA-SUB-Ghattas      # Project/Allocation name (req'd if you have more than 1)
#SBATCH --mail-user=twynn5403@gmail.com

# Any other commands must follow all #SBATCH directives...
module load tacc-apptainer


module list
pwd
date
export TRAIN_ENV=super
# Launch serial code...
./myprogram         # Do not use ibrun or any other MPI launcher
cd /scratch/10122/thomaswynn7394/latentDiffusion/autoencoderTraining
apptainer exec --nv /scratch/10122/thomaswynn7394/containers/ml_thomas_latest.sif \
    bash -c "pip install --user matplotlib --exists-action i -q && python3 /scratch/10122/thomaswynn7394/latentDiffusion/autoencoderTraining/training/trainAutoencoder_f8_noExplosion.py"
