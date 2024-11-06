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

#SBATCH -J testMua           # Job name
#SBATCH -o testMua.o%j       # Name of stdout output file
#SBATCH -e testMua.e%j       # Name of stderr error file
#SBATCH -p gpu-a100          # Queue (partition) name
#SBATCH -N 1               # Total # of nodes (must be 1 for serial)
#SBATCH -n 1               # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 00:00:50        # Run time (hh:mm:ss)
#SBATCH --mail-type=all    # Send email at begin and end of job
#SBATCH -A CDA23008      # Project/Allocation name (req'd if you have more than 1)
#SBATCH --mail-user=thomaswynn7394@tacc.utexas.edu

# Any other commands must follow all #SBATCH directives...
module load tacc-apptainer


module list
pwd
date

# Launch serial code...
./myprogram         # Do not use ibrun or any other MPI launcher
cd /scratch/10122/thomaswynn7394/latentDiffusion
apptainer exec --nv /scratch/10122/thomaswynn7394/containers/ml_thomas_latest.sif python3 /scratch/10122/thomaswynn7394/latentDiffusion/testMuaSliaces.py
