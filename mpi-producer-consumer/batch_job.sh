#!/bin/bash
#SBATCH -A ASC23013             # Account name
#SBATCH -J noncontiguous_lab    # Job name
#SBATCH -o noncontiguous_lab.%j # Name of the output and error file
#SBATCH -N 1                    # Total number of nodes requested
#SBATCH -n 2                     # Total number of tasks requested
#SBATCH -p normal               # Queue name (normal or development)
#SBATCH -t 00:10:00             # Expected maximum runtime (hh:mm:ss)

# Print the date before and after the execution
date

# Compile the MPI program
mpicc -o noncontiguous_lab noncontiguous_lab.c

# Run the MPI program with 4 processes
mpirun -np 4 ./noncontiguous_lab

date