#!/bin/bash
#SBATCH -A ASC23013
#SBATCH -J compute_pi       # job name
#SBATCH -o output.log      # output log for this job
#SBATCH -e error.log       # will store error if any occurs
#SBATCH -N 1               # total number of nodes requested
#SBATCH -n 1               # total number of tasks requested
#SBATCH -p normal          # queue name normal or development
#SBATCH -t 00:05:00        # expected maximum runtime (hh:mm:ss)

date

module load gcc

# Information of system architecture details using hwloc
hwloc-ls -v output.svg

# Set maximum number of cores to 128
MAX_CORES=128

# Compare performance for different execution policies
for OMP_PLACES in cores sockets; do
    for OMP_PROC_BIND in close spread; do
        export OMP_PLACES=$OMP_PLACES
        export OMP_PROC_BIND=$OMP_PROC_BIND
        export OMP_NUM_THREADS=$MAX_CORES  
        ./compute_pi 10000000 > results_${OMP_PLACES}_${OMP_PROC_BIND}.log
    done
done

# Varying number of threads
export OMP_PLACES=cores
export OMP_PROC_BIND=close

for ((THREADS=1; THREADS <= MAX_CORES; THREADS*=2)); do
    export OMP_NUM_THREADS=$THREADS
    ./compute_pi 10000000 > results_threads_${THREADS}.log
done

# Static VS Dynamic Scheduling (Chunk Size: 10, 100, 1000)
export OMP_NUM_THREADS=$MAX_CORES

for CHUNK in 10 100 1000; do
    export OMP_SCHEDULE="static,$CHUNK"
    ./compute_pi 10000000 > results_static_${CHUNK}.log

    export OMP_SCHEDULE="dynamic,$CHUNK"
    ./compute_pi 10000000 > results_dynamic_${CHUNK}.log
done

date
