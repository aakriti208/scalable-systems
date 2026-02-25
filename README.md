# Scalable Systems

Parallel and distributed computing exercises using OpenMP and MPI.

## Projects

### OpenMP
- **openmp-hello-world** - Basic parallel programming with OpenMP
- **openmp-pi-computation** - Parallel Pi calculation with performance benchmarking

### MPI
- **mpi-point-to-point** - Basic send/receive communication and ring topology
- **mpi-collective-ops** - Collective operations (Allreduce, Reduce) for statistical computations
- **mpi-nonblocking-comm** - Non-blocking communication with Irecv and Waitany
- **mpi-parallel-io** - Collective parallel I/O with file views
- **mpi-producer-consumer** - Producer-consumer pattern with broker implementation

## Requirements

- OpenMP-compatible C compiler (gcc with -fopenmp)
- MPI implementation (OpenMPI or MPICH)

## Running

Compile with appropriate flags:
```bash
gcc -fopenmp program.c -o program      # For OpenMP
mpicc program.c -o program             # For MPI
mpirun -np <num_procs> ./program       # Run MPI programs
```
