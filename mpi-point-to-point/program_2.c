#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

int main(int argc, char *argv[])
{
    int rank, nproc, num;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0)
    {
        srand(time(NULL));
        num = rand() % 100;
        printf("Process %d generated number: %d\n", rank, num);
        MPI_Send(&num, 1, MPI_INT, (rank + 1) % nproc, 123, MPI_COMM_WORLD);

        MPI_Recv(&num, 1, MPI_INT, nproc - 1, 123, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process %d received final number: %d\n", rank, num);
    }

    if (rank > 0)
    {
        MPI_Recv(&num, 1, MPI_INT, rank - 1, 123, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process %d received number: %d\n", rank, num);
        MPI_Send(&num, 1, MPI_INT, (rank + 1) % nproc, 123, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}