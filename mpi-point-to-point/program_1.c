#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0)
    {
        int data = rank;
        MPI_Send(&data, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
        printf("Process %d sent data: %d to process 1\n", rank, data);
    }
    else if (rank == 1)
    {
        int received_data;
        MPI_Status status;
        MPI_Recv(&received_data, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        printf("Process %d received data: %d from process 0\n", rank, received_data);
    }

    MPI_Finalize();
    return 0;
}
