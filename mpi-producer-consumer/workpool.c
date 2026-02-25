#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s <runtime in seconds>\n", argv[0]);
        exit(1);
    }

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    srand(time(NULL) + rank);

    long long consumed_count = 0, total_consumed = 0;
    MPI_Request send_request = MPI_REQUEST_NULL, recv_request = MPI_REQUEST_NULL;
    MPI_Status status;
    int work, received_work;
    double start_time, end_time;

    double run_time = atof(argv[1]);

    MPI_Irecv(&received_work, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &recv_request);

    start_time = MPI_Wtime();

    while (1)
    {
        end_time = MPI_Wtime();
        if (end_time - start_time >= run_time)
            break;

        work = rand() % 1000;
        int destination = rand() % size;

        MPI_Isend(&work, 1, MPI_INT, destination, 0, MPI_COMM_WORLD, &send_request);
        MPI_Wait(&send_request, MPI_STATUS_IGNORE);

        int flag = 0;
        MPI_Test(&recv_request, &flag, &status);
        if (flag)
        {
            consumed_count++;
            MPI_Irecv(&received_work, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &recv_request); // Post new receive
        }
    }

    MPI_Cancel(&recv_request);
    MPI_Request_free(&recv_request);
    MPI_Reduce(&consumed_count, &total_consumed, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        printf("Total number of messages consumed: %lld\n", total_consumed);
    }

    MPI_Finalize();
    return 0;
}
