#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#define MSG_WORK 0
#define MSG_REQUEST_WORK 1
#define MSG_ACK 2
#define MSG_ABORT 3
#define MSG_NO_WORK 4
#define MSG_CONSUMED_COUNT 5

void brokerLogic(int numProcs, int bufferCapacity, int simDuration);
void producerLogic(int producerCount, unsigned int seed);
void consumerLogic(unsigned int seed);
void sendAbortAndCollect(int totalProcs, int *totalWorkDone);

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, numProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    srand(time(NULL) + rank);
    int producerCount = (numProcs - 1) / 2;
    int bufferCapacity = 2 * producerCount;
    unsigned int seed = time(NULL) + rank;

    if (rank == 0)
    {
        brokerLogic(numProcs, bufferCapacity, atoi(argv[1]));
    }
    else if (rank <= producerCount)
    {
        producerLogic(producerCount, seed);
    }
    else
    {
        consumerLogic(seed);
    }

    MPI_Finalize();
    return 0;
}

void brokerLogic(int numProcs, int bufferCapacity, int simDuration)
{
    int workBuffer[bufferCapacity];
    int bufferFilled = 0;
    int pendingProducers[numProcs / 2];
    int pendingCount = 0;
    int totalWorkConsumed = 0;

    time_t startTime = time(NULL);

    while (1)
    {
        if (difftime(time(NULL), startTime) >= simDuration)
        {
            sendAbortAndCollect(numProcs, &totalWorkConsumed);
            printf("Total work consumed: %d\n", totalWorkConsumed);
            break;
        }

        int incoming;
        MPI_Status status;
        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &incoming, &status);
        if (incoming)
        {
            int work;
            MPI_Recv(&work, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            if (status.MPI_TAG == MSG_WORK)
            {
                if (bufferFilled < bufferCapacity)
                {
                    workBuffer[bufferFilled++] = work;
                    MPI_Send(NULL, 0, MPI_INT, status.MPI_SOURCE, MSG_ACK, MPI_COMM_WORLD);
                }
                else
                {
                    pendingProducers[pendingCount++] = status.MPI_SOURCE;
                }
            }
            else if (status.MPI_TAG == MSG_REQUEST_WORK)
            {
                if (bufferFilled > 0)
                {
                    MPI_Send(&workBuffer[--bufferFilled], 1, MPI_INT, status.MPI_SOURCE, MSG_WORK, MPI_COMM_WORLD);
                    if (pendingCount > 0)
                    {
                        MPI_Send(NULL, 0, MPI_INT, pendingProducers[0], MSG_ACK, MPI_COMM_WORLD);
                        memmove(pendingProducers, pendingProducers + 1, (pendingCount - 1) * sizeof(int));
                        pendingCount--;
                    }
                }
                else
                {
                    MPI_Send(NULL, 0, MPI_INT, status.MPI_SOURCE, MSG_NO_WORK, MPI_COMM_WORLD);
                }
            }
        }
    }
}

void producerLogic(int producerCount, unsigned int seed)
{
    while (1)
    {
        int work = rand() % 1000;
        MPI_Send(&work, 1, MPI_INT, 0, MSG_WORK, MPI_COMM_WORLD);

        for (int i = 0; i < 1000; i++)
        {
            work += rand() % 1000;
        }

        MPI_Status status;
        MPI_Recv(&work, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        if (status.MPI_TAG == MSG_ABORT)
        {
            break;
        }
    }
}

void consumerLogic(unsigned int seed)
{
    int workDone = 0;

    while (1)
    {
        MPI_Send(NULL, 0, MPI_INT, 0, MSG_REQUEST_WORK, MPI_COMM_WORLD);

        MPI_Status status;
        int receivedWork;
        MPI_Recv(&receivedWork, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        if (status.MPI_TAG == MSG_WORK)
        {
            for (int i = 0; i < 1000; i++)
            {
                receivedWork += rand() % 1000;
            }
            workDone++;
        }
        else if (status.MPI_TAG == MSG_ABORT)
        {
            break;
        }
    }

    MPI_Send(&workDone, 1, MPI_INT, 0, MSG_CONSUMED_COUNT, MPI_COMM_WORLD);
}

void sendAbortAndCollect(int totalProcs, int *totalWorkDone)
{
    for (int i = 1; i < totalProcs; i++)
    {
        MPI_Send(NULL, 0, MPI_INT, i, MSG_ABORT, MPI_COMM_WORLD);
    }

    int workDone;
    for (int i = totalProcs / 2 + 1; i < totalProcs; i++)
    {
        MPI_Recv(&workDone, 1, MPI_INT, i, MSG_CONSUMED_COUNT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        *totalWorkDone += workDone;
    }
}
