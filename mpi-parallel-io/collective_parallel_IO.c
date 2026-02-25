/* noncontiguous access with a single collective I/O function */
#include "mpi.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define FILESIZE 1024
#define INTS_PER_BLK 2

int main(int argc, char **argv)
{
    int *buf, rank, nprocs, nints, bufsize;
    MPI_File fh;
    MPI_Datatype filetype;

    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Get rank
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Get number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    bufsize = FILESIZE / nprocs;
    buf = (int *)malloc(bufsize);
    nints = bufsize / sizeof(int);
    memset(buf, 'A' + rank, nints * sizeof(int));

    // Create datatype
    MPI_Type_vector(nints / INTS_PER_BLK, INTS_PER_BLK, nprocs * INTS_PER_BLK, MPI_INT, &filetype);
    MPI_Type_commit(&filetype);

    // Open file
    MPI_File_open(MPI_COMM_WORLD, "result.out",
                  MPI_MODE_CREATE | MPI_MODE_RDWR,
                  MPI_INFO_NULL, &fh);

    // Setup file view
    MPI_File_set_view(fh, rank * INTS_PER_BLK * sizeof(int), MPI_INT, filetype, "native", MPI_INFO_NULL);

    // Collective MPI-IO write
    MPI_File_write_all(fh, buf, nints, MPI_INT, MPI_STATUS_IGNORE);

    MPI_File_set_view(fh, rank * INTS_PER_BLK * sizeof(int), MPI_INT, filetype, "native", MPI_INFO_NULL);
    MPI_File_read_all(fh, buf, nints, MPI_INT, MPI_STATUS_IGNORE);

    printf("Rank%d read: %c\n", rank, buf[0])

        // Close file
        MPI_File_close(&fh);

    // Free the datatype
    MPI_Type_free(&filetype);

    // Free buffer
    free(buf);

    // Finalize
    MPI_Finalize();

    return 0;
}
