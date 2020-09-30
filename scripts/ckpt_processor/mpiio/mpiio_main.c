#include <stdio.h>
#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <mpi.h>

#define FTI_BUFS 256
#define transferSize 16

int main(int argc, char **argv) {
    /*int rank, char* file, int fileSize,
     int nbApprocs, int nbNodes*/
    int rank = atoi(argv[1]);
    char file[FTI_BUFS];
    strcpy(file, argv[2]);
    int fileSize = atoi(argv[3]);
    int nbApprocs = atoi(argv[4]);
    int nbNodes = atoi(argv[5]);

    // Initialize the MPI environment
    MPI_Init(NULL, NULL);
    printf("Rank %d reading from shared code\n", rank);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // begin MPI-IO routine

    MPI_Info info;
    MPI_Info_create(&info);
    MPI_Info_set(info, "romio_cb_read", "enable");

    // set stripping unit to 4MB
    MPI_Info_set(info, "stripping_unit", "4194304");
    char gfn[FTI_BUFS], lfn[FTI_BUFS], gfp[FTI_BUFS];

    snprintf(gfn, FTI_BUFS, file);
    snprintf(lfn, FTI_BUFS, "tmp/Ckpt-mpiio-recovered.fti");
    snprintf(gfp, FTI_BUFS, "%s", gfn);

    // Open file
    MPI_File pfh;
    int buf = MPI_File_open(MPI_COMM_WORLD, gfp, MPI_MODE_RDWR, info, &pfh);
    if (buf != 0) {
        errno = 0;
        char mpi_err[FTI_BUFS];
        int reslen;
        MPI_Error_string(buf, mpi_err, &reslen);
        if (buf != MPI_ERR_NO_SUCH_FILE) {
            char str[FTI_BUFS];
            snprintf(str, FTI_BUFS,
             "Unable to access file [MPI ERROR - %i] %s", buf, mpi_err);
            printf("%s\n", str);
        }
    }
    // Compute Chunks

    int32_t* chunkSizes = (int32_t*) malloc(sizeof(int32_t) * nbApprocs*nbNodes);

    MPI_Allgather(&fileSize, 1, MPI_INT32_T, chunkSizes, 1,
     MPI_INT32_T, MPI_COMM_WORLD);

    MPI_Offset offset = 0;
    // set file offset
    int i;
    for (i = 0; i < rank; i++) {
        offset += chunkSizes[i];
    }
    printf("I am rank %d and i m starting at %lld\n", rank, offset);
    free(chunkSizes);

    // create the tmp recovery file

    mkdir("tmp", 0777); 
    FILE *lfd = fopen(lfn, "wb");
    if (lfd == NULL) {
        printf("Recovery: cannot open the local ckpt. file.");
        MPI_File_close(&pfh);
    }

    // Prepare reading MPIIO file

    char* readData = (char*) malloc(sizeof(char) * transferSize);
    int bSize = transferSize;
    int pos = 0;

    // File transfer from PFS
    while (pos < fileSize) {
        if ((fileSize - pos) < transferSize) {
            bSize = fileSize - pos;
        }
        // read block in parallel file
        buf = MPI_File_read_at(pfh, offset, readData, bSize, MPI_BYTE,
         MPI_STATUS_IGNORE);
        // check if successful
        if (buf != 0) {
            errno = 0;
            char mpi_err[FTI_BUFS];
            char str[FTI_BUFS];
            int reslen;
            MPI_Error_string(buf, mpi_err, &reslen);
            snprintf(str, FTI_BUFS, "R4 cannot read from the ckpt. file in"
            " the PFS. [MPI ERROR - %i] %s", buf, mpi_err);
            printf("%s\n", str);
            free(readData);
            MPI_File_close(&pfh);
            fclose(lfd);
        }

        fwrite(readData, sizeof(char), bSize, lfd);
        if (ferror(lfd)) {
            printf("Recovery: cannot write to the local ckpt. file.");
            free(readData);
            fclose(lfd);
            MPI_File_close(&pfh);
        }
        offset += bSize;
        pos = pos + bSize;
    }

    // Closing all pointers and files 
    free(readData);
    fclose(lfd);

    if (MPI_File_close(&pfh) != 0) {
        printf("Cannot close MPI file.");
    }

    // Finalizing MPI environment
    MPI_Finalize();
    return 0;
}
