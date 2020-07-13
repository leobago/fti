/**
 *  Copyright (c) 2017 Leonardo A. Bautista-Gomez
 *  All rights reserved
 *
 */
#include <stdlib.h>
#include <stdint.h>
#include "diff_test.h"

#define ALLOC_SIZE (100*MB)

int main(int argc, char* argv[]) {
    int exit_status = 0;
    if (argc < 1) {
        printf("This test requires an FTI config file path as the"
        " first argument.");
        exit(1);
    }
    MPI_Init(NULL, NULL);
    if (FTI_Init(argv[1], MPI_COMM_WORLD) != 0) {
        exit(-1);
    }

    dcp_info_t info;
    init(argv[1], &info, ALLOC_SIZE);

    protect_buffers(&info);
    if (FTI_Status() == 0) {
        reallocate_buffers(&info, ALLOC_SIZE, ALLOC_RANDOM);
        protect_buffers(&info);
        // FTI_Checkpoint(1, FTI_L4_DCP);
        checkpoint(&info, 1, FTI_L4_DCP);
        int i;
        for (i = 0; i < NUM_DCKPT-1; ++i) {
            uint32_t allocated = reallocate_buffers(&info,
             ALLOC_SIZE, ALLOC_RANDOM);
            protect_buffers(&info);
            xor_data(i, &info);
            // FTI_Checkpoint(i+2, FTI_L4_DCP);
            checkpoint(&info, i+2, FTI_L4_DCP);
        }
        reallocate_buffers(&info, ALLOC_SIZE, ALLOC_FULL);
        protect_buffers(&info);
        xor_data(i, &info);
        // FTI_Checkpoint(i+2, FTI_L4_DCP);
        checkpoint(&info, i+2, FTI_L4_DCP);
        // FTI_Checkpoint(i+3, FTI_L4_DCP);
        checkpoint(&info, i+3, FTI_L4_DCP);
        MPI_Barrier(FTI_COMM_WORLD);
        if (numHeads > 0) {
            int value = FTI_ENDW;
            MPI_Send(&value, 1, MPI_INT, headRank, finalTag, MPI_COMM_WORLD);
            MPI_Barrier(MPI_COMM_WORLD);
        }
    } else {
        if (FTI_Recover() != 0) {
            MPI_Abort(MPI_COMM_WORLD, EXIT_ID_ERROR_RECOVERY);
        }
        invert_data(&info);
        exit_status = (valid(&info)) ? EXIT_ID_SUCCESS : EXIT_ID_ERROR_DATA;
        MPI_Barrier(FTI_COMM_WORLD);
        FTI_Finalize();
    }
    deallocate_buffers(&info);
    MPI_Finalize();
    exit(exit_status);
}


