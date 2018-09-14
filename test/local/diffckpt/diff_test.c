#include "diff_test.h"

#define ALLOC_SIZE (200*MB)

int main() {
    
    int exit_status = 0;
    
    MPI_Init(NULL,NULL);
    if ( FTI_Init("config.fti", MPI_COMM_WORLD) != 0 ) {
        exit(-1);
    }
    
    dcp_info_t info;
    init( &info, ALLOC_SIZE );
   
    protect_buffers( &info );
    if (FTI_Status() == 0) {
        reallocate_buffers( &info, ALLOC_SIZE, ALLOC_RANDOM );
        protect_buffers( &info );
        FTI_Checkpoint( 1, FTI_L4_DCP );
        int i;
        for ( i=0; i<NUM_DCKPT-1; ++i ) {
            unsigned long allocated = reallocate_buffers( &info, ALLOC_SIZE, ALLOC_RANDOM );
            protect_buffers( &info );
            xor_data( i, &info );
            FTI_Checkpoint( i+2, FTI_L4_DCP );
        }
        reallocate_buffers( &info, ALLOC_SIZE, ALLOC_FULL );
        protect_buffers( &info );
        xor_data( i, &info );
        FTI_Checkpoint( i+2, FTI_L4_DCP );
        FTI_Checkpoint( i+3, FTI_L4_DCP );
        sleep(10);
        MPI_Abort(MPI_COMM_WORLD,0);
    } else {
        if ( FTI_Recover() != 0 ) {
            MPI_Abort( MPI_COMM_WORLD, EXIT_ID_ERROR_RECOVERY );
        }
        invert_data( &info );
        //memset(info.buffer[0], 0x45, 32);
        exit_status = ( valid( &info ) ) ? EXIT_ID_SUCCESS : EXIT_ID_ERROR_DATA;
        MPI_Barrier(FTI_COMM_WORLD);
        FTI_Finalize();
    }

    deallocate_buffers( &info );
    MPI_Finalize();
    exit(exit_status);
}


