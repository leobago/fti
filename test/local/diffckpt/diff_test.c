#include "diff_test.h"

#define ALLOC_SIZE (400*MB)

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
        sleep(5);
        int i;
        for ( i=0; i<NUM_DCKPT-1; ++i ) {
            unsigned long allocated = reallocate_buffers( &info, ALLOC_SIZE, ALLOC_RANDOM );
            protect_buffers( &info );
            xor_data( i, &info );
            FTI_Checkpoint( i+2, FTI_L4_DCP );
        sleep(10);
        }
        reallocate_buffers( &info, ALLOC_SIZE, ALLOC_FULL );
        protect_buffers( &info );
        xor_data( i, &info );
        FTI_Checkpoint( i+2, FTI_L4_DCP );
        sleep(10);
        FTI_Checkpoint( i+3, FTI_L4_DCP );
        sleep(10);
    } else {
        if ( FTI_Recover() != 0 ) {
            exit(-1);
        }
        invert_data( &info );
        //memset(info.buffer[0], 0x45, 32);
        exit_status = ( valid( &info ) ) ? 0 : -1;
        MPI_Barrier(FTI_COMM_WORLD);
        FTI_Finalize();
    }

    deallocate_buffers( &info );
    MPI_Finalize();
    exit(exit_status);
}


