#include "diff_test.h"

#define ALLOC_SIZE (10*MB)

int main() {
    
    int exit_status = 0;
    
    MPI_Init(NULL,NULL);
    FTI_Init("config.fti", MPI_COMM_WORLD);
    
    dcp_info_t info;
    init( &info, ALLOC_SIZE );
   
    allocate_buffers( &info, ALLOC_SIZE );
    generate_data( &info );
    protect_buffers( &info );
    
    if (FTI_Status() == 0) {
        FTI_Checkpoint( 1, 4 );
        int i;
        for ( i=0; i<NUM_DCKPT-1; ++i ) {
            reallocate_buffers( &info, ALLOC_SIZE, ALLOC_RANDOM );
            protect_buffers( &info );
            xor_data( i, &info );
            FTI_Checkpoint( i+2, 4 );
        }
        reallocate_buffers( &info, ALLOC_SIZE, ALLOC_FULL );
        protect_buffers( &info );
        xor_data( i, &info );
        FTI_Checkpoint( i+2, 4 );
    } else {
        FTI_Recover();
        invert_data( &info );
        exit_status = ( valid( &info ) ) ? 0 : -1;
        MPI_Barrier(FTI_COMM_WORLD);
        FTI_Finalize();
    }

    deallocate_buffers( &info );
    MPI_Finalize();
    exit(exit_status);
}


