#include "diff_test.h"

int A[1]  = {100};
int B[2]  = {30,70};
int C[3]  = {20,40,40};
int D[4]  = {30,20,40,10};
int E[5]  = {20,35,15,5,25};
int F[6]  = {17,23,12,16,8,24};
int G[7]  = {14,11,15,18,15,17,10};
int H[8]  = {8,12,23,7,15,8,13,14};
int I[9]  = {7,13,21,9,12,8,8,13,9};
int J[10] = {2,6,14,8,5,12,7,11,15,20};

void init_share() {
    
    SHARE = (int**) malloc( 10 * sizeof(int*) );
    SHARE[0] = A; 
    SHARE[1] = B;
    SHARE[2] = C;
    SHARE[3] = D;
    SHARE[4] = E;
    SHARE[5] = F;
    SHARE[6] = G;
    SHARE[7] = H;
    SHARE[8] = I;
    SHARE[9] = J;

}

double get_share_ratio() {
    srand(time(NULL));
    return ((double)(rand()%10000+1))/10000;
}

void init( dcp_info_t * info, unsigned long alloc_size ) {
    
    MPI_Comm_rank(MPI_COMM_WORLD, &grank);
    DBG_MSG("alloc_size: %lu",alloc_size);
    init_share();
    
    // init pattern
    pat = (uint32_t) rand();
    
    // protect pattern and xor_info
    FTI_InitType( &FTI_UI, UI_UNIT ); 
    FTI_Protect( PAT_ID, &pat, 1, FTI_UI );  
    FTI_InitType( &FTI_XOR_INFO, sizeof(xor_info_t) ); 
    FTI_Protect( XOR_INFO_ID, info->xor_info, NUM_CKPT-1, FTI_XOR_INFO );  
    FTI_Protect( NBUFFER_ID, &info->nbuffer,  1, FTI_INTG );  

    // check if alloc_size is sufficiant large
    if ( alloc_size < 101 ) EXIT_CFG_ERR("insufficiant allocation size"); 
    
    // determine number of buffers
    sleep(1*grank);
    srand(time(NULL));
    if ( FTI_Status() == 0 ) {
        info->nbuffer = rand()%10+1;
    } else {
        FTI_RecoverVar( NBUFFER_ID );
    }
    info->buffer = (void**) malloc(info->nbuffer*sizeof(void*));
    info->size = (unsigned long*) malloc(info->nbuffer*sizeof(unsigned long));
    info->hash = (unsigned char**) malloc(info->nbuffer*sizeof(unsigned char*));

    // allocate buffers
    int idx;
    unsigned long allocated = 0;
    for ( idx=0; idx<info->nbuffer; ++idx ) {
        double share = ((double)SHARE[info->nbuffer-1][idx])/100;
        info->size[idx] = (unsigned long)(share*alloc_size);
        info->buffer[idx] = malloc( info->size[idx] );
        if ( info->buffer[idx] == NULL ) {
            EXIT_STD_ERR("idx: %d, cannot allocate %lu bytes", idx, info->size[idx]);
        }
        allocated += info->size[idx];
        DBG_MSG("idx: %d, allocated (total): %lu, allocated (idx): %lu, share: %.2lf%%",idx,allocated, info->size[idx], share*100);
    }
    if ( allocated != alloc_size ) {
        DBG_MSG("allocated: %lu but to allocate is: %lu",allocated,alloc_size);
        unsigned long rest = alloc_size-allocated;
        allocated += rest;
        info->buffer[idx-1] = realloc( info->buffer[idx-1], rest+info->size[idx-1]);
        if ( info->buffer[idx-1] == NULL ) {
            EXIT_STD_ERR("idx: %d, cannot reallocate %lu bytes", idx-1, info->size[idx-1]);
        }
    }
    assert ( ( alloc_size == allocated ) );
    
    // init static random generator
    srand(STATIC_SEED);

    MD5_CTX ctx;
    for( idx=0; idx<info->nbuffer; ++idx) {
        info->hash[idx] = (unsigned char*) malloc(MD5_DIGEST_LENGTH);
        MD5_Init(&ctx);
        uintptr_t ptr = (uintptr_t) info->buffer[idx];
        uintptr_t ptr_e = (uintptr_t)info->buffer[idx] + (uintptr_t)info->size[idx];
        while ( ptr < ptr_e ) {
            unsigned int rui = (unsigned int) rand();
            int init_size = ( (ptr_e - ptr) > UI_UNIT ) ? UI_UNIT : ptr_e-ptr;
            memcpy((void*)ptr, &rui, init_size);
            MD5_Update( &ctx, (void*)ptr, init_size); 
            ptr += init_size;
        }
        assert( ptr == ptr_e );
        MD5_Final(info->hash[idx], &ctx);
    }

}

bool valid( dcp_info_t * info ) {
    unsigned char hash[MD5_DIGEST_LENGTH];
    bool success = true;
    int idx;
    for( idx=0; idx<info->nbuffer; ++idx ){
        MD5( info->buffer[idx], info->size[idx], hash );
        if ( memcmp( hash, info->hash[idx], MD5_DIGEST_LENGTH ) != 0 ) {
            MPI_Barrier(FTI_COMM_WORLD);
            printf("ran %d alive\n",grank);
            WARN_MSG("hashes for buffer id %d differ", idx);
            success = false;
        }
    }
}

void protect_buffers( dcp_info_t *info ) {
    int idx;
    for ( idx=0; idx<info->nbuffer; ++idx ) {
        FTI_Protect( idx, info->buffer[idx], info->size[idx], FTI_CHAR );
    }
}

void deallocate_buffers( dcp_info_t * info ) {
    int idx; 
    for( idx=0; idx<info->nbuffer; ++idx ) {
        free( info->buffer[idx] );
        free( info->hash[idx] );
    }
    free( info->buffer );
    free( info->hash );
    free( info->size );
}

void xor_data( int id, dcp_info_t *info ) {
    info->xor_info[id].share = get_share_ratio();
    if(grank==0){
        printf("share: %lf\n", info->xor_info[id].share);
    }
    srand(time(NULL));
    int idx;
    for ( idx=0; idx<info->nbuffer; ++idx ) {
        int max = ( RAND_MAX > info->size[idx] ) ? info->size[idx] : RAND_MAX;
        info->xor_info[id].offset[idx] = rand()%max;
        if(grank==0){
            printf("    offset: %d, size: %lu\n", info->xor_info[id].offset[idx], info->size[idx]);
        }
        assert(info->xor_info[id].offset[idx] > 0);
        unsigned long eff_size = info->size[idx] - info->xor_info[id].offset[idx];
        unsigned long nunits = ((unsigned long)(info->xor_info[id].share * eff_size))/UI_UNIT;
        unsigned long idxul;
        char *ptr = (char*)(void*)((uintptr_t)info->buffer[idx]+(uintptr_t)info->xor_info[id].offset[idx]);
        unsigned long cnt = 0;
        for ( idxul=0; idxul<nunits; ++idxul ) {
            uint32_t val;
            memcpy(&val, ptr, UI_UNIT);
            uint32_t xor_val = val^pat;
            memcpy(ptr, &xor_val, UI_UNIT);
            ++cnt;
            ptr += UI_UNIT;
        }
        DBG_MSG("changed: %lu, of: %lu, share requested: %.2lf, actual share: %.2lf", cnt*UI_UNIT, info->size[idx], nfo->xor_info[id].share, ((double)(cnt*UI_UNIT))/info->size[idx]);
    }
}

void invert_data( dcp_info_t *info ) {
    int id;
    for ( id=0; id<NUM_CKPT-1; ++id ) {
        int idx;
        if(grank==0){
            printf("share: %lf\n", info->xor_info[id].share);
        }
        for ( idx=0; idx<info->nbuffer; ++idx ) {
            if(grank==0){
                printf("    offset: %d, size: %lu\n", info->xor_info[id].offset[idx], info->size[idx]);
            }
            unsigned long eff_size = info->size[idx] - info->xor_info[id].offset[idx];
            unsigned long nunits = ((unsigned long)(info->xor_info[id].share * eff_size))/UI_UNIT;
            unsigned long idxul;
            char *ptr = (char*)(void*)((uintptr_t)info->buffer[idx]+(uintptr_t)info->xor_info[id].offset[idx]);
            unsigned long cnt = 0;
            for ( idxul=0; idxul<nunits; ++idxul ) {
                uint32_t val;
                memcpy(&val, ptr, UI_UNIT);
                uint32_t xor_val = val^pat;
                memcpy(ptr, &xor_val, UI_UNIT);
                ++cnt;
                ptr += UI_UNIT;
            }
        }
    }
}

