/**
 *  Copyright (c) 2017 Leonardo A. Bautista-Gomez
 *  All rights reserved
 *
 */
#include <stdint.h>
#include "diff_test.h"

int numHeads;
int finalTag;
int headRank;
int grank;

fti_id_t FTI_UI, FTI_XOR_INFO;
uint32_t pat;

int A[1]  = {100};
int B[2]  = {30, 70};
int C[3]  = {20, 40, 40};
int D[4]  = {30, 20, 40, 10};
int E[5]  = {20, 35, 15, 5, 25};
int F[6]  = {17, 23, 12, 16, 8, 24};
int G[7]  = {14, 11, 15, 18, 15, 17, 10};
int H[8]  = {8, 12, 23, 7, 15, 8, 13, 14};
int I[9]  = {7, 13, 21, 9, 12, 8, 8, 13, 9};
int J[10] = {2, 6, 14, 8, 5, 12, 7, 11, 15, 20};

int **SHARE;
unsigned int seed;

void init_share() {
    SHARE = (int**) malloc(10 * sizeof(int*));
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

void set_rand_seed() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    seed = 1000000 * tv.tv_sec + tv.tv_usec;
}

double get_share_ratio() {
    return ((double)(rand_r(&seed)%10000+1))/10000;
}

void init(char *fti_cfgfile, dcp_info_t * info, uint32_t alloc_size) {
    int wsize;
    MPI_Comm_size(MPI_COMM_WORLD, &wsize);
    MPI_Comm_rank(MPI_COMM_WORLD, &grank);

    dictionary* ini;

    if (access(fti_cfgfile, R_OK) == 0) {
        ini = iniparser_load(fti_cfgfile);
        if (ini == NULL) {
            WARN_MSG("failed to parse FTI config file!");
            exit(EXIT_FAILURE);
        }
    } else {
        EXIT_STD_ERR("cannot access FTI config file!");
    }

    finalTag = (int)iniparser_getint(ini, "Advanced:final_tag", 3107);
    numHeads = (int)iniparser_getint(ini, "Basic:head", 0);
    int nodeSize = (int)iniparser_getint(ini, "Basic:node_size", -1);

    headRank = grank - grank%nodeSize;

    char* env = getenv("TEST_MODE");
    if (env) {
        if (strcmp(env, "ICP") == 0) {
            info->test_mode = TEST_ICP;
            INFO_MSG("TEST MODE -> ICP");
        } else if (strcmp(env, "NOICP") == 0) {
            info->test_mode = TEST_NOICP;
            INFO_MSG("TEST MODE -> NOICP");
        } else {
            info->test_mode = TEST_NOICP;
            INFO_MSG("TEST MODE -> NOICP");
        }
    } else {
        info->test_mode = TEST_NOICP;
        INFO_MSG("TEST MODE -> NOICP");
    }

    // DBG_MSG_APP("alloc_size: %lu",0,alloc_size);
    init_share();

    // init pattern
    pat = (uint32_t) rand();

    // protect pattern and xor_info
    FTI_InitType(&FTI_UI, UI_UNIT);
    FTI_Protect(PAT_ID, &pat, 1, FTI_UI);
    FTI_InitType(&FTI_XOR_INFO, sizeof(xor_info_t));
    FTI_Protect(XOR_INFO_ID, info->xor_info, NUM_DCKPT, FTI_XOR_INFO);
    FTI_Protect(NBUFFER_ID, &info->nbuffer, 1, FTI_INTG);

    // check if alloc_size is sufficiant large
    if (alloc_size < 101) EXIT_CFG_ERR("insufficiant allocation size");

    // determine number of buffers
    usleep(5000*grank);
    set_rand_seed();
    // if (FTI_Status() == 0) {
        info->nbuffer = 5;  // rand()%10+1;
    // } else {
    //    FTI_RecoverVar(NBUFFER_ID);
    // }

    // initialize structure
    info->buffer = talloc(void*, info->nbuffer);
    info->size = talloc(uint32_t, info->nbuffer);
    info->oldsize = talloc(uint32_t, info->nbuffer);
    info->hash = (unsigned char**)malloc(info->nbuffer*sizeof(unsigned char*));
    int idx;
    for (idx = 0; idx < info->nbuffer; ++idx) {
        info->buffer[idx] = NULL;
        info->hash[idx] = (unsigned char*) malloc(MD5_DIGEST_LENGTH);
    }
    allocate_buffers(info, alloc_size);
    generate_data(info);
    set_rand_seed();
}

bool valid(dcp_info_t * info) {
    unsigned char hash[MD5_DIGEST_LENGTH];
    bool success = true;
    int idx;
    for (idx=0; idx < info->nbuffer; ++idx) {
        MD5(info->buffer[idx], info->size[idx], hash);
        if (memcmp(hash, info->hash[idx], MD5_DIGEST_LENGTH) != 0) {
            WARN_MSG("hashes for buffer id %d differ", idx);
            success = false;
        }
    }
    return success;
}

void protect_buffers(dcp_info_t *info) {
    int idx;
    for (idx=0; idx < info->nbuffer; ++idx) {
        FTI_Protect(idx, info->buffer[idx], info->size[idx], FTI_CHAR);
    }
}

void checkpoint(dcp_info_t *info, int ID, int level) {
    if (info->test_mode == TEST_ICP) {
        INFO_MSG("ICP: START CKPT");
        FTI_InitICP(ID, level, 1);
        int idx;
        for (idx = 0; idx < info->nbuffer; ++idx) {
            FTI_AddVarICP(idx);
        }
        FTI_AddVarICP(PAT_ID);
        FTI_AddVarICP(XOR_INFO_ID);
        FTI_AddVarICP(NBUFFER_ID);
        FTI_FinalizeICP();
        INFO_MSG("ICP: END CKPT");
    }
    if (info->test_mode == TEST_NOICP) {
        INFO_MSG("NOICP: START CKPT");
        FTI_Checkpoint(ID, level);
        INFO_MSG("NOICP: END CKPT");
    }
}

void deallocate_buffers(dcp_info_t * info) {
    int idx;
    for (idx=0; idx < info->nbuffer; ++idx) {
        free(info->buffer[idx]);
        free(info->hash[idx]);
    }
    free(info->buffer);
    free(info->hash);
    free(info->size);
}

void xor_data(int id, dcp_info_t *info) {
    info->xor_info[id].share = get_share_ratio();
    if (grank == 0) {
        // printf("%s:%d - info->xor_info[id].share: %.2lf\n", __FILE__,
        // __LINE__,info->xor_info[id].share);
    }
    int idx;
    int oldxinfooffset;
    uint32_t oldxinfonunits;
    uint32_t ckptsize = 0;
    uint32_t update = 0;
    uint32_t changed = 0;
    for (idx = 0; idx < info->nbuffer; ++idx) {
        ckptsize += info->size[idx];
        int max = (RAND_MAX > info->size[idx]) ? info->size[idx] : RAND_MAX;
        oldxinfooffset = info->xor_info[id].offset[idx];
        info->xor_info[id].offset[idx] = rand_r(&seed)%max;
        assert(info->xor_info[id].offset[idx] > 0);
        uint32_t eff_size = info->size[idx] -
         info->xor_info[id].offset[idx];
        oldxinfonunits = info->xor_info[id].nunits[idx];
        info->xor_info[id].nunits[idx] =
         ((uint32_t)(info->xor_info[id].share * eff_size))/UI_UNIT;
        assert(info->xor_info[id].nunits[idx]*UI_UNIT < info->size[idx]);
        uint32_t idxul;
        // DBG_MSG_APP("size[%d]: %lu, offset: %d, udsize: %lu",
        // 0,idx,info->size[idx],info->xor_info[id].offset[idx],
        // info->xor_info[id].nunits[idx]*UI_UNIT);
        char *ptr = (char*)(void*)((uintptr_t)info->buffer[idx]+
            (uintptr_t)info->xor_info[id].offset[idx]);
        uint32_t cnt = 0;
        for (idxul = 0; idxul < info->xor_info[id].nunits[idx]; ++idxul) {
            uint32_t val;
            memcpy(&val, ptr, UI_UNIT);
            uint32_t xor_val = val^pat;
            memcpy(ptr, &xor_val, UI_UNIT);
            ++cnt;
            changed += UI_UNIT;
            ptr += UI_UNIT;
        }
        if (info->size[idx] > info->oldsize[idx]) {
            update += info->size[idx]-info->oldsize[idx];
            if ((info->xor_info[id].offset[idx] + cnt*UI_UNIT) <
             info->oldsize[idx]) {
                update += cnt*UI_UNIT;
            } else if (info->xor_info[id].offset[idx] < info->oldsize[idx]) {
                update += info->oldsize[idx] - info->xor_info[id].offset[idx];
            }
        } else {
            update += cnt*UI_UNIT;
        }
    }
    if (oldxinfonunits != info->xor_info[id].nunits[idx]) {
        update += sizeof(uint32_t);
    }
    if (oldxinfooffset != info->xor_info[id].offset[idx]) {
        update += sizeof(int);
    }

    ckptsize += sizeof(int) + NUM_DCKPT*sizeof(xor_info_t) +
     sizeof(unsigned int);
    int32_t dcpStats[2];
    int32_t sendBuf[] = { ckptsize, update };
    MPI_Reduce(sendBuf, dcpStats, 2, MPI_INT32_T, MPI_SUM, 0, FTI_COMM_WORLD);
    DBG_MSG_APP("changed: %u, of: %u, expected dCP update (min): %.2lf", 0,
     dcpStats[1], dcpStats[0], 100*((double)dcpStats[1])/dcpStats[0]);
}

void invert_data(dcp_info_t *info) {
    int id;
    for (id=0; id < NUM_DCKPT; ++id) {
        int idx;
        for (idx = 0; idx < info->nbuffer; ++idx) {
            // unsigned long eff_size = info->size[idx] -
            // info->xor_info[id].offset[idx];
            // unsigned long nunits = ((unsigned long)(info->
            // xor_info[id].share * eff_size))/UI_UNIT;
            uint32_t idxul;
            char *ptr = (char*)(void*)((uintptr_t)info->buffer[idx]+
                (uintptr_t)info->xor_info[id].offset[idx]);
            uint32_t cnt = 0;
            for (idxul = 0; idxul < info->xor_info[id].nunits[idx]; ++idxul) {
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

void allocate_buffers(dcp_info_t * info, uint32_t alloc_size) {
    int idx;
    uint32_t allocated = 0;
    for (idx = 0; idx < info->nbuffer; ++idx) {
        double share = ((double)SHARE[info->nbuffer-1][idx])/100;
        info->size[idx] = (uint32_t)(share*alloc_size);
        info->oldsize[idx] = 0;
        info->buffer[idx] = malloc(info->size[idx]);
        if (info->buffer[idx] == NULL) {
            EXIT_STD_ERR("idx: %d, cannot allocate %u bytes", idx,
             info->size[idx]);
        }
        allocated += info->size[idx];
        // DBG_MSG_APP("idx: %d, allocated (variable): %lu, share:
        // %.2lf%%",0,idx, info->size[idx], share*100);
    }
    if (allocated != alloc_size) {
        // DBG_MSG_APP("allocated: %lu but to allocate is:
        // %lu",0,allocated,alloc_size);
        uint32_t rest = alloc_size-allocated;
        allocated += rest;
        info->size[idx-1] += rest;
        info->buffer[idx-1] = realloc(info->buffer[idx-1], info->size[idx-1]);
        if (info->buffer[idx-1] == NULL) {
            EXIT_STD_ERR("idx: %d, cannot reallocate %u bytes",
             idx-1, info->size[idx-1]);
        }
    }
    // UNUSED!!
    //uint32_t ckptsize = allocated + sizeof(int) +
    // NUM_DCKPT*sizeof(xor_info_t) + sizeof(unsigned int);
    // DBG_MSG_APP("allocated (total): %lu, [ckptsize: %lu]",
    // -1, allocated, ckptsize);
    assert((alloc_size == allocated));
}
uint32_t reallocate_buffers(dcp_info_t * info, uint32_t _alloc_size,
 enum ALLOC_FLAGS ALLOC_FLAG) {
    uint32_t alloc_size;
    if (ALLOC_FLAG == ALLOC_RANDOM) {
        alloc_size = ((uint32_t)(((uint64_t)rand() << 32) |
         rand()))%_alloc_size+1;
    } else {
        alloc_size = _alloc_size;
    }

    int idx;
    uint32_t allocated = 0;
    for (idx = 0; idx < info->nbuffer; ++idx) {
        double share = ((double)SHARE[info->nbuffer-1][idx])/100;
        info->oldsize[idx] = info->size[idx];
        info->size[idx] = (uint32_t)(share*alloc_size);
        allocated += info->size[idx];
        // DBG_MSG_APP("idx: %d, re-allocated (variable) :
        // %lu, share: %.2lf%%",0,idx, info->size[idx], share*100);
    }
    if (allocated != alloc_size) {
        // DBG_MSG_APP("reallocated: %lu but to reallocate is:
        // %lu",0,allocated,alloc_size);
        uint32_t rest = alloc_size-allocated;
        allocated += rest;
        info->size[idx-1] += rest;
    }
    // UNUSED!!
    //uint32_t ckptsize = allocated + sizeof(int) +
    // NUM_DCKPT*sizeof(xor_info_t) + sizeof(unsigned int);
    // DBG_MSG_APP("re-allocated (total): %lu, [ckptsize: %lu]",
    // -1, allocated, ckptsize);
    assert((alloc_size == allocated));
    return alloc_size;
}

// void update_data(dcp_info_t * info, uintptr_t *offset) {
//    // init static random generator
//    srand(STATIC_SEED);
//
//    MD5_CTX ctx;
//    int idx;
//    for(idx=0; idx<info->nbuffer; ++idx) {
//        MD5_Init(&ctx);
//        uintptr_t ptr = (uintptr_t) info->buffer[idx];
//        uintptr_t ptr_e = (uintptr_t)info->buffer[idx] +
//        (uintptr_t)info->size[idx];
//        while (ptr < ptr_e) {
//            unsigned int rui = (unsigned int) rand();
//            int init_size = ((ptr_e - ptr) > UI_UNIT) ? UI_UNIT : ptr_e-ptr;
//            if (ptr > offset[idx]) {
//                memcpy((void*)ptr, &rui, init_size);
//            }
//            MD5_Update(&ctx, (void*)ptr, init_size);
//            ptr += init_size;
//        }
//        assert(ptr == ptr_e);
//        MD5_Final(info->hash[idx], &ctx);
//    }
// }

void generate_data(dcp_info_t * info) {
    // init static random generator
    srand(STATIC_SEED);

    MD5_CTX ctx;
    int idx;
    for (idx = 0; idx < info->nbuffer; ++idx) {
        MD5_Init(&ctx);
        uintptr_t ptr = (uintptr_t) info->buffer[idx];
        uintptr_t ptr_e = (uintptr_t)info->buffer[idx] +
         (uintptr_t)info->size[idx];
        while (ptr < ptr_e) {
            unsigned int rui = (unsigned int) rand();
            int init_size = ((ptr_e - ptr) > UI_UNIT) ? UI_UNIT : ptr_e-ptr;
            memcpy((void*)ptr, &rui, init_size);
            MD5_Update(&ctx, (void*)ptr, init_size);
            ptr += init_size;
        }
        assert(ptr == ptr_e);
        MD5_Final(info->hash[idx], &ctx);
    }
}
