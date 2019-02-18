#ifndef _DIFF_TEST_H_
#define _DIFF_TEST_H_

#include <stdbool.h>
#include <assert.h>
#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <openssl/md5.h>
#include <fti.h>
#include <time.h>
#include <sys/time.h>
#include <errno.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include "../../../deps/iniparser/dictionary.h"
#include "../../../deps/iniparser/iniparser.h"

#ifndef NUM_DCKPT
#   define NUM_DCKPT 5
#endif
#define ERR_CONF (-1001)
#define ERR_STD (-1002)
#define XOR_INFO_ID 2001
#define PAT_ID 2002
#define NBUFFER_ID 2003

#define EXIT_ID_SUCCESS 0
#define EXIT_ID_ERROR_RECOVERY 1
#define EXIT_ID_ERROR_DATA 2

#define EXIT_CFG_ERR(MSG,...) do {                                                              \
    fprintf( stderr, "[ERROR-%d] " MSG "\n", grank, ##__VA_ARGS__);                             \
    exit(ERR_CONF);                                                                             \
} while (0)

#define EXIT_STD_ERR(MSG,...) do {                                                              \
    fprintf( stderr, "[ERROR-%d] " MSG " : %s\n", grank, ##__VA_ARGS__, strerror(errno));       \
    exit(EXIT_FAILURE);                                                                              \
} while (0)

#define WARN_MSG(MSG,...) do {                                                                  \
    printf( "[WARNING-%d] " MSG "\n", grank, ##__VA_ARGS__);                                    \
} while (0)

#define DBG_MSG(MSG,RANK,...) do { \
    int rank; \
    MPI_Comm_rank(FTI_COMM_WORLD,&rank); \
    if ( rank == RANK ) \
        printf( "%s:%d[DEBUG-%d] " MSG "\n", __FILE__,__LINE__,rank, ##__VA_ARGS__); \
    if ( RANK == -1 ) \
        printf( "%s:%d[DEBUG-%d] " MSG "\n", __FILE__,__LINE__,rank, ##__VA_ARGS__); \
} while (0)

#define INFO_MSG(MSG,...) do { \
    int rank; \
    MPI_Comm_rank(FTI_COMM_WORLD,&rank); \
    if ( rank == 0 ) \
        printf( "%s:%d[INFO] " MSG "\n", __FILE__,__LINE__,rank, ##__VA_ARGS__); \
} while (0)

#define KB (1024L)
#define MB (1024L*KB)
#define GB (1024L*MB)

#define TEST_ICP 1
#define TEST_NOICP 0

#define UI_UNIT sizeof(uint32_t)
#define STATIC_SEED 310793 

enum ALLOC_FLAGS {
    ALLOC_FULL,
    ALLOC_RANDOM
};

int grank;

extern int numHeads;
extern int finalTag;
extern int headRank;

static FTIT_type FTI_UI;
static FTIT_type FTI_XOR_INFO;
static uint32_t pat;
static double share_ratio;

extern int A[1];
extern int B[2];
extern int C[3];
extern int D[4];
extern int E[5];
extern int F[6];
extern int G[7];
extern int H[8];
extern int I[9];
extern int J[10];

extern int **SHARE;

void init_share();

typedef struct _xor_info {
    double share;
    int offset[256];
    unsigned long nunits[256];
} xor_info_t;

typedef struct _dcp_info {
    void **buffer;
    unsigned long *size;
    unsigned long *oldsize;
    int nbuffer;
    int test_mode;
    unsigned char **hash;
    xor_info_t xor_info[NUM_DCKPT];
} dcp_info_t;

/*
 * init a random amount of buffers with random data.
 * Allocate not more then 'alloc_size' in bytes.
*/
void init( dcp_info_t * info, unsigned long alloc_size );

/*
 * change 'share' percentage (integer) of data in buffer with 'id' 
 * and return size of changed data in bytes.
*/
void xor_data( int id, dcp_info_t *info );

void allocate_buffers( dcp_info_t * info, unsigned long alloc_size);
void update_data( dcp_info_t * info, uintptr_t *offset );
void generate_data( dcp_info_t * info );
unsigned long reallocate_buffers( dcp_info_t * info, unsigned long alloc_size, enum ALLOC_FLAGS ALLOC_FLAG );
void invert_data( dcp_info_t *info );
double get_share_ratio();
bool valid( dcp_info_t * info );
void protect_buffers( dcp_info_t *info );
void checkpoint( dcp_info_t *info, int ID, int level );
void deallocate_buffers( dcp_info_t * info );
#endif
