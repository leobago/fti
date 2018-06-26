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

#ifndef NUM_DCKPT
#   define NUM_DCKPT 5
#endif
#define ERR_CONF (-1001)
#define ERR_STD (-1002)
#define XOR_INFO_ID 2001
#define PAT_ID 2002
#define NBUFFER_ID 2003

#define EXIT_CFG_ERR(MSG,...) do {                                                              \
    fprintf( stderr, "[ERROR-%d] " MSG "\n", grank, ##__VA_ARGS__);                             \
    exit(ERR_CONF);                                                                             \
} while (0)

#define EXIT_STD_ERR(MSG,...) do {                                                              \
    fprintf( stderr, "[ERROR-%d] " MSG " : %s\n", grank, ##__VA_ARGS__, strerror(errno));       \
    exit(ERR_STD);                                                                              \
} while (0)

#define WARN_MSG(MSG,...) do {                                                                  \
    printf( "[WARNING-%d] " MSG "\n", grank, ##__VA_ARGS__);                                    \
} while (0)

#ifdef DEBUG 
#define DBG_MSG(MSG,...) do {                                                                   \
    printf( "[DEBUG-%d] " MSG "\n", grank, ##__VA_ARGS__);                                      \
} while (0)
#else
#define DBG_MSG(MSG,...)
#endif

#define KB (1024L)
#define MB (1024L*KB)
#define GB (1024L*MB)

#define UI_UNIT sizeof(uint32_t)
#define STATIC_SEED 310793 

enum ALLOC_FLAGS {
    ALLOC_FULL,
    ALLOC_RANDOM
};

int grank;

FTIT_type FTI_UI;
FTIT_type FTI_XOR_INFO;
uint32_t pat;
double share_ratio;

int A[1];
int B[2];
int C[3];
int D[4];
int E[5];
int F[6];
int G[7];
int H[8];
int I[9];
int J[10];

int **SHARE;

void init_share();

typedef struct _xor_info {
    double share;
    int offset[256];
    unsigned long nunits[256];
} xor_info_t;

typedef struct _dcp_info {
    void **buffer;
    unsigned long *size;
    int nbuffer;
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
void deallocate_buffers( dcp_info_t * info );
#endif
