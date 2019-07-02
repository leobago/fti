#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>
#include "md5Opt.h"
#include <pthread.h>
#include <fti.h>
#include <interface.h>
#define CPU 1
#define GPU 2
#define CFILE 3

int MD5GPU(FTIT_dataset *);
int MD5CPU(FTIT_dataset *);

pthread_t thread;
pthread_mutex_t worker;
pthread_mutex_t application;
long totalWork= 0;
long worker_exit = 0;
int deviceId;
unsigned char* (*cpuHash)( const unsigned char *data, unsigned long nBytes, unsigned char *hash );


typedef struct threadWork{
    FTIT_dataset *FTI_DataVar;
    FILE *f;
    unsigned int type;
}tw;

tw work[300];

#define CUDA_ERROR_CHECK(fun)                                                           \
    do {                                                                                    \
        cudaError_t err = fun;                                                              \
        char str[FTI_BUFS];                                                                 \
        if (err != cudaSuccess)                                                             \
        {                                                                                   \
            int device;                                                                       \
            cudaGetDevice(&device);                                                           \
            sprintf(str, "Cuda error %d %s:: %s device(%d)", __LINE__, __func__, cudaGetErrorString(err),device); \
            FTI_Print(str, FTI_WARN);                                                         \
            return FTI_NSCS;                                                                  \
        }                                                                                   \
    } while(0)

#define GETDIV(a,b) ((a/b) + (((a % b) == 0 )? 0:1))


MD5_u32plus *Hin,*Hout;
MD5_u32plus *in,*out,*tmp;
char *tempGpuBuffer; 
long tempBufferSize;
long md5ChunkSize;
cudaStream_t Gstream; 


#define F(x, y, z)			((z) ^ ((x) & ((y) ^ (z))))
#define G(x, y, z)			((y) ^ ((z) & ((x) ^ (y))))
#define H(x, y, z)			(((x) ^ (y)) ^ (z))
#define H2(x, y, z)			((x) ^ ((y) ^ (z)))
#define I(x, y, z)			((y) ^ ((x) | ~(z)))

#define STEP(f, a, b, c, d, x, t, s) \
    (a) += f((b), (c), (d)) + (x) + (t); \
(a) = (((a) << (s)) | (((a) & 0xffffffff) >> (32 - (s)))); \
(a) += (b);

#define SET(n) \
    (*(MD5_u32plus *)&ptr[(n) * 4])
#define GET(n) \
    SET(n)
/*
 * This processes one or more 64-byte data blocks, but does NOT update the bit
 * counters.  There are no alignment requirements.
 */

#define OUT(dst, src) \
    (dst)[0] = (unsigned char)(src); \
(dst)[1] = (unsigned char)((src) >> 8); \
(dst)[2] = (unsigned char)((src) >> 16); \
(dst)[3] = (unsigned char)((src) >> 24);


    __global__
void body(MD5_u32plus *out, const void *data, unsigned long size, long md5ChunkSize )
{
    const unsigned char *ptr;
    MD5_u32plus a, b, c, d;
    MD5_u32plus saved_a, saved_b, saved_c, saved_d;
    long tid = threadIdx.x  + blockIdx.x *blockDim.x; 
    long index = tid * md5ChunkSize;
    unsigned char block[64];
    int allocate = 0;
    if (index > size)
        return;
    //  unsigned char *block=&allBlock[blockIdx.x][0];
    ptr = (const unsigned char *)data;
    ptr = &ptr[index];
    long localSize = md5ChunkSize;

    if ( index+ md5ChunkSize > size){
        allocate=1;
        localSize = size-index;
        unsigned char *tmp = (unsigned char *) malloc (md5ChunkSize);
        memset(tmp, 0, md5ChunkSize);
        memcpy(tmp, ptr,localSize); 
        ptr = tmp;
        localSize = md5ChunkSize;
    }

    a = 0x67452301;
    b = 0xefcdab89;
    c  = 0x98badcfe;
    d = 0x10325476;

    MD5_u32plus hi = localSize >> 29;
    MD5_u32plus lo = localSize & 0x1fffffff;
    while (localSize >= 64 ){
        saved_a = a;
        saved_b = b;
        saved_c = c;
        saved_d = d;

        /* Round 1 */
        STEP(F, a, b, c, d, SET(0), 0xd76aa478, 7)
            STEP(F, d, a, b, c, SET(1), 0xe8c7b756, 12)
            STEP(F, c, d, a, b, SET(2), 0x242070db, 17)
            STEP(F, b, c, d, a, SET(3), 0xc1bdceee, 22)
            STEP(F, a, b, c, d, SET(4), 0xf57c0faf, 7)
            STEP(F, d, a, b, c, SET(5), 0x4787c62a, 12)
            STEP(F, c, d, a, b, SET(6), 0xa8304613, 17)
            STEP(F, b, c, d, a, SET(7), 0xfd469501, 22)
            STEP(F, a, b, c, d, SET(8), 0x698098d8, 7)
            STEP(F, d, a, b, c, SET(9), 0x8b44f7af, 12)
            STEP(F, c, d, a, b, SET(10), 0xffff5bb1, 17)
            STEP(F, b, c, d, a, SET(11), 0x895cd7be, 22)
            STEP(F, a, b, c, d, SET(12), 0x6b901122, 7)
            STEP(F, d, a, b, c, SET(13), 0xfd987193, 12)
            STEP(F, c, d, a, b, SET(14), 0xa679438e, 17)
            STEP(F, b, c, d, a, SET(15), 0x49b40821, 22)

            /* Round 2 */
            STEP(G, a, b, c, d, GET(1), 0xf61e2562, 5)
            STEP(G, d, a, b, c, GET(6), 0xc040b340, 9)
            STEP(G, c, d, a, b, GET(11), 0x265e5a51, 14)
            STEP(G, b, c, d, a, GET(0), 0xe9b6c7aa, 20)
            STEP(G, a, b, c, d, GET(5), 0xd62f105d, 5)
            STEP(G, d, a, b, c, GET(10), 0x02441453, 9)
            STEP(G, c, d, a, b, GET(15), 0xd8a1e681, 14)
            STEP(G, b, c, d, a, GET(4), 0xe7d3fbc8, 20)
            STEP(G, a, b, c, d, GET(9), 0x21e1cde6, 5)
            STEP(G, d, a, b, c, GET(14), 0xc33707d6, 9)
            STEP(G, c, d, a, b, GET(3), 0xf4d50d87, 14)
            STEP(G, b, c, d, a, GET(8), 0x455a14ed, 20)
            STEP(G, a, b, c, d, GET(13), 0xa9e3e905, 5)
            STEP(G, d, a, b, c, GET(2), 0xfcefa3f8, 9)
            STEP(G, c, d, a, b, GET(7), 0x676f02d9, 14)
            STEP(G, b, c, d, a, GET(12), 0x8d2a4c8a, 20)

            /* Round 3 */
            STEP(H, a, b, c, d, GET(5), 0xfffa3942, 4)
            STEP(H2, d, a, b, c, GET(8), 0x8771f681, 11)
            STEP(H, c, d, a, b, GET(11), 0x6d9d6122, 16)
            STEP(H2, b, c, d, a, GET(14), 0xfde5380c, 23)
            STEP(H, a, b, c, d, GET(1), 0xa4beea44, 4)
            STEP(H2, d, a, b, c, GET(4), 0x4bdecfa9, 11)
            STEP(H, c, d, a, b, GET(7), 0xf6bb4b60, 16)
            STEP(H2, b, c, d, a, GET(10), 0xbebfbc70, 23)
            STEP(H, a, b, c, d, GET(13), 0x289b7ec6, 4)
            STEP(H2, d, a, b, c, GET(0), 0xeaa127fa, 11)
            STEP(H, c, d, a, b, GET(3), 0xd4ef3085, 16)
            STEP(H2, b, c, d, a, GET(6), 0x04881d05, 23)
            STEP(H, a, b, c, d, GET(9), 0xd9d4d039, 4)
            STEP(H2, d, a, b, c, GET(12), 0xe6db99e5, 11)
            STEP(H, c, d, a, b, GET(15), 0x1fa27cf8, 16)
            STEP(H2, b, c, d, a, GET(2), 0xc4ac5665, 23)

            STEP(I, a, b, c, d, GET(0), 0xf4292244, 6)
            STEP(I, d, a, b, c, GET(7), 0x432aff97, 10)
            STEP(I, c, d, a, b, GET(14), 0xab9423a7, 15)
            STEP(I, b, c, d, a, GET(5), 0xfc93a039, 21)
            STEP(I, a, b, c, d, GET(12), 0x655b59c3, 6)
            STEP(I, d, a, b, c, GET(3), 0x8f0ccc92, 10)
            STEP(I, c, d, a, b, GET(10), 0xffeff47d, 15)
            STEP(I, b, c, d, a, GET(1), 0x85845dd1, 21)
            STEP(I, a, b, c, d, GET(8), 0x6fa87e4f, 6)
            STEP(I, d, a, b, c, GET(15), 0xfe2ce6e0, 10)
            STEP(I, c, d, a, b, GET(6), 0xa3014314, 15)
            STEP(I, b, c, d, a, GET(13), 0x4e0811a1, 21)
            STEP(I, a, b, c, d, GET(4), 0xf7537e82, 6)
            STEP(I, d, a, b, c, GET(11), 0xbd3af235, 10)
            STEP(I, c, d, a, b, GET(2), 0x2ad7d2bb, 15)
            STEP(I, b, c, d, a, GET(9), 0xeb86d391, 21)
            a += saved_a;
        b += saved_b;
        c += saved_c;
        d += saved_d;

        ptr += 64;
        localSize -=64;
    } 

    long used;
    for ( used = 0; used < localSize; used++){
        block[used] = ptr[used];
    }

    block[used++]=0x80;

    long available=64-used; 

    if ( available  < 8 ){
        ptr = block;
        saved_a = a;
        saved_b = b;
        saved_c = c;
        saved_d = d;
        /* Round 1 */
        STEP(F, a, b, c, d, SET(0), 0xd76aa478, 7)
            STEP(F, d, a, b, c, SET(1), 0xe8c7b756, 12)
            STEP(F, c, d, a, b, SET(2), 0x242070db, 17)
            STEP(F, b, c, d, a, SET(3), 0xc1bdceee, 22)
            STEP(F, a, b, c, d, SET(4), 0xf57c0faf, 7)
            STEP(F, d, a, b, c, SET(5), 0x4787c62a, 12)
            STEP(F, c, d, a, b, SET(6), 0xa8304613, 17)
            STEP(F, b, c, d, a, SET(7), 0xfd469501, 22)
            STEP(F, a, b, c, d, SET(8), 0x698098d8, 7)
            STEP(F, d, a, b, c, SET(9), 0x8b44f7af, 12)
            STEP(F, c, d, a, b, SET(10), 0xffff5bb1, 17)
            STEP(F, b, c, d, a, SET(11), 0x895cd7be, 22)
            STEP(F, a, b, c, d, SET(12), 0x6b901122, 7)
            STEP(F, d, a, b, c, SET(13), 0xfd987193, 12)
            STEP(F, c, d, a, b, SET(14), 0xa679438e, 17)
            STEP(F, b, c, d, a, SET(15), 0x49b40821, 22)
            /* Round 2 */
            STEP(G, a, b, c, d, GET(1), 0xf61e2562, 5)
            STEP(G, d, a, b, c, GET(6), 0xc040b340, 9)
            STEP(G, c, d, a, b, GET(11), 0x265e5a51, 14)
            STEP(G, b, c, d, a, GET(0), 0xe9b6c7aa, 20)
            STEP(G, a, b, c, d, GET(5), 0xd62f105d, 5)
            STEP(G, d, a, b, c, GET(10), 0x02441453, 9)
            STEP(G, c, d, a, b, GET(15), 0xd8a1e681, 14)
            STEP(G, b, c, d, a, GET(4), 0xe7d3fbc8, 20)
            STEP(G, a, b, c, d, GET(9), 0x21e1cde6, 5)
            STEP(G, d, a, b, c, GET(14), 0xc33707d6, 9)
            STEP(G, c, d, a, b, GET(3), 0xf4d50d87, 14)
            STEP(G, b, c, d, a, GET(8), 0x455a14ed, 20)
            STEP(G, a, b, c, d, GET(13), 0xa9e3e905, 5)
            STEP(G, d, a, b, c, GET(2), 0xfcefa3f8, 9)
            STEP(G, c, d, a, b, GET(7), 0x676f02d9, 14)
            STEP(G, b, c, d, a, GET(12), 0x8d2a4c8a, 20)
            /* Round 3 */
            STEP(H, a, b, c, d, GET(5), 0xfffa3942, 4)
            STEP(H2, d, a, b, c, GET(8), 0x8771f681, 11)
            STEP(H, c, d, a, b, GET(11), 0x6d9d6122, 16)
            STEP(H2, b, c, d, a, GET(14), 0xfde5380c, 23)
            STEP(H, a, b, c, d, GET(1), 0xa4beea44, 4)
            STEP(H2, d, a, b, c, GET(4), 0x4bdecfa9, 11)
            STEP(H, c, d, a, b, GET(7), 0xf6bb4b60, 16)
            STEP(H2, b, c, d, a, GET(10), 0xbebfbc70, 23)
            STEP(H, a, b, c, d, GET(13), 0x289b7ec6, 4)
            STEP(H2, d, a, b, c, GET(0), 0xeaa127fa, 11)
            STEP(H, c, d, a, b, GET(3), 0xd4ef3085, 16)
            STEP(H2, b, c, d, a, GET(6), 0x04881d05, 23)
            STEP(H, a, b, c, d, GET(9), 0xd9d4d039, 4)
            STEP(H2, d, a, b, c, GET(12), 0xe6db99e5, 11)
            STEP(H, c, d, a, b, GET(15), 0x1fa27cf8, 16)
            STEP(H2, b, c, d, a, GET(2), 0xc4ac5665, 23)

            STEP(I, a, b, c, d, GET(0), 0xf4292244, 6)
            STEP(I, d, a, b, c, GET(7), 0x432aff97, 10)
            STEP(I, c, d, a, b, GET(14), 0xab9423a7, 15)
            STEP(I, b, c, d, a, GET(5), 0xfc93a039, 21)
            STEP(I, a, b, c, d, GET(12), 0x655b59c3, 6)
            STEP(I, d, a, b, c, GET(3), 0x8f0ccc92, 10)
            STEP(I, c, d, a, b, GET(10), 0xffeff47d, 15)
            STEP(I, b, c, d, a, GET(1), 0x85845dd1, 21)
            STEP(I, a, b, c, d, GET(8), 0x6fa87e4f, 6)
            STEP(I, d, a, b, c, GET(15), 0xfe2ce6e0, 10)
            STEP(I, c, d, a, b, GET(6), 0xa3014314, 15)
            STEP(I, b, c, d, a, GET(13), 0x4e0811a1, 21)
            STEP(I, a, b, c, d, GET(4), 0xf7537e82, 6)
            STEP(I, d, a, b, c, GET(11), 0xbd3af235, 10)
            STEP(I, c, d, a, b, GET(2), 0x2ad7d2bb, 15)
            STEP(I, b, c, d, a, GET(9), 0xeb86d391, 21)
            a += saved_a;
        b += saved_b;
        c += saved_c;
        d += saved_d;
        used = 0;
        available = 64;

    }

    lo <<=3;
    memset(&block[used], 0, available-8);

    OUT( &block[56], lo);
    OUT( &block[60], hi);



    ptr = block;
    saved_a = a;
    saved_b = b;
    saved_c = c;
    saved_d = d;

    /* Round 1 */
    STEP(F, a, b, c, d, SET(0), 0xd76aa478, 7)
        STEP(F, d, a, b, c, SET(1), 0xe8c7b756, 12)
        STEP(F, c, d, a, b, SET(2), 0x242070db, 17)
        STEP(F, b, c, d, a, SET(3), 0xc1bdceee, 22)
        STEP(F, a, b, c, d, SET(4), 0xf57c0faf, 7)
        STEP(F, d, a, b, c, SET(5), 0x4787c62a, 12)
        STEP(F, c, d, a, b, SET(6), 0xa8304613, 17)
        STEP(F, b, c, d, a, SET(7), 0xfd469501, 22)
        STEP(F, a, b, c, d, SET(8), 0x698098d8, 7)
        STEP(F, d, a, b, c, SET(9), 0x8b44f7af, 12)
        STEP(F, c, d, a, b, SET(10), 0xffff5bb1, 17)
        STEP(F, b, c, d, a, SET(11), 0x895cd7be, 22)
        STEP(F, a, b, c, d, SET(12), 0x6b901122, 7)
        STEP(F, d, a, b, c, SET(13), 0xfd987193, 12)
        STEP(F, c, d, a, b, SET(14), 0xa679438e, 17)
        STEP(F, b, c, d, a, SET(15), 0x49b40821, 22)

        /* Round 2 */
        STEP(G, a, b, c, d, GET(1), 0xf61e2562, 5)
        STEP(G, d, a, b, c, GET(6), 0xc040b340, 9)
        STEP(G, c, d, a, b, GET(11), 0x265e5a51, 14)
        STEP(G, b, c, d, a, GET(0), 0xe9b6c7aa, 20)
        STEP(G, a, b, c, d, GET(5), 0xd62f105d, 5)
        STEP(G, d, a, b, c, GET(10), 0x02441453, 9)
        STEP(G, c, d, a, b, GET(15), 0xd8a1e681, 14)
        STEP(G, b, c, d, a, GET(4), 0xe7d3fbc8, 20)
        STEP(G, a, b, c, d, GET(9), 0x21e1cde6, 5)
        STEP(G, d, a, b, c, GET(14), 0xc33707d6, 9)
        STEP(G, c, d, a, b, GET(3), 0xf4d50d87, 14)
        STEP(G, b, c, d, a, GET(8), 0x455a14ed, 20)
        STEP(G, a, b, c, d, GET(13), 0xa9e3e905, 5)
        STEP(G, d, a, b, c, GET(2), 0xfcefa3f8, 9)
        STEP(G, c, d, a, b, GET(7), 0x676f02d9, 14)
        STEP(G, b, c, d, a, GET(12), 0x8d2a4c8a, 20)

        /* Round 3 */
        STEP(H, a, b, c, d, GET(5), 0xfffa3942, 4)
        STEP(H2, d, a, b, c, GET(8), 0x8771f681, 11)
        STEP(H, c, d, a, b, GET(11), 0x6d9d6122, 16)
        STEP(H2, b, c, d, a, GET(14), 0xfde5380c, 23)
        STEP(H, a, b, c, d, GET(1), 0xa4beea44, 4)
        STEP(H2, d, a, b, c, GET(4), 0x4bdecfa9, 11)
        STEP(H, c, d, a, b, GET(7), 0xf6bb4b60, 16)
        STEP(H2, b, c, d, a, GET(10), 0xbebfbc70, 23)
        STEP(H, a, b, c, d, GET(13), 0x289b7ec6, 4)
        STEP(H2, d, a, b, c, GET(0), 0xeaa127fa, 11)
        STEP(H, c, d, a, b, GET(3), 0xd4ef3085, 16)
        STEP(H2, b, c, d, a, GET(6), 0x04881d05, 23)
        STEP(H, a, b, c, d, GET(9), 0xd9d4d039, 4)
        STEP(H2, d, a, b, c, GET(12), 0xe6db99e5, 11)
        STEP(H, c, d, a, b, GET(15), 0x1fa27cf8, 16)
        STEP(H2, b, c, d, a, GET(2), 0xc4ac5665, 23)

        STEP(I, a, b, c, d, GET(0), 0xf4292244, 6)
        STEP(I, d, a, b, c, GET(7), 0x432aff97, 10)
        STEP(I, c, d, a, b, GET(14), 0xab9423a7, 15)
        STEP(I, b, c, d, a, GET(5), 0xfc93a039, 21)
        STEP(I, a, b, c, d, GET(12), 0x655b59c3, 6)
        STEP(I, d, a, b, c, GET(3), 0x8f0ccc92, 10)
        STEP(I, c, d, a, b, GET(10), 0xffeff47d, 15)
        STEP(I, b, c, d, a, GET(1), 0x85845dd1, 21)
        STEP(I, a, b, c, d, GET(8), 0x6fa87e4f, 6)
        STEP(I, d, a, b, c, GET(15), 0xfe2ce6e0, 10)
        STEP(I, c, d, a, b, GET(6), 0xa3014314, 15)
        STEP(I, b, c, d, a, GET(13), 0x4e0811a1, 21)
        STEP(I, a, b, c, d, GET(4), 0xf7537e82, 6)
        STEP(I, d, a, b, c, GET(11), 0xbd3af235, 10)
        STEP(I, c, d, a, b, GET(2), 0x2ad7d2bb, 15)
        STEP(I, b, c, d, a, GET(9), 0xeb86d391, 21)

        a += saved_a;
    b += saved_b;
    c += saved_c;
    d += saved_d;

    out[tid*4] = a;
    out[tid*4+1] = b;
    out[tid*4+2] = c;
    out[tid*4+3] = d;
    if (allocate ){
        free((void *)ptr); 
    }
    return;
}

int syncDevice(){
    CUDA_ERROR_CHECK(cudaStreamSynchronize(Gstream));
    return 1;
}

void *workerMain(void *){
    cudaSetDevice(deviceId);
    long l;
    int lock= 1;
    while (1){
        pthread_mutex_lock(&worker);
        if (worker_exit ){
            return NULL;
        }
        for ( l = 0; l < totalWork; l++){
            if( work[l].type == CPU ){
                MD5CPU(work[l].FTI_DataVar);
            }
            else if ( work[l].type == GPU ){
                MD5GPU(work[l].FTI_DataVar);
            }
            else if ( work[l].type == CFILE ){
                lock = 0;
                char str[100];
                double t0 = MPI_Wtime();
                fsync(fileno(work[l].f));
                fclose(work[l].f);
                double t1 = MPI_Wtime();
                sprintf(str,"In memory Ckpt Pushed in Stable Storage in : %.2f sec", t1-t0);
                FTI_Print(str,FTI_INFO);
            }
            else{
                FTI_Print("This should never happen work is either GPU or CPU\n",FTI_EROR);
            }
        }
        totalWork = 0;
        syncDevice();
        if ( lock ){
            pthread_mutex_unlock(&application);
        }
        else {
            lock = 1;
        }
    }
}

int FTI_initMD5(long cSize, long tempSize, FTIT_configuration *FTI_Conf){
    //this will be use by the application to sync
    cpuHash = FTI_Conf->dcpInfoPosix.hashFunc;
    pthread_attr_t attr;

    cudaGetDevice(&deviceId);

    if (pthread_mutex_init(&application, NULL) != 0){
        FTI_Print("Error in initing mutes", FTI_EROR);
        return 1;
    }
    pthread_mutex_lock(&application);

    // This will be used by the worker to sync
    if (pthread_mutex_init(&worker, NULL) != 0){
        FTI_Print("Error in initing mutes", FTI_EROR);
        return 1;
    }
    pthread_mutex_lock(&worker);

    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);

    pthread_attr_init(&attr);


    if(pthread_create(&thread, &attr, workerMain, NULL)) {
        FTI_Print( "Error creating thread",FTI_EROR);
    }


    tempBufferSize = tempSize;
    md5ChunkSize = cSize;
    size_t lvl1Chunks,free,total;
    CUDA_ERROR_CHECK(cudaMemGetInfo  ( &free, &total));  
    lvl1Chunks = ((total)/cSize)*4; // 4 ints per md5
    CUDA_ERROR_CHECK(cudaHostAlloc((void **)&Hin,  sizeof(MD5_u32plus*) * lvl1Chunks ,  cudaHostAllocMapped));
    CUDA_ERROR_CHECK(cudaHostAlloc((void **)&Hout,  sizeof(MD5_u32plus*) * lvl1Chunks ,  cudaHostAllocMapped));
    CUDA_ERROR_CHECK(cudaHostGetDevicePointer((void **)&in,  (void *) Hin, 0));
    CUDA_ERROR_CHECK(cudaHostGetDevicePointer((void **)&out,  (void *) Hout, 0));
    CUDA_ERROR_CHECK(cudaMallocManaged(&tempGpuBuffer, sizeof(char)*(tempBufferSize)));
    CUDA_ERROR_CHECK(cudaStreamCreate(&Gstream));
    return FTI_SCES;
}

int FTI_destroyMD5(){
    worker_exit = 1;
    pthread_mutex_unlock(&worker);

    /*
       char str[FTI_BUFS];
       sprintf(str,"FINALIZING MD5 %p %p %p %p",Hin, Hout, in, out);
       FTI_Print(str,FTI_WARN);
       CUDA_ERROR_CHECK(cudaFree(Hout));
       CUDA_ERROR_CHECK(cudaFreeHost(Hin));
       CUDA_ERROR_CHECK(cudaFree(tempGpuBuffer));
       CUDA_ERROR_CHECK(cudaStreamDestroy(Gstream));
     */
    return FTI_SCES;
}



int MD5GPU(FTIT_dataset *FTI_DataVar){
    size_t size = FTI_DataVar->size;
    unsigned long lvl1Chunks = GETDIV(size,md5ChunkSize);// + (((size % md5ChunkSize) == 0 )? 0:1);
    long i;
    unsigned char block[md5ChunkSize];
    long numKernels= GETDIV(size,md5ChunkSize);
    long numThreads = min(numKernels,1024L);
    long numGroups = GETDIV(numKernels,numThreads);// + ((( numKernels % numThreads ) == 0 ) ? 0:1);
    unsigned char *tmp = (unsigned char*) malloc (sizeof(char)*size);
    body<<<numGroups,numThreads,0,Gstream>>>((MD5_u32plus *) FTI_DataVar->dcpInfoPosix.currentHashArray, FTI_DataVar->devicePtr, size, md5ChunkSize);
//    body<<<numGroups,numThreads>>>((MD5_u32plus *) FTI_DataVar->dcpInfoPosix.currentHashArray, FTI_DataVar->devicePtr, size, md5ChunkSize);
/*    CUDA_ERROR_CHECK(cudaMemcpy(tmp,FTI_DataVar->devicePtr,size,cudaMemcpyDeviceToHost));
    syncDevice();
    for (size_t i =0 ; i < size; i+=md5ChunkSize){
        int chunkSize = size-i;
        int hashIdx = i/md5ChunkSize;
        unsigned char tmpHash[16];

        if( chunkSize < md5ChunkSize ) {
            memset( block, 0x0, md5ChunkSize );
            memcpy( block, &tmp[i], chunkSize );
            cpuHash( block, md5ChunkSize , tmpHash );
        } else {
            cpuHash( &tmp[i], md5ChunkSize , tmpHash );
        }
        int val = memcmp( &(FTI_DataVar->dcpInfoPosix.currentHashArray[hashIdx*16]), tmpHash , 16 );
        if ( val != 0 ){
            char str[100];
            char str1[33],str2[33];
            FTI_GetHashHexStr(tmpHash,16,str1);
            FTI_GetHashHexStr(&FTI_DataVar->dcpInfoPosix.currentHashArray[hashIdx*16],16,str2);
            sprintf(str,"Indexes %ld differ from total %ld (%s %s)",i,size,str1,str2);
            FTI_Print(str,FTI_WARN);
        }
    }

    free(tmp);
*/

    /*
       i = lvl1Chunks*16;
       while ( i > 1 ){
       tmp = out;
       out = in;
       in = tmp;

       tmp = Hin;
       Hin = Hout;
       Hout = tmp;

       long numKernels= i/md5ChunkSize + (((i% md5ChunkSize) == 0 )? 0:1);
       long numThreads = min(numKernels,1024L);
       long numGroups = numKernels / numThreads + ((( numKernels % numThreads ) == 0 ) ? 0:1);
       body<<<numGroups,numThreads,0,Gstream>>>(out, in, i, md5ChunkSize);
       i = GETDIV(i,converge);
       }

       char str[FTI_BUFS];

       CUDA_ERROR_CHECK(cudaMemcpyAsync(chunk,out,16,cudaMemcpyDeviceToHost,Gstream));
     */
    return FTI_SCES;
}



long md5Level( char *tempGpuBuffer, const char *data , MD5_u32plus *Dmd5hash, unsigned long size){
    unsigned long numIntermediateSteps = GETDIV(size, (tempBufferSize));
    long converge = md5ChunkSize/16L;
    long l;
    long outputSize =0;
    long hashesPerIter = 4*(tempBufferSize/(md5ChunkSize));
    for ( l = 0 ; l < numIntermediateSteps; l++){
        long problemSize = min((size-l*tempBufferSize),tempBufferSize);
        long numKernels= GETDIV((problemSize),md5ChunkSize); 
        long numThreads = min(numKernels,1024L);
        long numGroups = numKernels / numThreads + ((( numKernels % numThreads ) == 0 ) ? 0:1);
        CUDA_ERROR_CHECK(cudaMemcpyAsync(tempGpuBuffer,&data[l*tempBufferSize],problemSize,cudaMemcpyHostToDevice,Gstream));
        body<<<numGroups,numThreads,0,Gstream>>>(&Dmd5hash[l*hashesPerIter], tempGpuBuffer , problemSize, md5ChunkSize);
        outputSize += (tempBufferSize/converge); 
    }
    return outputSize;
}


int MD5CPU(FTIT_dataset *FTI_DataVar){
    unsigned long dataSize = FTI_DataVar->size;
    unsigned char block[md5ChunkSize];
    size_t i;
    unsigned char *ptr = (unsigned char *) FTI_DataVar->ptr;
    for ( i = 0 ; i < FTI_DataVar->size; i+=md5ChunkSize){
        unsigned int blockId = i/md5ChunkSize;
        unsigned int hashIdx = blockId*16;
        unsigned int chunkSize = ( (dataSize-i) < md5ChunkSize ) ? dataSize-i: md5ChunkSize;
        if( chunkSize < md5ChunkSize ) {
            memset( block, 0x0, md5ChunkSize );
            memcpy( block, &ptr[i], chunkSize );
            cpuHash( block, md5ChunkSize , &FTI_DataVar->dcpInfoPosix.currentHashArray[hashIdx] );
        } else {
            cpuHash( &ptr[i], md5ChunkSize , &FTI_DataVar->dcpInfoPosix.currentHashArray[hashIdx] );
        }
    }
    return FTI_SCES;
}

    /*
    size_t size = FTI_DataVar->size;
    unsigned long lvl1Chunks = GETDIV(size,md5ChunkSize);// + (((size % md5ChunkSize) == 0 )? 0:1);
    long remainingSize = md5Level(tempGpuBuffer, (char*) FTI_DataVar->ptr, (MD5_u32plus*) FTI_DataVar->dcpInfoPosix.currentHashArray , size);
       while ( i > 1 ){ 
       tmp = out;
       out = in;
       in = tmp;

       tmp = Hin;
       Hin = Hout;
       Hout = tmp;

       long numKernels= i/md5ChunkSize + (((i% md5ChunkSize) == 0 )? 0:1);
       long numThreads = min(numKernels,1024L);
       long numGroups = numKernels / numThreads + ((( numKernels % numThreads ) == 0 ) ? 0:1);
       body<<<numGroups,numThreads,0,Gstream>>>(out, in, i, md5ChunkSize);
       i = GETDIV(i,converge);
       }
       CUDA_ERROR_CHECK(cudaMemcpyAsync(chunk,out,16,cudaMemcpyDeviceToHost,Gstream));
    return FTI_SCES;
}

     */

int FTI_MD5CPU(FTIT_dataset *FTI_DataVar){
    work[totalWork].FTI_DataVar= FTI_DataVar;
    work[totalWork].type= CPU;
    totalWork++;
    return 1;
}

int FTI_MD5GPU(FTIT_dataset *FTI_DataVar){
    work[totalWork].FTI_DataVar= FTI_DataVar;
    work[totalWork].type= GPU;
    totalWork++;
    return 1;
}

int FTI_CLOSE_ASYNC(FILE *f){
    work[totalWork].f= f;
    work[totalWork].type= CFILE;
    totalWork++;
    pthread_mutex_unlock(&worker);
}

int FTI_SyncMD5(){
    pthread_mutex_lock(&application);
    return FTI_SCES;
}

int FTI_startMD5(){
    pthread_mutex_unlock(&worker);
    return FTI_SCES;
}

