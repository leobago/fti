/**
 *  @file   api_cuda.h
 *  @author Max M. Baird(maxbaird.gy@gmail.com)
 *  @date   May, 2018
 *  @brief  Header file for functions that add CUDA support to FTI.
 */

#ifndef _API_CUDA_H
#define _API_CUDA_H

#ifdef GPUSUPPORT
#include <cuda_runtime_api.h>

#define CUDA_ERROR_CHECK(fun)                                                           \
    do {                                                                                    \
        cudaError_t err = fun;                                                              \
        char str[FTI_BUFS];                                                                 \
        sprintf(str, "Cuda error %d %s:: %s", __LINE__, __func__, cudaGetErrorString(err)); \
        if (err != cudaSuccess)                                                             \
        {                                                                                   \
            FTI_Print(str, FTI_WARN);                                                         \
            return FTI_NSCS;                                                                  \
        }                                                                                   \
    } while(0)

#endif
#include "interface.h"

#define FTI_DEFAULT_CHOSTBUF_SIZE_MB 32

#define MIN(x, y) (x < y ? x : y)

typedef unsigned char BYTE; 

typedef enum FTIT_ptrtype {
    FTIT_PTRTYPE_CPU = 0,
    FTIT_PTRTYPE_GPU
} FTIT_ptrtype;

/** @typedef    FTIT_ptrinfo
 *  @brief      Information of a data pointer.
 *
 *  This type describes necessary information of a data pointer.
 */
typedef struct FTIT_ptrinfo {
    FTIT_ptrtype    type;               /**< Type of this data pointer      */
    int             deviceID;           /**< ID of the GPU that this pointer belongs to */
} FTIT_ptrinfo;


typedef struct FTIT_data_prefetch{
    unsigned char *dptr; // Pointer pointing to data to be fetched;
    size_t totalBytesToFetch; // total number of bytes to be streamed;
    size_t fetchSize;  // Bytes I will fetch each time;
    int isDevice;     //Check Whether Bytes are stored in device
    int Id;           //Points to the current stream
    bool end;                  // Equals to 1 when I have prefetched all the data         
    int requestedData;         //Size of data I requested on previous iteration.
#ifdef GPUSUPPORT
    cudaStream_t streams[2];  // Streams Used by FTI to get/set data to the gpu side
#endif
}FTIT_data_prefetch;


typedef int (*FTIT_fwritefunc)(void *src, size_t size, void *opaque);

int FTI_get_pointer_info(const void *ptr, FTIT_ptrinfo *ptrInfo);
int FTI_copy_from_device(void *dst, const void *src, size_t count,  FTIT_execution *exec);
int FTI_copy_to_device(void *dst, const void *src, size_t count,  FTIT_execution *exec);
int FTI_TransferDeviceMemToFileAsync(FTIT_dataset *FTI_Data,  FTIT_fwritefunc fwritefunc, void *opaque);
int FTI_InitPrefetcher(FTIT_data_prefetch *dfls);
int FTI_getPrefetchedData( FTIT_data_prefetch *dfls, size_t *size, unsigned  char **fetchedData );
int FTI_destroyPrefetcher(FTIT_data_prefetch *dfls);
int FTI_DestroyDevices();
int FTI_InitDevices ( int HostBuffSize );
int FTI_TransferFileToDeviceAsync(FILE *fd, void *dptr, int numBytes);
BYTE *FTI_getHostBuffer( int id );
size_t FTI_getHostBuffSize();
int FTI_copy_to_device_async(void *dst, const void *src, size_t count);
int FTI_device_sync();
#endif
