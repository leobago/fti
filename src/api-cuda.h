/**
 *  Copyright (c) 2017 Leonardo A. Bautista-Gomez
 *  All rights reserved

 *  @file   api_cuda.h
 *  @author Max M. Baird(maxbaird.gy@gmail.com)
 *  @date   May, 2018
 *  @brief  Header file for functions that add CUDA support to FTI.
 */

#ifndef FTI_SRC_API_CUDA_H_
#define FTI_SRC_API_CUDA_H_

#include "interface.h"

#ifdef GPUSUPPORT

#include <cuda_runtime_api.h>

#define CUDA_ERROR_CHECK(fun) \
    do { \
        cudaError_t err = fun; \
        char str[FTI_BUFS]; \
        snprintf(str, sizeof(str), "Cuda error %d %s:: %s", __LINE__, __func__,
         cudaGetErrorString(err)); \
        if (err != cudaSuccess) { \
            FTI_Print(str, FTI_WARN); \
            return FTI_NSCS; \
        } \
    } while (0)

#endif

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
    /**< Type of this data pointer      */
    FTIT_ptrtype    type;
    /**< ID of the GPU that this pointer belongs to */
    int             deviceID;
} FTIT_ptrinfo;


typedef struct FTIT_data_prefetch {
    // Pointer pointing to data to be fetched;
    unsigned char *dptr;
    // total number of bytes to be streamed;
    int64_t totalBytesToFetch;
    // Bytes I will fetch each time;
    int64_t fetchSize;
    // Check Whether Bytes are stored in device
    int isDevice;
    // Points to the current stream
    int Id;
    // Equals to 1 when I have prefetched all the data
    bool end;
    // Size of data I requested on previous iteration.
    int requestedData;
#ifdef GPUSUPPORT
    // Streams Used by FTI to get/set data to the gpu side
    cudaStream_t streams[2];
#endif
} FTIT_data_prefetch;


typedef int (*FTIT_fwritefunc)(void *src, int64_t size, void *opaque);

int FTI_get_pointer_info(const void *ptr, FTIT_ptrinfo *ptrInfo);
int FTI_copy_from_device(void *dst, const void *src, int64_t count,
 FTIT_execution *exec);
int FTI_copy_to_device(void *dst, const void *src, int64_t count,
 FTIT_execution *exec);
int FTI_TransferDeviceMemToFileAsync(FTIT_dataset *data,
 FTIT_fwritefunc fwritefunc, void *opaque);
int FTI_InitPrefetcher(FTIT_data_prefetch *dfls);
int FTI_getPrefetchedData(FTIT_data_prefetch *dfls, int64_t *size,
 unsigned char **fetchedData);
int FTI_destroyPrefetcher(FTIT_data_prefetch *dfls);
int FTI_DestroyDevices();
int FTI_InitDevices(int HostBuffSize);
int FTI_TransferFileToDeviceAsync(FILE *fd, void *dptr, int numBytes);
BYTE *FTI_getHostBuffer(int id);
int64_t FTI_getHostBuffSize();
int FTI_copy_to_device_async(void *dst, const void *src, int64_t count);
int FTI_device_sync();

#endif  // FTI_SRC_API_CUDA_H_
