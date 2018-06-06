#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>
#include "api_cuda.h"
#include "interface.h"


/*-------------------------------------------------------------------------*/
/**
  @brief      It gets the information of this pointer (CPU or GPU, which device, etc.).
  @param      ptr             The pointer to be checked.
  @param      ptrInfo         Output the information regarding this pointer.
  @return     integer         FTI_SCES if successful.

  This function returns the information of the pointer. It is useful for
  determining the location of data.

 **/
/*-------------------------------------------------------------------------*/
int FTI_get_pointer_info(const void *ptr, FTIT_ptrinfo *ptrInfo)
{
    char message[FTI_BUFS];
    struct cudaPointerAttributes attributes;

    cudaError_t err = cudaPointerGetAttributes(&attributes, ptr);

    if (err != cudaSuccess && err != cudaErrorInvalidValue) {
        sprintf(message, "Cuda error %d %s:: %s", __LINE__, __func__, cudaGetErrorString(err));
        FTI_Print(message, FTI_EROR);
        return FTI_NSCS;
    }
  
    if (err == cudaErrorInvalidValue || attributes.memoryType == cudaMemoryTypeHost || attributes.isManaged) {
        ptrInfo->type = FTIT_PTRTYPE_CPU;
        sprintf(message, "Ptr %p is a CPU pointer", ptr);
    }
    else {
        ptrInfo->type = FTIT_PTRTYPE_GPU;
        ptrInfo->deviceID = attributes.device;
        sprintf(message, "Ptr %p is a GPU pointer", ptr);
    }
  
    FTI_Print(message, FTI_DBUG);
  
    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      It copies data from GPU to CPU.
  @param      dst             CPU destination memory address.
  @param      src             GPU source memory address.
  @param      count           Size in bytes to copy.
  @param      ptrInfo         Information of the src pointer.
  @param      exec            Information of this execution.
  @return     integer         FTI_SCES if successful.

  This function copies count bytes from the GPU memory area pointed to by src
  to the CPU memory area pointed to by dst.
 **/
/*-------------------------------------------------------------------------*/
int FTI_copy_from_device(void *dst, const void *src, size_t count, FTIT_ptrinfo *ptrInfo, FTIT_execution *exec)
{
    cudaStream_t stream = exec->cStreams[ptrInfo->deviceID];
    CUDA_ERROR_CHECK(cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToHost, stream));
    CUDA_ERROR_CHECK(cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost));
    CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));
    FTI_Print("Copied data from GPU", FTI_DBUG);
    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      It copies data from CPU to GPU.
  @param      dst             GPU destination memory address.
  @param      src             CPU source memory address.
  @param      count           Size in bytes to copy.
  @param      ptrInfo         Information of the dst pointer.
  @param      exec            Information of this execution.
  @return     integer         FTI_SCES if successful.

  This function copies count bytes from the CPU memory area pointed to by src
  to the GPU memory area pointed to by dst.
 **/
/*-------------------------------------------------------------------------*/
int FTI_copy_to_device(void *dst, const void *src, size_t count, FTIT_ptrinfo *ptrInfo, FTIT_execution *exec)
{
    cudaStream_t stream = exec->cStreams[ptrInfo->deviceID];
    CUDA_ERROR_CHECK(cudaMemcpyAsync(dst, src, count, cudaMemcpyHostToDevice, stream));
    CUDA_ERROR_CHECK(cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));
    FTI_Print("Copied data to GPU", FTI_DBUG);
    return FTI_SCES;
}

