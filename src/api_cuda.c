#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>
#include "api_cuda.h"
#include "interface.h"

#define CUDA_ERROR_CHECK(fun)                                                           \
do{                                                                                     \
    cudaError_t err = fun;                                                              \
    char str[FTI_BUFS];                                                                 \
    sprintf(str, "Cuda error %d %s:: %s", __LINE__, __func__, cudaGetErrorString(err)); \
    if(err != cudaSuccess)                                                              \
    {                                                                                   \
      FTI_Print(str, FTI_EROR);                                                         \
      return FTI_NSCS;                                                                  \
    }                                                                                   \
}while(0);

/*-------------------------------------------------------------------------*/
/**
  @brief      It determines whether the pointer type is a GPU or CPU pointer.
  @param      ptr             The pointer to be checked.
  @param      pointer_type    Output parameter specifying pointer type.
  @return     integer         FTI_SCES if successful.

  This function checks the pointer so that it can determine if the pointer
  is a reference to CPU or GPU memory.

 **/
/*-------------------------------------------------------------------------*/
int FTI_determine_pointer_type(const void *ptr, int *pointer_type)
{
  *pointer_type = CPU_POINTER;
  struct cudaPointerAttributes attributes;
  cudaError_t err = cudaPointerGetAttributes(&attributes, ptr);

  if(err == cudaErrorInvalidDevice){
    return FTI_NSCS;
  }

  char str[FTI_BUFS];

  if(attributes.memoryType == cudaMemoryTypeDevice){
    *pointer_type = GPU_POINTER;
  }

  if(attributes.isManaged == 1){
    *pointer_type = CPU_POINTER;
  }

  sprintf(str, "Pointer type: %s", (*pointer_type == CPU_POINTER) ? "CPU Pointer" : "GPU Pointer"); 
  FTI_Print(str, FTI_DBUG);

  return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      It copies data from GPU to CPU.
  @param      dst             CPU destination memory address.
  @param      src             GPU source memory address.
  @param      count           Size in bytes to copy.
  @return     integer         FTI_SCES if successful.

  This function copies count bytes from the GPU memory area pointed to by src
  to the CPU memory area pointed to by dst.
 **/
/*-------------------------------------------------------------------------*/
int FTI_copy_from_device(void* dst, void* src, long count)
{
  char str[FTI_BUFS];
  sprintf(str, "Copying data from GPU");
  FTI_Print(str, FTI_DBUG);
  CUDA_ERROR_CHECK(cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost));
  return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      It copies data from CPU to GPU.
  @param      dst             GPU destination memory address.
  @param      src             CPU source memory address.
  @param      count           Size in bytes to copy.
  @return     integer         FTI_SCES if successful.

  This function copies count bytes from the CPU memory area pointed to by src
  to the GPU memory area pointed to by dst.
 **/
/*-------------------------------------------------------------------------*/
int FTI_copy_to_device(void *dst, void *src, long count)
{
  char str[FTI_BUFS];
  sprintf(str, "Copying data to GPU");
  FTI_Print(str, FTI_DBUG);
  CUDA_ERROR_CHECK(cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice));
  return FTI_SCES;
}
