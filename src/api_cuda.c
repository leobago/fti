#include <stdio.h>
#include <stdlib.h>
#include "api_cuda.h"
#include "interface.h"

#ifdef GPUSUPPORT
#include <cuda_runtime_api.h>
#endif

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
#ifdef GPUSUPPORT
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

#endif    
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
int FTI_copy_from_device(void *dst, const void *src, size_t count,  FTIT_execution *exec)
{
#ifdef GPUSUPPORT
  cudaStream_t stream = exec->cStream;
  CUDA_ERROR_CHECK(cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToHost, stream));
  CUDA_ERROR_CHECK(cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost));
  CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));
  FTI_Print("Copied data from GPU", FTI_DBUG);
#endif
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
int FTI_copy_to_device(void *dst, const void *src, size_t count, FTIT_execution *exec)
{
#ifdef GPUSUPPORT
  cudaStream_t stream = exec->cStream;
  CUDA_ERROR_CHECK(cudaMemcpyAsync(dst, src, count, cudaMemcpyHostToDevice, stream));
  CUDA_ERROR_CHECK(cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice));
  CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));
  FTI_Print("Copied data to GPU", FTI_DBUG);
#endif
  return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Streaming data from GPU to storage
  @param      FTI_Data        Pointer to the dataset to be copied.
  @param      ptrInfo         Information of the src pointer.
  @param      FTI_Exec        Information of this execution.
  @param      FTI_Conf        Configuration parameters.      
  @param      fwritefunc      Pointer to the function that perform file writes.
  @param      opaque          Additional data to be passed to fwritefunc.                 
  @return     integer         FTI_SCES if successful.

  This function streams (pipelines) data from the GPU memory to the storage.
  The actual writes to the storage are performed by fwritefunc.
 **/
/*-------------------------------------------------------------------------*/
int FTI_pipeline_gpu_to_storage(FTIT_dataset *FTI_Data,  FTIT_execution *FTI_Exec, FTIT_configuration *FTI_Conf, FTIT_fwritefunc fwritefunc, void *opaque)
{
#ifdef GPUSUPPORT
  FTI_Print("Piplining GPU -> Storage", FTI_DBUG);

  if (FTI_Data->size == 0)
    return FTI_SCES;

  int res;

  char *ptr = (char *)FTI_Data->devicePtr;
  size_t remaining_size = FTI_Data->size;
  size_t copy_size = MIN(remaining_size, FTI_Conf->cHostBufSize);

  char str[FTI_BUFS];
  size_t valid_data_sizes[2];

  CUDA_ERROR_CHECK(cudaMemcpyAsync(FTI_Exec->cHostBufs[0], ptr, copy_size, cudaMemcpyDeviceToHost, FTI_Exec->cStream));


  if (FTI_Data->size <= FTI_Conf->cHostBufSize) {
    CUDA_ERROR_CHECK(cudaStreamSynchronize(FTI_Exec->cStream));
    res = fwritefunc(FTI_Exec->cHostBufs[0], FTI_Data->size, opaque);
    return res;
  }

  CUDA_ERROR_CHECK(cudaEventRecord(FTI_Exec->cEvents[0], FTI_Exec->cStream));

  valid_data_sizes[0] = copy_size;

  ptr += copy_size;
  remaining_size -= copy_size;
  copy_size = MIN(remaining_size, FTI_Conf->cHostBufSize);

  CUDA_ERROR_CHECK(cudaMemcpyAsync(FTI_Exec->cHostBufs[1], ptr, copy_size, cudaMemcpyDeviceToHost, FTI_Exec->cStream));

  CUDA_ERROR_CHECK(cudaEventRecord(FTI_Exec->cEvents[1], FTI_Exec->cStream));

  valid_data_sizes[1] = copy_size;

  ptr += copy_size;
  remaining_size -= copy_size;
  copy_size = MIN(remaining_size, FTI_Conf->cHostBufSize);

  bool is_event_active[2] = { true, true };
  int id = 0;

  while (is_event_active[0] || is_event_active[1]) {
    CUDA_ERROR_CHECK(cudaEventSynchronize(FTI_Exec->cEvents[id]));
    if ((res = fwritefunc(FTI_Exec->cHostBufs[id], valid_data_sizes[id], opaque)) != FTI_SCES)
      return res;


    if (remaining_size > 0) {
      CUDA_ERROR_CHECK(cudaMemcpyAsync(FTI_Exec->cHostBufs[id], ptr, copy_size, cudaMemcpyDeviceToHost, FTI_Exec->cStream));
      CUDA_ERROR_CHECK(cudaEventRecord(FTI_Exec->cEvents[id], FTI_Exec->cStream));

      valid_data_sizes[id] = copy_size;

      ptr += copy_size;
      remaining_size -= copy_size;
      copy_size = MIN(remaining_size, FTI_Conf->cHostBufSize);
    }
    else
      is_event_active[id] = false;

    id = (id == 0) ? 1 : 0;
  }
#endif
  return FTI_SCES;
}


int FTI_InitPrefetcher(FTIT_data_prefetch *dfls){
  dfls->Id = 0;
  dfls->end = false;
#ifdef GPUSUPPORT
  dfls->Id = 0;
  if (dfls->isDevice){
    CUDA_ERROR_CHECK(cudaStreamCreate(&(dfls->streams[0])));
    CUDA_ERROR_CHECK(cudaStreamCreate(&(dfls->streams[1])));
    size_t copy_size = MIN(dfls->fetchSize, dfls->totalBytesToFetch);
    CUDA_ERROR_CHECK(cudaMemcpyAsync(dfls->FTI_Exec->cHostBufs[dfls->Id], dfls->dptr, copy_size, cudaMemcpyDeviceToHost, dfls->streams[dfls->Id] ));
    dfls->dptr += copy_size;
    dfls->totalBytesToFetch-=copy_size;
    dfls->requestedData = copy_size;
  }
#endif
  return FTI_SCES;
}

int FTI_getPrefetchedData( FTIT_data_prefetch *dfls, size_t *size, unsigned  char **fetchedData ){
  if (dfls->end ){
    *fetchedData = NULL;
    return FTI_SCES;
  }
#ifdef GPUSUPPORT
  int prevId = dfls->Id; 
  dfls->Id  = (prevId + 1)%2;
  if ( dfls->isDevice ){
    *size = dfls->requestedData;
    if ( dfls->totalBytesToFetch > 0 ){
      size_t copy_size = MIN(dfls->fetchSize, dfls->totalBytesToFetch);
      CUDA_ERROR_CHECK(cudaMemcpyAsync(dfls->FTI_Exec->cHostBufs[dfls->Id], dfls->dptr, copy_size, cudaMemcpyDeviceToHost, dfls->streams[dfls->Id] ));
      dfls->dptr += copy_size;
      dfls->totalBytesToFetch-=copy_size;
      dfls->requestedData = copy_size;
    }
    else {
      dfls->end = true;
    }
    CUDA_ERROR_CHECK(cudaStreamSynchronize(dfls->streams[prevId]));   
    *fetchedData = dfls->FTI_Exec->cHostBufs[prevId];
  }
  else{
    *fetchedData = dfls->dptr;
    *size = dfls->totalBytesToFetch;
    dfls->end = true;
  }
#else
  *fetchedData = dfls->dptr;
  *size = dfls->totalBytesToFetch;
  dfls->end = true;
#endif
  return FTI_SCES;
}

int FTI_destroyPrefetcher(FTIT_data_prefetch *dfls){
#ifdef GPUSUPPORT
  CUDA_ERROR_CHECK(cudaStreamDestroy(dfls->streams[0]));
  CUDA_ERROR_CHECK(cudaStreamDestroy(dfls->streams[1]));
#endif
  return FTI_SCES;
}



