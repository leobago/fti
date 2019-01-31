#include <stdio.h>
#include <stdlib.h>
#include "api_cuda.h"
#include "interface.h"

#ifdef GPUSUPPORT
#include <cuda_runtime_api.h>
#endif



void *hostBuffers[2];
size_t bufferSize;
cudaStream_t Gstream; 


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
    cudaGetLastError();

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
    CUDA_ERROR_CHECK(cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost));
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
    CUDA_ERROR_CHECK(cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice));
    FTI_Print("Copied data to GPU", FTI_DBUG);
#endif
    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Streaming data from GPU to storage
  @param      FTI_Data        Pointer to the dataset to be copied.
  @param      fwritefunc      Pointer to the function that perform file writes.
  @param      opaque          Additional data to be passed to fwritefunc.                 
  @return     integer         FTI_SCES if successful.

  This function streams (pipelines) data from the GPU memory to the storage.
  The actual writes to the storage are performed by fwritefunc.
 **/
/*-------------------------------------------------------------------------*/
int FTI_TransferDeviceMemToFileAsync(FTIT_dataset *FTI_Data,  FTIT_fwritefunc fwritefunc, void *opaque)
{
#ifdef GPUSUPPORT
    FTI_Print("Piplining GPU -> Storage", FTI_DBUG);
    if (FTI_Data->size == 0)
        return FTI_SCES;

    int res;

    FTIT_data_prefetch prefetcher;
    prefetcher.fetchSize = bufferSize; 
    prefetcher.totalBytesToFetch = FTI_Data->size;
    prefetcher.isDevice = FTI_Data->isDevicePtr;
    prefetcher.dptr = FTI_Data->devicePtr;
    FTI_InitPrefetcher(&prefetcher);
    size_t bytesToWrite;
    unsigned char *basePtr = NULL;

    if ( FTI_Try ( FTI_getPrefetchedData(&prefetcher,&bytesToWrite, &basePtr), "Fetching next memory block from memory") != FTI_SCES){
        return FTI_NSCS;
    }
    while (basePtr){
        if ((res = fwritefunc(basePtr, bytesToWrite, opaque)) != FTI_SCES)
            return res;
        basePtr = NULL;
        if ( FTI_Try ( FTI_getPrefetchedData(&prefetcher,&bytesToWrite, &basePtr), "Fetching next memory block from memory") != FTI_SCES)
            return FTI_NSCS;
    }

    FTI_destroyPrefetcher(&prefetcher);
#endif
    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief     Initializes a Prefetcher 
  @param     A pointer to the Prefetcher struct 
  @return     integer         FTI_SCES if successful.

  This function initializes the streaming procedure for data in the GPU.
  If the data re in the GPU side it also performs the first request to the GPU
  to copy data.
 **/
/*-------------------------------------------------------------------------*/
int FTI_InitPrefetcher(FTIT_data_prefetch *dfls){
    dfls->Id = 0;
    dfls->end = false;
    char str[FTI_BUFS];
    sprintf(str,"FTI_InitPrefetcer:: I am Initializing GPU Prefetcher, isGPU: %d, GPU buffer Size %ld Fetch Size is %ld", dfls->isDevice, bufferSize , dfls->fetchSize); 
    FTI_Print(str,FTI_DBUG);
#ifdef GPUSUPPORT
    dfls->Id = 0;
    if (dfls->isDevice){
        CUDA_ERROR_CHECK(cudaStreamCreate(&(dfls->streams[0])));
        CUDA_ERROR_CHECK(cudaStreamCreate(&(dfls->streams[1])));
        size_t copy_size = MIN(dfls->fetchSize, dfls->totalBytesToFetch);

        if ( copy_size > bufferSize ){
            FTI_Print("I am requesting more bytes than the ones reserved for GPU usage",FTI_WARN);
            FTI_Print("I will reallocate the address space to match the new size",FTI_WARN);
            FTI_Try(FTI_DestroyDevices(), "Destroying allocated host memory for gpu tranfers");
            FTI_Try(FTI_InitDevices(copy_size), "Allocating larger host memory");
            bufferSize = copy_size;
        }
        CUDA_ERROR_CHECK(cudaMemcpyAsync(hostBuffers[dfls->Id], dfls->dptr, copy_size, cudaMemcpyDeviceToHost, dfls->streams[dfls->Id] ));
        dfls->dptr += copy_size;
        dfls->totalBytesToFetch-=copy_size;
        dfls->requestedData = copy_size;
    }
#endif
    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief     Gets gpu-fetched data. 
  @param     dfls A pointer to the Prefetcher struct 
  @param     size  the amount of data in bytes copied from the gpu to the cpu 
  @param     fetchData pointer pointing to the data fetched. 
  @return     integer         FTI_SCES if successful.

    This data returns a pointer to data copied from the GPU side. It also
    requests more data from the GPU side.
 **/
/*-------------------------------------------------------------------------*/
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
            CUDA_ERROR_CHECK(cudaMemcpyAsync(hostBuffers[dfls->Id], dfls->dptr, copy_size, cudaMemcpyDeviceToHost, dfls->streams[dfls->Id] ));
            dfls->dptr += copy_size;
            dfls->totalBytesToFetch-=copy_size;
            dfls->requestedData = copy_size;
        }
        else {
            dfls->end = true;
        }
        CUDA_ERROR_CHECK(cudaStreamSynchronize(dfls->streams[prevId]));   
        *fetchedData = hostBuffers[prevId];
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

/*-------------------------------------------------------------------------*/
/**
  @brief     Allocates Host buffers which will be used for asynchronous communication with the GPU
  @param     HostBuffSize Size in bytes of each buffer. 
  @return     integer         FTI_SCES if successful.
    
 Allocates 2 host buffers which are going to be used for asynchronous communication
 with the GPU.
 **/
/*-------------------------------------------------------------------------*/

int FTI_InitDevices ( int HostBuffSize ){
#ifdef GPUSUPPORT
    char str[FTI_BUFS];
    sprintf(str, "GPU Device Init:: Allocation of 2 GPU-Host Buffers: Total MBytes: %d", HostBuffSize);
    FTI_Print(str, FTI_INFO);
    bufferSize = HostBuffSize;
    CUDA_ERROR_CHECK(cudaHostAlloc(&hostBuffers[0], HostBuffSize, cudaHostAllocDefault));
    CUDA_ERROR_CHECK(cudaHostAlloc(&hostBuffers[1], HostBuffSize, cudaHostAllocDefault));
    CUDA_ERROR_CHECK(cudaStreamCreate(&Gstream));
#endif
    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief     De-allocates Host buffers which will be used for asynchronous communication with the GPU
  @return     integer         FTI_SCES if successful.
    
 De-Allocates 2 host buffers which are going to be used for asynchronous communication
 with the GPU.
 **/
/*-------------------------------------------------------------------------*/
int FTI_DestroyDevices(){
#ifndef GPUSUPPORT
    return FTI_SCES;
#else
    CUDA_ERROR_CHECK(cudaFreeHost(hostBuffers[0]));
    CUDA_ERROR_CHECK(cudaFreeHost(hostBuffers[1]));
    CUDA_ERROR_CHECK(cudaStreamDestroy(Gstream));
    return FTI_SCES;
#endif
}


/*-------------------------------------------------------------------------*/
/**
  @brief    Copies data from a file to the GPU asynchronously 
  @param    fd Pointer to the file
  @param    dptr    Pointer to the data
  @param    numBytes number of bytes to be written 
  @return     integer         FTI_SCES if successful.
    
 Uses 2 streams to copy data from a file to the GPU device memory.

 **/
/*-------------------------------------------------------------------------*/
int FTI_TransferFileToDeviceAsync(FILE *fd, void *dptr, int numBytes){
#ifdef GPUSUPPORT
    int id = 0 ;
    int prevId = 1;
    int copy_size;
    cudaStream_t streams[2]; 
    copy_size= MIN(numBytes, bufferSize);
    CUDA_ERROR_CHECK(cudaStreamCreate(&(streams[0])));
    CUDA_ERROR_CHECK(cudaStreamCreate(&(streams[1])));

    while(numBytes){
        int bytesRead = fread(hostBuffers[id], 1, copy_size, fd);
        if (bytesRead != copy_size){
          FTI_Print("Could Not read entire file",FTI_EROR);
          fclose(fd);
          return FTI_NSCS;
        }
        CUDA_ERROR_CHECK(cudaMemcpyAsync( dptr,hostBuffers[id], copy_size, cudaMemcpyHostToDevice, streams[id]));
        numBytes = numBytes - copy_size;
        dptr+=copy_size;

        if (ferror(fd)) {
            FTI_Print("Could not read FTI checkpoint file.", FTI_EROR);
            fclose(fd);
            return FTI_NSCS;
        }
        CUDA_ERROR_CHECK(cudaStreamSynchronize(streams[prevId]));   
        prevId = id;
        id = (id +1)%2;
        copy_size= MIN(numBytes, bufferSize);
    }
    CUDA_ERROR_CHECK(cudaStreamSynchronize(streams[prevId]));   
    CUDA_ERROR_CHECK(cudaStreamDestroy(streams[0]));
    CUDA_ERROR_CHECK(cudaStreamDestroy(streams[1]));
#endif
    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief     Returns the size of each host buffer in bytes 
  @return     integer        The size of the buffer.

  Returns the size of each host buffer in bytes 

 **/
/*-------------------------------------------------------------------------*/
size_t FTI_getHostBuffSize(){
    return bufferSize;
}


/*-------------------------------------------------------------------------*/
/**
  @brief     Returns the pointer of a specific host buffer 
  @param     id The id of the host buffer
  @return    pointer To the location of the data buffer.

  Returns the pointer of a specific host buffer in bytes 

 **/
/*-------------------------------------------------------------------------*/
BYTE *FTI_getHostBuffer( int id ){
    return hostBuffers[id];
}
/*-------------------------------------------------------------------------*/
/**
  @brief    Copies asynchronously memory from host to device 
  @param    dst   Memory location to copy data to. 
  @param    src   Memory location to copy data from.
  @param    count Number of bytes to copy
  @return     integer        FTI_SCES on success.

  Copies data from a CPU memory Location to a GPU
  memory location. Attention neither src or dest can
  be de allocated until the transaction is finished. 

 **/
/*-------------------------------------------------------------------------*/

int FTI_copy_to_device_async(void *dst, const void *src, size_t count)
{
#ifdef GPUSUPPORT
  CUDA_ERROR_CHECK(cudaMemcpyAsync(dst, src, count, cudaMemcpyHostToDevice,Gstream));
#endif
  return FTI_SCES;
}


/*-------------------------------------------------------------------------*/
/**
  @brief      Synchronizes the devices 
  @return     integer        FTI_SCES on success.

 **/
/*-------------------------------------------------------------------------*/
int FTI_device_sync()
{
#ifdef GPUSUPPORT
  CUDA_ERROR_CHECK(cudaStreamSynchronize(Gstream));
#endif
  return FTI_SCES;
}
