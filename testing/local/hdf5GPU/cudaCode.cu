#include "wrapperFunc.h"
#include <stdio.h>
#include <cuda_runtime_api.h>

#define CUDA_CALL_SAFE(f)                                                                       \
  do {                                                                                            \
    cudaError_t _e = f;                                                                          \
    if(_e != cudaSuccess) {                                                                    \
      fprintf(stderr, "Cuda error %s %d %s:: %s\n", __FILE__,__LINE__, __func__, cudaGetErrorString(_e));  \
      exit(EXIT_FAILURE);                                                                       \
    }                                                                                           \
  } while(0)

__global__ void mykernel(threeD* data){
  const size_t threadsPerBlock = blockDim.x * blockDim.y * blockDim.z;
  const size_t threadNumInBlock = threadIdx.x + threadIdx.y *(blockDim.x) + threadIdx.z * (blockDim.x*blockDim.y);
  const size_t blockNumInGrid = blockIdx.x  + blockIdx.y * gridDim.x + blockIdx.z * ( gridDim.x * gridDim.y);
  const size_t globalThreadNum = blockNumInGrid * threadsPerBlock + threadNumInBlock;
  int idX = threadIdx.x + (blockIdx.x * (gridDim.x*blockDim.x));
  int idY = threadIdx.y + blockIdx.y * (gridDim.y *blockDim.y);
  int idZ = threadIdx.z + blockIdx.z * (gridDim.z *blockDim.z);
  threeD *me = &data[globalThreadNum];
  me->id = globalThreadNum;
  me->x = idX;
  me->y = idY;
  me->z = idZ;
}

void allocateMemory(void **ptr, size_t size){
  CUDA_CALL_SAFE(cudaMalloc(ptr, size));
  return;
}

void cudaCopy(void *src, void *dest, size_t size){
  CUDA_CALL_SAFE(cudaMemcpy(dest, src, size, cudaMemcpyHostToDevice));
}

void hostCopy(void *src, void *dest, size_t size){
  CUDA_CALL_SAFE(cudaMemcpy(dest, src, size, cudaMemcpyDeviceToHost));
}

void freeCuda( void *ptr ){
  CUDA_CALL_SAFE(cudaFree(ptr));
}

void executeKernel( threeD *ptr){
  dim3 dimBlock(BLKX, BLKY, BLKZ);
  dim3 dimGrid(GRDX, GRDY, GRDZ);
  mykernel<<<dimGrid, dimBlock>>>(ptr);
  CUDA_CALL_SAFE(cudaPeekAtLastError());
  CUDA_CALL_SAFE(cudaDeviceSynchronize());
}

int getProperties(){
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  return nDevices;
}

void setDevice(int id){
  CUDA_CALL_SAFE(cudaSetDevice(id));
}
