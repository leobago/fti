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

__global__ void mykernel(int* data, int start){
  int myId = blockIdx.x * blockDim.x +  threadIdx.x;
  data[myId] = start+myId;
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

void deviceMemset(void *ptr, int size){
  CUDA_CALL_SAFE( cudaMemset(ptr, 0, size) );
}

void executeKernel( int *ptr, int start){
  mykernel<<<1024, 1024>>>(ptr, start);
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
