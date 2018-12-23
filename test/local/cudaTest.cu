#include <stdio.h>
#include "wrapperFunc.h"


#define CUDA_CALL_SAFE(f)                                                                       \
  do {                                                                                            \
    cudaError_t _e = f;                                                                          \
    if(_e != cudaSuccess) {                                                                    \
      fprintf(stderr, "Cuda error %s %d %s:: %s\n", __FILE__,__LINE__, __func__, cudaGetErrorString(_e));  \
      exit(EXIT_FAILURE);                                                                       \
    }                                                                                           \
  } while(0)

__global__
void vecmultGPU(double* A, double* B, size_t asize) {
  size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  if (i < asize) {
    A[i] = A[i] * B[i];
  }
}


void allocateMemory(void **ptr, size_t size){
  CUDA_CALL_SAFE(cudaMalloc(ptr, size));
  return;
}

void cudaCopy(void *src, void *dest, size_t size){
  CUDA_CALL_SAFE(cudaMemcpy(dest, src, size, cudaMemcpyHostToDevice));
}

void executeVecMult(int blocks, int threads, double *A, double *B, size_t size, double *ha){
  vecmultGPU<<< blocks, threads >>>(A, B, size);
  CUDA_CALL_SAFE(cudaDeviceSynchronize());
  CUDA_CALL_SAFE(cudaMemcpy(ha, A, size*sizeof(double) , cudaMemcpyDeviceToHost));
}

void freeCuda( void *ptr ){
  CUDA_CALL_SAFE(cudaFree(ptr));
}


void executeVecMultUnified(int blocks, int threads, double *A, double *B, size_t size, double *ha){
  vecmultGPU<<< blocks, threads >>>(A, B, size);
  CUDA_CALL_SAFE(cudaDeviceSynchronize());
}


void allocateManaged( void **ptr , size_t size){
  CUDA_CALL_SAFE(cudaMallocManaged(ptr, size));
}

int getProperties(){
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  return nDevices;
}

void setDevice(int id){
  CUDA_CALL_SAFE(cudaSetDevice(id));
}
