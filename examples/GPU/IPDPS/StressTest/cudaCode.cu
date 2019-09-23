#include "wrapperFunc.h"
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#include <curand_kernel.h>

#define CUDA_CALL_SAFE(f)                                                                       \
  do {                                                                                            \
    cudaError_t _e = f;                                                                          \
    if(_e != cudaSuccess) {                                                                    \
      fprintf(stderr, "Cuda error %s %d %s:: %s\n", __FILE__,__LINE__, __func__, cudaGetErrorString(_e));  \
      exit(EXIT_FAILURE);                                                                       \
    }                                                                                           \
  } while(0)

__global__ void mykernel(char * data,long numHashes, float ratio, int *result){
  long myId = blockIdx.x * blockDim.x +  threadIdx.x;
  if ( myId>= numHashes )
    return;
  long index= myId*16*1024;
  curandState state;
  curand_init((unsigned long long)clock() + myId, 0, 0, &state);
  double rand1 = curand_uniform_double(&state);
  if ( rand1 < ratio ){
    long i;
    char *ptr = &data[index];
    for ( i = 0; i < 16*1024; i++){
      int tmp =curand_uniform_double(&state)*16314; 
      char val = tmp%256;
      *ptr = val; 
      ptr++;
    }
    atomicAdd(result,1);
  }
}

void getMemoryUsage(int rank, int device, char *str){
  size_t total,free;
  CUDA_CALL_SAFE(cudaMemGetInfo   (   &free, &total )); 
  printf("%s Rank %d Device %d, Total memory %ld Mb, Free memory %ld Mb\n",str, rank, device, total/(1024*1024),free/(1024*1024));

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

float executeKernel( char *ptr, long numElements, float ratio ){
  int *dRes;
  int hRes = 0;
  allocateMemory((void **)&dRes,sizeof(int));
  deviceMemset(dRes, sizeof(int));
  unsigned int numHashes = numElements/(16*1024);
  if (numHashes < 1 ){
    fprintf(stderr,"This should not happen\n");
  }
  dim3 blocks={1024};
  dim3 grids={(numHashes/1024)+(((numHashes%1024)==0)?0:1)};
  mykernel<<<grids, blocks>>>(ptr, numHashes, ratio,dRes);
  CUDA_CALL_SAFE(cudaPeekAtLastError());
  CUDA_CALL_SAFE(cudaDeviceSynchronize());
  
  hostCopy(dRes,&hRes, sizeof(int));
  freeCuda(dRes);
  return (float) hRes / (float) numHashes;
}

int getProperties(){
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  return nDevices;
}

void setDevice(int id){
  CUDA_CALL_SAFE(cudaSetDevice(id));
}

void getError(){
  CUDA_CALL_SAFE(cudaGetLastError());
}
