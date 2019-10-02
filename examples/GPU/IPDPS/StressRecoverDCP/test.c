#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>

#define CUDA_CALL_SAFE(f)                                                                       \
  do {                                                                                            \
    cudaError_t _e = f;                                                                          \
    if(_e != cudaSuccess) {                                                                    \
      fprintf(stderr, "Cuda error %s %d %s:: %s\n", __FILE__,__LINE__, __func__, cudaGetErrorString(_e));  \
      exit(EXIT_FAILURE);                                                                       \
    }                                                                                           \
  } while(0)

int getProperties(){
  int nDevices;
  CUDA_CALL_SAFE(cudaGetDeviceCount(&nDevices));
  return nDevices;
}

int main(int argc, char *argv[]){
  printf("Number of devices are %d\n", getProperties());
}
