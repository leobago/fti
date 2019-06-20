#include <cuda_runtime_api.h>
#include "cudaWrap.h"
#include <stdio.h>


#define BLKX 32
#define BLKY 32

cudaStream_t gstream;


#define CUDA_CALL_SAFE(f)                                                                       \
  do {                                                                                            \
    cudaError_t _e = f;                                                                          \
    if(_e != cudaSuccess) {                                                                    \
      fprintf(stderr, "Cuda error %s %d %s:: %s\n", __FILE__,__LINE__, __func__, cudaGetErrorString(_e));  \
      exit(EXIT_FAILURE);                                                                       \
    }                                                                                           \
  } while(0)


void hostCopy(void *src, void *dest, size_t size){
  CUDA_CALL_SAFE(cudaMemcpyAsync(dest, src, size, cudaMemcpyDeviceToHost,gstream));
}

void deviceCopy(void *src, void *dest, size_t size){
  CUDA_CALL_SAFE(cudaMemcpyAsync(dest, src, size, cudaMemcpyHostToDevice,gstream));
}

void setDevice(int id){
  CUDA_CALL_SAFE(cudaSetDevice(id));
}

int getProperties(){
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  return nDevices;
}

void initStream(){
  cudaStreamCreate(&gstream);
}

void syncStream(){
  CUDA_CALL_SAFE(cudaStreamSynchronize(gstream));
}

void destroyStream(){
  CUDA_CALL_SAFE(cudaStreamDestroy(gstream));
}

void freeCuda( void *ptr ){
  CUDA_CALL_SAFE(cudaFree(ptr));
}

void freeCudaHost(void *ptr){ 
  CUDA_CALL_SAFE(cudaFreeHost(ptr));
}


__global__ void initData(int nbLines, int M, double *h, double *g)
{
  long idX = threadIdx.x + blockIdx.x * blockDim.x;

  if (idX > nbLines * M)
    return;

  h[idX] = 0.0L;
  g[idX] = 0.0L;
  if ( idX >= M +1  && idX  < 2*M-1 ){ 
    h[idX] = 100.0;
    g[idX] = 100.0;
  }
}

__global__ void gpuWork(double *g, double *h, double *error,  int M, int nbLines){

  // This moves thread (0,0) to position (1,1) on the grid
  long idX = threadIdx.x + blockIdx.x * blockDim.x +1; 
  long idY = threadIdx.y + blockIdx.y * blockDim.y +1;
  long threadId = threadIdx.y * blockDim.x + threadIdx.x;
  long tidX = threadIdx.x + blockIdx.x * blockDim.x; 
  long tidY = threadIdx.y + blockIdx.y * blockDim.y;

  register double temp;
  long xSize = M+2;

  __shared__ double errors[BLKX*BLKY];

  errors[threadId] = 0.0;

  if (tidX < M && tidY < nbLines ){
    temp = 0.25*(h[(idY-1)*xSize +idX]
        +h[((idY+1)*xSize)+idX]
        +h[(idY*xSize)+idX-1]
        +h[(idY*xSize)+idX+1]);
    errors[threadId] = fabs(temp - h[(idY*xSize)+idX]); 
    g[(idY*xSize)+idX] = temp;
  }
  else{
    return;
  }

  __syncthreads();


  for (unsigned long s = (blockDim.x*blockDim.y)/2; s>0; s=s>>1){
    if ( threadId < s ){
      errors[threadId] =  fmax(errors[threadId], errors[threadId+s]);
    }
    __syncthreads();
  }


  if ( threadId == 0 ){
    int id = blockIdx.y * (gridDim.x) + blockIdx.x;
    error[id] = errors[0];
  }
  return;
}


void allocateMemory(void **ptr, size_t size){
  CUDA_CALL_SAFE(cudaMallocManaged(ptr, size));
  return;
}

void allocateSafeHost(void **ptr, size_t size){
  CUDA_CALL_SAFE(cudaMallocHost(ptr, size));
  return;
}

void allocateErrorMemory( void **lerror, void **derror, long xElem, long yElem){
  long numGridsX = ceil(xElem/BLKX);
  long numGridsY = ceil(yElem)/BLKY;
  *lerror = (double*) malloc(sizeof(double)*numGridsX*numGridsY);
  allocateMemory(derror, sizeof(double) * numGridsX*numGridsY );
}

double executekernel(long xElem, long yElem, double *in, double *out, double *halo[2],  double *dError, double *lError, int rank){
  long numGridsX = ceil(xElem/BLKX);
  long numGridsY = ceil(yElem)/BLKY;
  dim3 dimGrid(numGridsX,numGridsY);
  dim3 dimBlock(BLKX,BLKY);
  double localError;
  gpuWork<<<dimGrid,dimBlock,0,gstream>>>(out, in, dError,  xElem , yElem);
  CUDA_CALL_SAFE(cudaPeekAtLastError());
  hostCopy(dError,lError, sizeof(double) *((xElem*yElem)/(numGridsX*numGridsY)+1));
  localError=0.0;
  for (long  j = 0; j < numGridsX*numGridsY; j++){
    localError = fmax(localError, lError[j]);
  }
  return localError;
}

void init(double *h, double *g, long Y, long X){
  long numBlocks = ceil((X*Y)/1024.0);
  initData<<<numBlocks ,1024,0,gstream>>>(Y, X, h, g);
  CUDA_CALL_SAFE(cudaStreamSynchronize(gstream));
}


