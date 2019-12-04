#include <cuda_runtime_api.h>
#include "cudaWrap.h"
#include <stdio.h>
#include <unistd.h>


#define BLKX (32)
#define BLKY (32)
#define BLK (BLKX*BLKY)

cudaStream_t gstream;
#define MIN(a,b) (((a)<(b))?(a):(b))


#define CUDA_CALL_SAFE(f)                                                                       \
  do {                                                                                            \
    cudaError_t _e = f;                                                                          \
    if(_e != cudaSuccess) {                                                                    \
      fprintf(stderr, "Cuda error %s %d %s:: %s\n", __FILE__,__LINE__, __func__, cudaGetErrorString(_e));  \
      exit(EXIT_FAILURE);                                                                       \
    }                                                                                           \
  } while(0)


void hostCopy(void *src, void *dest, size_t size){
  CUDA_CALL_SAFE(cudaMemcpy(dest, src, size, cudaMemcpyDeviceToHost));
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

void initStream(cudaStream_t *stream){
  cudaStreamCreate(stream);
}

void syncStream(cudaStream_t *stream){
  CUDA_CALL_SAFE(cudaStreamSynchronize((*stream)));
}

void destroyStream(cudaStream_t *stream){
  CUDA_CALL_SAFE(cudaStreamDestroy(*stream));
}

void freeCuda( void *ptr ){
  CUDA_CALL_SAFE(cudaFree(ptr));
}

void freeCudaHost(void *ptr){ 
  CUDA_CALL_SAFE(cudaFreeHost(ptr));
}


__global__ void initData(long nbLines, long M, double *h, double *g, int rank)
{
  long idX = threadIdx.x + blockIdx.x * blockDim.x;

  if (idX > nbLines * M)
    return;

  h[idX] = 0.0L;
  g[idX] = 0.0L;
  if ( idX >= M +1  && idX  < 2*M-1 ){ 
    h[idX] = 32768.0;
    g[idX] = 0.0;
  }
  else{
    h[idX] = 0.0L;
    g[idX] = 0.0L;
  }
}

__global__ void gpuWork(double *g, double *h,  long M, long nbLines, double *error, long offset, int rank){

  // This moves thread (0,0) to position (1,1) on the grid
  long idX = threadIdx.x + blockIdx.x * blockDim.x +1; 
  long idY = threadIdx.y + blockIdx.y * blockDim.y +1;
  long threadId =  threadIdx.x + threadIdx.y * blockDim.x;
  long tidX = threadIdx.x + blockIdx.x * blockDim.x; 
  long tidY =  idY-1;

  register double temp;
  long xSize = M+2;

  __shared__ double errors[BLK];

  errors[threadId] = 0.0;
  __syncthreads();

  if (tidX < M && tidY < nbLines  ){
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

  __syncthreads();
  if ( threadId == 0 ){
    long id =  (gridDim.x) * blockIdx.y + blockIdx.x;
    error[id] = errors[0];
  }
  __syncthreads();
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

void allocateErrorMemory( void **lerror, void **derror, long xElem, long yElem, int rank){
  long gridSizeX = (xElem +BLKX-1)/BLKX;
  long gridSizeY = (yElem +BLKY-1)/BLKY;
  long numGridsX = gridSizeX * gridSizeY;
  *lerror = (double*) malloc(sizeof(double)*numGridsX);
  allocateMemory(derror, sizeof(double) * numGridsX );
  if ( rank == 0 )
    printf("In function value is %p\n", *derror);
}

double executekernel(long xElem, long yElem, double *in, double *out,  double *dError, double *lError, int rank, int deviceId){
  cudaStream_t CpuToDev;
  cudaStream_t Execute;
  cudaStream_t DevToCpu;

  initStream(&CpuToDev);
  initStream(&Execute);
  initStream(&DevToCpu);
  long linesToSend = 128;// (1024*1024)/(float)(sizeof(double) * (xElem+2)) +1;
//  linesToSend = linesToSend - ( (yElem+2)%linesToSend );

  long i;
  long erroIndex = 0;
    double localError;
  localError=0.0;



  long remainingLines = yElem ;
  //  linesToSend = remainingLines;

  for ( i =0; i < yElem; i+=linesToSend ){


    long gridSizeX = (xElem +BLKX-1)/BLKX;
    long min = MIN( remainingLines, linesToSend);
    long gridSizeY = (min+BLKY-1)/BLKY;
    dim3 grid(gridSizeX,gridSizeY,1);
    dim3 block(BLKX,BLKY,1);

    syncStream(&Execute);
    if ( erroIndex > 0 ){
      for (long  j = 0; j < erroIndex ; j++){
        localError = fmax(localError, dError[j]);
      }
    }

    gpuWork<<<grid,block,0,Execute>>>(&out[i*(xElem+2)], &in[i*(xElem+2)],  xElem , min, dError, erroIndex, rank);

    erroIndex = gridSizeX*gridSizeY;

    remainingLines -= linesToSend;
  }
  remainingLines += linesToSend;

  syncStream(&Execute);

  syncStream(&DevToCpu);
  destroyStream(&Execute);
  destroyStream(&DevToCpu);
  destroyStream(&CpuToDev);


  return localError;

}

void init(double *h, double *g, long Y, long X, int rank){
  memset(g,0,  X *Y);
  memset(h,0,  X *Y);
  long numBlocks = ceil((X*2)/1024.0);
  initData<<<numBlocks ,1024,0,gstream>>>(Y, X, h, g, rank);
  CUDA_CALL_SAFE(cudaStreamSynchronize(gstream));
}


