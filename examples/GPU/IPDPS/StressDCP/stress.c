#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <fti.h>
#include "wrapperFunc.h"
#include <unistd.h>
#include <time.h>

void executeCPUKernel(char *ptr, long numElements, float ratio){
  long numHashes = numElements/(16*1024) ;
  long i,j;
  for ( i = 0; i < numHashes; i++){
    if ( ratio > (float)rand()/(float)RAND_MAX){
      char *chunk = &ptr[i*16384];
      for ( j = 0; j < 16384; j++){
          chunk[j] = rand()%256;
      }
    }
  }
}

void init(char *ptr, long numElements){
  for ( long i = 0; i < numElements; i++){
      ptr[i]= rand()%256;
  }
}


int main ( int argc, char *argv[]){
  MPI_Init(&argc, &argv);
  int world_size;
  int world_rank;

  int numDevices = getProperties();
  srand( time(NULL));

  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int myDevice = (world_rank)%numDevices;
  printf("I am rank %d, Size of world is %d Number of devices are %d I am setting my device to %d\n", world_rank,world_size, numDevices, myDevice); 
  setDevice(myDevice);

  FTI_Init(argv[3],MPI_COMM_WORLD);

  MPI_Comm_size(FTI_COMM_WORLD, &world_size);
  MPI_Comm_rank(FTI_COMM_WORLD, &world_rank);

  
  long totalGBBytes = atoi(argv[1]);
  long size = totalGBBytes * 1024*1024*1024;

  long mySize = (double) size/(double) world_size;
  float ratioOfMemory = atof(argv[2]);
  float ratioOfChange = atof(argv[4]);

  float numIters = atoi(argv[5]);

  long dSize = ratioOfMemory * mySize;
  long hSize = (1.0 - ratioOfMemory) * mySize;

  char *dPtr;
  char *lPtr;

  allocateMemory((void**) &dPtr,dSize);
  lPtr = (char*) malloc (sizeof(char) * hSize);
  initKernel(dPtr,dSize);
  memset(lPtr,'5',hSize);
  int iter = 0;
  FTI_Protect(0, dPtr, dSize , FTI_CHAR);
  FTI_Protect(1, lPtr, hSize , FTI_CHAR);
  FTI_Protect(2, &iter, 1, FTI_INTG);
  
  if (FTI_Status()!= 0){
    FTI_Recover();
  }
  int i = iter;
  printf("Iter is equal to %d\n", iter);


  for ( ; i < numIters ; i++){
    executeKernel(dPtr,dSize, ratioOfChange);
    executeCPUKernel(lPtr,hSize, ratioOfChange);
    iter++;
    FTI_Checkpoint(iter,5);
    sleep(60);
  }
  
  freeCuda(dPtr);
  free(lPtr);

  FTI_Finalize();
  MPI_Finalize();

}

