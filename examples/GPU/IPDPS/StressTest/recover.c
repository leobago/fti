#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <fti.h>
#include "wrapperFunc.h"
#include "../../../src/profiler/profiler.h"
#include <unistd.h>


int main ( int argc, char *argv[]){
  MPI_Init(&argc, &argv);
  int world_size;
  int world_rank;

  int numDevices = getProperties();


  FTI_Init(argv[3],MPI_COMM_WORLD);

  MPI_Comm_size(FTI_COMM_WORLD, &world_size);
  MPI_Comm_rank(FTI_COMM_WORLD, &world_rank);

  int myDevice = (world_rank)%numDevices;
  setDevice(myDevice);

  long totalGBBytes = atoi(argv[1]);
  long size = totalGBBytes * 1024*1024*1024;

  long mySize = (double) size/(double) world_size;
  float ratioOfMemory = atof(argv[2]);

  long dSize = ratioOfMemory * mySize;
  long hSize = (1.0 - ratioOfMemory) * mySize;

  char *dPtr;
  char *lPtr;

  allocateMemory((void**) &dPtr,dSize);
  lPtr = (char*) malloc (sizeof(char) * hSize);

  FTI_Protect(0, dPtr, dSize , FTI_CHAR);
  FTI_Protect(1, lPtr, hSize , FTI_CHAR);
  
  if ( FTI_Status() == 1 ){
    startCount("TotalRecover");
    MPI_Barrier(FTI_COMM_WORLD);
    FTI_Recover();
    MPI_Barrier(FTI_COMM_WORLD);
    stopCount("TotalRecover");
    freeCuda(dPtr);
    free(lPtr);
    MPI_Barrier(FTI_COMM_WORLD);
  }
  else{
    getError(); 
    FTI_Checkpoint(0,1);
    getError();
    freeCuda(dPtr);
    getError();
    free(lPtr);
    MPI_Barrier(FTI_COMM_WORLD);
    sleep(10);
    FTI_CleanGPU();
    sleep(5);
    MPI_Barrier(FTI_COMM_WORLD);
    exit(0);
  }


  FTI_Finalize();
  MPI_Finalize();

}

