#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "wrapperFunc.h"
#include "mpi.h"
#include "fti.h"
#include "../../../../src/deps/iniparser/iniparser.h"
#include "../../../../src/deps/iniparser/dictionary.h"


#define CNTRLD_EXIT 10
#define RECOVERY_FAILED 20
#define DATA_CORRUPT 30
#define WRONG_ENVIRONMENT 50
#define KEEP 2
#define RESTART 1
#define INIT 0

#define SIZE (( 1<<20 )*16) 

int main ( int argc, char *argv[]){
  int i;
  int state;
  int sizeOfDimension;
  int success = 1;
  int FTI_APP_RANK;
  int *ptr;
  int *devPtr;
  int result;
  int numGpus = getProperties();
  
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&FTI_APP_RANK);

  setDevice(FTI_APP_RANK%numGpus);

  result = FTI_Init(argv[1], MPI_COMM_WORLD);

  if (result == FTI_NREC) {
    exit(RECOVERY_FAILED);
  }
  int crash = atoi(argv[2]);

  MPI_Comm_rank(FTI_COMM_WORLD,&FTI_APP_RANK);


  dictionary *ini = iniparser_load( argv[1] );
  int grank;    
  MPI_Comm_rank(MPI_COMM_WORLD,&grank);
  int nbHeads = (int)iniparser_getint(ini, "Basic:head", -1); 
  int finalTag = (int)iniparser_getint(ini, "Advanced:final_tag", 3107);
  int nodeSize = (int)iniparser_getint(ini, "Basic:node_size", -1);
  int headRank = grank - grank%nodeSize;


  if ( (nbHeads<0) || (nodeSize<0) ) {
    printf("wrong configuration (for head or node-size settings)! %d %d\n",nbHeads, nodeSize);
    MPI_Abort(MPI_COMM_WORLD, -1);
  }

  i = 0;

  allocateMemory((void **) &devPtr, SIZE*sizeof(int));
  deviceMemset(devPtr, SIZE*sizeof(int));
  FTI_Protect(0, devPtr,  SIZE ,FTI_INTG);
  FTI_Protect(1, &i,  1 ,FTI_INTG);

  state = FTI_Status();

  if (state != INIT) {
    result = FTI_Recover();
    if (result != FTI_SCES) {
      exit(RECOVERY_FAILED);
    }
  }
  
  int copyData = 1024*1024;
  while( i < SIZE/(copyData)){
    executeKernel(&(devPtr[i*copyData]), i*copyData);
    i++;
    FTI_Checkpoint(i-1,FTI_L4_DCP);
    if ( (i % 4) == 0){
      if ( crash ) {
        if( nbHeads > 0 ) { 
          int value = FTI_ENDW;
          MPI_Send(&value, 1, MPI_INT, headRank, finalTag, MPI_COMM_WORLD);
          MPI_Barrier(MPI_COMM_WORLD);
        }
        MPI_Finalize();
        freeCuda(devPtr);
        exit(0);
      }
    }
  }

  ptr = (int *) malloc(sizeof(int)*SIZE);

  hostCopy(devPtr, ptr,SIZE*sizeof(int));

  result = 0;
  for ( i = 0 ; i < SIZE; i++){
    if (ptr[i] != i ){
      result = 1;
      break;
    }
  }
  if (state == RESTART || state == KEEP) {
    int tmp;
    MPI_Allreduce(&result, &tmp, 1, MPI_INT, MPI_SUM, FTI_COMM_WORLD);
    result = tmp;
  }

  freeCuda(devPtr);

  if (FTI_APP_RANK == 0 && (state == RESTART || state == KEEP)) {
    if (result == 0) {
      printf("[SUCCESSFUL]\n");
    } else {
      printf("[NOT SUCCESSFUL]\n");
      success=0;
    }
  }

  MPI_Barrier(FTI_COMM_WORLD);
  FTI_Finalize();
  MPI_Finalize();

  if (success == 1)
    return 0;
  else
    exit(DATA_CORRUPT);
}

