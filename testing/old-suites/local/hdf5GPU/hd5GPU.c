#include <stdio.h>
#include <stdlib.h>
#include <hdf5.h>
#include <string.h>
#include "wrapperFunc.h"
#include "mpi.h"
#include "fti.h"
#include "../../../../src/deps/iniparser/iniparser.h"
#include "../../../../src/deps/iniparser/dictionary.h"


#define MYFILE "hd5GPUTest.h5"


#define CNTRLD_EXIT 10
#define RECOVERY_FAILED 20
#define DATA_CORRUPT 30
#define WRONG_ENVIRONMENT 50
#define KEEP 2
#define RESTART 1
#define INIT 0


void initData(threeD *ptr){
  int gx,gz,gy, bx,bz,by;
  size_t threadsPerBlock = BLKX * BLKY * BLKZ;
  int index = 0;
  for ( gz = 0 ; gz < GRDZ; gz++)
    for ( gy = 0 ; gy < GRDY; gy++)
      for ( gx = 0 ; gx < GRDX; gx++)
        for ( bz = 0 ; bz < BLKZ; bz++)
          for ( by = 0 ; by < BLKY; by++)
            for ( bx = 0 ; bx < BLKX; bx++){
              size_t threadNumInBlock = bx + by *(BLKX) + bz * (BLKX*BLKY);
              size_t blockNumInGrid = gx  + gy * GRDX  + gz * ( GRDX * GRDY);
              size_t index= blockNumInGrid * threadsPerBlock + threadNumInBlock;
              ptr[index].id = index;
              ptr[index].x = bx + gx * (GRDX * BLKX);
              ptr[index].y = by + gy * (GRDY * BLKY);
              ptr[index].z = bz + gz * (GRDZ * BLKZ);
            }
}


threeD ***allocateLinearMemory(size_t x, size_t y, size_t z)
{
  threeD *p = (threeD*) malloc(x * y * z * sizeof(threeD));
  threeD ***q = (threeD***) malloc(z * sizeof(threeD**));
  int i;
  int j;
  for (i = 0; i < z; i++)
  {
    q[i] = (threeD **) malloc(y * sizeof(threeD *));
    for (j = 0; j < y; j++)
    {

      int idx = i*x*y + j*x;
      q[i][j] = &p[idx];
    }
  }
  return q;
} 

void deallocateLinearMemory(size_t x, threeD ***q)
{
  free(q[0][0]);
  size_t i;
  for(i = 0; i < x; i++)
  {
    free(q[i]);
  }
  free(q);    
}

void printDimensions(threeD *ptr, int size){
  int i;
  for ( i = 0 ; i < size ; i ++)
    printf("ID %d (%d %d %d)\n",ptr[i].id, ptr[i].x, ptr[i].y, ptr[i].z); 
}


void validateData( threeD *gPtr, threeD *lPtr){
  int i,j,k;
  int index = 0;
  for ( i = 0 ; i < ZSIZE; i++)
    for ( j = 0 ; j < YSIZE; j++)
      for ( k = 0 ;  k < XSIZE; k++){
        if (memcmp(&gPtr[index],&lPtr[index],sizeof(threeD))!=0) 
          printf("%d (%d,%d, %d,%d), (%d,%d,%d,%d)\n", index, gPtr[index].id, gPtr[index].x, gPtr[index].y, gPtr[index].z
              , lPtr[index].id, lPtr[index].x, lPtr[index].y, lPtr[index].z);
        index++;
      }

}

int main ( int argc, char *argv[]){
  int i;
  int state;
  int sizeOfDimension;
  int success = 1;
  int FTI_APP_RANK;
  herr_t status;
  threeD ***ptr = allocateLinearMemory(XSIZE, YSIZE, ZSIZE );
  threeD *devPtr;
  int result;
  MPI_Init(&argc, &argv);
  result = FTI_Init(argv[1], MPI_COMM_WORLD);
  if (result == FTI_NREC) {
    exit(RECOVERY_FAILED);
  }
  int crash = atoi(argv[2]);
  int level = atoi(argv[3]);



  memset(&ptr[0][0][0],0, sizeof(threeD) * (XSIZE * YSIZE * ZSIZE));

  int numGpus = getProperties();
  MPI_Comm_rank(FTI_COMM_WORLD,&FTI_APP_RANK);

  setDevice(FTI_APP_RANK%numGpus);

  dictionary *ini = iniparser_load( argv[1] );
  int grank;    
  MPI_Comm_rank(MPI_COMM_WORLD,&grank);
  int nbHeads = (int)iniparser_getint(ini, "Basic:head", -1); 
  int finalTag = (int)iniparser_getint(ini, "Advanced:final_tag", 3107);
  int nodeSize = (int)iniparser_getint(ini, "Basic:node_size", -1);
  int headRank = grank - grank%nodeSize;

  FTIT_complexType coordinateDef;
  FTIT_Datatype threeDType;
  FTI_AddSimpleField( &coordinateDef, &FTI_INTG, offsetof( threeD, x),0, "X"); 
  FTI_AddSimpleField( &coordinateDef, &FTI_INTG, offsetof( threeD, y),1, "y"); 
  FTI_AddSimpleField( &coordinateDef, &FTI_INTG, offsetof( threeD, z),2, "z"); 
  FTI_AddSimpleField( &coordinateDef, &FTI_INTG, offsetof( threeD, id),3, "id"); 
  FTI_InitComplexType(&threeDType, &coordinateDef, 4 , sizeof(threeD), "ThreeD", NULL);
  

  if ( (nbHeads<0) || (nodeSize<0) ) {
    printf("wrong configuration (for head or node-size settings)! %d %d\n",nbHeads, nodeSize);
    MPI_Abort(MPI_COMM_WORLD, -1);
  }

  allocateMemory((void **) &devPtr, (XSIZE * YSIZE * ZSIZE*sizeof(threeD)));
  FTI_Protect(0, devPtr,  (XSIZE * YSIZE * ZSIZE),threeDType);
  int dimLength[3] = {ZSIZE,YSIZE,XSIZE};
  if (grank == 0)
    for ( i =0 ; i < 3; i++){
      printf("Dimension is %d size is %d\n", dimLength[i], XSIZE*YSIZE*ZSIZE*sizeof(threeDType) / (1024*1024));
    }
  FTI_DefineDataset(0, 3, dimLength , "GPU TOPOLOGY" , NULL);
  state = FTI_Status();
  if ( state == INIT ){
    executeKernel(devPtr);
    FTI_Checkpoint(1,level);
    if ( crash ) {
      if( nbHeads > 0 ) { 
        int value = FTI_ENDW;
        MPI_Send(&value, 1, MPI_INT, headRank, finalTag, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
      }
      MPI_Finalize();
      exit(0);
    }
  }else{
    result = FTI_Recover();
    if (result != FTI_SCES) {
      exit(RECOVERY_FAILED);
    }
    hostCopy(devPtr, &ptr[0][0][0],(XSIZE * YSIZE * ZSIZE*sizeof(threeD)));
  }
  threeD ***validationMemory= allocateLinearMemory(XSIZE, YSIZE, ZSIZE );
  initData(&validationMemory[0][0][0]);

  if (state == RESTART || state == KEEP) {
    int tmp;
    result =  memcmp(&validationMemory[0][0][0], &ptr[0][0][0],(XSIZE * YSIZE * ZSIZE*sizeof(threeD)));
    MPI_Allreduce(&result, &tmp, 1, MPI_INT, MPI_SUM, FTI_COMM_WORLD);
    result = tmp;

  }

  deallocateLinearMemory(ZSIZE , ptr);
  deallocateLinearMemory(ZSIZE , validationMemory);
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

