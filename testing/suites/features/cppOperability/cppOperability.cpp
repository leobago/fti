/**
 *  @file   cppOperability.cpp
 *  @author Sawsane Ouchtal
 *  @date   October, 2020
 *  @brief  Tests four of FTI's APIs 
 *          for string operability in C++
 */

#include <iostream>
#include <cstring>
#include <string>
#include "mpi.h"
#include "fti.h"


#define Failed_Init 101
#define Failed_Config 102
#define Failed_Reco 103


void allocArray(int *array[], int *sizes, int N) {
  int i;
  for (i = 0; i < N; i++) array[i] = (int *)malloc(sizeof(int) * sizes[i]);
}

void initArray(int *array[], int *sizes, int N) {
  int i;
  int j;
  for (i = 0; i < N; i++) {
    for (j = 0; j < sizes[i]; j++) array[i][j] = i;
  }
}


int main(int argc, char *argv[]) {
  int *array[10];
  int sizes[10] = {22, 54, 19, 58, 24, 31, 77, 21, 50, 49};
  std::string names[] = {"zero", "one", "two", "three", "four",
                   "five", "six", "seven", "eight", "nine"};
  int i;
  // allocate arrays
  allocArray(array, sizes, 10);

	int ckpt_id = 0;
	unsigned char state;
	int nrank, nbProcs;
    
  MPI_Init(&argc, &argv);

  // test FTI_Init
  std::string config = argv[1];
  if (FTI_Init(config.c_str(), MPI_COMM_WORLD) != 0) {
  	exit(Failed_Init);
  }

  MPI_Comm_size(FTI_COMM_WORLD, &nbProcs);
  MPI_Comm_rank(FTI_COMM_WORLD, &nrank);

  // test FTI_GetConfig
  FTIT_allConfiguration configStruct = FTI_GetConfig(config.c_str(),
   FTI_COMM_WORLD);
 	if (configStruct.configuration.ioMode == -1) {
 		exit(Failed_Config); 
 	}

  state = FTI_Status();
  if (state == 0) {
  	// test FTI_setIDFromString
    initArray(array, sizes, 10);
    int i;
    for (i = 0; i < 10; i++) {
      int FTI_id = FTI_setIDFromString(names[i].c_str());
      FTI_Protect(FTI_id, array[i], sizes[i], FTI_INTG);
    }
    FTI_Checkpoint(ckpt_id++, 1);
  } else if (state == 1 || state == 2) {
  	// test FTI_getIDFromString
    int res = FTI_RecoverVarInit();
    int i;
    for (i = 0; i < 10; i++) {
      int FTI_id = FTI_getIDFromString(names[i].c_str());
      FTI_Protect(FTI_id, array[i], sizes[i], FTI_INTG);
      res += FTI_RecoverVar(FTI_id);
    }
    res += FTI_RecoverVarFinalize();
    if (res != FTI_SCES) {
      exit(Failed_Reco);
    }
  }

  // finalizing the test application
  MPI_Barrier(FTI_COMM_WORLD);
	FTI_Finalize();
	MPI_Finalize();

  return 0;
}

