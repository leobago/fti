/**
 *  Copyright (c) 2017 Leonardo A. Bautista-Gomez
 *  All rights reserved
 *
 *  @file   test.c
 *  @author Kai Keller (kellekai@gmx.de)
 *  @date   June, 2017
 *  @brief  FTI testing program.
 **/

#include <fti.h>
#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "../../../../src/deps/iniparser/dictionary.h"
#include "../../../../src/deps/iniparser/iniparser.h"

#define N 1024

unsigned int seed = 42;

int main(int argc, char* argv[]) {
  MPI_Init(NULL, NULL);
  FTI_Init(argv[1], MPI_COMM_WORLD);

  int rank, grank;
  MPI_Comm_rank(FTI_COMM_WORLD, &rank);
  MPI_Comm_rank(MPI_COMM_WORLD, &grank);

  dictionary* ini = iniparser_load(argv[1]);

  int nbHeads = (int)iniparser_getint(ini, "Basic:head", -1);
  int finalTag = (int)iniparser_getint(ini, "Advanced:final_tag", 3107);
  int nodeSize = (int)iniparser_getint(ini, "Basic:node_size", -1);
  int headRank = grank - grank % nodeSize;

  if ((nbHeads < 0) || (nodeSize < 0)) {
    printf("wrong configuration (for head or node-size settings)!\n");
    MPI_Abort(MPI_COMM_WORLD, -1);
  }

  uint32_t* buffer = talloc(uint32_t, N);
  int i = 0;
  srand(time(NULL));
  for (; i < N; ++i) {
    uint32_t rval = rand_r(&seed);
    memcpy(&buffer[i], &rval, sizeof(uint32_t));
  }

  fti_id_t FTI_UINT32_T = FTI_InitType(sizeof(uint32_t));

  int id_cnt = 1;
  FTI_Protect(0, buffer, N, FTI_UINT32_T);
  FTI_Protect(1, &id_cnt, 1, FTI_INTG);

  if (FTI_Status() == 0) {
    FTI_Checkpoint(id_cnt++, 1);
    FTI_Checkpoint(id_cnt++, 1);
    FTI_Checkpoint(id_cnt++, 4);
    FTI_Checkpoint(id_cnt++, 2);
    FTI_Checkpoint(id_cnt++, 2);
    FTI_Checkpoint(id_cnt++, 4);
    FTI_Checkpoint(id_cnt++, 3);
    FTI_Checkpoint(id_cnt++, 3);
    if (nbHeads > 0) {
      int value = FTI_ENDW;
      MPI_Send(&value, 1, MPI_INT, headRank, finalTag, MPI_COMM_WORLD);
      MPI_Barrier(MPI_COMM_WORLD);
    }
    MPI_Finalize();
    exit(0);
  }

  if (FTI_Status() != 0) {
    FTI_Recover();
    FTI_Checkpoint(id_cnt++, 4);
    FTI_Checkpoint(id_cnt++, 1);
    FTI_Checkpoint(id_cnt++, 4);
  }

  FTI_Finalize();

  if (rank == 0) {
    printf("[L4 CP ID's that must been kept are: 3, 6, 9, 11...]\n");
  }
  MPI_Finalize();

  return EXIT_SUCCESS;
}
