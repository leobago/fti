/**
 *  Copyright (c) 2017 Leonardo A. Bautista-Gomez
 *  All rights reserved
 * 
 *  @file   ckptHierarchy.c
 *  @author Kai Keller (kellekai@gmx.de)
 *  @date   June, 2017
 *  @brief  FTI testing program.
**/
#include <fti.h>
#include <stdio.h>
#include <stdlib.h>

#include "../../../../src/deps/iniparser/dictionary.h"
#include "../../../../src/deps/iniparser/iniparser.h"

#define ARRAY_SIZE 1024 * 1024
#define DATASET_SIZE (ARRAY_SIZE / 4)
#define FIRST array
#define SECOND (array + DATASET_SIZE)
#define THIRD (array + DATASET_SIZE * 2)
#define FOURTH (array + DATASET_SIZE * 3)

int* array;
int world_rank;
int world_size;
int global_world_rank;
int global_world_size;
int checkpoint_level[4];
int initStatus;

char* configFile;

void simulateCrash() {
  dictionary* ini = iniparser_load(configFile);
  int heads = (int)iniparser_getint(ini, "Basic:head", -1);
  int nodeSize = (int)iniparser_getint(ini, "Basic:node_size", -1);
  int general_tag = (int)iniparser_getint(ini, "Advanced:general_tag", 2612);
  int final_tag = (int)iniparser_getint(ini, "Advanced:final_tag", 3107);
  int res;
  if (checkpoint_level[3] != 1) {
    int isInline = -1;
    int heads = (int)iniparser_getint(ini, "Basic:head", -1);
    switch (checkpoint_level[3]) {
      case 2:
        isInline = (int)iniparser_getint(ini, "Basic:inline_l2", 1);
        break;
      case 3:
        isInline = (int)iniparser_getint(ini, "Basic:inline_l3", 1);
        break;
      case 4:
        isInline = (int)iniparser_getint(ini, "Basic:inline_l4", 1);
        break;
    }
    if (isInline == 0) {
      // waiting untill head do Post-checkpointing
      MPI_Recv(&res, 1, MPI_INT,
               global_world_rank - (global_world_rank % nodeSize), general_tag,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }
  iniparser_freedict(ini);
  if (heads > 0) {
    res = FTI_ENDW;
    // sending END WORK to head to stop listening
    MPI_Send(&res, 1, MPI_INT,
             global_world_rank - (global_world_rank % nodeSize), final_tag,
             MPI_COMM_WORLD);
    // Barrier needed for heads (look FTI_Finalize() in api.c)
    MPI_Barrier(MPI_COMM_WORLD);
  }
  MPI_Barrier(FTI_COMM_WORLD);
  // There is no FTI_Finalize(), because want to recover also from L1, L2, L3
  MPI_Finalize();
  free(array);
  exit(0);
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &global_world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &global_world_rank);

  int i, crash, reco;

  if (argc != 8) {
    if (global_world_rank == 0)
      printf(
          "Argc doesn't equeal 7! (run: ./ckptHierarchy configFile 1stCkpt "
          "2ndCkpt 3rdCkpt 4thCkpt (1/2/3/4) ifCrash ifReco(0/1) \n");
    MPI_Barrier(MPI_COMM_WORLD);
    return 1;
  }

  configFile = argv[1];
  for (i = 0; i < 4; i++)
    checkpoint_level[i] = atoi(argv[i + 2]);  // First number is in [2]
  crash = atoi(argv[6]);
  reco = atoi(argv[7]);

  initStatus = FTI_Init(configFile, MPI_COMM_WORLD);
  MPI_Comm_size(FTI_COMM_WORLD, &world_size);
  MPI_Comm_rank(FTI_COMM_WORLD, &world_rank);

  array = malloc(sizeof(int) * ARRAY_SIZE);
  if (reco == 0) {
    for (i = 0; i < ARRAY_SIZE; i++) {
      array[i] = (i + world_rank);
    }
  }

  FTI_Protect(1, FIRST, DATASET_SIZE, FTI_INTG);
  FTI_Protect(2, SECOND, DATASET_SIZE, FTI_INTG);
  FTI_Protect(3, THIRD, DATASET_SIZE, FTI_INTG);
  FTI_Protect(4, FOURTH, DATASET_SIZE, FTI_INTG);

  if (reco == 0) {
    for (i = 0; i < 4; i++) {
      FTI_Checkpoint(i + 1, checkpoint_level[i]);
    }
  } else {
    FTI_Recover();
  }

  if (crash == 1) {
    simulateCrash();
  }

  FTI_Finalize();
  MPI_Finalize();
  free(array);
  return 0;
}
