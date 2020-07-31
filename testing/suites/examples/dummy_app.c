/**
 *  Copyright (c) 2017 Leonardo A. Bautista-Gomez
 *  All rights reserved
 *
 *  @file   dummy_app.c
 *  @author Alexandre de Limas Santana (alexandre.delimassantana@bsc.es)
 *  @date   June, 2020
 *  @brief  ITF example program.
 **/

#include <fti.h>
#include <stdio.h>

int main(int argc, char** argv) {
  MPI_Init(NULL, NULL);

  // First argument is the path to an FTI configuration file
  FTI_Init(argv[1], MPI_COMM_WORLD);

  int world_size, world_rank, name_len;
  char processor_name[MPI_MAX_PROCESSOR_NAME];

  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Get_processor_name(processor_name, &name_len);

  printf("Hello world from processor %s, rank %d out of %d processors\n",
         processor_name, world_rank, world_size);

  // If the second argument is non-zero, simulate a crash
  if (atoi(argv[2]) != 0)
    FTI_Finalize();
  MPI_Finalize();

  // The third argument is the return value for the application
  return atoi(argv[3]);
}