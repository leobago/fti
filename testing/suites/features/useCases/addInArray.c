/**
 *  Copyright (c) 2017 Leonardo A. Bautista-Gomez
 *  All rights reserved
 *
 *  @file   addInArray.c
 *  @author Karol Sierocinski (ksiero@man.poznan.pl)
 *  @date   Feburary, 2017
 *  @brief  FTI testing program.
 *
 *  Testing FTI_Init, FTI_Checkpoint, FTI_Status, FTI_Recover, FTI_Finalize,
 *  saving last checkpoint to PFS
 *
 *  Program adds number in array, does MPI_Allgather each iteration and
 *  checkpoint every ITER_CHECK interations with level passed in argv, but
 *  recovery is always from L4, because of FTI_Finalize() call.
 *
 *  First execution this program should be with fail flag = 1, because
 *  then FTI saves checkpoint and program stops after ITER_STOP iteration.
 *  Second execution must be with the same #defines and flag = 0 to
 *  properly recover data. It is important that FTI config file got
 *  keep_last_ckpt = 1.
 */

#include <fcntl.h>
#include <fti.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define ITERATIONS 120  // iterations for every level
#define ITER_CHECK 10   // every ITER_CHECK iterations make checkpoint
#define ITER_STOP 63    // stop work after ITER_STOP iterations

#define WORK_DONE 0
#define CHECKPOINT_FAILED 1
#define RECOVERY_FAILED 2

#define VERIFY_SUCCESS 0
#define VERIFY_FAILED 1

/*-------------------------------------------------------------------------*/
/**
    @brief      Do work to makes checkpoints
    @param      array     Pointer to array, length == app. proc.
    @param      wRank     FTI_COMM rank
    @param      wSize     FTI_COMM size
    @param      ckptLevel Checkpont level to all checkpoints
    @param      fail      True if stop after ITER_STOP, false if resuming work
    @return     integer   WORK_DONE if successful.
 **/
/*-------------------------------------------------------------------------*/
int do_work(int *array, int wRank, int wSize, int ckptLevel, int fail) {
  int res, number = wRank;
  int i = 0;
  // adding variables to protect
  FTI_Protect(1, &i, 1, FTI_INTG);
  FTI_Protect(2, &number, 1, FTI_INTG);

  // checking if this is recovery run
  if (FTI_Status() != 0 && fail == 0) {
    res = FTI_Recover();
    if (res != 0) {
      printf("%d: FTI_Recover returned %d.\n", wRank, res);
      return RECOVERY_FAILED;
    }
  }
  // if recovery, but recover values don't match
  if (fail == 0 && i != (ITER_STOP - ITER_STOP % ITER_CHECK))
    return RECOVERY_FAILED;
  if (wRank == 0) printf("Starting work at i = %d.\n", i);

  for (; i < ITERATIONS; i++) {
    // checkpoints after every ITER_CHECK iterations
    if (i % ITER_CHECK == 0) {
      res = FTI_Checkpoint(i / ITER_CHECK + 1, ckptLevel);
      if (res != FTI_DONE) {
        printf("%d: FTI_Checkpoint returned %d.\n", wRank, res);
        return CHECKPOINT_FAILED;
      }
    }
    MPI_Allgather(&number, 1, MPI_INT, array, 1, MPI_INT, FTI_COMM_WORLD);

    // stoping after ITER_STOP iterations
    if (fail && i >= ITER_STOP) {
      if (wRank == 0) {
        printf("Work stopped at i = %d.\n", ITER_STOP);
      }
      break;
    }
    number += 1;
  }
  return WORK_DONE;
}

/*-------------------------------------------------------------------------*/
/**
    @return     integer     0 if successful, 1 otherwise
 **/
/*-------------------------------------------------------------------------*/
int main(int argc, char **argv) {
  // Arguments
  int checkpoint_level, fail;
  char *config;

  // Variables
  int world_rank, world_size;  // FTI_COMM rank and size
  int *array;
  int rtn;

  if (argc != 4) {
    printf(
        "Missing arguments!! Usage: [ configFile (string), ckptLevel (int), "
        "simulateCrash (bool)]");
    return 0;
  }

  config = argv[1];
  checkpoint_level = atoi(argv[2]);
  fail = atoi(argv[3]);

  MPI_Init(&argc, &argv);
  FTI_Init(config, MPI_COMM_WORLD);

  MPI_Comm_rank(FTI_COMM_WORLD, &world_rank);
  MPI_Comm_size(FTI_COMM_WORLD, &world_size);

  array = (int*) malloc(sizeof(int)*world_size);
  rtn = do_work(array, world_rank, world_size, checkpoint_level, fail);

  // Verify if array values are correct, if necessary
  int i;
  if (world_rank == 0 && rtn == 0 && !fail) {
    for (i = 0; i < world_size; i++) {
      if (array[i] != ITERATIONS + (i - 1)) {
        printf("Failure: array[%d] = %d, should be %d.\n", i, array[i],
               ITERATIONS + (i - 1));
        rtn = VERIFY_FAILED;
        break;
      }
    }
  }
  free(array);
  FTI_Finalize();
  MPI_Finalize();
  return rtn;
}
