/**
 *  Copyright (c) 2017 Leonardo A. Bautista-Gomez
 *  All rights reserved
 *
 *  @file   massive.c
 *  @author Kai Keller (kellekai@gmx.de)
 *  @date   June, 2017
 *  @brief  FTI testing program.
 **/

#include <assert.h>
#include <errno.h>
#include <mpi.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "fti.h"

#define EXIT_FAIL(MSG)                                          \
  do {                                                          \
    if (rank == 0) {                                            \
      printf("%s:%d [ERROR] -> %s\n", __FILE__, __LINE__, MSG); \
    }                                                           \
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);                    \
  } while (0)

#define FILE_SIZE 1024  // 1KB
#define NUM_ITER 100L
#define FILES_PER_ITER 10L
#define CLEAN_FREQ 10
#define F_BUFF 512
#define F_FORM "file-rank%04d-iter%05lu-numb%02lu.fti"
#define REMOTE_DIR "./rdir"

void createFile(char *fn);
bool check_status(int request_counter, int *reqID, bool printout);

int rank, size;
unsigned long num_files;

int main(int argc, char *argv[]) {
  if (argc < 1) {
    printf("This test requires an FTI config file path as the first argument.");
    exit(1);
  }

  MPI_Init(NULL, NULL);
  FTI_Init(argv[1], MPI_COMM_WORLD);
  MPI_Comm_rank(FTI_COMM_WORLD, &rank);
  MPI_Comm_size(FTI_COMM_WORLD, &size);

  // total number of staged files
  num_files = ((unsigned long)size) * FILES_PER_ITER * NUM_ITER;

  // request ID array
  int *reqID = (int *)malloc(FILES_PER_ITER * NUM_ITER * sizeof(int));

  // set stage and remote directory
  char ldir[F_BUFF / 2]; // Divided buffer by half to allow merging
  char rdir[F_BUFF / 2] = REMOTE_DIR;
  if (FTI_GetStageDir(ldir, F_BUFF) != FTI_SCES) {
    EXIT_FAIL("Failed to get the local directory.");
  }

  // create remote directory
  errno = 0;
  if (mkdir(rdir, (mode_t)0700) != 0) {
    if (errno != EEXIST) {
      char msg[F_BUFF];
      snprintf(msg, F_BUFF, "unable to create remote directory ('%s').", rdir);
      EXIT_FAIL(msg);
    }
  }

  // allocate filename arrays
  char *lfile[FILES_PER_ITER];
  char *rfile[FILES_PER_ITER];
  char *filename[FILES_PER_ITER];
  unsigned long i;
  for (i = 0; i < FILES_PER_ITER; ++i) {
    lfile[i] = (char *)malloc(F_BUFF);
    rfile[i] = (char *)malloc(F_BUFF);
    filename[i] = (char *)malloc(F_BUFF);
  }

  // perform staging
  unsigned long request_counter = 0;
  for (i = 0; i < NUM_ITER; ++i) {
    unsigned long j = 0;
    for (; j < FILES_PER_ITER; ++j) {
      snprintf(filename[j], F_BUFF, F_FORM, rank, i, j);
      snprintf(lfile[j], F_BUFF, "%s/%s", ldir, filename[j]);
      snprintf(rfile[j], F_BUFF, "%s/%s", rdir, filename[j]);
      createFile(lfile[j]);
      if ((reqID[i * FILES_PER_ITER + j] = FTI_SendFile(lfile[j], rfile[j])) ==
          FTI_NSCS) {
        char msg[F_BUFF];
        snprintf(msg, F_BUFF, "Failed to stage %s.", filename[j]);
        EXIT_FAIL(msg);
      }
      request_counter++;
    }
    if (i % CLEAN_FREQ == 0) {
      check_status(request_counter, reqID, true);
    }
  }

  while (check_status(request_counter, reqID, true)) {
    sleep(2);
  }

  FTI_Finalize();

  // remove files
  for (i = 0; i < NUM_ITER; ++i) {
    unsigned long j = 0;
    for (; j < FILES_PER_ITER; ++j) {
      snprintf(filename[j], F_BUFF, F_FORM, rank, i, j);
      snprintf(rfile[j], F_BUFF, "%s/%s", rdir, filename[j]);
      errno = 0;
      if (remove(rfile[j]) != 0) {
        if (errno != ENOENT) {
          char msg[F_BUFF];
          snprintf(msg, F_BUFF, "Failed to remove %s ('%s').", filename[j],
                   strerror(errno));
          EXIT_FAIL(msg);
        }
      }
    }
  }

  MPI_Finalize();

  return EXIT_SUCCESS;
}

void createFile(char *fn) {
  FILE *fstream = fopen(fn, "wb+");
  fsync(fileno(fstream));
  fclose(fstream);
  truncate(fn, FILE_SIZE);
}

bool check_status(int request_counter, int *reqID, bool printout) {
  unsigned long completed = 0;
  unsigned long active = 0;
  unsigned long j;

  bool wait = true;

  for (j = 0; j < request_counter; ++j) {
    if (reqID[j] == -1) {
      completed++;
      continue;
    }
    int status = FTI_GetStageStatus(reqID[j]);
    if (status == FTI_SI_FAIL) {
      char msg[F_BUFF];
      snprintf(msg, F_BUFF,
               "Stage request with ID = %d returned status failure.", reqID[j]);
      EXIT_FAIL(msg);
    }
    if (status == FTI_SI_SCES) {
      reqID[j] = -1;
      completed++;
      continue;
    }
    assert(status != FTI_SI_NINI);
    active++;
  }

  if (printout) {
    unsigned long completed_all;
    double acc;
    MPI_Allreduce(&completed, &completed_all, 1, MPI_UNSIGNED_LONG, MPI_SUM,
                  FTI_COMM_WORLD);
    acc = 100 * ((double)completed_all) / num_files;
    if (rank == 0) {
      printf("[ %.2lf%% of staging completed ( %lu / %lu )... ]\n", acc,
             completed_all, num_files);
    }
    wait = !(completed_all == num_files);
  }

  return wait;
}
