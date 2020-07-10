/**
 *  Copyright (c) 2017 Leonardo A. Bautista-Gomez
 *  All rights reserved
 *
 *  @file   heatdis.c
 *  @author Leonardo A. Bautista Gomez
 *  @date   May, 2014
 *  @brief  Heat distribution code to test FTI.
 */


#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>
#include <fti.h>
#include <math.h>
#include "cudaWrap.h"

#define PRECISION   0.005
#define ITER_TIMES  5
#define ITER_OUT    5
#define WORKTAG     50
#define REDUCE      5


void communicate(int numprocs, int rank, int32_t xElem, int32_t yElem,
 double *in, double *out, double *hallowArrays[], MPI_Comm comm) {
  MPI_Request req1[2], req2[2];
  MPI_Status status1[2], status2[2];

  if (rank > 0) {
    hostCopy(&in[yElem], hallowArrays[0], sizeof(double)*xElem);
    syncStream();
    MPI_Isend(hallowArrays[0], xElem, MPI_DOUBLE, rank-1,
     WORKTAG, comm , &req1[0]);
    MPI_Irecv(hallowArrays[1], xElem, MPI_DOUBLE, rank-1,
     WORKTAG, comm, &req1[1]);
  }

  if (rank < numprocs-1) {
    hostCopy(&in[(yElem-2)*xElem], hallowArrays[2], sizeof(double)*xElem);
    syncStream();
    MPI_Isend(hallowArrays[2], xElem, MPI_DOUBLE, rank+1,
     WORKTAG, comm, &req2[0]);
    MPI_Irecv(hallowArrays[3], xElem, MPI_DOUBLE, rank+1,
     WORKTAG, comm, &req2[1]);
  }

  if (rank > 0) {
    MPI_Waitall(2, req1, status1);
    deviceCopy(hallowArrays[1], &in[(yElem-2)*xElem], sizeof(double)*xElem);
  }
  if (rank < numprocs-1) {
    MPI_Waitall(2, req2, status2);
    deviceCopy(hallowArrays[3], &in[yElem], sizeof(double)*xElem);
  }
  MPI_Barrier(FTI_COMM_WORLD);
}



int main(int argc, char *argv[]) {
  int nbProcs;
  int32_t nbLines, i, j, M, arg;
  int rank;
  double wtime, *h, *g, memSize, localerror, globalerror = 1;
  double *hallowArrays[4];
  double *lError, *dError;

  MPI_Init(&argc, &argv);
  FTI_Init(argv[2], MPI_COMM_WORLD);

  MPI_Comm_size(FTI_COMM_WORLD, &nbProcs);
  MPI_Comm_rank(FTI_COMM_WORLD, &rank);

  int numDevices = getProperties();

  int myDevice = (rank)%numDevices;
  printf("My device is %d\n", myDevice);
  setDevice(myDevice);

  int32_t sizePerNode = (atol(argv[1]) *1024*512)/sizeof(double);
  M = sqrt(sizePerNode);
  nbLines = sizePerNode/M;
  initStream();


  allocateMemory((void **) &h, sizeof(double) * (M+2) * (nbLines+2));
  allocateMemory((void **) &g, sizeof(double) * (M+2) * (nbLines+2));
  allocateSafeHost((void **) &hallowArrays[0], sizeof(double)*(M+2));
  allocateSafeHost((void **) &hallowArrays[1], sizeof(double)*(M+2));
  allocateSafeHost((void **) &hallowArrays[2], sizeof(double)*(M+2));
  allocateSafeHost((void **) &hallowArrays[3], sizeof(double)*(M+2));
  allocateErrorMemory((void**) &lError, (void **) &dError, M+2 , nbLines+2);


  // These arrays will be used to tranfer data from the GPU to the host
  // and then send them to the neighbor

  if (rank == 0) {
    memSize = (double)(M * nbLines * 2 * sizeof(double)) /
    (double) (1024 * 1024 *1024);
    printf("Local data size is %ld x %ld = %g MB.\n", M, nbLines, memSize);
    printf("Allocated Extra %ld MB For padding .\n", 4*M/(1024*1024));
    printf("Target precision : %f \n", PRECISION);
    printf("Maximum number of iterations : %d \n", ITER_TIMES);
  }

  init(h, g, nbLines +2, M+2);

  double *temp;
  double localError;
  MPI_Barrier(FTI_COMM_WORLD);

  FTI_Protect(0, h, (M+2) * (nbLines+2) , FTI_DBLE);
  FTI_Protect(1, g, (M+2) * (nbLines+2) , FTI_DBLE);
  FTI_Protect(2, &i, 1 , FTI_INTG);
  syncStream();
  for (i = 0; i < 5; i++) {
    if (i == 1) {
      FTI_Checkpoint(i, 4);
    }
    communicate(nbProcs, rank, M+2, nbLines+2 , h, g, hallowArrays,
     FTI_COMM_WORLD);
    localError = executekernel(M+2, nbLines+2 , h, g, hallowArrays,
      dError, lError, rank);
    MPI_Allreduce(&localError, &globalerror, 1, MPI_DOUBLE, MPI_MAX,
     FTI_COMM_WORLD);
    temp = g;
    g = h;
    h = temp;
    if (rank == 0) {
      printf("%d) Step : %ld, error = %g\n", rank, i, globalerror);
    }
  }

  freeCuda(h);
  freeCuda(g);
  freeCuda(dError);
  freeCudaHost(hallowArrays[0]);
  freeCudaHost(hallowArrays[1]);
  freeCudaHost(hallowArrays[2]);
  freeCudaHost(hallowArrays[3]);
  free(lError);
  destroyStream();
  FTI_Finalize();
  MPI_Finalize();

  return 0;
}
