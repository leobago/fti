/**
 *  @file   heatdis.c
 *  @author Leonardo A. Bautista Gomez
 *  @date   May, 2014
 *  @brief  Heat distribution code to test FTI.
 */


#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <fti.h>
#include <math.h>
#include <cuda_runtime_api.h>
#include "cudaWrap.h"

#define PRECISION   0.000005
#define ITER_TIMES  15000
#define ITER_OUT    5
#define WORKTAG     50
#define REDUCE      5


void communicate(int numprocs, int rank, long xElem, long yElem, double *in, double *out, double *hallowArrays[], MPI_Comm comm, cudaStream_t *appStream){
  MPI_Request req1[2], req2[2];
  MPI_Status status1[2], status2[2];

  if (rank > 0) {
    hostCopy( &in[xElem],hallowArrays[0], sizeof(double)*xElem);
    syncStream(appStream);
    MPI_Isend(hallowArrays[0], xElem, MPI_DOUBLE, rank-1, WORKTAG, comm , &req1[0]);
    MPI_Irecv(hallowArrays[1], xElem, MPI_DOUBLE, rank-1, WORKTAG, comm, &req1[1]);
  }

  if (rank < numprocs-1) {
    hostCopy(&in[(yElem-2)*xElem],hallowArrays[2],  sizeof(double)*xElem);
    syncStream(appStream);
    MPI_Isend(hallowArrays[2], xElem, MPI_DOUBLE, rank+1, WORKTAG, comm, &req2[0]);
    MPI_Irecv(hallowArrays[3], xElem, MPI_DOUBLE, rank+1, WORKTAG, comm, &req2[1]);
  }

  if (rank > 0) {
    MPI_Waitall(2,req1,status1);
    deviceCopy(hallowArrays[1],&in[0], sizeof(double)*xElem);
  }
  if (rank < numprocs-1) {
    MPI_Waitall(2,req2,status2);
    deviceCopy(hallowArrays[3],&in[(yElem-1)*xElem], sizeof(double)*xElem);
  }
  MPI_Barrier(FTI_COMM_WORLD);
}



int main(int argc, char *argv[])
{
  int nbProcs;
  double start, end;
  long nbLines, i,j, M, arg;
  int rank;
  double wtime, *h, *g, memSize, localerror, globalerror = 1;
  double *hallowArrays[4];
  double *lError, *dError;

  MPI_Init(&argc, &argv);
  int level = atoi(argv[3]);
  int version = atoi(argv[4]);

  int numDevices = getProperties();
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int myDevice = (rank)%numDevices;

  setDevice(myDevice);

  FTI_Init(argv[2],MPI_COMM_WORLD);


  MPI_Comm_size(FTI_COMM_WORLD, &nbProcs);
  MPI_Comm_rank(FTI_COMM_WORLD, &rank);



  long sizePerNode = (atol(argv[1]) *1024*1024*512)/sizeof(double);
  M = (int)sqrt(sizePerNode);
  nbLines = sizePerNode/M;
  cudaStream_t appStream;
  initStream(&appStream);


  allocateMemory((void **) &h, sizeof(double) * (M+2) * (nbLines+2));
  allocateMemory((void **) &g, sizeof(double) * (M+2) * (nbLines+2));
  allocateSafeHost((void **) &hallowArrays[0], sizeof(double)*(M+2) );
  allocateSafeHost((void **) &hallowArrays[1], sizeof(double)*(M+2) );
  allocateSafeHost((void **) &hallowArrays[2], sizeof(double)*(M+2) );
  allocateSafeHost((void **) &hallowArrays[3], sizeof(double)*(M+2) );



  allocateErrorMemory( (void**) &lError, (void **) &dError, M+2 , nbLines+2, rank);


  // These arrays will be used to tranfer data from the GPU to the host
  // and then send them to the neighbor

  if (rank == 0 ){
    memSize = (double)(M * nbLines * 2 * sizeof(double)) /(double) (1024 * 1024 *1024);
    printf("Local data size is %ld x %ld = %g GB.\n", M, nbLines, memSize );
    printf("Allocated Extra %ld MB For padding .\n", 4*M/(1024*1024));
    printf("Target precision : %f \n", PRECISION);
    printf("Maximum number of iterations : %d \n", ITER_TIMES);
    printf("Maximum number of iterations : %ld -- %ld \n", M, nbLines);
  }

  init(h,g, nbLines +2, M+2, rank);

  double *temp;
  double localError;
  MPI_Barrier(FTI_COMM_WORLD);

  FTI_Protect(0, h, (M+2) * (nbLines+2) , FTI_DBLE);
  FTI_Protect(1, g, (M+2) * (nbLines+2) , FTI_DBLE);
  FTI_Protect(2, &i, 1 , FTI_INTG);
  MPI_Barrier(FTI_COMM_WORLD); /* IMPORTANT */

  if (version){
    if ( FTI_Status() != 0){
#ifdef __RECOVER__ 
    startCount("TotalRecover");
    MPI_Barrier(FTI_COMM_WORLD);
    FTI_Recover();
    MPI_Barrier(FTI_COMM_WORLD);
    stopCount("TotalRecover");
    MPI_Barrier(FTI_COMM_WORLD);
    FTI_Finalize();
    MPI_Finalize();
    return 0;
#endif
    }
  }

  MPI_Barrier(FTI_COMM_WORLD);
  start = MPI_Wtime();
  double total = start;
  double interval = 0.0;
  double elapsed = 0;
  i = 0;
  while( elapsed < 20.0 ){
    MPI_Barrier(FTI_COMM_WORLD);
    interval = MPI_Wtime() - start;
    double avgTime;
    MPI_Allreduce(&interval, &avgTime, 1, MPI_DOUBLE, MPI_SUM, FTI_COMM_WORLD);
    avgTime = avgTime/(double)nbProcs;
    if (avgTime > 30.0 && version){
     if (rank == 0 )
       printf("Elapsed time is %g %ld, Total Time is %g\n", interval,i,MPI_Wtime()-total);
     FTI_Checkpoint(i,level);
     MPI_Barrier(FTI_COMM_WORLD);
     start = MPI_Wtime();
#ifdef __RECOVER__ 
#warning compiling recover code
        sleep(60);
        FTI_CleanGPU();
        sleep(5);
        exit(-1);
#endif
    }
    communicate(nbProcs, rank, M+2, nbLines+2 , h, g, hallowArrays, FTI_COMM_WORLD, &appStream);
    localError = executekernel(M, nbLines , h, g,  dError, lError, rank, myDevice);
    if ( localError < 0.0 )
      break;

    MPI_Allreduce(&localError, &globalerror, 1, MPI_DOUBLE, MPI_MAX, FTI_COMM_WORLD);
    if (rank == 0) { /* use time on master node */
      printf("Error is %g Elapsed Time %g\n", globalerror,avgTime);
    }
    temp = g;
    g = h;
    h = temp;
    MPI_Barrier(FTI_COMM_WORLD);
    elapsed  = MPI_Wtime()-total;
    MPI_Allreduce(&elapsed, &avgTime, 1, MPI_DOUBLE, MPI_SUM, FTI_COMM_WORLD);
    elapsed = avgTime/(double)nbProcs;
    elapsed = elapsed/60.0; // Minutes;
    i++;
  }

  MPI_Barrier(FTI_COMM_WORLD); /* IMPORTANT */
  end = MPI_Wtime();
  if (rank == 0) { /* use time on master node */
    printf("Runtime = %f Computed %ld iterations\n", end-total,i);
  }


  freeCuda(h);
  freeCuda(g);
  freeCuda(dError);
  freeCudaHost(hallowArrays[0]);
  freeCudaHost(hallowArrays[1]);
  freeCudaHost(hallowArrays[2]);
  freeCudaHost(hallowArrays[3]);
  free(lError);
  FTI_Finalize();
  MPI_Finalize();

  return 0;
}
