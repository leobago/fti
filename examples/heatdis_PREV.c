/**
 *  @file   heatdis.c
 *  @author Leonardo A. Bautista Gomez
 *  @date   May, 2014
 *  @brief  Heat distribution code to test FTI.
 */


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fti.h>
#include <assert.h>
#include <string.h>
#include <time.h>

#define PRECISION   0.005
#define ITER_TIMES  6000
#define ITER_OUT    100
#define WORKTAG     50
#define REDUCE      1


void initData(int nbLines, int M, int rank, double *h)
{
    int i, j;
    for (i = 0; i < nbLines; i++) {
        for (j = 0; j < M; j++) {
            h[(i*M)+j] = 0;
        }
    }
    if (rank == 0) {
        for (j = (M*0.1); j < (M*0.9); j++) {
            h[j] = 100;
        }
    }
}


double doWork(int numprocs, int rank, int M, int nbLines, double *g, double *h)
{
    int i,j;
    MPI_Request req1[2], req2[2];
    MPI_Status status1[2], status2[2];
    double localerror;
    localerror = 0;
    for(i = 0; i < nbLines; i++) {
        for(j = 0; j < M; j++) {
            h[(i*M)+j] = g[(i*M)+j];
        }
    }
    if (rank > 0) {
        MPI_Isend(g+M, M, MPI_DOUBLE, rank-1, WORKTAG, FTI_COMM_WORLD, &req1[0]);
        MPI_Irecv(h,   M, MPI_DOUBLE, rank-1, WORKTAG, FTI_COMM_WORLD, &req1[1]);
    }
    if (rank < numprocs-1) {
        MPI_Isend(g+((nbLines-2)*M), M, MPI_DOUBLE, rank+1, WORKTAG, FTI_COMM_WORLD, &req2[0]);
        MPI_Irecv(h+((nbLines-1)*M), M, MPI_DOUBLE, rank+1, WORKTAG, FTI_COMM_WORLD, &req2[1]);
    }
    if (rank > 0) {
        MPI_Waitall(2,req1,status1);
    }
    if (rank < numprocs-1) {
        MPI_Waitall(2,req2,status2);
    }
    for(i = 1; i < (nbLines-1); i++) {
        for(j = 0; j < M; j++) {
            g[(i*M)+j] = 0.25*(h[((i-1)*M)+j]+h[((i+1)*M)+j]+h[(i*M)+j-1]+h[(i*M)+j+1]);
            if(localerror < fabs(g[(i*M)+j] - h[(i*M)+j])) {
                localerror = fabs(g[(i*M)+j] - h[(i*M)+j]);
            }
        }
    }
    if (rank == (numprocs-1)) {
        for(j = 0; j < M; j++) {
            g[((nbLines-1)*M)+j] = g[((nbLines-2)*M)+j];
        }
    }
    return localerror;
}


int main(int argc, char *argv[])
{
    int rank, nbProcs, nbLines, i, M, arg;
    double wtime, *h, *g, memSize, localerror, globalerror = 1;
    srand(time(NULL));

    MPI_Init(&argc, &argv);
    FTI_Init(argv[2], MPI_COMM_WORLD);

    MPI_Comm_size(FTI_COMM_WORLD, &nbProcs);
    MPI_Comm_rank(FTI_COMM_WORLD, &rank);

    arg = atoi(argv[1]);
    M = (int)sqrt((double)(arg * 1024.0 * 512.0 * nbProcs)/sizeof(double));
    nbLines = (M / nbProcs)+3;
    h = (double *) malloc(sizeof(double) * M * nbLines);
    g = (double *) malloc(sizeof(double) * M * nbLines);

    double *aux_h,*aux_g;int j;
    aux_h = (double *) malloc(sizeof(double) * M * nbLines);
    aux_g = (double *) malloc(sizeof(double) * M * nbLines);

    initData(nbLines, M, rank, g);
    memSize = M * nbLines * 2 * sizeof(double) / (1024 * 1024);

    if (rank == 0) {
        printf("Local data size is %d x %d = %f MB (%d).\n", M, nbLines, memSize, arg);
        printf("Target precision : %f \n", PRECISION);
        printf("Maximum number of iterations : %d \n", ITER_TIMES);
    }

    FTI_Protect(0, &i, 1, FTI_INTG);
    FTI_Protect(1, h, M*nbLines, FTI_DBLE);
    FTI_Protect(2, g, M*nbLines, FTI_DBLE);

    int itersAfterCKPT=-1;
    wtime = MPI_Wtime();
    for (i = 0; i < ITER_TIMES; i++) {
        if(itersAfterCKPT>=0) itersAfterCKPT++;
        int checkpointed = FTI_Snapshot();
        if(FTI_CheckCheckpointDone()){
//            itersAfterCKPT=0;
//        }
//        if(itersAfterCKPT==400){
            memcpy(aux_h, h, M*nbLines*sizeof(double));
            memcpy(aux_g, g, M*nbLines*sizeof(double));
            j=i;
//            int r;
//            r = rand();
//            if((r%3)==0)
//                FTI_DestroyData(&i, 1*sizeof(int));
            if(rank>=0){
                FTI_DestroyData(&i, 1*sizeof(int));
                FTI_DestroyData(h, M*nbLines*sizeof(double));
                FTI_DestroyData(g, M*nbLines*sizeof(double));
            }

        }

        int *status_array;
        if(FTI_GlobalErrDetected(&status_array)==FTI_SCES){
            if(status_array[rank]==1){
                FTI_RecoverLocalCkpt();
            }
            assert( memcmp(h,aux_h,M*nbLines*sizeof(double)) ==0);
            assert( memcmp(g,aux_g,M*nbLines*sizeof(double)) ==0);
            assert(i==j);
            /* application recovery routine */
            /* Agree about iteration:
             * what happens if survivors in different ones?? Not taken into account now.
             * */
//            int current_index;
//            int aux_current_index = (status_array[rank]==1? -1: i);
//            MPI_Allreduce(&aux_current_index, &current_index, 1,
//                    MPI_INT, MPI_MAX, FTI_COMM_WORLD);
//            MPI_Request req1[2], req2[2];
//            MPI_Status status1[2], status2[2];
//            int exchange_rows=nbLines/2;
//
////            if(status_array[rank]==1){ /* FAILED PROC */
////                i=current_index;
////            }
//
//            if(status_array[rank]==1){ /* FAILED PROC */
//                i=current_index;
//
//                int k,l;
//                for(k = 0; k < nbLines; k++) {
//                    for(l = 0; l < M; l++) {
//                        h[(k*M)+l] = g[(k*M)+l];
//                    }
//                }
//
//
//                int nops_1=0,nops_2=0;
//                if (rank > 0) {
//                    if(status_array[rank-1]==0){
//                        printf("Failed %d recv from survivor %d in G(1:%d,:)\n",rank, rank-1,exchange_rows);
//                        /* Receive what I originally sent */
//                        MPI_Irecv(g+M, M*exchange_rows, MPI_DOUBLE, rank-1, WORKTAG, FTI_COMM_WORLD, &req1[nops_1]);
//                        nops_1++;
//                    }
//                }
//                if (rank < nbProcs-1) {
//                    if(status_array[rank+1]==0){
//                        printf("Failed %d recv from survivor %d in G(%d:%d,:)\n",rank, rank+1,nbLines-2+1-exchange_rows, nbLines-2);
//                        MPI_Irecv(g+((nbLines-2+1-exchange_rows)*M), M*exchange_rows, MPI_DOUBLE, rank+1, WORKTAG, FTI_COMM_WORLD, &req2[nops_2]);
//                        nops_2++;
//                    }
//                }
//                if (rank > 0) {
//                    MPI_Waitall(nops_1,req1,status1);
//                }
//                if (rank < nbProcs-1) {
//                    MPI_Waitall(nops_2,req2,status2);
//                }
////                for(k = 1; k < (nbLines-1); k++) {
////                    for(l = 0; l < M; l++) {
////                        g[(k*M)+l] = 0.25*(h[((k-1)*M)+l]+h[((k+1)*M)+l]+h[(k*M)+l-1]+h[(k*M)+l+1]);
////                        if(localerror < fabs(g[(k*M)+l] - h[(k*M)+l])) {
////                            localerror = fabs(g[(k*M)+l] - h[(k*M)+l]);
////                        }
////                    }
////                }
////
////                if (rank == (nbProcs-1)) {
////                    for(j = 0; j < M; j++) {
////                        g[((nbLines-1)*M)+j] = g[((nbLines-2)*M)+j];
////                    }
////                }
//            }else{ /* SURVIVOR PROC */
//                if (rank > 0) {
//                    if(status_array[rank-1]==1){
//                        /*Send what I originally received*/
//                        printf("Survivor %d send to failed %d H(0:%d,:)\n",rank, rank-1,exchange_rows);
//                        MPI_Send(g,   M*exchange_rows, MPI_DOUBLE, rank-1, WORKTAG, FTI_COMM_WORLD);
//                    }
//                }
//                if (rank < nbProcs-1) {
//                    if(status_array[rank+1]==1){
//                        /*Send what I originally received*/
//                        printf("Survivor %d send to failed %d H(%d:%d,:)\n",rank, rank+1,nbLines-1+1-exchange_rows,nbLines-1);
//                        MPI_Send(g+((nbLines-1+1-exchange_rows)*M), M*exchange_rows, MPI_DOUBLE, rank+1, WORKTAG, FTI_COMM_WORLD);
//                    }
//                }
//            }

            /* ... */
            FTI_FinishRecovery(&status_array);
        }

        localerror = doWork(nbProcs, rank, M, nbLines, g, h);
        if ((i%REDUCE) == 0) {
            MPI_Allreduce(&localerror, &globalerror, 1, MPI_DOUBLE, MPI_MAX, FTI_COMM_WORLD);
        }
//        if (((i%ITER_OUT) == 0) && (rank == 0)) {
//            printf("Step : %d, error = %f\n", i, globalerror);
//        }
        if (((i%ITER_OUT) == 0) /*&& (rank == 0)*/) {
            printf("%d Step : %d, error = %f local %f\n",rank, i, globalerror, localerror);
        }

        if(globalerror < PRECISION) {
            printf("Step : %d, error = %f: done, less than precision %f \n", i, globalerror, PRECISION);
            break;
        }
    }
    if (rank == 0) {
        printf("Execution finished in %lf seconds.\n", MPI_Wtime() - wtime);
    }

    free(h);
    free(g);
    free(aux_h);
    free(aux_g);

    FTI_Finalize();
    MPI_Finalize();
    return 0;
}
