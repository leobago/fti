/**
 *  @file   heatdis.c
 *  @author Leonardo A. Bautista Gomez
 *  @date   May, 2014
 *  @brief  Heat distribution code to test FTI.
 */


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <fti.h>
#include <time.h>


#define PRECISION   0.0000001
#define ITER_TIMES  1000000
#define ITER_OUT    5
#define WORKTAG     50
#define REDUCE      5

int FIRST_CHECKPOINT;
int SECOND_CHECKPOINT;
int CHECKPOINT_ITER;
int LAST_CHECKPOINT;
bool isRestart;
bool toRecover;
bool toFinalize;
bool isCheckpointed;

void Snapshot_distance( int iter, double wtime, int rank ) {
    
    static double ckpt1_t;

    if ( isRestart ) {
        if ( toRecover ) {
            FTI_Recover();
            toRecover = false;
        }
        if ( toFinalize ) {
            FTI_Finalize();
            MPI_Finalize();
            exit(EXIT_SUCCESS);
        }
        if ( (iter%5) == 0 ) {
            toFinalize = true;
        }
    } else {
        if ( iter == FIRST_CHECKPOINT ) {
            FTI_Checkpoint( iter, 2 );
            ckpt1_t = MPI_Wtime();
        }
        if ( iter == SECOND_CHECKPOINT ) {
            if (rank == 0) {
                printf("Second checkpoint after %lf seconds.\n", MPI_Wtime() - ckpt1_t);
            }
            FTI_Checkpoint( iter, 2 );
            isCheckpointed = true;
        }
        if ( toFinalize ) {
            if (rank == 0) {
                printf("Execution finished in %lf seconds.\n", MPI_Wtime() - wtime);
            }
            MPI_Finalize();
            exit(EXIT_SUCCESS);
        }
        if ( (iter%5) == 0 && isCheckpointed ) {
            toFinalize = true;
        }
    }
}

void Snapshot( int iter, double wtime, int rank, double *ptr1, double *ptr2 ) {
    
    static double ckpt1_t;

    if ( isRestart ) {
        if ( toRecover ) {
            FTI_Recover();
            toRecover = false;
        }
        if ( toFinalize ) {
            FTI_Finalize();
            MPI_Finalize();
            exit(EXIT_SUCCESS);
        }
        if ( (iter%5) == 0 ) {
            toFinalize = true;
        }
    } else {
        if ( (iter%CHECKPOINT_ITER == 0) && iter > 0 ) {
            FTI_Checkpoint( iter, 2 );
            ckpt1_t = MPI_Wtime();
        }
        if ( iter == LAST_CHECKPOINT ) {
            if (rank == 0) {
                printf("Execution finished in %lf seconds.\n", MPI_Wtime() - wtime);
            }
            free(ptr1);
            free(ptr2);
            FTI_Finalize();
            MPI_Finalize();
            exit(EXIT_SUCCESS);
        }
    }
}

void initData(int nbLines, int M, int rank, double *h)
{
    int i, j;
    srand(time(NULL));
    for (i = 0; i < nbLines; i++) {
        for (j = 0; j < M; j++) {
            if((i%2 == 0) && (rank%512 == 0)) {
                if(j%2 == 0) {
                    h[(i*M)+j] = rand()%100+1;
                } else {
                    h[(i*M)+j] = 0;
                }
            }
            else if((i%2 != 0) && (rank%512 == 0)) {
                if(j%2 != 0) {
                    h[(i*M)+j] = rand()%100+1;
                } else {
                    h[(i*M)+j] = 0;
                }
            } else {
                h[(i*M)+j] = 0;
            }
        }
    }
//    if (rank%2 == 0) {
//        for (j = (M*0.1); j < (M*0.9); j++) {
//            h[j] = rand()%100+1;
//        }
//    }
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

    MPI_Init(&argc, &argv);
    FTI_Init(argv[2], MPI_COMM_WORLD);
    
    MPI_Comm_size(FTI_COMM_WORLD, &nbProcs);
    MPI_Comm_rank(FTI_COMM_WORLD, &rank);
    
    toFinalize = false;
    isCheckpointed = false;
    if ( FTI_Status() == 1 ) {
        toRecover = true;
        isRestart = true;
    }

    //FIRST_CHECKPOINT = atoi(getenv("FTI_FIRST_CKPT"));
    //SECOND_CHECKPOINT = atoi(getenv("FTI_SECOND_CKPT"));
    CHECKPOINT_ITER = atoi(getenv("FTI_CKPT_ITER"));
    LAST_CHECKPOINT = atoi(getenv("FTI_LAST_CKPT"));
    
    arg = atoi(argv[1]);
    M = (int)sqrt((double)(arg * 1024.0 * 512.0 * nbProcs)/sizeof(double));
    nbLines = (M / nbProcs)+3;
    h = (double *) malloc(sizeof(double *) * M * nbLines);
    g = (double *) malloc(sizeof(double *) * M * nbLines);
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

    wtime = MPI_Wtime();
    for (i = 0; i < ITER_TIMES; i++) {
        Snapshot( i, wtime, rank, h, g );
        localerror = doWork(nbProcs, rank, M, nbLines, g, h);
        if ((i%REDUCE) == 0) {
            MPI_Allreduce(&localerror, &globalerror, 1, MPI_DOUBLE, MPI_MAX, FTI_COMM_WORLD);
        }
        if (((i%ITER_OUT) == 0) && (rank == 0)) {
            printf("Step : %d, error = %f\n", i, globalerror);
        }
        if(globalerror < PRECISION) {
            break;
        }
    }
    if (rank == 0) {
        printf("Execution finished in %lf seconds.\n", MPI_Wtime() - wtime);
    }

    free(h);
    free(g);

    FTI_Finalize();
    MPI_Finalize();
    return 0;
}
