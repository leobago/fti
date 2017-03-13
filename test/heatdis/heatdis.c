/**
 *  @file   heatdis.c
 *  @author Leonardo A. Bautista Gomez (../../exaples/heatdis.c) & Karol Sierocinski
 *  @date   March, 2017
 *  @brief  Heat distribution code to test FTI.
 *
 *  Program tests FTI_Snapshot.
 *
 *  First execution this program should be with fail flag = 1, because
 *  then FTI saves checkpoint and program stops after ITER_STOP iteration.
 *  Second execution must be with the same #defines and flag = 0 to
 *  properly recover data. It is important that FTI config file got
 *  keep_last_ckpt = 1.
 */


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fti.h>

//do not change this defines (static verifying)
#define PRECISION   0.005
#define ITER_TIMES  5000
#define ITER_OUT    500
#define WORKTAG     50
#define REDUCE      5

#define ITER_STOP 3100    //simulate failure after ITER_STOP iterations

#define WORK_DONE 0
#define WORK_STOPED 1
#define SNAPSHOT_FAILED 2

#define VERIFY_SUCCESS 0
#define VERIFY_FAILED 1

void initData(int nbLines, int M, int rank, double *h)
{
    int i, j;
    for (i = 0; i < nbLines; i++)
    {
        for (j = 0; j < M; j++)
        {
            h[(i*M)+j] = 0;
        }
    }
    if (rank == 0)
    {
        for (j = (M*0.1); j < (M*0.9); j++)
        {
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
    for(i = 0; i < nbLines; i++)
    {
        for(j = 0; j < M; j++)
        {
            h[(i*M)+j] = g[(i*M)+j];
        }
    }
    if (rank > 0)
    {
        MPI_Isend(g+M, M, MPI_DOUBLE, rank-1, WORKTAG, FTI_COMM_WORLD, &req1[0]);
        MPI_Irecv(h,   M, MPI_DOUBLE, rank-1, WORKTAG, FTI_COMM_WORLD, &req1[1]);
    }
    if (rank < numprocs-1)
    {
        MPI_Isend(g+((nbLines-2)*M), M, MPI_DOUBLE, rank+1, WORKTAG, FTI_COMM_WORLD, &req2[0]);
        MPI_Irecv(h+((nbLines-1)*M), M, MPI_DOUBLE, rank+1, WORKTAG, FTI_COMM_WORLD, &req2[1]);
    }
    if (rank > 0)
    {
        MPI_Waitall(2,req1,status1);
    }
    if (rank < numprocs-1)
    {
        MPI_Waitall(2,req2,status2);
    }
    for(i = 1; i < (nbLines-1); i++)
    {
        for(j = 0; j < M; j++)
        {
            g[(i*M)+j] = 0.25*(h[((i-1)*M)+j]+h[((i+1)*M)+j]+h[(i*M)+j-1]+h[(i*M)+j+1]);
            if(localerror < fabs(g[(i*M)+j] - h[(i*M)+j]))
            {
                localerror = fabs(g[(i*M)+j] - h[(i*M)+j]);
            }
        }
    }
    if (rank == (numprocs-1))
    {
        for(j = 0; j < M; j++)
        {
            g[((nbLines-1)*M)+j] = g[((nbLines-2)*M)+j];
        }
    }
    return localerror;
}

int init(char** argv, int* fail) {
    int rtn = 0;    //return value
    if (argv[1] == NULL) {
        printf("Missing first parameter (config file).\n");
        rtn = 1;
    }
    if (argv[2] == NULL) {
        printf("Missing third parameter (if fail).\n");
        rtn = 1;
    } else {
        *fail = atoi(argv[2]);
    }
    return rtn;
}

int verify (double globalerror, int rank){
    if (fabs(globalerror - 0.004998) <= 0.000001) {
        return 0;
    }
    printf("%d: globalerror = %f, should be 0.004998\n", rank, globalerror);
    return 1;
}

/*-------------------------------------------------------------------------*/
/**
    @return     integer     0 if successful, 1 otherwise
 **/
/*-------------------------------------------------------------------------*/
int main(int argc, char** argv)
{
    int fail, rank, nbProcs, nbLines, i, M, arg;
    double wtime, *h, *g, memSize, localerror, globalerror = 1;

    if (init(argv, &fail)) return 0;   //verify args

    MPI_Init(&argc, &argv);
    FTI_Init(argv[1], MPI_COMM_WORLD);

    MPI_Comm_size(FTI_COMM_WORLD, &nbProcs);
    MPI_Comm_rank(FTI_COMM_WORLD, &rank);

    arg = 4;
    M = (int)sqrt((double)(arg * 1024.0 * 512.0 * nbProcs)/sizeof(double));
    nbLines = (M / nbProcs)+3;
    h = (double *) malloc(sizeof(double *) * M * nbLines);
    g = (double *) malloc(sizeof(double *) * M * nbLines);
    initData(nbLines, M, rank, g);
    memSize = M * nbLines * 2 * sizeof(double) / (1024 * 1024);

    if (rank == 0) printf("Local data size is %d x %d = %f MB (%d).\n", M, nbLines, memSize, arg);
    if (rank == 0) printf("Target precision : %f \n", PRECISION);
    if (rank == 0) printf("Maximum number of iterations : %d \n", ITER_TIMES);

    //adding variables to protect
    FTI_Protect(0, &i, 1, FTI_INTG);
    FTI_Protect(1, h, M*nbLines, FTI_DBLE);
    FTI_Protect(2, g, M*nbLines, FTI_DBLE);
    int iTmp = 0;
    wtime = MPI_Wtime();
    for(i = 0; i < ITER_TIMES; i++)
    {
        iTmp = i;
        int checkpointed = FTI_Snapshot();
        if (!(checkpointed != FTI_SCES || checkpointed != FTI_DONE)) {
            printf("%d: Snapshot failed! Returned %d.\n", rank, checkpointed);
            free(h);
            free(g);
            FTI_Finalize();
            MPI_Finalize();
            return 1;
        } else if (rank == 0 && checkpointed == FTI_DONE) {
            printf("Checkpoint made i = %d\n", i);
        } else if (rank == 0 && checkpointed == FTI_SCES && i != iTmp) {
            printf("Recovered! i = %d\n", i);
        }
        localerror = doWork(nbProcs, rank, M, nbLines, g, h);
        if (((i%ITER_OUT) == 0) && (rank == 0)) printf("Step : %d, error = %f\n", i, globalerror);
        if ((i%REDUCE) == 0) MPI_Allreduce(&localerror, &globalerror, 1, MPI_DOUBLE, MPI_MAX, FTI_COMM_WORLD);
        if (globalerror < PRECISION) break;
        if (fail && i >= ITER_STOP) {
            printf("%d: Stoped at i = %d.\n", rank, i);
            break;
        }
    }
    if (rank == 0) printf("Execution finished in %lf seconds. Error = %f\n", MPI_Wtime() - wtime, globalerror);

    int rtn = 0; //return value
    if (!fail) {
        rtn = verify(globalerror, rank);
    }

    free(h);
    free(g);
    FTI_Finalize();
    MPI_Finalize();
    return rtn;
}
