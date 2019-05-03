/**
 *  @file   heatdis-fti.c
 *  @author Leonardo A. Bautista Gomez and Sheng Di
 *  @date   January, 2014
 *  @brief  Heat distribution code to test FTI.
 */


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fti.h>


#define ITER_TIMES  5000
#define ITER_OUT    1
#define CKPT_OUT    10
#define PRECISION   0.001
#define WORKTAG     26
#define GRIDSIZE    512


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


void print_solution (char *filename, double *grid)
{
    int i, j;
    FILE *outfile;
    outfile = fopen(filename,"w");
    if (outfile == NULL) {
        printf("Can't open output file.");
        exit(-1);
    }
    for (i = 0; i < GRIDSIZE; i++) {
        for (j = 0; j < GRIDSIZE; j++) {
            fprintf (outfile, "%6.2f\n", grid[(i*GRIDSIZE)+j]);
        }
        fprintf(outfile, "\n");
    }
    fclose(outfile);
}


double doWork(int numprocs, int rank, int M, int nbLines, double *g, double *h)
{
    int i,j;
    MPI_Request req1[2], req2[2];
    MPI_Status status1[2], status2[2];
    double localerror, globalerror;
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
    for (i = 1; i < (nbLines-1); i++) {
        for (j = 0; j < M; j++) {
            g[(i*M)+j] = 0.25*(h[((i-1)*M)+j]+h[((i+1)*M)+j]+h[(i*M)+j-1]+h[(i*M)+j+1]);
            if (localerror < fabs(g[(i*M)+j] - h[(i*M)+j])) {
                localerror = fabs(g[(i*M)+j] - h[(i*M)+j]);
            }
        }
    }
    if (rank == (numprocs-1)) {
        for(j = 0; j < M; j++) {
            g[((nbLines-1)*M)+j] = g[((nbLines-2)*M)+j];
        }
    }
    MPI_Allreduce(&localerror, &globalerror, 1, MPI_DOUBLE, MPI_MAX, FTI_COMM_WORLD);
    return globalerror;
}


int main(int argc, char *argv[])
{
    int rank, nbProcs, nbLines, i, j, N, M, res;
    double wtime, *h, *g, *grid, globalerror = 1;
    char fn[32];

    MPI_Init(&argc, &argv);
    FTI_Init(argv[1], MPI_COMM_WORLD);
    MPI_Comm_size(FTI_COMM_WORLD, &nbProcs);
    MPI_Comm_rank(FTI_COMM_WORLD, &rank);

    M = GRIDSIZE;
    N = GRIDSIZE;
    nbLines = (N / nbProcs)+3;
    h = (double *) malloc(sizeof(double *) * M * nbLines);
    g = (double *) malloc(sizeof(double *) * M * nbLines);
    grid = (double *) malloc(sizeof(double *) * M * (nbLines-2) * nbProcs);
    initData(nbLines, M, rank, g);
    if (rank == 0) {
        printf("Data initialized. Global grid size is %d x %d.\n", M, (nbLines-2)*nbProcs);
    }

    // Create your data structure
    typedef struct cInfo {
        int id;
        int level;
    } cInfo;
    // Define and initialize the datastructure
    cInfo myCkpt = {1,1};
    // Create a new FTI data type
    FTIT_type ckptInfo;
    // Initialize the new FTI data type
    FTI_InitType(&ckptInfo, 2*sizeof(int));

    FTI_Protect(0, &i, 1, FTI_INTG);
    FTI_Protect(1, &myCkpt, 1, ckptInfo);
    FTI_Protect(2, h, M*nbLines, FTI_DBLE);
    FTI_Protect(3, g, M*nbLines, FTI_DBLE);

    MPI_Barrier(FTI_COMM_WORLD);
    wtime = MPI_Wtime();
    for(i = 0; i < ITER_TIMES; i++) { // Check execution status
        if (FTI_Status() != 0) {
            res = FTI_Recover();
            if (res != 0) {
                exit(1);
            }
            else { // Update ckpt. id & level
                myCkpt.level = (myCkpt.level+1)%5; myCkpt.id++;
            }
        }
        else {
            if (((i+1)%CKPT_OUT) == 0) { // Checkpoint every ITER_OUT steps
                res = FTI_Checkpoint(myCkpt.id, FTI_L4); // Ckpt ID 5 is ignored because level = 0
                if (res == 0) {
                    myCkpt.level = (myCkpt.level+1)%5; myCkpt.id++;
                } // Update ckpt. id & level
            }
        }
        globalerror = doWork(nbProcs, rank, M, nbLines, g, h);
        if ((i%ITER_OUT) == 0) {
            if (rank == 0) {
                printf("Step : %d, current error = %f; target = %f\n", i, globalerror, PRECISION);
            }
            MPI_Gather(g+M, (nbLines-2)*M, MPI_DOUBLE, grid, (nbLines-2)*M, MPI_DOUBLE, 0, FTI_COMM_WORLD);
            sprintf(fn, "results/vis-%d.dat", i);
            if (rank == 0) {
                print_solution(fn, grid);
            }
        }
        if (globalerror < PRECISION) {
            break;
        }
    }
    if (rank == 0) {
        printf("Execution finished in %lf seconds.\n", MPI_Wtime() - wtime);
    }

    free(h);
    free(g);
    free(grid);

    FTI_Finalize();
    MPI_Finalize();
    return 0;
}
