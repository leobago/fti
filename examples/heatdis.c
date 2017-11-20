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
#include<signal.h>

#define PRECISION   0.005
#define ITER_TIMES  6000
#define ITER_OUT    100
#define WORKTAG     50
#define REDUCE      5
//#define PRINT_RESULTS
//#define PRINT_THERMOMETER
#define PRINT_THERMOMETER_JUSTFAILED

#define N_THERMOMETER 100
#define N_THERMOMETER_LINES 10


//#define GRIDSIZE    4096
#define GRIDSIZE    512

//#define FAILURE
#define RANK_TO_FAIL 0

#ifdef PRINT_RESULTS
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
#else
#ifdef PRINT_THERMOMETER

void print_solution (char *filename, double *grid, int procs, int index)
{
    int i, j;
    FILE *outfile;
    outfile = fopen(filename,"aw");
    if (outfile == NULL) {
        printf("Can't open output file.");
        exit(-1);
    }

    if(index==0){
        fprintf(outfile, "# PROCS %d SamplesIter %d \n", procs, N_THERMOMETER);
    }

    for (i = 0; i < procs; i++) {
        for (j = 0; j < N_THERMOMETER; j++) {
            fprintf (outfile, "%d %d %6.2f \n", index, (i*N_THERMOMETER)+j, grid[(i*N_THERMOMETER)+j] );
        }
    }
    fprintf (outfile, "\n");

    fclose(outfile);
}
#else
#ifdef PRINT_THERMOMETER_JUSTFAILED

void print_solution (char *filename, double *grid, int procs, int index)
{
    int i, j;
    FILE *outfile;
    outfile = fopen(filename,"aw");
    if (outfile == NULL) {
        printf("Can't open output file.");
        exit(-1);
    }

    if(index==0){
        fprintf(outfile, "# PROCS %d SamplesIter %d FAILED %d \n", procs, N_THERMOMETER, RANK_TO_FAIL);
    }

    fprintf (outfile, "%d ", index );
    for (j = 0; j < N_THERMOMETER; j++) {
        fprintf (outfile, "%6.2f ", grid[j] );
    }
    fprintf (outfile, "\n");

    fclose(outfile);
}
#endif
#endif
#endif

void initData(int nbLines, int M, int rank, double *h, int numprocs)
{
    int i, j;
    for (i = 0; i < nbLines; i++) {
        for (j = 0; j < M; j++) {
            h[(i*M)+j] = 0;
        }
    }
//    if (rank == 0) {
//        for (j = (M*0.1); j < (M*0.9); j++) {
//            h[j] = 100;
//        }
//    }

    if ((rank % 2)==0 ) {
        for (i = ((nbLines/numprocs)*rank); i < ((nbLines/numprocs)*(rank+1)); i++) {
//            for (j = (((M/numprocs)*rank)); j < (((M/numprocs)*(rank+1))); j++) {
            for (j = 0; j < M*0.8; j++) {
                h[(i*M)+j] = 100;
            }
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

int count=2;
int ids[2];
void sig_handler(int signo)
{
    if (signo == SIGUSR1){
        /* Notify FTI that rank affected by a soft error */
        FTI_RankAffectedBySoftError();
        /* Select id to be recovered.
         * In the future this will be done by selecting those IDs contained
         * in the page affected by the soft error.
         * */

        /* never destroy logical vars: counter i in this case,
         * we only protect computation data */
        count=2;
        ids[0]=1; ids[1]=2;

//        FTI_DestroyData(ids, count);

//        /* Recover could be invoke in the handler or after snapshot */
//        FTI_RecoverLocalCkptVars(ids, count);
    }
}
int main(int argc, char *argv[])
{
    int rank, nbProcs, nbLines, i, M, arg;
    double wtime, *h, *g, memSize, localerror, globalerror = 1;
    srand(time(NULL));

#ifdef PRINT_RESULTS
    char fn[32];
    double *grid;
#endif
#ifdef PRINT_THERMOMETER
    char fn[32];
    double *grid;
    double *aux_grid;
#endif
#ifdef PRINT_THERMOMETER_JUSTFAILED
    char fn[32];
    double *grid;
#endif

    if (signal (SIGUSR1, sig_handler) == SIG_IGN)
        signal (SIGINT, SIG_IGN);


    MPI_Init(&argc, &argv);
    FTI_Init(argv[2], MPI_COMM_WORLD);

    MPI_Comm_size(FTI_COMM_WORLD, &nbProcs);
    MPI_Comm_rank(FTI_COMM_WORLD, &rank);


    arg = atoi(argv[1]);
#if defined(PRINT_RESULTS)||defined(PRINT_THERMOMETER)||defined(PRINT_THERMOMETER_JUSTFAILED)
    M = GRIDSIZE;
    int N = GRIDSIZE;
    nbLines = (N / nbProcs)+3;
#else
    M = (int)sqrt((double)(arg * 1024.0 * 512.0 * nbProcs)/sizeof(double));
    nbLines = (M / nbProcs)+3;
#endif
    h = (double *) malloc(sizeof(double) * M * nbLines);
    g = (double *) malloc(sizeof(double) * M * nbLines);
#ifdef PRINT_RESULTS
    grid = (double *) malloc(sizeof(double) * M * (nbLines-2) * nbProcs);
#else
#ifdef PRINT_THERMOMETER
    grid = (double *) malloc(sizeof(double) * N_THERMOMETER * nbProcs);
    aux_grid =  (double *) malloc(sizeof(double) * N_THERMOMETER );
#else
#ifdef PRINT_THERMOMETER_JUSTFAILED
    grid = (double *) malloc(sizeof(double) * N_THERMOMETER);
#endif
#endif
#endif

    initData(nbLines, M, rank, g, nbProcs);
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
        if(checkpointed==FTI_SB_FAIL){
            int*status_array;
            FTI_GetGlobalStatus(&status_array);

            /* Application recovery code: just index*/
            /* We are not protecting logical data, only computation data */
//            int current_index;
//            int aux_current_index = (status_array[rank]==1? -1: i);
//            MPI_Allreduce(&aux_current_index, &current_index, 1,
//                    MPI_INT, MPI_MAX, FTI_COMM_WORLD);

            if(status_array[rank]==FTI_FAILED){
                /* Recover could be invoke in the handler or after snapshot */
                FTI_RecoverLocalCkptVars(ids, count);
            }
        }

        /*************************************/
        /* CODE FOR INTRODUCING A SOFT ERROR */
#ifdef FAILURE
        if(FTI_CheckCheckpointDone()){
            itersAfterCKPT=0;
        }
        if(itersAfterCKPT==401){
            if(rank==RANK_TO_FAIL){
                int pid = getpid();
                char buffS[100];
                sprintf(buffS,"./signal_soft_error.sh %d &",  pid);
                system(buffS);
            }
        }
#endif
        /*************************************/

        localerror = doWork(nbProcs, rank, M, nbLines, g, h);
        if ((i%REDUCE) == 0) {
            MPI_Allreduce(&localerror, &globalerror, 1, MPI_DOUBLE, MPI_MAX, FTI_COMM_WORLD);
        }

        if (((i%ITER_OUT) == 0) ) {
            if (rank == 0) {
                printf("%d Step : %d, error = %f local %f\n",rank, i, globalerror, localerror);
            }
#ifdef PRINT_RESULTS
            MPI_Gather(g+M, (nbLines-2)*M, MPI_DOUBLE, grid, (nbLines-2)*M, MPI_DOUBLE, 0, FTI_COMM_WORLD);
            sprintf(fn, "results/vis-%d.dat", i);
            if (rank == 0) {
                print_solution(fn, grid);
            }
#endif
#ifdef PRINT_THERMOMETER

            int auxi,auxj, pos=0, c=0;

            for(auxi=0; auxi<N_THERMOMETER_LINES;auxi++){
                pos= auxi*(nbLines/N_THERMOMETER_LINES)*M + M ; /* row */
                for(auxj= 0; auxj<N_THERMOMETER/N_THERMOMETER_LINES; auxj++){
                    pos+= (M/(N_THERMOMETER/N_THERMOMETER_LINES));
                    aux_grid[c]=g[pos];
                    c++;
                }
            }
            MPI_Gather(aux_grid, N_THERMOMETER, MPI_DOUBLE, grid, N_THERMOMETER, MPI_DOUBLE, 0, FTI_COMM_WORLD);
            sprintf(fn, "results/vis-thermometers.dat");
            if (rank == 0) {
                print_solution(fn, grid, nbProcs, i);
            }

#endif
#ifdef PRINT_THERMOMETER_JUSTFAILED
            if(rank==RANK_TO_FAIL){
                int auxi,auxj, pos=0, c=0;
                for(auxi=0; auxi<N_THERMOMETER_LINES;auxi++){
                    pos= auxi*(nbLines/N_THERMOMETER_LINES)*M + M ; /* row */
                    for(auxj= 0; auxj<N_THERMOMETER/N_THERMOMETER_LINES; auxj++){
                        pos+= (M/(N_THERMOMETER/N_THERMOMETER_LINES));
                        grid[c]=g[pos];
                        c++;
                    }
                }
                sprintf(fn, "results/vis-thermometers.dat");
                print_solution(fn, grid, nbProcs, i);
            }

#endif
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
#ifdef PRINT_RESULTS
    free(grid);
#endif

#ifdef PRINT_THERMOMETER
    free(grid);
    free(aux_grid);
#endif
#ifdef PRINT_THERMOMETER_JUSTFAILED
    free(grid);
#endif
    FTI_Finalize();
    MPI_Finalize();
    return 0;
}
