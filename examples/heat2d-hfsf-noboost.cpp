/**
 *  @file   heatdis-fti.c
 *  @author Leonardo A. Bautista Gomez and Sheng Di
 *  @date   January, 2014
 *  @brief  Heat distribution code to test FTI.
 */


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <hdf5.h>
#include <fti.h>
#include <interface-paper.h>
#include <mpi.h>
#include <mpi-ext.h>
#include <iostream>

// Create your data structure
typedef struct cInfo {
    int id;
    int level;
} cInfo;

// Create a new FTI data type
FTIT_type ckptInfo;

void handle_error( int & i, cInfo & myCkpt, int & nbLines, hsize_t & nbLinesGlobal,  
        double* & h, double* & g );
void protect( int & i, cInfo & myCkpt, double* & h, double* & g, const int & nbLines, const hsize_t & nbLinesGlobal );
void initTopology( int & nbLines, double* & h, double* & g );

#define ITER_TIMES  5000
#define ITER_OUT    100
#define CKPT_OUT    1000
#define FAIL_ITER   1200
#define PRECISION   0.001
#define WORKTAG     26
#define GRIDSIZE    1024
#define M GRIDSIZE
#define N GRIDSIZE

int rank, nbProc;

void initData( int nbLines, double *h )
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

double doWork( int nbLines, double *g, double *h ) //throw( mpi::exception )
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
    if (rank < nbProc-1) {
        MPI_Isend(g+((nbLines-2)*M), M, MPI_DOUBLE, rank+1, WORKTAG, FTI_COMM_WORLD, &req2[0]);
        MPI_Irecv(h+((nbLines-1)*M), M, MPI_DOUBLE, rank+1, WORKTAG, FTI_COMM_WORLD, &req2[1]);
    }
    if (rank > 0) {
        MPI_Waitall(2,req1,status1);
    }
    if (rank < nbProc-1) {
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
    if (rank == (nbProc-1)) {
        for(j = 0; j < M; j++) {
            g[((nbLines-1)*M)+j] = g[((nbLines-2)*M)+j];
        }
    }
    MPI_Allreduce(&localerror, &globalerror, 1, MPI_DOUBLE, MPI_MAX, FTI_COMM_WORLD);

    return globalerror;
}


int main(int argc, char *argv[])
{
    int nbLines, i, j, res;
    hsize_t nbLinesGlobal = N;
    double wtime, *h = NULL, *g = NULL, globalerror = 1;
    char fn[32];

    MPI::Init(argc, argv);
    
    FTI_Init(argv[1], MPI_COMM_WORLD);
    
    MPI_Comm_set_errhandler( FTI_COMM_WORLD, MPI::ERRORS_THROW_EXCEPTIONS );
    //MPI_Comm_set_errhandler( FTI_COMM_WORLD, MPI_ERRORS_RETURN );
    MPI_Comm_size(FTI_COMM_WORLD, &nbProc);
    MPI_Comm_rank(FTI_COMM_WORLD, &rank);

    initTopology( nbLines, h, g );

    initData( nbLines, g );
    
    if (rank == 0) {
        printf("Data initialized. Global grid size is %d x %d.\n", M, nbLinesGlobal);
    }

    // Define and initialize the datastructure
    cInfo myCkpt = {1,1};
    // Initialize the new FTI data type
    FTI_InitType(&ckptInfo, 2*sizeof(int));

    protect( i, myCkpt, h, g, nbLines, nbLinesGlobal );

    MPI_Barrier(FTI_COMM_WORLD);
    
    wtime = MPI_Wtime();
    for(i = 0; i < ITER_TIMES; i++) { // Check execution status
        if (((i+1)%CKPT_OUT) == 0) { // Checkpoint every CKPT_OUT steps
            res = FTI_Checkpoint(myCkpt.id, FTI_L4_H5_SINGLE); // Ckpt ID 5 is ignored because level = 0
            if (res == 0) {
                myCkpt.level = (myCkpt.level+1)%5; myCkpt.id++;
            } // Update ckpt. id & level
        }
        try {
            if( rank == 1 && i == FAIL_ITER ) 
                *(int*)NULL = 1; 
            globalerror = doWork( nbLines, g, h );
        } catch ( MPI::Exception e ) {
            handle_error( i, myCkpt, nbLines, nbLinesGlobal, h, g );
        }
        if ((i%ITER_OUT) == 0) {
            if (rank == 0) {
                printf("Step : %d, current error = %f; target = %f\n", i, globalerror, PRECISION);
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

    FTI_Finalize();
    return 0;
}

void handle_error( int & i, cInfo & myCkpt, int & nbLines, hsize_t & nbLinesGlobal,  
        double* & h, double* & g )
{
    MPIX_Comm_revoke( FTI_COMM_WORLD );
    MPI_Comm NEW_FTI_COMM_WORLD;
    MPIX_Comm_shrink( FTI_COMM_WORLD, &NEW_FTI_COMM_WORLD );
    int rank; MPI_Comm_rank( NEW_FTI_COMM_WORLD, &rank );
    if( rank == 0 ) XFTI_updateKeyCfg( "Restart", "failure", "3" ); 
    MPI_Barrier( NEW_FTI_COMM_WORLD );
    MPI_Comm_free(&FTI_COMM_WORLD);
    MPI_Barrier( NEW_FTI_COMM_WORLD );
    FTI_Init( "config.fti", NEW_FTI_COMM_WORLD );
    FTI_InitType(&ckptInfo, 2*sizeof(int));
    MPI_Comm_set_errhandler( FTI_COMM_WORLD, MPI::ERRORS_THROW_EXCEPTIONS );
    initTopology( nbLines, h, g );
    protect( i, myCkpt, h, g, nbLines, nbLinesGlobal ); 
    FTI_Recover();
}

void initTopology( int & nbLines, double* & h, double* & g )
{
    nbLines = ( N / nbProc ) + 2;
    if( rank == nbProc-1 ) nbLines += static_cast<int>(N)%static_cast<int>(nbProc) ;
    h = (double *) realloc( h, sizeof(double *) * M * nbLines);
    g = (double *) realloc( g, sizeof(double *) * M * nbLines);
}

void protect( int & i, cInfo & myCkpt, double* & h, double* & g, const int & nbLines, const hsize_t & nbLinesGlobal )
{
    int id = 0;
    
    hsize_t dim_i = 1;
    hsize_t dim_myCkpt = 1;
    hsize_t dim_gh[2] = { M, nbLinesGlobal };
    FTI_DefineGlobalDataset( 0, 1, &dim_i, "iteration counter", NULL, FTI_INTG );
    FTI_DefineGlobalDataset( 1, 1, &dim_myCkpt, "checkpoint info", NULL, ckptInfo );
    FTI_DefineGlobalDataset( 2, 2, dim_gh, "temperature distribution (h)", NULL, FTI_DBLE );
    FTI_DefineGlobalDataset( 3, 2, dim_gh, "computation buffer (g)", NULL, FTI_DBLE );
    
    hsize_t offset_ickpt = 0;
    hsize_t count_ickpt = 1;
    FTI_Protect(id, &i, 1, FTI_INTG);
    FTI_Protect(id+1, &myCkpt, 1, ckptInfo);
    FTI_AddSubset( id, 1, &offset_ickpt, &count_ickpt, 0 );
    FTI_AddSubset( id+1, 1, &offset_ickpt, &count_ickpt, 1 );
    id += 2;
    hsize_t stride = N / nbProc;
    hsize_t offset_N = stride * rank;
    hsize_t offset_gh[2] = { 0, offset_N };
    hsize_t count_gh[2] = { M, 1 };
    for( i=1; i<nbLines-1; i++) {
        FTI_Protect(id, h+i*M, M, FTI_DBLE);
        FTI_Protect(id+1, g+i*M, M, FTI_DBLE);
        FTI_AddSubset( id, 2, offset_gh, count_gh, 2 );
        FTI_AddSubset( id+1, 2, offset_gh, count_gh, 3 );
        offset_gh[1]++;
        id += 2;
    }
}
