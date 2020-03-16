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
#include <boost/mpi/environment.hpp>
#include <boost/mpi/exception.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/request.hpp> 
#include <boost/mpi/status.hpp> 
#include <boost/mpi/operations.hpp>
#include <boost/mpi/timer.hpp>
    
namespace mpi = boost::mpi;

// Create your data structure
typedef struct cInfo {
    int id;
    int level;
} cInfo;

// Create a new FTI data type
FTIT_type ckptInfo;

void handle_error( int & i, cInfo & myCkpt, int & nbLines, hsize_t & nbLinesGlobal,  
        double* & h, double* & g, mpi::communicator &comm );
void protect( int & i, cInfo & myCkpt, double* & h, double* & g, const int & nbLines, const hsize_t & nbLinesGlobal, mpi::communicator & comm );
void initTopology( int & nbLines, double* & h, double* & g, const mpi::communicator & comm  );

#define ITER_TIMES  5000
#define ITER_OUT    100
#define CKPT_OUT    1000
#define FAIL_ITER   1200
#define PRECISION   0.001
#define WORKTAG     26
#define GRIDSIZE    1024
#define M GRIDSIZE
#define N GRIDSIZE


void initData(int nbLines, double *h, mpi::communicator & comm )
{
    int i, j;
    for (i = 0; i < nbLines; i++) {
        for (j = 0; j < M; j++) {
            h[(i*M)+j] = 0;
        }
    }
    if (comm.rank() == 0) {
        for (j = (M*0.1); j < (M*0.9); j++) {
            h[j] = 100;
        }
    }
}

double doWork( int nbLines, double *g, double *h, mpi::communicator & comm ) //throw( mpi::exception )
{
    int i,j;
    mpi::request req1[2], req2[2];
    mpi::status status1[2], status2[2];
    double localerror, globalerror;
    localerror = 0;
    int rank = comm.rank();
    int numprocs = comm.size();
    for(i = 0; i < nbLines; i++) {
        for(j = 0; j < M; j++) {
            h[(i*M)+j] = g[(i*M)+j];
        }
    }
    if (rank > 0) {
        req1[0] = comm.isend<double>( rank-1, WORKTAG, g+M, M );
        req1[1] = comm.irecv<double>( rank-1, WORKTAG, h, M );
    }
    if (rank < numprocs-1) {
        req2[0] = comm.isend<double>( rank+1, WORKTAG, g+((nbLines-2)*M), M );
        req2[1] = comm.irecv<double>( rank+1, WORKTAG, h+((nbLines-1)*M), M );
    }
    if (rank > 0) {
        mpi::wait_all(req1,req1+2,status1);
    }
    if (rank < numprocs-1) {
        mpi::wait_all(req2,req2+2,status2);
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
    mpi::all_reduce<double>( comm, localerror, globalerror, mpi::maximum<double>() );

    return globalerror;
}

int crash;
int main(int argc, char *argv[])
{
    int nbLines, i, j, res;
    hsize_t nbLinesGlobal = N;
    double wtime, *h = NULL, *g = NULL, globalerror = 1;
    char fn[32];

    crash = 1;
    mpi::environment world;
    mpi::timer timer;
    FTI_Init(argv[1], MPI_COMM_WORLD);
    MPI_Comm_set_errhandler(FTI_COMM_WORLD, MPI_ERRORS_RETURN);
    mpi::communicator comm(FTI_COMM_WORLD, mpi::comm_duplicate);

    initTopology( nbLines, h, g, comm );

    initData( nbLines, g, comm );
    
    if (comm.rank() == 0) {
        printf("Data initialized. Global grid size is %d x %d.\n", M, nbLinesGlobal);
    }

    // Define and initialize the datastructure
    cInfo myCkpt = {1,1};
    // Initialize the new FTI data type
    FTI_InitType(&ckptInfo, 2*sizeof(int));

    protect( i, myCkpt, h, g, nbLines, nbLinesGlobal, comm );

    comm.barrier();
    
    wtime = timer.elapsed();
    for(i = 0; i < ITER_TIMES; i++) { // Check execution status
        if (((i+1)%CKPT_OUT) == 0) { // Checkpoint every CKPT_OUT steps
            if (comm.rank() == 0) {
                printf("Ckpt Step : %d, current error = %f; target = %f\n", i, globalerror, PRECISION);
            }
            res = FTI_Checkpoint(myCkpt.id, FTI_L4_H5_SINGLE); // Ckpt ID 5 is ignored because level = 0
            if (res == 0) {
                myCkpt.level = (myCkpt.level+1)%5; myCkpt.id++;
            } // Update ckpt. id & level
        }
        if (FTI_Status() != 0) {
            res = FTI_Recover();
            if (res != 0) {
                exit(1);
            }
            else { // Update ckpt. id & level
                myCkpt.level = (myCkpt.level+1)%5; myCkpt.id++;
            }
        }
        try {
            if( comm.rank() == 1 && i == FAIL_ITER && crash ) 
                *(int*)NULL = 1; 
            globalerror = doWork( nbLines, g, h, comm );
        } catch ( mpi::exception e ) {
            handle_error( i, myCkpt, nbLines, nbLinesGlobal, h, g, comm );
        }
        if ((i%ITER_OUT) == 0) {
            if (comm.rank() == 0) {
                printf("Step : %d, current error = %f; target = %f\n", i, globalerror, PRECISION);
            }
        }
        if (globalerror < PRECISION) {
            break;
        }
    }
    if (comm.rank() == 0) {
        printf("Execution finished in %lf seconds (err=%f).\n", timer.elapsed() - wtime, globalerror);
    }

    free(h);
    free(g);

    FTI_Finalize();
    return 0;
}

void handle_error( int & i, cInfo & myCkpt, int & nbLines, hsize_t & nbLinesGlobal,  
        double* & h, double* & g, mpi::communicator &comm )
{
    crash = 0;
    i=0;
    free(g); g=NULL;
    free(h); h=NULL;
    MPIX_Comm_revoke( MPI_Comm( comm ) );
    MPI_Comm NEW_FTI_COMM_WORLD;
    MPIX_Comm_shrink( FTI_COMM_WORLD, &NEW_FTI_COMM_WORLD );
    int rank; MPI_Comm_rank( NEW_FTI_COMM_WORLD, &rank );
    if( rank == 0 ) XFTI_updateKeyCfg( "Restart", "failure", "3" ); 
    MPI_Barrier( NEW_FTI_COMM_WORLD );
    MPI_Comm_free(&FTI_COMM_WORLD);
    MPI_Barrier( NEW_FTI_COMM_WORLD );
    FTI_Init( "config.fti", NEW_FTI_COMM_WORLD );
    FTI_InitType(&ckptInfo, 2*sizeof(int));
    mpi::communicator newcomm(FTI_COMM_WORLD, mpi::comm_duplicate);
    newcomm.barrier();
    comm = newcomm;
    initTopology( nbLines, h, g, comm );
    initData( nbLines, g, comm );
    protect( i, myCkpt, h, g, nbLines, nbLinesGlobal, comm ); 
    FTI_Recover();
    doWork( nbLines, g, h, comm );
}

void initTopology( int & nbLines, double* & h, double* & g, const mpi::communicator & comm  )
{
    nbLines = ( N / comm.size() ) + 2;
    if( comm.rank() == comm.size()-1 ) nbLines += static_cast<int>(N)%static_cast<int>(comm.size()) ;
    h = (double *) realloc( h, sizeof(double *) * M * nbLines);
    g = (double *) realloc( g, sizeof(double *) * M * nbLines);
}

void protect( int & i, cInfo & myCkpt, double* & h, double* & g, const int & nbLines, const hsize_t & nbLinesGlobal, mpi::communicator & comm )
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
    hsize_t stride = N / comm.size();
    hsize_t offset_N = stride * comm.rank();
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
