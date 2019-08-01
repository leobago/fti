/**
 *  @file   heat2d.cpp
 *  @author Leonardo A. Bautista Gomez and Sheng Di and Kai Keller
 *  @date   July, 2019
 *  @brief  Heat distribution code in C++ to test FTI.
 */

#include <hdf5.h>
#include <fti.h>
#include <interface-paper.h>
#include <mpi.h>
#include <mpi-ext.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <assert.h>

// GRID PARAMETERS
const size_t M = 1024;
const size_t N = 1024;

// SIMULATION PARAMETERS
const int ITER_MAX = 10000;
const int ITER_OUT = 500;
const double PRECISION = 0.001;

// FAULT TOLERANCE
const int ITER_CHK = 1000;
const int ITER_FAIL =1100;

class SEnvironment {

    public:
        void init( int & argc, char** & argv ) {
            MPI::Init( argc, argv );
            m_global_comm = MPI_COMM_WORLD;
            MPI_Comm_set_errhandler( MPI_COMM_WORLD, MPI::ERRORS_THROW_EXCEPTIONS );
            
            if( argc > 1 ) {
                m_crash = atoi( argv[1] );
            } else {
                m_crash = false;
            }

            FTI_Init( "config.fti", MPI_COMM_WORLD );
            m_comm = FTI_COMM_WORLD;
            MPI_Comm_rank( m_comm, &m_rank );
            MPI_Comm_size( m_comm, &m_size );
        }
        void finalize( void ) {
            FTI_Finalize();
            MPI_Finalize();
        }
        const bool & crash() const { return m_crash; }
        const int & rank() const { return m_rank; }
        const int & size() const { return m_size; }
        const MPI_Comm & comm() const { return m_comm; }
        void reset( void ) {
            
            // propagate error and shrink communicator
            MPIX_Comm_revoke( m_global_comm );
            MPI_Comm new_global_comm;
            MPIX_Comm_shrink( m_global_comm, &new_global_comm );
            //MPI_Comm_free( &m_global_comm );
            m_global_comm = new_global_comm;
            MPI_Comm_rank( m_global_comm, &m_global_rank );
            
            // enable VPR restart
            if( m_global_rank == 0 ) XFTI_updateKeyCfg( "Restart", "failure", "3" ); 

            // reset FTI
            MPI_Comm_free(&FTI_COMM_WORLD);
            
            FTI_Init( "config.fti", m_global_comm );
            m_comm = FTI_COMM_WORLD;
            MPI_Comm_rank( m_comm, &m_rank );
            MPI_Comm_size( m_comm, &m_size );
            
            m_crash = false;
            
            MPI_Barrier( m_global_comm );
        
        }

    private:

        MPI_Comm m_comm;
        MPI_Comm m_global_comm;
        int m_rank;
        int m_global_rank;
        int m_size;
        double m_t0;
        bool m_crash;

};

class TDist {
    
    public:
        void print_progress( int i, const SEnvironment & env ) {
            if( (i%ITER_OUT == 0) && (env.rank() == 0) )
                std::cout << "Step : " << i << ", current error = " << m_error << "; target = " << PRECISION << std::endl;
        }
        void init( const size_t & M, const size_t & N, const SEnvironment & env ) {
            m_chk_id = 1;
            m_Mloc = ( M / env.size() );
            if( env.rank() == env.size()-1 ) m_Mloc += M%static_cast<size_t>(env.size());
            m_Nloc = N;

            m_num_ghosts = 0;
            m_max_dist_row = m_Mloc-1; 
            m_max_data_row = m_Mloc-1;
            
            m_has_down_neighbor = ( env.rank() > 0 ) ? true : false ; 
            m_has_up_neighbor = ( env.rank() < (env.size()-1) ) ? true : false ;
            if( m_has_down_neighbor ) {
                m_num_ghosts++;
                m_max_dist_row++;
            }
            if( m_has_up_neighbor ) {
                m_num_ghosts++;
                m_max_dist_row++;
            }
            alloc();
            init_data( env );
            
            hsize_t dim_i = 1;
            hsize_t dim_data[2] = { M, N };
            FTI_DefineGlobalDataset( 0, 1, &dim_i, "iteration counter", NULL, FTI_INTG );
            FTI_DefineGlobalDataset( 1, 2, dim_data, "temperature distribution (h)", NULL, FTI_DBLE );

        }
        void alloc( void ) {
            m_dist = new double*[ m_Mloc + m_num_ghosts ];
            m_dist_cpy = new double*[ m_Mloc + m_num_ghosts ];
            m_data = new double[ m_Mloc * m_Nloc ];
            if( m_has_down_neighbor ) {
                m_ghost_down = new double[ m_Nloc ];
                m_dist[0] = m_ghost_down;
                m_dist_cpy[0] = new double[ m_Nloc ];
            }
            if( m_has_up_neighbor ) { 
                m_ghost_up = new double[ m_Nloc ];
                m_dist[m_max_dist_row] = m_ghost_up;
                m_dist_cpy[m_max_dist_row] = new double[ m_Nloc ];
            }
            for( int m = m_has_down_neighbor ? 1 : 0, _m = 0; _m <= m_max_data_row; ++m, ++_m ) {
                m_dist[m] = &m_data[(_m)*m_Nloc];
                m_dist_cpy[m] = new double[ m_Nloc ];
            }
        }
        void init_data( const SEnvironment & env ) { 
            for (int m = 0; m <= m_max_dist_row; ++m) {
                for (int n = 0; n < m_Nloc; n++) {
                    m_dist[m][n] = 0;
                }
            }
            if (env.rank() == 0) {
                int min = static_cast<int>(m_Nloc*0.1);
                int max = static_cast<int>(m_Nloc*0.9);
                //for(int m = 0; m < m_Mloc/2; ++m) { 
                    for (int n = min; n < max; ++n) {
                        m_dist[0][n] = 1000;
                    }
                //}
            }
        }
        void compute_step( const SEnvironment & env ) {
            MPI_Request req1[2], req2[2];
            MPI_Status status1[2], status2[2];
            double localerror;
            
            localerror = 0;
            for(int m = 0; m <= m_max_dist_row; ++m) {
                for(int n = 0; n < m_Nloc; ++n) {
                    m_dist_cpy[m][n] = m_dist[m][n];
                }
            }
            if ( m_has_down_neighbor ) {
                MPI_Isend(&m_dist[1][0], m_Nloc, MPI_DOUBLE, env.rank()-1, 0, env.comm(), &req1[0]);
                MPI_Irecv(&m_dist_cpy[0][0],   m_Nloc, MPI_DOUBLE, env.rank()-1, 0, env.comm(), &req1[1]);
            }
            if ( m_has_up_neighbor ) {
                MPI_Isend(&m_dist[m_max_dist_row-1][0], m_Nloc, MPI_DOUBLE, env.rank()+1, 0, env.comm(), &req2[0]);
                MPI_Irecv(&m_dist_cpy[m_max_dist_row][0], m_Nloc, MPI_DOUBLE, env.rank()+1, 0, env.comm(), &req2[1]);
            }
            if ( m_has_down_neighbor ) {
                MPI_Waitall(2,req1,status1);
            }
            if ( m_has_up_neighbor ) {
                MPI_Waitall(2,req2,status2);
            }
            for (int m = 0; m <= m_max_dist_row; ++m) {
                for (int n=0; n<m_Nloc; ++n) {
                    double val = m_dist_cpy[m][n];
                    int norm = 1;
                    if( m > 0 ) {
                        val += m_dist_cpy[m-1][n];
                        norm++;
                    }
                    if( m < m_max_dist_row ) {
                        val += m_dist_cpy[m+1][n];
                        norm++;
                    }
                    if( n > 0 ) {
                        val += m_dist_cpy[m][n-1];
                        norm++;
                    }
                    if( n < m_Nloc-1 ) {
                        val += m_dist_cpy[m][n+1];
                        norm++;
                    }
                    m_dist[m][n] = val/norm;
                    double error = std::fabs( m_dist[m][n] - m_dist_cpy[m][n] ); 
                    if ( error > localerror ) {
                        localerror = error; 
                    }
                }
            }
            
            MPI_Allreduce( &localerror, &m_error, 1, MPI_DOUBLE, MPI_MAX, env.comm() );

        }
        void checkpoint( int & i, SEnvironment & env, bool force=false ) {
            if( ((i%ITER_CHK == 0) && (i>0)) || force ) {
                FTI_Checkpoint( m_chk_id++, FTI_L4_H5_SINGLE );
            }
        }
        void protect( int & i, SEnvironment & env ) {
            int id = 0;

            hsize_t offset_i = 0;
            hsize_t count_i = 1;
            FTI_Protect(id, &i, 1, FTI_INTG);
            FTI_AddSubset( id, 1, &offset_i, &count_i, 0 );
            id += 1;

            hsize_t stride = M / env.size();
            hsize_t offset_M = stride * env.rank();
            hsize_t offset_data[2] = { offset_M, 0 };
            hsize_t count_data[2] = { m_Mloc, m_Nloc };
            FTI_Protect(id, m_data, m_Mloc*m_Nloc, FTI_DBLE);
            FTI_AddSubset( id, 2, offset_data, count_data, 1 );
        }
        double get_error( void ) { return m_error; } 
        void finalize( void ) {
            delete m_data;
            if( m_has_up_neighbor ) delete m_ghost_up;
            if( m_has_down_neighbor ) delete m_ghost_down;
            delete m_dist;
            delete[] m_dist_cpy; 
        }
        bool condition() {
            return m_error < PRECISION;
        }
        void handle_error( int & i, SEnvironment & env ) {
            
            // reset environment
            env.reset();

            // reset pointers
            finalize();

            // initialize
            init( M, N, env );
            protect( i, env );
            FTI_Recover();

        }
        void inject_failure( int i, SEnvironment & env, int rank = 0 ) {
            if( (i%ITER_FAIL == 0) && (i>0) && (env.rank() == rank) && env.crash() ) {
                *(int*)NULL = 1;
            }
        }

    private:

        int m_chk_id;

        double m_error;
        double** m_dist;
        double** m_dist_cpy;
        double* m_data;
        double* m_ghost_down;
        double* m_ghost_up;
        int m_num_ghosts;

        size_t m_Mloc;
        size_t m_Nloc;

        bool m_has_up_neighbor;
        bool m_has_down_neighbor;

        size_t m_max_dist_row;
        size_t m_max_data_row;
};

static TDist dist;
static SEnvironment env;

int main( int argc, char** argv )
{

    int i;

    // INITIALIZE SIMULATION
    env.init( argc, argv );
    dist.init( M, N, env );
    dist.protect( i, env );

    // MAINLOOP
    for(i=0; i<ITER_MAX; ++i) {

        dist.print_progress( i, env );
        
        dist.checkpoint( i, env );
        if( FTI_Status() ) {
            FTI_Recover();
        }
        
        try {

            dist.inject_failure( i, env );
            dist.compute_step( env );
        
        } catch ( MPI::Exception ) {
            
            dist.handle_error( i, env );
        
        }
        
        if( dist.condition() ) break;
        
    }
    
    // FINALIZE
    dist.finalize();
    env.finalize();
    
    return 0;

}
