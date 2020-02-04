/**
 *  @file   heat2d.cpp
 *  @author Leonardo A. Bautista Gomez and Sheng Di and Kai Keller
 *  @date   July, 2019
 *  @brief  Heat distribution code in C++ to test FTI.
 */

#include <hdf5.h>
#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <assert.h>
#include <random>
#include <chrono>
#include <sstream>
#include <fstream>

typedef enum CLOCK_ID {
    ITER,
    CKPT,
    RECO,
    EXEC,
    FTI_INIT,
    NUM_CLOCK_IDS
} CLOCK_ID;

class Timer{
  public:
    Timer() {
        _t0 = std::chrono::system_clock::now();
        _dt = new double[NUM_CLOCK_IDS];
        _t  = new double[NUM_CLOCK_IDS];
        std::fill( _t, _t+NUM_CLOCK_IDS, 0 );
        _tp = new std::chrono::time_point<std::chrono::system_clock>[NUM_CLOCK_IDS];
    }
    
    ~Timer() {
        delete[] _dt;
        delete[] _t;
        delete[] _tp;
    };

    inline void start(const int i) { 
        _tp[i] = std::chrono::system_clock::now(); 
    }
    
    inline void stop (const int i) {
      std::chrono::duration<double> tmp = std::chrono::system_clock::now() - _tp[i];
      _dt[i] = tmp.count();
      _t [i]+= _dt[i];
    }
    
    inline double get_t(const int i) {
        return _t[i]; 
    }
    
    inline double get_dt(const int i) {
        return _dt[i]; 
    }

    inline double get_dT() { 
        return  std::chrono::duration<double>(std::chrono::system_clock::now() - _t0).count(); 
    }

    inline double log(const int i, std::string msg) { 
        double dt, dT, t;
        auto now = std::chrono::system_clock::now();
        dt = get_dt( i ); 
        dT = get_dT();
        t = get_t( i );
        _log << msg << " (total_time | delta_t | system_time_stamp) : "; 
        _log << t;  
        _log << " : " << dt; 
        _log << " : " << std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count(); 
        _log << std::endl;
    }

    inline void flush( std::string fn, const bool write ) {
        if( write ) {
            std::ofstream instrumentFile( fn, std::ofstream::app );
            instrumentFile << _log.str();
            instrumentFile.close();
            _log.str(std::string());
        }
    }

  private:
    std::chrono::time_point<std::chrono::system_clock> _t0;
    std::chrono::time_point<std::chrono::system_clock> *_tp;
    double *_dt;
    double *_t;
    std::stringstream _log;
};

size_t M;               // largest (768*256);
size_t N;               // largest (768*256);

// SIMULATION PARAMETERS
std::string m_logdir;
int ITER_MAX;
const int ITER_OUT = 1;
const double PRECISION = 0.000;

class SEnvironment {

    public:
        
        void init( int & argc, char** & argv ) {
            
            if( argc != 4 ) {
                printf("usage: %s IterMax", argv[0]);
                exit(-1);
            }
            
            ITER_MAX  = atoi( argv[1] );
            M = 256*24*atoi(argv[2]);
            N = (768*256);               // largest (768*256);
            std::stringstream ss;
            ss << argv[3] << "/timing/";
            m_logdir = ss.str();
            ss.str(std::string());
            
            Timer loc;
            loc.start(EXEC); 
            loc.stop(EXEC);
            loc.log(EXEC, "T1" );
            
            MPI::Init( argc, argv );
            m_global_comm = MPI_COMM_WORLD;
           
            MPI_Comm_group( MPI_COMM_WORLD, &m_world_group );
            
            MPI_Comm_size(MPI_COMM_WORLD, &m_global_size);
            MPI_Comm_rank(MPI_COMM_WORLD, &m_global_rank);

            m_comm = MPI_COMM_WORLD;
            MPI_Comm_rank( m_comm, &m_rank );
            MPI_Comm_size( m_comm, &m_size );
            
            ss << m_logdir << "timestamps";
            loc.flush( ss.str(), m_rank == 0 );
            ss.str(std::string());
        }
        
        void finalize( void ) {
            
            Timer loc;
            loc.start(EXEC); 
            loc.stop(EXEC);
            loc.log(EXEC, "T2" );
            std::stringstream ss;
            ss << m_logdir << "timestamps";
            loc.flush( ss.str(), m_rank == 0 );
            ss.str(std::string());
            
            MPI_Finalize();
        
        }

        const int & rank() const { return m_rank; }
        
        const int & size() const { return m_size; }
        
        const MPI_Comm & comm() const { return m_comm; }
        

    private:

        MPI_Comm m_comm;
        MPI_Comm m_global_comm;
        int m_rank;
        int m_global_rank;
        int m_size;
        int m_global_size;
        double m_t0;
        MPI_Group m_world_group;

};

class TDist {
    
    public:
        
        void print_progress( int i, const SEnvironment & env ) {
            
            if( (i%ITER_OUT == 0) && (env.rank() == 0) )
                std::cout << "Step : " << i << ", current error = " << m_error << "; target = " << PRECISION << std::endl;
        
        }
        
        void init( const size_t & M, const size_t & N, const SEnvironment & env ) {
            
            m_Mloc = ( M / env.size() );
            int Mloc_rest = M%static_cast<size_t>(env.size());
            if( env.rank() < Mloc_rest ) m_Mloc += 1;
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
            for( int m = (m_has_down_neighbor ? 1 : 0), _m = 0; _m <= m_max_data_row; ++m, ++_m ) {
                m_dist[m] = &m_data[(_m)*m_Nloc];
                m_dist_cpy[m] = new double[ m_Nloc ];
            }
        
        }
        
        void init_data( const SEnvironment & env ) { 
            
            std::uniform_real_distribution<double> random(0,1000);
            std::default_random_engine re(31071980);

            for (size_t m = 0; m <= m_max_dist_row; ++m) {
                for (size_t n = 0; n < m_Nloc; n++) {
                    m_dist[m][n] = 0;
                }
            }

            size_t range_begin = env.rank() * (M/env.size());
            size_t range_end = range_begin + m_Mloc;
            size_t minM = M*0.2;
            size_t maxM = M*0.8;
            
            if ( (range_end >= minM) && (range_begin <= maxM) ) {
                size_t minN = m_Nloc*0.2;
                size_t maxN = m_Nloc*0.8;
                for(size_t m = 0, pos = range_begin; m <= m_max_dist_row; ++m, ++pos) { 
                    for (size_t n = minN; n < maxN; ++n) {
                        if( pos >= minM && pos <= maxM ) {
                            m_dist[m][n] = pos%static_cast<size_t>(M);
                        }   
                    }
                }
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
            
            for (int m = (m_has_down_neighbor ? 1 : 0); m <= (m_has_up_neighbor ? m_max_dist_row-1 : m_max_dist_row); ++m) {
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

            double globalerror;
            MPI_Allreduce( &localerror, &globalerror, 1, MPI_DOUBLE, MPI_MAX, env.comm() );

            m_error = globalerror;

        }
        
        double get_error( void ) { return m_error; } 
        
        bool condition() {
            
            return m_error < PRECISION;
        
        }
        
        void finalize( void ) {
           
            delete m_data;
            if( m_has_up_neighbor ) delete m_ghost_up;
            if( m_has_down_neighbor ) delete m_ghost_down;
            delete m_dist;
            delete[] m_dist_cpy; 
            
        }
        
        Timer & clock() { return m_clock; }
        
    private:

        double m_error;
        double** m_dist;
        double** m_dist_cpy;
        double* m_data;
        double* m_ghost_down;
        double* m_ghost_up;
        int m_num_ghosts;
        
        Timer m_clock;
        
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
        
    // MAINLOOP
    for(i=0; i<ITER_MAX; ++i) {

        dist.clock().start(ITER);

        dist.compute_step( env );
        dist.print_progress( i, env );

        std::stringstream ss;
        ss << "Iteration[" << i << "]";
        dist.clock().stop( ITER );
        dist.clock().log( ITER, ss.str() );

        if( dist.condition() ) break;

    }
    
    // FINALIZE
    dist.finalize();
    std::stringstream ss;
    ss << m_logdir << "rank-" << env.rank();
    dist.clock().flush( ss.str(), 1 );
    ss.str(std::string());
    env.finalize();
    
    return 0;

}
