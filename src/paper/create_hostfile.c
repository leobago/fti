#include "interface-paper.h"
#include "../interface.h"
#include <sys/types.h>
#include <unistd.h>

#ifdef __INTEL_COMPILER
#   define HFPAT "%s:%d\n"
#else
#   define HFPAT "%s slots=%d\n"
#endif

void XFTI_LiberateHeads()
{
    if( topo->nodeRank == 0 + topo->nbHeads ) {
        int dummy;
        MPI_Send( &dummy, 1, MPI_INT, topo->headRank, conf->failedTag, exec->globalComm );
    }
}
int XFTI_updateKeyCfg( const char* tag, const char* key, const char* value )
{

    // Load dictionary
    dictionary* ini = iniparser_load(conf->cfgFile);
    if (ini == NULL) {
        FTI_Print("Iniparser failed to parse the conf. file.", FTI_WARN);
        return FTI_NSCS;
    }

    char str[FTI_BUFS];
    snprintf(str, FTI_BUFS, "%s:%s", tag, key);
    iniparser_set(ini, str, value);

    FILE* fd = fopen(conf->cfgFile, "w");
    if (fd == NULL) {
        FTI_Print("FTI failed to open the configuration file.", FTI_EROR);

        iniparser_freedict(ini);

        return FTI_NSCS;
    }

    // Write new configuration
    iniparser_dump_ini(ini, fd);

    if (fclose(fd) != 0) {
        FTI_Print("FTI failed to close the configuration file.", FTI_EROR);

        iniparser_freedict(ini);

        return FTI_NSCS;
    }

    // Free dictionary
    iniparser_freedict(ini);

    return FTI_SCES;

}

void XFTI_CrashNodes( int nbNodes ) 
{
    MPI_Barrier(FTI_COMM_WORLD);
    MPI_Comm affectedNodes;
    MPI_Comm_split( FTI_COMM_WORLD, topo->nodeID < nbNodes, 0, &affectedNodes );
    if( topo->nodeID < nbNodes ) {
        int rank, size;
        MPI_Comm_size(affectedNodes, &size);
        MPI_Comm_rank(affectedNodes, &rank);
        if( (topo->nodeRank == 1) && (topo->nbHeads == 1) ) {
            int value = 1;
            MPI_Ssend(&value, 1, MPI_INT, topo->headRank, conf->killTag, exec->globalComm);
            MPI_Recv(&value, 1, MPI_INT, topo->headRank, conf->killTag, exec->globalComm, MPI_STATUS_IGNORE);
        }
        MPI_Barrier( affectedNodes );
        sleep(2);
        DBG_MSG("I WILL DIE (pid:%d|rank:%d)",-1, getpid(), topo->myRank);
        XFTI_CRASH;
    } else {
        //sleep(4);
        //DBG_MSG("I WILL SURVIVE",-1);
        DBG_MSG("I WILL SURVIVE (pid:%d|rank:%d)",-1, getpid(), topo->myRank);
    }
    //sleep(2);
}

void XFTI_Crash() 
{
    if( topo->nbHeads == 1 ) {
        int value = FTI_ENDW;
        MPI_Send(&value, 1, MPI_INT, topo->headRank, conf->finalTag, exec->globalComm);
    }
    MPI_Barrier( exec->globalComm );
    MPI_Abort( exec->globalComm, -1 );
}

int XFTI_GetNbNodes() { return topo->nbNodes; }
int XFTI_GetNbApprocs() { return topo->nbApprocs; }

int XFTI_CreateHostfile( int nbNodes, const char* nodeList, const char* fn )
{
    char hostname[FTI_BUFS]; hostname[FTI_BUFS-1]='\0';
    int i;

    if( nodeList == NULL ) {
        if( topo->nodeID < nbNodes ) { 
            if( (topo->splitRank%topo->nbApprocs == 0) && !(topo->splitRank == 0) ) {
                gethostname( hostname, FTI_BUFS-1 );
                MPI_Send( hostname, FTI_BUFS, MPI_CHAR, 0, conf->generalTag, FTI_COMM_WORLD ); 
            }
        }

        if( topo->splitRank == 0 ) {
            FILE* out = fopen( fn, "w" );
            gethostname( hostname, FTI_BUFS-1 );
            fprintf( out, HFPAT, hostname, topo->nodeSize );  
            for(i=1; i<nbNodes; i++) {
                MPI_Recv( hostname, FTI_BUFS, MPI_CHAR, i*topo->nbApprocs, conf->generalTag, FTI_COMM_WORLD, MPI_STATUS_IGNORE );
                fprintf( out, HFPAT, hostname, topo->nodeSize );  
            }
            fclose(out);
        }
    }

    return FTI_SCES;
}
