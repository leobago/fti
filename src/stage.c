#include "interface.h"


int FTI_InitStage( FTIT_execution *FTI_Exec, FTIT_configuration *FTI_Conf, FTIT_topology *FTI_Topo ) 
{
    
    // allocate nbApproc instances of Stage info for heads but only one for application processes
    int num = ( FTI_Topo->amIaHead ) ? FTI_Topo->nbApprocs : 1;
    FTI_Exec->stageInfo = malloc( num * sizeof(FTIT_StageInfo) );

    // create node communicator
    MPI_Info info;
    MPI_Comm_split_type( FTI_Exec->globalComm, MPI_COMM_TYPE_SHARED, FTI_Topo->myRank, info, &FTI_Exec->nodeComm ); 
    
    int size;
    MPI_Comm_size( FTI_Exec->nodeComm, &size );
    if ( size != FTI_Topo->nbApprocs ) {
        FTI_Print("Wrong size of node communicator, disable staging!", FTI_WARN );
        FTI_Conf->stagingEnabled = false;
        MPI_Comm_free( &FTI_Exec->nodeComm );
        return FTI_NSCS;
    }

    // store rank
    MPI_Comm_rank( FTI_Exec->nodeComm, &FTI_Topo->nodeRank );

}

int FTI_InitStageRequestApp( FTIT_execution *FTI_Exec, FTIT_topology *FTI_Topo, uint32_t ID ) 
{
    
    FTIT_StageAppInfo *sInfo = malloc( sizeof(FTIT_StageAppInfo) );
    if( sInfo == NULL ) {
        FTI_Print( "failed to allocate memory for 'sInfo'", FTI_EROR );
        return FTI_NSCS;
    }
    
    // store request in fti execution
    if( FTI_Exec->stageInfo->firstReq == NULL ) {  // init request list
        FTI_Exec->stageInfo->firstReq = (void*) sInfo;
        FTI_Exec->stageInfo->lastReq = (void*) sInfo;
        sInfo->next = NULL;
        sInfo->previous = NULL;
    } else { // append request to request list
        sInfo->previous = FTI_SI_APTR(FTI_Exec->stageInfo->lastReq);
        FTI_SI_APTR(FTI_Exec->stageInfo->lastReq)->next = sInfo;
        FTI_Exec->stageInfo->lastReq = sInfo;
        sInfo->next = NULL;
    }

    sInfo->ID = ID;
    FTI_Exec->stageInfo->status[ID] = FTI_STAGE_PEND; // init status fail
    sInfo->mpiReq = MPI_REQUEST_NULL;
}

int FTI_InitStageRequestHead( char* lpath, char *rpath, FTIT_execution *FTI_Exec, 
        FTIT_topology *FTI_Topo, int source, uint32_t ID ) 
{
    
    FTIT_StageHeadInfo *sInfo = malloc( sizeof(FTIT_StageHeadInfo) );
    if( sInfo == NULL ) {
        FTI_Print( "failed to allocate memory for 'sInfo'", FTI_EROR );
        return FTI_NSCS;
    }

    // store request in fti execution
    if( FTI_Exec->stageInfo[source].firstReq == NULL ) {  // init request list
        FTI_Exec->stageInfo[source].firstReq = (void*) sInfo;
        FTI_Exec->stageInfo[source].lastReq = (void*) sInfo;
        sInfo->previous = NULL;
        sInfo->next = NULL;
    } else { // append request to request list
        sInfo->previous = FTI_SI_HPTR(FTI_Exec->stageInfo[source].lastReq);
        FTI_SI_HPTR(FTI_Exec->stageInfo[source].lastReq)->next = (void*) sInfo;
        FTI_Exec->stageInfo[source].lastReq = (void*) sInfo;
        sInfo->next = NULL;
    }

    strncpy( sInfo->lpath, lpath, FTI_BUFS );
    strncpy( sInfo->rpath, rpath, FTI_BUFS );
    sInfo->offset = 0;
    sInfo->size = 0;
    
    // TODO
    // here lock window and put the new value FTI_STAGE_ACTV to status
}

int FTI_SyncStage( char* lpath, char *rpath, FTIT_execution *FTI_Exec, FTIT_configuration *FTI_Conf, uint32_t ID ) 
{

    FTI_Exec->stageInfo->status[ID] = FTI_STAGE_ACTV; // init status fail
    
    char errstr[FTI_BUFS];

    // set buffer size for the file data transfer
    size_t bs = FTI_Conf->transferSize;

    // check local file and get file size
    struct stat st;
    if(  stat( lpath, &st ) == -1 ) {
        snprintf( errstr, FTI_BUFS, "Could not stat the local file ('%s') for staging", lpath );
        FTI_Print( errstr, FTI_EROR );
        errno = 0;
        FTI_Exec->stageInfo->status[ID] = FTI_STAGE_FAIL; // init status fail
        return FTI_NSCS;
    }
    if( !S_ISREG(st.st_mode) ) {
        snprintf( errstr, FTI_BUFS, "'%s' is not a regular file, staging failed.", lpath );
        FTI_Print( errstr, FTI_EROR );
        errno = 0;
        FTI_Exec->stageInfo->status[ID] = FTI_STAGE_FAIL; // init status fail
        return FTI_NSCS;
    }

    off_t eof = st.st_size;

    // open local file
    int fd_local = open( lpath, O_RDONLY );
    if( fd_local == -1 ) {
        FTI_Print("Could not open the local file for staging", FTI_EROR);
        FTI_Exec->stageInfo->status[ID] = FTI_STAGE_FAIL; // init status fail
        return FTI_NSCS;
    }
    // open file on remote fs
    int fd_global = open( rpath, O_WRONLY|O_CREAT, (mode_t) 0600 );
    if( fd_global == -1 ) {
        FTI_Print("Could not open the destination file for staging", FTI_EROR);
        close( fd_local );
        FTI_Exec->stageInfo->status[ID] = FTI_STAGE_FAIL; // init status fail
        return FTI_NSCS;
    }
    // allocate buffer
    char *buf = (char*) malloc( bs );

    // move file to destination
    off_t pos = 0;
    ssize_t read_bytes, write_bytes;
    size_t buf_bytes;
    while( pos < eof ) {
        buf_bytes = ( (eof - pos) < bs ) ? eof - pos : bs;
        // for the case we have written less then we have read
        if( lseek( fd_local, pos, SEEK_SET ) == -1 ) {
            snprintf( errstr, FTI_BUFS, "unable to seek in '%s'.", lpath );
            FTI_Print( errstr, FTI_EROR );
            errno = 0;
            free( buf );
            close( fd_local );
            close( fd_global );
            FTI_Exec->stageInfo->status[ID] = FTI_STAGE_FAIL; // init status fail
            return FTI_NSCS;
        }
        if( (read_bytes = read( fd_local, buf, buf_bytes )) == -1 ) {
            snprintf( errstr, FTI_BUFS, "unable to read from '%s'.", lpath );
            FTI_Print( errstr, FTI_EROR );
            errno = 0;
            free( buf );
            close( fd_local );
            close( fd_global );
            FTI_Exec->stageInfo->status[ID] = FTI_STAGE_FAIL; // init status fail
            return FTI_NSCS;
        }
        if( (write_bytes = write( fd_global, buf, read_bytes )) == -1 ) {  
            snprintf( errstr, FTI_BUFS, "unable to write to '%s'.", rpath );
            FTI_Print( errstr, FTI_EROR );
            errno = 0;
            free( buf );
            close( fd_local );
            close( fd_global );
            FTI_Exec->stageInfo->status[ID] = FTI_STAGE_FAIL; // init status fail
            return FTI_NSCS;
        }
        pos += write_bytes;
        printf("pos: %lu, write_bytes: %lu, read_bytes: %lu, buf_bytes: %lu\n", pos, write_bytes, read_bytes, buf_bytes);
    }

    // deallocate buffer and close file descriptors
    free( buf );
    close( fd_local );
    fsync( fd_global );
    close( fd_global );

    if( remove( lpath ) == -1 ) {
        snprintf( errstr, FTI_BUFS, "Could not remove local file '%s'.", lpath );
        FTI_Print( errstr, FTI_WARN );
        errno = 0;
        FTI_Exec->stageInfo->status[ID] = FTI_STAGE_FAIL; // init status fail
        return FTI_NSCS;
    }

    FTI_Exec->stageInfo->status[ID] = FTI_STAGE_SCES;
    
    return FTI_SCES;

}
int FTI_StageRequestHead( char *lpath, char *rpath, FTIT_configuration *FTI_Conf, 
        FTIT_execution *FTI_Exec, FTIT_topology *FTI_Topo, uint32_t ID ) {
    
    // serialize request before sending to the head
    void *buf_ser = malloc ( FTI_BUFS + sizeof(uint32_t) );
    int pos = 0;
    memcpy( buf_ser, lpath, FTI_BUFS );   
    pos += FTI_BUFS;
    memcpy( buf_ser + pos, rpath, FTI_BUFS );   
    pos += FTI_BUFS;
    memcpy( buf_ser + pos, &ID, sizeof(uint32_t) );   
    pos += sizeof(uint32_t);
    MPI_Datatype buf_t;
    MPI_Type_contiguous( pos, MPI_CHAR, &buf_t );
    MPI_Type_commit( &buf_t );
    
    // send request to head
    int ierr = MPI_Isend( buf_ser, 1, buf_t, FTI_Topo->headRank, 
            FTI_Conf->stageTag, FTI_Exec->globalComm, &(FTI_SI_APTR(FTI_Exec->stageInfo->lastReq)->mpiReq) );
    if ( ierr != MPI_SUCCESS ) {
        char errstr[FTI_BUFS], mpierrbuf[FTI_BUFS];
        int reslen;
        MPI_Error_string( ierr, mpierrbuf, &reslen );
        snprintf( errstr, FTI_BUFS, "MPI_Isend failed: %s", mpierrbuf );  
        FTI_Print( errstr, FTI_WARN );
        return FTI_NSCS;
    }

    MPI_Type_free( &buf_t );
}

int FTI_HandleStageRequest(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt, int source)
{      
    char errstr[FTI_BUFS];

    size_t buf_ser_size = 2*FTI_BUFS + sizeof(uint32_t);
    void *buf_ser = malloc ( buf_ser_size );
    MPI_Datatype buf_t;
    MPI_Type_contiguous( buf_ser_size, MPI_CHAR, &buf_t );
    MPI_Type_commit( &buf_t );
    MPI_Recv( buf_ser, 1, buf_t, source, FTI_Conf->stageTag, FTI_Exec->globalComm, MPI_STATUS_IGNORE );
  
    // set local file path
    char lpath[FTI_BUFS];
    char rpath[FTI_BUFS];
    memset( lpath, 0x0, FTI_BUFS );
    memset( rpath, 0x0, FTI_BUFS );

    strncpy( lpath, buf_ser, FTI_BUFS );
    strncpy( rpath, buf_ser+FTI_BUFS, FTI_BUFS );
    uint32_t ID = *(uint32_t*)(buf_ser+2*FTI_BUFS);
    
    // init Head staging meta data
    if ( FTI_InitStageRequestHead( lpath, rpath, FTI_Exec, FTI_Topo, source, ID ) != FTI_SCES ) {
        FTI_Print( "failed to initialize stage request meta info!", FTI_WARN );
        // TODO
        // here lock window and put the new value FTI_STAGE_FAIL to status
        return FTI_NSCS;
    }
   
    // set buffer size for the file data transfer
    size_t bs = FTI_Conf->transferSize;

    // check local file and get file size
    struct stat st;
    if(  stat( lpath, &st ) == -1 ) {
        snprintf( errstr, FTI_BUFS, "Could not stat the local file ('%s') for staging", lpath );
        FTI_Print( errstr, FTI_EROR );
        errno = 0;
        // TODO
        // here lock window and put the new value FTI_STAGE_FAIL to status
        return FTI_NSCS;
    }
    if( !S_ISREG(st.st_mode) ) {
        snprintf( errstr, FTI_BUFS, "'%s' is not a regular file, staging failed.", lpath );
        FTI_Print( errstr, FTI_EROR );
        errno = 0;
        // TODO
        // here lock window and put the new value FTI_STAGE_FAIL to status
        return FTI_NSCS;
    }

    off_t eof = st.st_size;

    // open local file
    int fd_local = open( lpath, O_RDONLY );
    if( fd_local == -1 ) {
        FTI_Print("Could not open the local file for staging", FTI_EROR);
        // TODO
        // here lock window and put the new value FTI_STAGE_FAIL to status
        return FTI_NSCS;
    }
    // open file on remote fs
    int fd_global = open( rpath, O_WRONLY|O_CREAT, (mode_t) 0600 );
    if( fd_global == -1 ) {
        FTI_Print("Could not open the destination file for staging", FTI_EROR);
        close( fd_local );
        // TODO
        // here lock window and put the new value FTI_STAGE_FAIL to status
        return FTI_NSCS;
    }
    // allocate buffer
    char *buf = (char*) malloc( bs );

    // move file to destination
    off_t pos = 0;
    ssize_t read_bytes, write_bytes;
    size_t buf_bytes;
    while( pos < eof ) {
        buf_bytes = ( (eof - pos) < bs ) ? eof - pos : bs;
        // for the case we have written less then we have read
        if( lseek( fd_local, pos, SEEK_SET ) == -1 ) {
            snprintf( errstr, FTI_BUFS, "unable to seek in '%s'.", lpath );
            FTI_Print( errstr, FTI_EROR );
            errno = 0;
            free( buf );
            close( fd_local );
            close( fd_global );
            // TODO
            // here lock window and put the new value FTI_STAGE_FAIL to status
            return FTI_NSCS;
        }
        if( (read_bytes = read( fd_local, buf, buf_bytes )) == -1 ) {
            snprintf( errstr, FTI_BUFS, "unable to read from '%s'.", lpath );
            FTI_Print( errstr, FTI_EROR );
            errno = 0;
            free( buf );
            close( fd_local );
            close( fd_global );
            // TODO
            // here lock window and put the new value FTI_STAGE_FAIL to status
            return FTI_NSCS;
        }
        if( (write_bytes = write( fd_global, buf, read_bytes )) == -1 ) {  
            snprintf( errstr, FTI_BUFS, "unable to write to '%s'.", rpath );
            FTI_Print( errstr, FTI_EROR );
            errno = 0;
            free( buf );
            close( fd_local );
            close( fd_global );
            // TODO
            // here lock window and put the new value FTI_STAGE_FAIL to status
            return FTI_NSCS;
        }
        pos += write_bytes;
        printf("pos: %lu, write_bytes: %lu, read_bytes: %lu, buf_bytes: %lu\n", pos, write_bytes, read_bytes, buf_bytes);
    }

    // deallocate buffer and close file descriptors
    free( buf );
    close( fd_local );
    fsync( fd_global );
    close( fd_global );

    if( remove( lpath ) == -1 ) {
        snprintf( errstr, FTI_BUFS, "Could not remove local file '%s'.", lpath );
        FTI_Print( errstr, FTI_WARN );
        errno = 0;
        // TODO
        // here lock window and put the new value FTI_STAGE_FAIL to status
        return FTI_NSCS;
    }

    // TODO
    // here lock window and put the new value FTI_STAGE_SCES to status
      
    return FTI_SCES;
}
