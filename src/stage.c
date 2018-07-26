#include "interface.h"

/* NOTE: 
 *
 * The status variable is assumed to be written in an atomic operation since it is
 * a 32-bit data type (https://stackoverflow.com/questions/24931456/how-does-sig-atomic-t-actually-work)
 *
 * Therefor we do not use MPI_Accumulate with MPI_REPLACE operation (which ensures to replace each 
 * element in an atomic operation).
 * 
 * However, we need to assure, that operations that consist of multiple instructions are performed
 * on a copy of the status field. The final value of the copy is then assigned to the  original 
 * status field variable in memory.
 *
 */

// location of ID corresponding request array element (rank local)
static uint32_t *idxRequest;

int FTI_GetRequestIdx( int ID ) {
    if ( FTI_GetRequestField( ID, FTI_SIF_ALL ) ) {
        return FTI_GetRequestField( ID, FTI_SIF_IDX );
    } else {
        return -1;
    }
}

int FTI_InitStage( FTIT_execution *FTI_Exec, FTIT_configuration *FTI_Conf, FTIT_topology *FTI_Topo ) 
{
    
    // memory window size
    size_t win_size = FTI_SI_MAX_NUM * sizeof(uint8_t) * !(FTI_Topo->amIaHead);
    
    // requestIdx array size
    size_t arr_size = FTI_SI_MAX_NUM * sizeof(uint32_t);

    // allocate nbApproc instances of Stage info for heads but only one for application processes
    int num = ( FTI_Topo->amIaHead ) ? FTI_Topo->nbApprocs : 1;
    FTI_Exec->stageInfo = malloc( num * sizeof(FTIT_StageInfo) );

    // allocate request idx array and init to 0x0
    idxRequest = calloc( 1, arr_size );

    // set init request value (important since we call realloc)
    FTI_Exec->stageInfo->request = NULL;

    // set request counter to 0
    FTI_Exec->stageInfo->nbRequest = 0;

    // create node communicator
    // NOTE: head is assigned the highest rank. This is important in order
    // to access the stageInfo array by FTI_Topo->nodeRank (i.e. source)
    int key = (FTI_Topo->amIaHead) ? 1 : 0;
    if ( FTI_Conf->test ) {
        int color = FTI_Topo->nodeID;
        MPI_Comm_split( FTI_Exec->globalComm, color, key, &FTI_Exec->nodeComm );
    } else {
        MPI_Comm_split_type( FTI_Exec->globalComm, MPI_COMM_TYPE_SHARED, key, MPI_INFO_NULL, &FTI_Exec->nodeComm ); 
    }
    
    // check for a consistant communicator size
    int size;
    MPI_Comm_size( FTI_Exec->nodeComm, &size );
    if ( size != (FTI_Topo->nbApprocs+1) ) {
        FTI_Print("Wrong size of node communicator, disable staging!", FTI_WARN );
        FTI_Conf->stagingEnabled = false;
        MPI_Comm_free( &FTI_Exec->nodeComm );
        return FTI_NSCS;
    }

    // store rank
    MPI_Comm_rank( FTI_Exec->nodeComm, &FTI_Topo->nodeRank );
    
    // store head rank in node communicator
    // NOTE: should be the highest rank number
    FTI_Topo->headRankNode = size-1;

    // create shared memory window
    int disp = sizeof(uint8_t);
    MPI_Info win_info;
    MPI_Info_create(&win_info);
    MPI_Info_set(win_info, "alloc_shared_noncontig", "true");
    MPI_Win_allocate_shared( win_size, disp, win_info, FTI_Exec->nodeComm, &(FTI_Exec->stageInfo->status), &(FTI_Exec->stageInfo->stageWin) );
    MPI_Info_free(&win_info);

    // wait for windows syncronization
    MPI_Win_sync(FTI_Exec->stageInfo->stageWin);
    MPI_Barrier(FTI_Exec->nodeComm);

    // init shared memory window segments to 0x0
    if ( !(FTI_Topo->amIaHead) ) {
        MPI_Aint qsize;
        int qdisp;
        MPI_Win_shared_query( FTI_Exec->stageInfo->stageWin, FTI_Topo->nodeRank, &qsize, &qdisp, &(FTI_Exec->stageInfo->status) );
        memset( FTI_Exec->stageInfo->status, 0x0, win_size );
    }

    // create stage directory
    snprintf(FTI_Conf->stageDir, FTI_BUFS, "%s/stage", FTI_Conf->localDir);
    if (mkdir(FTI_Conf->stageDir, 0777) == -1) {
        if (errno != EEXIST) {
            FTI_Print("Cannot create stage directory", FTI_EROR);
        }
    }
    
}

int FTI_InitStageRequestApp( FTIT_execution *FTI_Exec, FTIT_topology *FTI_Topo, uint32_t ID ) 
{
    
    int idx = FTI_Exec->stageInfo->nbRequest++;
    FTI_Exec->stageInfo->request = realloc( FTI_Exec->stageInfo->request, sizeof(FTIT_StageAppInfo) * FTI_Exec->stageInfo->nbRequest );
    if( FTI_Exec->stageInfo->request == NULL ) {
        // TODO this is a fatal error, finalize staging feature required
        FTI_Print( "failed to allocate memory", FTI_EROR );
        return FTI_NSCS;
    }

    // init structure
    FTI_SetStatusField( FTI_Exec, FTI_Topo, ID, FTI_SI_PEND, FTI_SIF_VAL, FTI_Topo->nodeRank );
    FTI_SI_APTR(FTI_Exec->stageInfo->request)[idx].mpiReq = MPI_REQUEST_NULL;

    FTI_SetRequestField( ID, FTI_SI_IALL, FTI_SIF_ALL );
    FTI_SetRequestField( ID, idx, FTI_SIF_IDX );

    return FTI_SCES;

}

int FTI_InitStageRequestHead( char* lpath, char *rpath, FTIT_execution *FTI_Exec, 
        FTIT_topology *FTI_Topo, int source, uint32_t ID ) 
{
    
    FTIT_StageInfo *si = &(FTI_Exec->stageInfo[source]); 
    int idx = si->nbRequest++;
    si->request = realloc( si->request, sizeof(FTIT_StageHeadInfo) * si->nbRequest );
    if( si->request == NULL ) {
        // TODO this is a fatal error, finalize staging feature required
        FTI_Print( "failed to allocate memory", FTI_EROR );
        return FTI_NSCS;
    }

    strncpy( FTI_SI_HPTR(si->request)[idx].lpath, lpath, FTI_BUFS );
    strncpy( FTI_SI_HPTR(si->request)[idx].rpath, rpath, FTI_BUFS );
    FTI_SI_HPTR(si->request)[idx].offset = 0;
    FTI_SI_HPTR(si->request)[idx].size = 0;

    FTI_SetStatusField( FTI_Exec, FTI_Topo, ID, FTI_SI_ACTV, FTI_SIF_VAL, source );

    FTI_SetRequestField( ID, FTI_SI_IALL, FTI_SIF_ALL );
    FTI_SetRequestField( ID, idx, FTI_SIF_IDX );
    
    return FTI_SCES;
    
}

void FTI_FreeStageRequest( FTIT_execution *FTI_Exec, FTIT_topology *FTI_Topo, int ID ) 
{ 
    if ( !FTI_GetRequestField( ID, FTI_SIF_ALL ) ) {
        return;
    }
    
    int idx = FTI_GetRequestField( ID, FTI_SIF_IDX );

    size_t type_size = ( FTI_Topo->amIaHead ) ? sizeof(FTIT_StageHeadInfo) : sizeof(FTIT_StageAppInfo);
    size_t offset_ID = ( FTI_Topo->amIaHead ) ? offsetof( FTIT_StageHeadInfo, ID ) : offsetof( FTIT_StageAppInfo, ID );

    // ptr is all we need
    void *ptr = FTI_Exec->stageInfo->request;

    // if last element in array, we just need to truncate the array.
    if ( idx == (FTI_Exec->stageInfo->nbRequest-1) ) {
        ptr = realloc( ptr, type_size * (--FTI_Exec->stageInfo->nbRequest) );
    } 
    // if not last element, we need to truncate array and move elements (before truncation)
    else {
        void *dest = ptr + type_size*idx;
        void *src = ptr + type_size*(idx+1);
        size_t mem_size = (FTI_Exec->stageInfo->nbRequest - (idx+1)) * type_size; 
        memmove( dest, src, mem_size );
        ptr = realloc( ptr, type_size * (--FTI_Exec->stageInfo->nbRequest) );
        // re-assign the correct values
        int i = idx;
        void *pos = ptr + type_size*idx;
        for ( ; i<FTI_Exec->stageInfo->nbRequest; ++i, pos+=type_size ) {
            int ID = *(int*)(pos + offset_ID);
            FTI_SetRequestField( ID, i, FTI_SIF_IDX );
        }
    }

    FTI_SetRequestField( ID, FTI_SI_NALL, FTI_SIF_ALL );

}

int FTI_SyncStage( char* lpath, char *rpath, FTIT_execution *FTI_Exec, FTIT_configuration *FTI_Conf, uint32_t ID ) 
{

    FTI_Exec->stageInfo->status[ID] = FTI_SI_ACTV; // init status fail
    
    char errstr[FTI_BUFS];

    // set buffer size for the file data transfer
    size_t bs = FTI_Conf->transferSize;

    // check local file and get file size
    struct stat st;
    if(  stat( lpath, &st ) == -1 ) {
        snprintf( errstr, FTI_BUFS, "Could not stat the local file ('%s') for staging", lpath );
        FTI_Print( errstr, FTI_EROR );
        errno = 0;
        //FTI_Exec->stageInfo->status[ID] = FTI_SI_FAIL; // init status fail
        return FTI_NSCS;
    }
    if( !S_ISREG(st.st_mode) ) {
        snprintf( errstr, FTI_BUFS, "'%s' is not a regular file, staging failed.", lpath );
        FTI_Print( errstr, FTI_EROR );
        errno = 0;
        //FTI_Exec->stageInfo->status[ID] = FTI_SI_FAIL; // init status fail
        return FTI_NSCS;
    }

    off_t eof = st.st_size;

    // open local file
    int fd_local = open( lpath, O_RDONLY );
    if( fd_local == -1 ) {
        FTI_Print("Could not open the local file for staging", FTI_EROR);
        //FTI_Exec->stageInfo->status[ID] = FTI_SI_FAIL; // init status fail
        return FTI_NSCS;
    }
    // open file on remote fs
    int fd_global = open( rpath, O_WRONLY|O_CREAT, (mode_t) 0600 );
    if( fd_global == -1 ) {
        FTI_Print("Could not open the destination file for staging", FTI_EROR);
        close( fd_local );
        //FTI_Exec->stageInfo->status[ID] = FTI_SI_FAIL; // init status fail
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
            //FTI_Exec->stageInfo->status[ID] = FTI_SI_FAIL; // init status fail
            return FTI_NSCS;
        }
        if( (read_bytes = read( fd_local, buf, buf_bytes )) == -1 ) {
            snprintf( errstr, FTI_BUFS, "unable to read from '%s'.", lpath );
            FTI_Print( errstr, FTI_EROR );
            errno = 0;
            free( buf );
            close( fd_local );
            close( fd_global );
            //FTI_Exec->stageInfo->status[ID] = FTI_SI_FAIL; // init status fail
            return FTI_NSCS;
        }
        if( (write_bytes = write( fd_global, buf, read_bytes )) == -1 ) {  
            snprintf( errstr, FTI_BUFS, "unable to write to '%s'.", rpath );
            FTI_Print( errstr, FTI_EROR );
            errno = 0;
            free( buf );
            close( fd_local );
            close( fd_global );
            //FTI_Exec->stageInfo->status[ID] = FTI_SI_FAIL; // init status fail
            return FTI_NSCS;
        }
        pos += write_bytes;
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
        //FTI_Exec->stageInfo->status[ID] = FTI_SI_FAIL; // init status fail
        return FTI_NSCS;
    }

    //FTI_Exec->stageInfo->status[ID] = FTI_SI_SCES;
    
    return FTI_SCES;

}
int FTI_AsyncStage( char *lpath, char *rpath, FTIT_configuration *FTI_Conf, 
        FTIT_execution *FTI_Exec, FTIT_topology *FTI_Topo, uint32_t ID ) {
    
    // serialize request before sending to the head
    void *buf_ser = malloc ( 2*FTI_BUFS + sizeof(uint32_t) );
    int pos = 0;
    memcpy( buf_ser, lpath, FTI_BUFS );   
    pos += FTI_BUFS;
    memcpy( buf_ser + pos, rpath, FTI_BUFS );   
    pos += FTI_BUFS;
    memcpy( buf_ser + pos, &ID, sizeof(uint32_t) );   
    pos += sizeof(uint32_t);
    MPI_Datatype buf_t;
    MPI_Type_contiguous( pos, MPI_BYTE, &buf_t );
    MPI_Type_commit( &buf_t );
   
    // send request to head
    int idx = FTI_GetRequestField( ID, FTI_SIF_IDX );
    int ierr = MPI_Isend( buf_ser, 1, buf_t, FTI_Topo->headRankNode, 
            FTI_Conf->stageTag, FTI_Exec->nodeComm, &(FTI_SI_APTR(FTI_Exec->stageInfo->request)[idx].mpiReq) );
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
    MPI_Recv( buf_ser, 1, buf_t, source, FTI_Conf->stageTag, FTI_Exec->nodeComm, MPI_STATUS_IGNORE );
  
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
        // here lock window and put the new value FTI_SI_FAIL to status
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
        // here lock window and put the new value FTI_SI_FAIL to status
        return FTI_NSCS;
    }
    if( !S_ISREG(st.st_mode) ) {
        snprintf( errstr, FTI_BUFS, "'%s' is not a regular file, staging failed.", lpath );
        FTI_Print( errstr, FTI_EROR );
        errno = 0;
        // TODO
        // here lock window and put the new value FTI_SI_FAIL to status
        return FTI_NSCS;
    }

    off_t eof = st.st_size;

    // open local file
    int fd_local = open( lpath, O_RDONLY );
    if( fd_local == -1 ) {
        FTI_Print("Could not open the local file for staging", FTI_EROR);
        // TODO
        // here lock window and put the new value FTI_SI_FAIL to status
        return FTI_NSCS;
    }
    // open file on remote fs
    int fd_global = open( rpath, O_WRONLY|O_CREAT, (mode_t) 0600 );
    if( fd_global == -1 ) {
        FTI_Print("Could not open the destination file for staging", FTI_EROR);
        close( fd_local );
        // TODO
        // here lock window and put the new value FTI_SI_FAIL to status
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
            // here lock window and put the new value FTI_SI_FAIL to status
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
            // here lock window and put the new value FTI_SI_FAIL to status
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
            // here lock window and put the new value FTI_SI_FAIL to status
            return FTI_NSCS;
        }
        pos += write_bytes;
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
        // here lock window and put the new value FTI_SI_FAIL to status
        return FTI_NSCS;
    }

    // TODO
    // here lock window and put the new value FTI_SI_SCES to status
    
    FTI_SetStatusField( FTI_Exec, FTI_Topo, ID, FTI_SI_SCES, FTI_SIF_VAL, source );
    FTI_FreeStageRequest( FTI_Exec, FTI_Topo, ID );

    return FTI_SCES;
}

int FTI_GetRequestField( int ID, FTIT_StatusField val ) 
{

    if( (val < FTI_SIF_ALL) || (val > FTI_SIF_IDX) ) {
        FTI_Print( "invalid argument for 'FTI_GetRequestIdxField'", FTI_WARN );
        return FTI_NSCS;
    }
    
    const uint32_t all_mask = 0x00080000;
    const uint32_t idx_mask = 0x0007FFFF; // 524,288 ID's, max ID: 524,287

    uint32_t field = idxRequest[ID];

    switch( val ) {

        case FTI_SIF_ALL:
            return (int)((field & all_mask) >> 19);
        case FTI_SIF_IDX:
            return ((int)(field & idx_mask));

    }

}

int FTI_SetRequestField( int ID, uint32_t entry, FTIT_StatusField val )
{

    if( (val < FTI_SIF_ALL) || (val > FTI_SIF_IDX) ) {
        FTI_Print( "invalid argument for 'FTI_SetRequestIdxField'", FTI_WARN );
        return FTI_NSCS;
    }

    const uint32_t all_mask = 0x00080000;
    const uint32_t idx_mask = 0x0007FFFF; // 524,288 ID's, max ID: 524,287

    int ierr = (int) entry;

    uint32_t field = idxRequest[ID];

    switch( val ) {

        case FTI_SIF_ALL:
            if ( ( entry > 0x1 ) ) { 
                FTI_Print( "invalid argument for 'FTI_SetRequestIdxField'", FTI_WARN );
                ierr = FTI_NSCS;
                break;
            }
            field = (entry << 19) | ((~all_mask) & field);
            break;
        case FTI_SIF_IDX:
            if ( entry > idx_mask ) {
                FTI_Print( "invalid argument for 'FTI_SetRequestIdxField'", FTI_WARN );
                ierr = FTI_NSCS;
                break;
            }
            field = entry | ((~idx_mask) & field);
            break;

    }
    
    if ( ierr != FTI_NSCS ) {
        idxRequest[ID] = field;
    }
    
    return ierr;

}
    
int FTI_GetStatusField( FTIT_execution *FTI_Exec, FTIT_topology *FTI_Topo, int ID, FTIT_StatusField val, int source ) 
{

    if( (val < 0) || (val > FTI_SIF_VAL) ) {
        FTI_Print( "invalid argument for 'FTI_GetStatusField'", FTI_WARN );
        return FTI_NSCS;
    }

    const uint8_t val_mask = 0xE;
    const uint8_t avl_mask = 0x1;

    int disp;
    MPI_Aint size;
    MPI_Win_shared_query( FTI_Exec->stageInfo->stageWin, source, &size, &disp, &(FTI_Exec->stageInfo->status) ); 
    uint8_t status = FTI_Exec->stageInfo->status[ID];

    switch( val ) {

        case FTI_SIF_VAL:
            return ((int)(status & val_mask)) >> 1;
        case FTI_SIF_AVL:
            return ((int)(status & avl_mask));

    }

}

int FTI_SetStatusField( FTIT_execution *FTI_Exec, FTIT_topology *FTI_Topo, int ID, uint8_t entry, FTIT_StatusField val, int source )
{

    if ( (val < 0) || (val > FTI_SIF_VAL) ) {
        FTI_Print( "invalid argument for 'FTI_GetStatusField'", FTI_WARN );
        return FTI_NSCS;
    }

    const uint8_t val_mask = 0xE; // in bits: 1110
    const uint8_t avl_mask = 0x1; // in bits: 0001
    
    int ierr = FTI_SCES;

    int disp;
    MPI_Aint size;
    MPI_Win_shared_query( FTI_Exec->stageInfo->stageWin, source, &size, &disp, &(FTI_Exec->stageInfo->status) ); 
    uint8_t status = FTI_Exec->stageInfo->status[ID];
    
    switch( val ) {

        case FTI_SIF_VAL:
            if ( (entry > 0x4) ) { 
                FTI_Print( "invalid argument for 'FTI_SetStatusField'", FTI_WARN );
                ierr = FTI_NSCS;
                break;
            }
            status = (entry << 1) | ((~val_mask) & status);
            break;
        case FTI_SIF_AVL:
            if ( entry > 0x1 ) {
                FTI_Print( "invalid argument for 'FTI_SetStatusField'", FTI_WARN );
                ierr = FTI_NSCS;
                break;
            }
            status = entry | ((~avl_mask) & status);
            break;

    }
    
    if ( ierr == FTI_SCES ) {
        FTI_Exec->stageInfo->status[ID] = status;
    }
    
    return ierr;

}

// stage request counter returns -1 if too many requests.
int FTI_GetRequestID( FTIT_execution *FTI_Exec, FTIT_topology *FTI_Topo ) 
{

    int ID = -1;

    static int req_cnt = 0;
    if( req_cnt < FTI_SI_MAX_NUM ) {
        ID = req_cnt++;
        FTI_SetStatusField( FTI_Exec, FTI_Topo, ID, FTI_SI_NAVL, FTI_SIF_AVL, FTI_Topo->nodeRank );
    } else {
        int i;
        // return first free status element index (i.e. ID)
        for( i=0; i<FTI_SI_MAX_NUM; ++i ) {
            if ( FTI_GetStatusField( FTI_Exec, FTI_Topo, i, FTI_SIF_AVL, FTI_Topo->nodeRank ) == FTI_SI_IAVL ) {
                ID = i;
                FTI_SetStatusField( FTI_Exec, FTI_Topo, ID, FTI_SI_NAVL, FTI_SIF_AVL, FTI_Topo->nodeRank );
            }
        }
    }

    return ID;

}

void FTI_PrintStatus( FTIT_execution *FTI_Exec, FTIT_topology *FTI_Topo, int ID, int source ) 
{
    
    MPI_Aint qsize;
    int qdisp;
    MPI_Win_shared_query( FTI_Exec->stageInfo->stageWin, source, &qsize, &qdisp, &(FTI_Exec->stageInfo->status) );

    uint8_t status = FTI_Exec->stageInfo->status[ID];

    int val;
    
    // get avl string
    val = FTI_GetStatusField( FTI_Exec, FTI_Topo, ID, FTI_SIF_AVL, source );
    char avlstr[FTI_BUFS];
    if ( val == FTI_SI_IAVL ) {
        strcpy( avlstr, "available" );
    } else if ( val == FTI_SI_NAVL ) {
        strcpy( avlstr, "not available" );
    } else {
        snprintf( avlstr, FTI_BUFS, "not valid ('%d')", val );
    }

    // get idx string
    val = FTI_GetRequestField( ID, FTI_SIF_IDX );
    char idxstr[FTI_BUFS];
    if ( val < 0 ) {
        snprintf(idxstr, FTI_BUFS, "not valid ('%d')", val );
    } else {
        snprintf(idxstr, FTI_BUFS, "%u", (uint8_t)val );
    }
    
    // get status value
    val = FTI_GetStatusField( FTI_Exec, FTI_Topo, ID, FTI_SIF_VAL, source );
    char valstr[FTI_BUFS];
    switch ( val ) {
        case FTI_SI_NINI:
            snprintf( valstr, FTI_BUFS, "not initialized" );
            break;
        case FTI_SI_FAIL:
            snprintf( valstr, FTI_BUFS, "failure" );
            break;
        case FTI_SI_SCES:    
            snprintf( valstr, FTI_BUFS, "success" );
            break;
        case FTI_SI_ACTV:
            snprintf( valstr, FTI_BUFS, "active" );
            break;
        case FTI_SI_PEND:
            snprintf( valstr, FTI_BUFS, "pending" );
            break;
        default:
            snprintf( valstr, FTI_BUFS, "not valid ('%d')", val );
    }

    printf("[rank(g|l):%d|%d][ID:%d] status is 'avl:%s', 'idx:%s', 'val:%s'\n", FTI_Topo->myRank, FTI_Topo->nodeRank, ID, avlstr, idxstr, valstr );

}

