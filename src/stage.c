#include "interface.h"

/* TODO
 * (1)  supply function to remove all files from staging folder.
 * (2)  add flag to FTI_SendFile( ..., FTI_SI_RM ) that indicates if the local file
 *      shall be deleted at success or failure.
 */

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
static uint8_t *status;                    /**< status of request              */
static MPI_Win stageWin;

bool *enableStagingPtr;

int FTI_GetRequestIdx( int ID ) {

    char str[FTI_BUFS];

    if ( FTI_GetRequestField( ID, FTI_SIF_ALL ) ) {
        return FTI_GetRequestField( ID, FTI_SIF_IDX );
    } else {
        return -1;
    }
}

int FTI_InitStage( FTIT_execution *FTI_Exec, FTIT_configuration *FTI_Conf, FTIT_topology *FTI_Topo ) 
{
    
    if ( !FTI_Conf->stagingEnabled ) {
        FTI_Print( "Staging disabled, invalid call to 'FTI_InitStage'", FTI_WARN );
        return FTI_NSCS;
    }

    // memory window size
    size_t win_size = FTI_SI_MAX_NUM * sizeof(uint8_t) * !(FTI_Topo->amIaHead);
    
    // requestIdx array size
    size_t arr_size = FTI_SI_MAX_NUM * sizeof(uint32_t);

    // keep ptr to enableStaging flag local to file
    enableStagingPtr = &FTI_Conf->stagingEnabled;

    // allocate nbApproc instances of Stage info for heads but only one for application processes
    int num = ( FTI_Topo->amIaHead ) ? FTI_Topo->nbApprocs : 1;
    FTI_Exec->stageInfo = calloc( num, sizeof(FTIT_StageInfo) );
    if ( FTI_Exec->stageInfo == NULL ) {
        FTI_DISABLE_STAGING;
        FTI_Print( "Failed to allocate memory for 'FTI_Exec->stageInfo'", FTI_EROR );
        return FTI_NSCS;
    }

    // allocate request idx array and init to 0x0. Head does not have one (would double memory consumption).
    if ( FTI_Topo->amIaHead ) {
        idxRequest = NULL;
    } else {
        idxRequest = calloc( 1, arr_size );
        if ( idxRequest == NULL ) {
            FTI_DISABLE_STAGING;
            FTI_Print( "Failed to allocate memory for 'idxRequest'", FTI_EROR );
            free( FTI_Exec->stageInfo );
            return FTI_NSCS;
        }
    }

    // create node communicator
    // NOTE: head is assigned rank 0. This is important in order 
    // to access the stageInfo array at the  head rank in the 
    // implemented way (array[app_rank-1], app_ranks = 1 -> nodeSize-1)
    int key = (FTI_Topo->amIaHead) ? 0 : 1;
    if ( FTI_Conf->test ) {
        int color = FTI_Topo->nodeID;
        MPI_Comm_split( FTI_Exec->globalComm, color, key, &FTI_Exec->nodeComm );
    } else {
        MPI_Comm_split_type( FTI_Exec->globalComm, MPI_COMM_TYPE_SHARED, key, MPI_INFO_NULL, &FTI_Exec->nodeComm ); 
    }
    
    // check for a consistant communicator size
    int size;
    MPI_Comm_size( FTI_Exec->nodeComm, &size );
    if ( size != FTI_Topo->nodeSize ) {
        FTI_DISABLE_STAGING;
        char str[FTI_BUFS];
        snprintf(str, FTI_BUFS, "Wrong size (%d != %d) of node communicator, disable staging!", size, FTI_Topo->nodeSize );
        FTI_Print(str, FTI_WARN );
        MPI_Comm_free( &FTI_Exec->nodeComm );
        free( FTI_Exec->stageInfo );
        free( idxRequest );
        return FTI_NSCS;
    }

    // store rank
    MPI_Comm_rank( FTI_Exec->nodeComm, &FTI_Topo->nodeRank );
    
    // store head rank in node communicator
    // NOTE: must(!!) be the lowest rank number
    FTI_Topo->headRankNode = 0;
    
    // create shared memory window
    int disp = sizeof(uint8_t);
    MPI_Info win_info;
    MPI_Info_create(&win_info);
    MPI_Info_set(win_info, "alloc_shared_noncontig", "true");
    MPI_Info_set(win_info, "no_locks", "true");
    MPI_Win_allocate_shared( win_size, disp, win_info, FTI_Exec->nodeComm, &status, &stageWin );
    MPI_Info_free(&win_info);

    // init shared memory window segments to 0x0
    if ( !(FTI_Topo->amIaHead) ) {
        MPI_Aint qsize;
        int qdisp;
        MPI_Win_shared_query( stageWin, FTI_Topo->nodeRank, &qsize, &qdisp, &status );
        memset( status, 0x0, win_size );
    }

    // create stage directory
    snprintf(FTI_Conf->stageDir, FTI_BUFS, "%s/stage", FTI_Conf->localDir);
    if (mkdir(FTI_Conf->stageDir, 0777) == -1) {
        if (errno != EEXIST) {
            FTI_DISABLE_STAGING;
            MPI_Win_free( &stageWin );
            MPI_Comm_free( &FTI_Exec->nodeComm );
            free( FTI_Exec->stageInfo );
            free( idxRequest );
            FTI_Print("Cannot create stage directory", FTI_EROR);
        }
    }
    
}

int FTI_InitStageRequestApp( FTIT_execution *FTI_Exec, FTIT_topology *FTI_Topo, uint32_t ID ) 
{
   
    if ( !FTI_SI_ENABLED ) {
        FTI_Print( "Staging disabled! Invalid call to 'FTI_InitStageRequestApp'", FTI_WARN );
        return FTI_NSCS;
    }

    if ( ID  > FTI_SI_MAX_ID ) {
        FTI_Print( "passed invalid ID to 'FTI_InitStageRequestApp'", FTI_WARN );
        return FTI_NSCS;
    }

    void *ptr = realloc( FTI_Exec->stageInfo->request, sizeof(FTIT_StageAppInfo) * (FTI_Exec->stageInfo->nbRequest+1) );
    if( ptr == NULL ) {
        FTI_Print( "failed to allocate memory for 'FTI_Exec->stageInfo->request'", FTI_EROR );
        return FTI_NSCS;
    }
    FTI_Exec->stageInfo->request = ptr;
    int idx = FTI_Exec->stageInfo->nbRequest++;

    FTI_SI_APTR(FTI_Exec->stageInfo->request)[idx].mpiReq = MPI_REQUEST_NULL;
    FTI_SI_APTR(FTI_Exec->stageInfo->request)[idx].sendBuf = NULL;
    
    FTI_SetStatusField( FTI_Exec, FTI_Topo, ID, FTI_SI_PEND, FTI_SIF_VAL, FTI_Topo->nodeRank );
    FTI_SetRequestField( ID, FTI_SI_IALL, FTI_SIF_ALL );
    FTI_SetRequestField( ID, idx, FTI_SIF_IDX );
    
    FTI_SI_APTR(FTI_Exec->stageInfo->request)[idx].ID = ID;

    return FTI_SCES;

}

int FTI_InitStageRequestHead( char* lpath, char *rpath, FTIT_execution *FTI_Exec, 
        FTIT_topology *FTI_Topo, int source, uint32_t ID ) 
{
   
    if ( !FTI_SI_ENABLED ) {
        FTI_Print( "Staging disabled, invalid call to 'FTI_InitStageRequestHead'", FTI_WARN );
        return FTI_NSCS;
    }

    if ( ID  > FTI_SI_MAX_ID ) {
        FTI_Print( "passed invalid ID to 'FTI_InitStageRequestHead'", FTI_WARN );
        return FTI_NSCS;
    }
    
    void *ptr = realloc( FTI_Exec->stageInfo[source-1].request, sizeof(FTIT_StageHeadInfo) * (FTI_Exec->stageInfo[source-1].nbRequest+1) );
    if( ptr == NULL ) {
        FTI_Print( "failed to allocate memory", FTI_EROR );
        return FTI_NSCS;
    }
    FTIT_StageInfo *si = &(FTI_Exec->stageInfo[source-1]); 
    si->request = ptr;
    int idx = si->nbRequest++;

    strncpy( FTI_SI_HPTR(si->request)[idx].lpath, lpath, FTI_BUFS );
    strncpy( FTI_SI_HPTR(si->request)[idx].rpath, rpath, FTI_BUFS );
    FTI_SI_HPTR(si->request)[idx].offset = 0;
    FTI_SI_HPTR(si->request)[idx].size = 0;
    FTI_SI_HPTR(si->request)[idx].ID = ID;

    FTI_SetStatusField( FTI_Exec, FTI_Topo, ID, FTI_SI_ACTV, FTI_SIF_VAL, source );
    
    return FTI_SCES;
    
}

int FTI_FreeStageRequest( FTIT_execution *FTI_Exec, FTIT_topology *FTI_Topo, int ID, int source ) 
{ 

    if ( !FTI_SI_ENABLED ) {
        FTI_Print( "Staging disabled, invalid call to 'FTI_FreeStageRequest'", FTI_WARN );
        return FTI_NSCS;
    }

    int status;
    status = FTI_GetStatusField( FTI_Exec, FTI_Topo, ID, FTI_SIF_VAL, source );  

    if ( !FTI_Topo->amIaHead ) {
        
        // if request already free, just return
        if ( !FTI_GetRequestField( ID, FTI_SIF_ALL ) ) {
            return FTI_SCES;
        }
        
        int idx = FTI_GetRequestField( ID, FTI_SIF_IDX );
    
        size_t type_size = sizeof(FTIT_StageAppInfo);
  
        // ptr is all we need
        FTIT_StageAppInfo *ptr = FTI_SI_APTR(FTI_Exec->stageInfo->request);

        // if last element in array, we just need to truncate the array.
        if ( idx == (FTI_Exec->stageInfo->nbRequest-1) ) {
            void *ptr_cpy = ptr;
            void *buf_ptr = ptr->sendBuf;
            ptr = realloc( ptr, type_size * (FTI_Exec->stageInfo->nbRequest-1) );
            if ( (ptr == NULL) && (FTI_Exec->stageInfo->nbRequest > 1) ) {
                FTI_Print( "failed to allocate memory for 'ptr' in 'FTI_FreeStageRequest'", FTI_EROR );
                return FTI_NSCS;
            }
            free( buf_ptr );
            --FTI_Exec->stageInfo->nbRequest;
            FTI_Exec->stageInfo->request = (FTI_Exec->stageInfo->nbRequest == 0) ? NULL : (void*)ptr;
        } 
        // if not last element, we need to truncate array and move elements (before truncation)
        else {
            void *dest = &ptr[idx];
            void *dest_cpy = malloc( type_size );
            if ( dest_cpy == NULL ) {
                FTI_Print( "failed to allocate memory for 'dest_cpy' in 'FTI_FreeStageRequest'", FTI_EROR );
                return FTI_NSCS;
            }
            memcpy( dest_cpy, dest, type_size );
            void *src = &ptr[idx+1];
            size_t mem_size = (FTI_Exec->stageInfo->nbRequest - (idx+1)) * type_size; 
            memmove( dest, src, mem_size );
            ptr = realloc( ptr, type_size * (FTI_Exec->stageInfo->nbRequest-1) );
            if ( ptr == NULL ) {
                FTI_Print( "failed to allocate memory for 'ptr' in 'FTI_FreeStageRequest'", FTI_EROR );
                memmove( src, dest, mem_size );
                memcpy( dest, dest_cpy, type_size );
                free( dest_cpy);
                return FTI_NSCS;
            }
            free( dest_cpy );
            FTI_Exec->stageInfo->request = (void*)ptr;
            --FTI_Exec->stageInfo->nbRequest;
            // re-assign the correct values
            int i = idx;
            for ( ; i<FTI_Exec->stageInfo->nbRequest; ++i ) {
                int ID_tmp = ptr[i].ID;
                FTI_SetRequestField( ID_tmp, i, FTI_SIF_IDX );
            }
        }

        FTI_SetRequestField( ID, FTI_SI_NALL, FTI_SIF_ALL );
    
    } else {
        
        int nbRequest = FTI_Exec->stageInfo[source-1].nbRequest;
        FTIT_StageHeadInfo *ptr = FTI_SI_HPTR(FTI_Exec->stageInfo[source-1].request);
        int idx;
        // locate idx, heads do not have a look-up table
        for( idx=0; idx<nbRequest; ++idx ) {
            if(ptr->ID == ID) {
                break;
            }
        }
        if ( idx == nbRequest ) {
            FTI_Print("invalid ID! Failed to free stage request meta info (FTI head process).", FTI_WARN);
            return FTI_NSCS;
        }
        
        size_t type_size = sizeof(FTIT_StageHeadInfo);
        
        // if last element in array, we just need to truncate the array.
        if ( idx == (nbRequest-1) ) {
            ptr = realloc( ptr, type_size * (nbRequest-1) );
            if ( (ptr == NULL) && (nbRequest > 1) ) {
                FTI_Print( "failed to allocate memory for 'ptr' in 'FTI_FreeStageRequest'", FTI_EROR );
                return FTI_NSCS;
            }
            --FTI_Exec->stageInfo[source-1].nbRequest;
            FTI_Exec->stageInfo[source-1].request = (FTI_Exec->stageInfo[source-1].nbRequest == 0) ? NULL : (void*)ptr;
        } 
        
        // if not last element, we need to truncate array and move elements (before truncation)
        else {
            void *dest = &(ptr[idx]);
            void *dest_cpy = malloc( type_size );
            if ( dest_cpy == NULL ) {
                FTI_Print( "failed to allocate memory for 'dest_cpy' in 'FTI_FreeStageRequest'", FTI_EROR );
                return FTI_NSCS;
            }
            memcpy( dest_cpy, dest, type_size );
            void *src = &(ptr[idx+1]);
            size_t mem_size = (nbRequest - (idx+1)) * type_size; 
            memmove( dest, src, mem_size );
            ptr = realloc( ptr, type_size * (nbRequest-1) );
            if ( ptr == NULL ) {
                FTI_Print( "failed to allocate memory for 'ptr' in 'FTI_FreeStageRequest'", FTI_EROR );
                memmove( src, dest, mem_size );
                memcpy( dest, dest_cpy, type_size );
                free( dest_cpy);
                return FTI_NSCS;
            }
            free( dest_cpy);
            FTI_Exec->stageInfo[source-1].request = ptr;
            --FTI_Exec->stageInfo[source-1].nbRequest;
        }

    }

    return FTI_SCES;

}

int FTI_SyncStage( char* lpath, char *rpath, FTIT_execution *FTI_Exec, FTIT_topology *FTI_Topo, FTIT_configuration *FTI_Conf, uint32_t ID ) 
{
    
    if ( !FTI_SI_ENABLED ) {
        FTI_Print( "Staging disabled, invalid call to 'FTI_SyncStage'", FTI_WARN );
        return FTI_NSCS;
    }

    char errstr[FTI_BUFS];
    
    int source = FTI_Topo->nodeRank;

    // for consistency
    FTI_SetStatusField( FTI_Exec, FTI_Topo, ID, FTI_SI_ACTV, FTI_SIF_VAL, source );

    // set buffer size for the file data transfer
    size_t bs = FTI_Conf->transferSize;

    // check local file and get file size
    struct stat st;
    if(  stat( lpath, &st ) == -1 ) {
        FTI_SetStatusField( FTI_Exec, FTI_Topo, ID, FTI_SI_FAIL, FTI_SIF_VAL, source );
        snprintf( errstr, FTI_BUFS, "Could not stat the local file ('%s') for staging", lpath );
        FTI_Print( errstr, FTI_EROR );
        return FTI_NSCS;
    }
    if( !S_ISREG(st.st_mode) ) {
        FTI_SetStatusField( FTI_Exec, FTI_Topo, ID, FTI_SI_FAIL, FTI_SIF_VAL, source );
        snprintf( errstr, FTI_BUFS, "'%s' is not a regular file, staging failed.", lpath );
        FTI_Print( errstr, FTI_EROR );
        errno = 0;
        return FTI_NSCS;
    }

    off_t eof = st.st_size;

    // open local file
    int fd_local = open( lpath, O_RDONLY );
    if( fd_local == -1 ) {
        FTI_SetStatusField( FTI_Exec, FTI_Topo, ID, FTI_SI_FAIL, FTI_SIF_VAL, source );
        FTI_Print("Could not open the local file for staging", FTI_EROR);
        return FTI_NSCS;
    }
    // open file on remote fs
    int fd_global = open( rpath, O_WRONLY|O_CREAT, (mode_t) 0600 );
    if( fd_global == -1 ) {
        FTI_SetStatusField( FTI_Exec, FTI_Topo, ID, FTI_SI_FAIL, FTI_SIF_VAL, source );
        FTI_Print("Could not open the destination file for staging", FTI_EROR);
        close( fd_local );
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
            return FTI_NSCS;
        }
        if( (read_bytes = read( fd_local, buf, buf_bytes )) == -1 ) {
            FTI_SetStatusField( FTI_Exec, FTI_Topo, ID, FTI_SI_FAIL, FTI_SIF_VAL, source );
            snprintf( errstr, FTI_BUFS, "unable to read from '%s'.", lpath );
            FTI_Print( errstr, FTI_EROR );
            errno = 0;
            free( buf );
            close( fd_local );
            close( fd_global );
            return FTI_NSCS;
        }
        if( (write_bytes = write( fd_global, buf, read_bytes )) == -1 ) {  
            FTI_SetStatusField( FTI_Exec, FTI_Topo, ID, FTI_SI_FAIL, FTI_SIF_VAL, source );
            snprintf( errstr, FTI_BUFS, "unable to write to '%s'.", rpath );
            FTI_Print( errstr, FTI_EROR );
            errno = 0;
            free( buf );
            close( fd_local );
            close( fd_global );
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
        return FTI_NSCS;
    }
    
    FTI_SetStatusField( FTI_Exec, FTI_Topo, ID, FTI_SI_SCES, FTI_SIF_VAL, source );
    
    return FTI_SCES;

}
int FTI_AsyncStage( char *lpath, char *rpath, FTIT_configuration *FTI_Conf, 
        FTIT_execution *FTI_Exec, FTIT_topology *FTI_Topo, int ID ) {
    
    if ( !FTI_SI_ENABLED ) {
        FTI_Print( "Staging disabled, invalid call to 'FTI_AsyncStage'", FTI_WARN );
        return FTI_NSCS;
    }

    int idx = FTI_GetRequestField( ID, FTI_SIF_IDX );
    
    // serialize request before sending to the head
    void *buf_ser = malloc ( 2*FTI_BUFS + sizeof(int) );
    if ( buf_ser == NULL ) {
        FTI_Print("failed to allocate memory for 'buf_ser' in FTI_AsyncStage'", FTI_EROR );
        return FTI_NSCS;
    }
    int pos = 0;
    memcpy( buf_ser, lpath, FTI_BUFS );   
    pos += FTI_BUFS;
    memcpy( buf_ser + pos, rpath, FTI_BUFS );   
    pos += FTI_BUFS;
    memcpy( buf_ser + pos, &ID, sizeof(int) );   
    pos += sizeof(int);
    MPI_Datatype buf_t;
    MPI_Type_contiguous( pos, MPI_BYTE, &buf_t );
    MPI_Type_commit( &buf_t );
   
    // send request to head
    MPI_Request *mpiReq = &(FTI_SI_APTR(FTI_Exec->stageInfo->request)[idx].mpiReq);
    MPI_Isend( buf_ser, 1, buf_t, FTI_Topo->headRankNode, FTI_Conf->stageTag, FTI_Exec->nodeComm, mpiReq );
    
    // keep send buffer until it may be freed (MPI_Test check in 'FTI_GetStageStatus') 
    FTI_SI_APTR(FTI_Exec->stageInfo->request)[idx].sendBuf = buf_ser;
    MPI_Type_free( &buf_t );

    return FTI_SCES;
}

int FTI_HandleStageRequest(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt, int source)
{      

    if ( !FTI_SI_ENABLED ) {
        FTI_Print( "Staging disabled, invalid call to 'FTI_HandleStageRequest'", FTI_WARN );
        return FTI_NSCS;
    }

    char errstr[FTI_BUFS];
 
    size_t buf_ser_size = 2*FTI_BUFS + sizeof(int);
    void *buf_ser = malloc ( buf_ser_size );
    if ( buf_ser == NULL ) {
        FTI_Print( "failed to allocate memory for 'buf_ser' in FTI_HandleStageRequest", FTI_EROR );
        return FTI_NSCS; 
    }
    MPI_Datatype buf_t;
    MPI_Type_contiguous( buf_ser_size, MPI_CHAR, &buf_t );
    MPI_Type_commit( &buf_t );
    MPI_Recv( buf_ser, 1, buf_t, source, FTI_Conf->stageTag, FTI_Exec->nodeComm, MPI_STATUS_IGNORE );
  
    // set local file path
    char lpath[FTI_BUFS];
    char rpath[FTI_BUFS];

    strncpy( lpath, buf_ser, FTI_BUFS );
    strncpy( rpath, buf_ser+FTI_BUFS, FTI_BUFS );
    lpath[FTI_BUFS-1] = '\0';
    rpath[FTI_BUFS-1] = '\0';
    int ID = *(int*)(buf_ser+2*FTI_BUFS);
    
    free( buf_ser );
    MPI_Type_free( &buf_t );

    // init Head staging meta data
    if ( FTI_InitStageRequestHead( lpath, rpath, FTI_Exec, FTI_Topo, source, ID ) != FTI_SCES ) {
        FTI_Print( "failed to initialize stage request meta info!", FTI_WARN );
        FTI_SetStatusField( FTI_Exec, FTI_Topo, ID, FTI_SI_FAIL, FTI_SIF_VAL, source );
        return FTI_NSCS;
    }
    
    // set buffer size for the file data transfer
    size_t bs = FTI_Conf->transferSize;

    // check local file and get file size
    struct stat st;
    if(  stat( lpath, &st ) == -1 ) {
        FTI_FreeStageRequest( FTI_Exec, FTI_Topo, ID, source );
        FTI_SetStatusField( FTI_Exec, FTI_Topo, ID, FTI_SI_FAIL, FTI_SIF_VAL, source );
        snprintf( errstr, FTI_BUFS, "Could not stat the local file ('%s') for staging.", lpath );
        FTI_Print( errstr, FTI_EROR );
        return FTI_NSCS;
    }
    if( !S_ISREG(st.st_mode) ) {
        FTI_FreeStageRequest( FTI_Exec, FTI_Topo, ID, source );
        FTI_SetStatusField( FTI_Exec, FTI_Topo, ID, FTI_SI_FAIL, FTI_SIF_VAL, source );
        snprintf( errstr, FTI_BUFS, "'%s' is not a regular file, staging failed.", lpath );
        FTI_Print( errstr, FTI_EROR );
        return FTI_NSCS;
    }

    off_t eof = st.st_size;

    // open local file
    int fd_local = open( lpath, O_RDONLY );
    if( fd_local == -1 ) {
        FTI_FreeStageRequest( FTI_Exec, FTI_Topo, ID, source );
        FTI_SetStatusField( FTI_Exec, FTI_Topo, ID, FTI_SI_FAIL, FTI_SIF_VAL, source );
        FTI_Print("Could not open the local file for staging", FTI_EROR);
        return FTI_NSCS;
    }
    // open file on remote fs
    int fd_global = open( rpath, O_WRONLY|O_CREAT, (mode_t) 0600 );
    if( fd_global == -1 ) {
        FTI_FreeStageRequest( FTI_Exec, FTI_Topo, ID, source );
        FTI_SetStatusField( FTI_Exec, FTI_Topo, ID, FTI_SI_FAIL, FTI_SIF_VAL, source );
        FTI_Print("Could not open the destination file for staging", FTI_EROR);
        close( fd_local );
        return FTI_NSCS;
    }
    // allocate buffer
    char *buf = (char*) malloc( bs );
    if ( buf == NULL ) {
        close ( fd_local );
        close ( fd_global );
        FTI_FreeStageRequest( FTI_Exec, FTI_Topo, ID, source );
        FTI_SetStatusField( FTI_Exec, FTI_Topo, ID, FTI_SI_FAIL, FTI_SIF_VAL, source );
        FTI_Print( "failed to allocate memory for 'buf' in 'FTI_HandleStageRequest'", FTI_EROR );
        return FTI_NSCS;
    }

    // move file to destination
    off_t pos = 0;
    ssize_t read_bytes, write_bytes;
    size_t buf_bytes;
    while( pos < eof ) {
        buf_bytes = ( (eof - pos) < bs ) ? eof - pos : bs;
        // for the case we have written less then we have read
        if( lseek( fd_local, pos, SEEK_SET ) == -1 ) {
            FTI_FreeStageRequest( FTI_Exec, FTI_Topo, ID, source );
            FTI_SetStatusField( FTI_Exec, FTI_Topo, ID, FTI_SI_FAIL, FTI_SIF_VAL, source );
            snprintf( errstr, FTI_BUFS, "unable to seek in '%s'.", lpath );
            FTI_Print( errstr, FTI_EROR );
            errno = 0;
            free( buf );
            close( fd_local );
            close( fd_global );
            return FTI_NSCS;
        }
        if( (read_bytes = read( fd_local, buf, buf_bytes )) == -1 ) {
            FTI_FreeStageRequest( FTI_Exec, FTI_Topo, ID, source );
            FTI_SetStatusField( FTI_Exec, FTI_Topo, ID, FTI_SI_FAIL, FTI_SIF_VAL, source );
            snprintf( errstr, FTI_BUFS, "unable to read from '%s'.", lpath );
            FTI_Print( errstr, FTI_EROR );
            errno = 0;
            free( buf );
            close( fd_local );
            close( fd_global );
            return FTI_NSCS;
        }
        if( (write_bytes = write( fd_global, buf, read_bytes )) == -1 ) {  
            FTI_FreeStageRequest( FTI_Exec, FTI_Topo, ID, source );
            FTI_SetStatusField( FTI_Exec, FTI_Topo, ID, FTI_SI_FAIL, FTI_SIF_VAL, source );
            snprintf( errstr, FTI_BUFS, "unable to write to '%s'.", rpath );
            FTI_Print( errstr, FTI_EROR );
            errno = 0;
            free( buf );
            close( fd_local );
            close( fd_global );
            return FTI_NSCS;
        }
        pos += write_bytes;
    }

    // deallocate buffer and close file descriptors
    free( buf );
    close( fd_local );
    fsync( fd_global );
    close( fd_global );
 
    FTI_SetStatusField( FTI_Exec, FTI_Topo, ID, FTI_SI_SCES, FTI_SIF_VAL, source );
    FTI_FreeStageRequest( FTI_Exec, FTI_Topo, ID, source );

    return FTI_SCES;
}

int FTI_GetRequestField( int ID, FTIT_StatusField val ) 
{

    if ( !FTI_SI_ENABLED ) {
        FTI_Print( "Staging disabled, invalid call to 'FTI_GetRequestField'", FTI_WARN );
        return FTI_NSCS;
    }

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

    if ( !FTI_SI_ENABLED ) {
        FTI_Print( "Staging disabled, invalid call to 'FTI_SetRequestField'", FTI_WARN );
        return FTI_NSCS;
    }

    if( (val < FTI_SIF_ALL) || (val > FTI_SIF_IDX) ) {
        FTI_Print( "invalid argument for 'FTI_SetRequestIdxField'", FTI_WARN );
        return FTI_NSCS;
    }

    const uint32_t all_mask = 0x00080000;
    const uint32_t idx_mask = 0x0007FFFF; // 524,288 ID's, max ID: 524,287

    bool err = false;

    uint32_t field = idxRequest[ID];

    switch( val ) {

        case FTI_SIF_ALL:
            if ( ( entry > 0x1 ) ) { 
                FTI_Print( "invalid argument for 'FTI_SetRequestIdxField'", FTI_WARN );
                err = true;
                break;
            }
            field = (entry << 19) | ((~all_mask) & field);
            break;
        case FTI_SIF_IDX:
            if ( entry > idx_mask ) {
                FTI_Print( "invalid argument for 'FTI_SetRequestIdxField'", FTI_WARN );
                err = true;
                break;
            }
            field = entry | ((~idx_mask) & field);
            break;

    }
    
    if ( !err ) {
        idxRequest[ID] = field;
        return entry;
    } else {
        return FTI_NSCS;
    }
    
}
    
int FTI_GetStatusField( FTIT_execution *FTI_Exec, FTIT_topology *FTI_Topo, int ID, FTIT_StatusField val, int source ) 
{   
    
    if ( !FTI_SI_ENABLED ) {
        FTI_Print( "Staging disabled, invalid call to 'FTI_GetStatusField'", FTI_WARN );
        return FTI_NSCS;
    }

    if( (val < 0) || (val > FTI_SIF_VAL) ) {
        FTI_Print( "invalid argument for 'FTI_GetStatusField'", FTI_WARN );
        return FTI_NSCS;
    }

    const uint8_t val_mask = 0xE;
    const uint8_t avl_mask = 0x1;

    int disp;
    MPI_Aint size;
    MPI_Win_shared_query( stageWin, source, &size, &disp, &(status) ); 
    uint8_t status_cpy = status[ID];

    switch( val ) {

        case FTI_SIF_VAL:
            return ((int)(status_cpy & val_mask)) >> 1;
        case FTI_SIF_AVL:
            return ((int)(status_cpy & avl_mask));

    }

}

int FTI_SetStatusField( FTIT_execution *FTI_Exec, FTIT_topology *FTI_Topo, int ID, uint8_t entry, FTIT_StatusField val, int source )
{

    if ( !FTI_SI_ENABLED ) {
        FTI_Print( "Staging disabled, invalid call to 'FTI_SetStatusField'", FTI_WARN );
        return FTI_NSCS;
    }

    if ( (val < 0) || (val > FTI_SIF_VAL) ) {
        FTI_Print( "invalid argument for 'FTI_GetStatusField'", FTI_WARN );
        return FTI_NSCS;
    }

    const uint8_t val_mask = 0xE; // in bits: 1110
    const uint8_t avl_mask = 0x1; // in bits: 0001
    
    int ierr = FTI_SCES;

    int disp;
    MPI_Aint size;
    MPI_Win_shared_query( stageWin, source, &size, &disp, &(status) ); 
    uint8_t status_cpy = status[ID];
    
    switch( val ) {

        case FTI_SIF_VAL:
            if ( (entry > 0x4) ) { 
                FTI_Print( "invalid argument for 'FTI_SetStatusField'", FTI_WARN );
                ierr = FTI_NSCS;
                break;
            }
            status_cpy = (entry << 1) | ((~val_mask) & status_cpy);
            break;
        case FTI_SIF_AVL:
            if ( entry > 0x1 ) {
                FTI_Print( "invalid argument for 'FTI_SetStatusField'", FTI_WARN );
                ierr = FTI_NSCS;
                break;
            }
            status_cpy = entry | ((~avl_mask) & status_cpy);
            break;

    }
    
    if ( ierr == FTI_SCES ) {
        status[ID] = status_cpy;
    }
    
    return ierr;

}

// stage request counter returns -1 if too many requests.
int FTI_GetRequestID( FTIT_execution *FTI_Exec, FTIT_topology *FTI_Topo ) 
{

    if ( !FTI_SI_ENABLED ) {
        FTI_Print( "Staging disabled, invalid call to 'FTI_GetRequestID'", FTI_WARN );
        return FTI_NSCS;
    }

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

// FOR DEBUGGING
void FTI_PrintStageStatus( FTIT_execution *FTI_Exec, FTIT_topology *FTI_Topo, int ID, int source ) 
{
    
    if ( !FTI_SI_ENABLED ) {
        FTI_Print( "Staging disabled, invalid call to 'FTI_PrintStageStatus'", FTI_WARN );
        return;
    }

    MPI_Aint qsize;
    int qdisp;
    MPI_Win_shared_query( stageWin, source, &qsize, &qdisp, &(status) );

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
        snprintf(idxstr, FTI_BUFS, "%u", (uint32_t)val );
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

