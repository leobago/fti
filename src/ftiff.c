#include "interface.h"

// allocate static memory
FTIFF_MPITypeInfo FTIFF_MPITypes[FTIFF_NUM_MPI_TYPES];

int FTIFF_Recover( FTIT_execution *FTI_Exec, FTIT_dataset *FTI_Data, FTIT_checkpoint *FTI_Ckpt ) 
{
    if (FTI_Exec->initSCES == 0) {
        FTI_Print("FTI is not initialized.", FTI_WARN);
        return FTI_NSCS;
    }
    if (FTI_Exec->initSCES == 2) {
        FTI_Print("No checkpoint files to make recovery.", FTI_WARN);
        return FTI_NSCS;
    }

    char fn[FTI_BUFS]; //Path to the checkpoint file
    char str[FTI_BUFS]; //For console output

    //Check if nubmer of protected variables matches
    if (FTI_Exec->nbVar != FTI_Exec->meta[FTI_Exec->ckptLvel].nbVar[0]) {
        sprintf(str, "Checkpoint has %d protected variables, but FTI protects %d.",
                FTI_Exec->meta[FTI_Exec->ckptLvel].nbVar[0], FTI_Exec->nbVar);
        FTI_Print(str, FTI_WARN);
        return FTI_NREC;
    }
    //Check if sizes of protected variables matches
    int i;
    for (i = 0; i < FTI_Exec->nbVar; i++) {
        if (FTI_Data[i].size != FTI_Exec->meta[FTI_Exec->ckptLvel].varSize[i]) {
            sprintf(str, "Cannot recover %ld bytes to protected variable (ID %d) size: %ld",
                    FTI_Exec->meta[FTI_Exec->ckptLvel].varSize[i], FTI_Exec->meta[FTI_Exec->ckptLvel].varID[i],
                    FTI_Data[i].size);
            FTI_Print(str, FTI_WARN);
            return FTI_NREC;
        }
    }
    //Recovering from local for L4 case in FTI_Recover
    if (FTI_Exec->ckptLvel == 4) {
        sprintf(fn, "%s/%s", FTI_Ckpt[1].dir, FTI_Exec->meta[1].ckptFile);
    }
    else {
        sprintf(fn, "%s/%s", FTI_Ckpt[FTI_Exec->ckptLvel].dir, FTI_Exec->meta[FTI_Exec->ckptLvel].ckptFile);
    }
    // get filesize
    struct stat st;
    stat(fn, &st);
    int ferr;
    char strerr[FTI_BUFS];

    // block size for memcpy of pointer.
    long membs = 1024*1024*16; // 16 MB
    long cpybuf, cpynow, cpycnt;

    if (!FTI_Exec->firstdb) {
        FTI_Print( "FTIFF: RecoveryGlobal - No db meta information. Nothing to recover.", FTI_WARN );
        return FTI_NREC;
    }

    // open checkpoint file for read only
    int fd = open( fn, O_RDONLY, 0 );
    if (fd == -1) {
        sprintf( strerr, "FTIFF: RecoveryGlobal - could not open '%s' for reading.", fn);
        FTI_Print(strerr, FTI_EROR);
        return FTI_NREC;
    }

    // map file into memory
    char* fmmap = (char*) mmap(0, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (fmmap == MAP_FAILED) {
        sprintf( strerr, "FTIFF: RecoveryGlobal - could not map '%s' to memory.", fn);
        FTI_Print(strerr, FTI_EROR);
        close(fd);
        return FTI_NREC;
    }

    // file is mapped, we can close it.
    close(fd);

    FTIFF_db *currentdb, *nextdb;
    FTIFF_dbvar *currentdbvar = NULL;
    char *destptr, *srcptr;
    int dbvar_idx, pvar_idx, dbcounter=0;

    // MD5 context for checksum of data chunks
    MD5_CTX mdContext;
    unsigned char hash[MD5_DIGEST_LENGTH];

    int isnextdb;

    currentdb = FTI_Exec->firstdb;

    do {

        isnextdb = 0;

        for(dbvar_idx=0;dbvar_idx<currentdb->numvars;dbvar_idx++) {

            currentdbvar = &(currentdb->dbvars[dbvar_idx]);

            // get source and destination pointer
            destptr = (char*) FTI_Data[currentdbvar->idx].ptr + currentdbvar->dptr;
            srcptr = (char*) fmmap + currentdbvar->fptr;

            MD5_Init( &mdContext );
            cpycnt = 0;
            while ( cpycnt < currentdbvar->chunksize ) {
                cpybuf = currentdbvar->chunksize - cpycnt;
                cpynow = ( cpybuf > membs ) ? membs : cpybuf;
                cpycnt += cpynow;
                memcpy( destptr, srcptr, cpynow );
                MD5_Update( &mdContext, destptr, cpynow );
                destptr += cpynow;
                srcptr += cpynow;
            }

            // debug information
            sprintf(str, "FTIFF: RecoveryGlobal -  dataBlock:%i/dataBlockVar%i id: %i, idx: %i"
                    ", destptr: %ld, fptr: %ld, chunksize: %ld, "
                    "base_ptr: 0x%" PRIxPTR " ptr_pos: 0x%" PRIxPTR ".", 
                    dbcounter, dbvar_idx,  
                    currentdbvar->id, currentdbvar->idx, currentdbvar->dptr,
                    currentdbvar->fptr, currentdbvar->chunksize,
                    FTI_Data[currentdbvar->idx].ptr, destptr);
            FTI_Print(str, FTI_DBUG);

            MD5_Final( hash, &mdContext );

            if ( memcmp( currentdbvar->hash, hash, MD5_DIGEST_LENGTH ) != 0 ) {
                sprintf( strerr, "FTIFF: RecoveryGlobal - dataset with id:%i has been corrupted! Discard recovery.", currentdbvar->id);
                FTI_Print(strerr, FTI_WARN);
                return FTI_NREC;
            }

        }

        if (currentdb->next) {
            currentdb = currentdb->next;
            isnextdb = 1;
        }

        dbcounter++;

    } while( isnextdb );

    // unmap memory
    if ( munmap( fmmap, st.st_size ) == -1 ) {
        FTI_Print("FTIFF: RecoveryGlobal - unable to unmap memory", FTI_WARN);
    }

    FTI_Exec->reco = 0;

    return FTI_SCES;
}

int FTIFF_RecoverVar( int id, FTIT_execution *FTI_Exec, FTIT_dataset *FTI_Data, FTIT_checkpoint *FTI_Ckpt )
{
    if (FTI_Exec->initSCES == 0) {
        FTI_Print("FTI is not initialized.", FTI_WARN);
        return FTI_NSCS;
    }

    if(FTI_Exec->reco==0){
        /* This is not a restart: no actions performed */
        return FTI_SCES;
    }

    if (FTI_Exec->initSCES == 2) {
        FTI_Print("No checkpoint files to make recovery.", FTI_WARN);
        return FTI_NSCS;
    }

    //Check if sizes of protected variables matches
    int i;
    for (i = 0; i < FTI_Exec->nbVar; i++) {
        if (id == FTI_Exec->meta[FTI_Exec->ckptLvel].varID[i]) {
            if (FTI_Data[i].size != FTI_Exec->meta[FTI_Exec->ckptLvel].varSize[i]) {
                char str[FTI_BUFS];
                sprintf(str, "Cannot recover %ld bytes to protected variable (ID %d) size: %ld",
                        FTI_Exec->meta[FTI_Exec->ckptLvel].varSize[i], FTI_Exec->meta[FTI_Exec->ckptLvel].varID[i],
                        FTI_Data[i].size);
                FTI_Print(str, FTI_WARN);
                return FTI_NREC;
            }
        }
    }

    char fn[FTI_BUFS]; //Path to the checkpoint file

    //Recovering from local for L4 case in FTI_Recover
    if (FTI_Exec->ckptLvel == 4) {
        sprintf(fn, "%s/%s", FTI_Ckpt[1].dir, FTI_Exec->meta[1].ckptFile);
    }
    else {
        sprintf(fn, "%s/%s", FTI_Ckpt[FTI_Exec->ckptLvel].dir, FTI_Exec->meta[FTI_Exec->ckptLvel].ckptFile);
    }

    char str[FTI_BUFS];

    sprintf(str, "Trying to load FTI checkpoint file (%s)...", fn);
    FTI_Print(str, FTI_DBUG);
    
    // get filesize
    struct stat st;
    stat(fn, &st);
    int ferr;
    char strerr[FTI_BUFS];

    // block size for memcpy of pointer.
    long membs = 1024*1024*16; // 16 MB
    long cpybuf, cpynow, cpycnt;

    if (!FTI_Exec->firstdb) {
        FTI_Print( "FTIFF: RecoveryLocal - No db meta information. Nothing to recover.", FTI_WARN );
        return FTI_NREC;
    }

    // open checkpoint file for read only
    int fd = open( fn, O_RDONLY, 0 );
    if (fd == -1) {
        sprintf( strerr, "FTIFF: RecoveryLocal - could not open '%s' for reading.", fn);
        FTI_Print(strerr, FTI_EROR);
        return FTI_NREC;
    }

    // map file into memory
    char* fmmap = (char*) mmap(0, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (fmmap == MAP_FAILED) {
        sprintf( strerr, "FTIFF: RecoveryLocal - could not map '%s' to memory.", fn);
        FTI_Print(strerr, FTI_EROR);
        close(fd);
        return FTI_NREC;
    }

    // file is mapped, we can close it.
    close(fd);

    FTIFF_db *currentdb, *nextdb;
    FTIFF_dbvar *currentdbvar = NULL;
    char *destptr, *srcptr;
    int dbvar_idx, pvar_idx, dbcounter=0;

    // MD5 context for checksum of data chunks
    MD5_CTX mdContext;
    unsigned char hash[MD5_DIGEST_LENGTH];

    int isnextdb;

    currentdb = FTI_Exec->firstdb;

    do {

        isnextdb = 0;

        for(dbvar_idx=0;dbvar_idx<currentdb->numvars;dbvar_idx++) {

            currentdbvar = &(currentdb->dbvars[dbvar_idx]);

            if (currentdbvar->id == id) {
                // get source and destination pointer
                destptr = (char*) FTI_Data[currentdbvar->idx].ptr + currentdbvar->dptr;
                srcptr = (char*) fmmap + currentdbvar->fptr;

                MD5_Init( &mdContext );
                cpycnt = 0;
                while ( cpycnt < currentdbvar->chunksize ) {
                    cpybuf = currentdbvar->chunksize - cpycnt;
                    cpynow = ( cpybuf > membs ) ? membs : cpybuf;
                    cpycnt += cpynow;
                    memcpy( destptr, srcptr, cpynow );
                    MD5_Update( &mdContext, destptr, cpynow );
                    destptr += cpynow;
                    srcptr += cpynow;
                }

                // debug information
                sprintf(str, "FTIFF: RecoveryLocal -  dataBlock:%i/dataBlockVar%i id: %i, idx: %i"
                        ", destptr: %ld, fptr: %ld, chunksize: %ld, "
                        "base_ptr: 0x%" PRIxPTR " ptr_pos: 0x%" PRIxPTR ".", 
                        dbcounter, dbvar_idx,  
                        currentdbvar->id, currentdbvar->idx, currentdbvar->dptr,
                        currentdbvar->fptr, currentdbvar->chunksize,
                        FTI_Data[currentdbvar->idx].ptr, destptr);
                FTI_Print(str, FTI_DBUG);

                MD5_Final( hash, &mdContext );

                if ( memcmp( currentdbvar->hash, hash, MD5_DIGEST_LENGTH ) != 0 ) {
                    sprintf( strerr, "FTIFF: RecoveryLocal - dataset with id:%i has been corrupted! Discard recovery.", currentdbvar->id);
                    FTI_Print(strerr, FTI_WARN);
                    return FTI_NREC;
                }

            }

        }

        if (currentdb->next) {
            currentdb = currentdb->next;
            isnextdb = 1;
        }

        dbcounter++;

    } while( isnextdb );

    // unmap memory
    if ( munmap( fmmap, st.st_size ) == -1 ) {
        FTI_Print("FTIFF: RecoveryLocal - unable to unmap memory", FTI_WARN);
    }

    return FTI_SCES;
}

void FTIFF_FreeDbFTIFF(FTIFF_db* last)
{
    if (last) {
        FTIFF_db *current = last;
        FTIFF_db *previous;
        while( current ) {
            previous = current->previous;
            free(current->dbvars);
            free(current);
            current = previous;
        }
    }
}

int FTIFF_Checksum(FTIT_execution* FTI_Exec, FTIT_dataset* FTI_Data, char* checksum)
{       
    
    MD5_CTX mdContext;
    MD5_Init (&mdContext);
    FTIFF_db *currentdb = FTI_Exec->firstdb;
    FTIFF_dbvar *currentdbvar = NULL;
    char *dptr;
    int dbvar_idx, pvar_idx, dbcounter=0;

    int isnextdb;

    //MD5_Update (&mdContext, &(FTI_Exec->FTIFFMeta), sizeof(FTIFF_metaInfo));

    do {

        isnextdb = 0;

        MD5_Update (&mdContext, &(currentdb->numvars), sizeof(int));
        MD5_Update (&mdContext, &(currentdb->dbsize), sizeof(long));

        for(dbvar_idx=0;dbvar_idx<currentdb->numvars;dbvar_idx++) {

            currentdbvar = &(currentdb->dbvars[dbvar_idx]);
            MD5_Update (&mdContext, currentdbvar, sizeof(FTIFF_dbvar));

        }

        for(dbvar_idx=0;dbvar_idx<currentdb->numvars;dbvar_idx++) {

            currentdbvar = &(currentdb->dbvars[dbvar_idx]);
            dptr = (char*)(FTI_Data[currentdbvar->idx].ptr) + currentdb->dbvars[dbvar_idx].dptr;
            MD5_Update (&mdContext, dptr, currentdbvar->chunksize);

        }

        if (currentdb->next) {
            currentdb = currentdb->next;
            isnextdb = 1;
        }

        dbcounter++;

    } while( isnextdb );

    unsigned char hash[MD5_DIGEST_LENGTH];
    MD5_Final (hash, &mdContext);
    int ii = 0, i;
    for(i = 0; i < MD5_DIGEST_LENGTH; i++) {
        sprintf(&checksum[ii], "%02x", hash[i]);
        ii += 2;
    }
}

int FTIFF_InitMpiTypes() 
{
    
    // define MPI datatypes
    // headInfo
    DECL_MPI_TYPES( headInfo );
    MBR_CNT( headInfo ) =  6;
    MBR_BLK_LEN( headInfo ) = { 1, 1, FTI_BUFS, 1, 1, 1 };
    MBR_TYPES( headInfo ) = { MPI_INT, MPI_INT, MPI_CHAR, MPI_LONG, MPI_LONG, MPI_LONG };
    MBR_DISP( headInfo ) = {  
        offsetof( FTIFF_headInfo, exists), 
        offsetof( FTIFF_headInfo, nbVar), 
        offsetof( FTIFF_headInfo, ckptFile), 
        offsetof( FTIFF_headInfo, maxFs), 
        offsetof( FTIFF_headInfo, fs), 
        offsetof( FTIFF_headInfo, pfs) 
    };
    FTIFF_MPITypes[FTIFF_HEAD_INFO].mbrCnt = headInfo_mbrCnt;
    FTIFF_MPITypes[FTIFF_HEAD_INFO].mbrBlkLen = headInfo_mbrBlkLen;
    FTIFF_MPITypes[FTIFF_HEAD_INFO].mbrTypes = headInfo_mbrTypes;
    FTIFF_MPITypes[FTIFF_HEAD_INFO].mbrDisp = headInfo_mbrDisp;
   
    // commit MPI types
    int i;
    for(i=0; i<FTIFF_NUM_MPI_TYPES; i++) {
        MPI_Type_create_struct( 
                FTIFF_MPITypes[i].mbrCnt, 
                FTIFF_MPITypes[i].mbrBlkLen, 
                FTIFF_MPITypes[i].mbrDisp, 
                FTIFF_MPITypes[i].mbrTypes, 
                &FTIFF_MPITypes[i].raw );
        MPI_Type_get_extent( 
                FTIFF_MPITypes[i].raw, 
                &lb, 
                &extent );
        MPI_Type_create_resized( 
                FTIFF_MPITypes[i].raw, 
                lb, 
                extent, 
                &FTIFF_MPITypes[i].final);
        MPI_Type_commit( &FTIFF_MPITypes[i].final );
    }
 

}

/*-------------------------------------------------------------------------*/
/**
    @brief      updates datablock structure for FTI File Format.
    @param      FTI_Conf        Configuration metadata.
    @param      FTI_Exec        Execution metadata.
    @param      FTI_Topo        Topology metadata.
    @param      FTI_Ckpt        Checkpoint metadata.
    @param      FTI_Data        Dataset metadata.
    @return     integer         FTI_SCES if successful.
    
    Updates information about the checkpoint file. Updates file pointers
    in the dbvar structures and updates the db structure.

 **/
/*-------------------------------------------------------------------------*/
int FTIFF_UpdateDatastructFTIFF( FTIT_execution* FTI_Exec, 
      FTIT_dataset* FTI_Data )
{

    int dbvar_idx, pvar_idx, num_edit_pvars = 0;
    int *editflags = (int*) calloc( FTI_Exec->nbVar, sizeof(int) ); // 0 -> nothing changed, 1 -> new pvar, 2 -> size changed
    FTIFF_dbvar *dbvars = NULL;
    int isnextdb;
    long offset = sizeof(FTIFF_metaInfo), chunksize;
    long *FTI_Data_oldsize, dbsize;
    
    // first call, init first datablock
    if(!FTI_Exec->firstdb) { // init file info
        dbsize = FTI_dbstructsize + sizeof(FTIFF_dbvar) * FTI_Exec->nbVar;
        FTIFF_db *dblock = (FTIFF_db*) malloc( sizeof(FTIFF_db) );
        dbvars = (FTIFF_dbvar*) malloc( sizeof(FTIFF_dbvar) * FTI_Exec->nbVar );
        dblock->previous = NULL;
        dblock->next = NULL;
        dblock->numvars = FTI_Exec->nbVar;
        dblock->dbvars = dbvars;
        for(dbvar_idx=0;dbvar_idx<dblock->numvars;dbvar_idx++) {
            dbvars[dbvar_idx].fptr = offset + dbsize;
            dbvars[dbvar_idx].dptr = 0;
            dbvars[dbvar_idx].id = FTI_Data[dbvar_idx].id;
            dbvars[dbvar_idx].idx = dbvar_idx;
            dbvars[dbvar_idx].chunksize = FTI_Data[dbvar_idx].size;
            dbsize += dbvars[dbvar_idx].chunksize; 
        }
        FTI_Exec->nbVarStored = FTI_Exec->nbVar;
        dblock->dbsize = dbsize;
        
        // set as first datablock
        FTI_Exec->firstdb = dblock;
        FTI_Exec->lastdb = dblock;
    
    } else {
       
        /*
         *  - check if protected variable is in file info
         *  - check if size has changed
         */
        
        FTI_Data_oldsize = (long*) calloc( FTI_Exec->nbVarStored, sizeof(long) );
        FTI_Exec->lastdb = FTI_Exec->firstdb;
        
        // iterate though datablock list
        do {
            isnextdb = 0;
            for(dbvar_idx=0;dbvar_idx<FTI_Exec->lastdb->numvars;dbvar_idx++) {
                for(pvar_idx=0;pvar_idx<FTI_Exec->nbVarStored;pvar_idx++) {
                    if(FTI_Exec->lastdb->dbvars[dbvar_idx].id == FTI_Data[pvar_idx].id) {
                        chunksize = FTI_Exec->lastdb->dbvars[dbvar_idx].chunksize;
                        FTI_Data_oldsize[pvar_idx] += chunksize;
                    }
                }
            }
            offset += FTI_Exec->lastdb->dbsize;
            if (FTI_Exec->lastdb->next) {
                FTI_Exec->lastdb = FTI_Exec->lastdb->next;
                isnextdb = 1;
            }
        } while( isnextdb );

        // check for new protected variables
        for(pvar_idx=FTI_Exec->nbVarStored;pvar_idx<FTI_Exec->nbVar;pvar_idx++) {
            editflags[pvar_idx] = 1;
            num_edit_pvars++;
        }
        
        // check if size changed
        for(pvar_idx=0;pvar_idx<FTI_Exec->nbVarStored;pvar_idx++) {
            if(FTI_Data_oldsize[pvar_idx] != FTI_Data[pvar_idx].size) {
                editflags[pvar_idx] = 2;
                num_edit_pvars++;
            }
        }
                
        // if size changed or we have new variables to protect, create new block. 
        dbsize = FTI_dbstructsize + sizeof(FTIFF_dbvar) * num_edit_pvars;
       
        int evar_idx = 0;
        if( num_edit_pvars ) {
            for(pvar_idx=0; pvar_idx<FTI_Exec->nbVar; pvar_idx++) {
                switch(editflags[pvar_idx]) {

                    case 1:
                        // add new protected variable in next datablock
                        dbvars = (FTIFF_dbvar*) realloc( dbvars, (evar_idx+1) * sizeof(FTIFF_dbvar) );
                        dbvars[evar_idx].fptr = offset + dbsize;
                        dbvars[evar_idx].dptr = 0;
                        dbvars[evar_idx].id = FTI_Data[pvar_idx].id;
                        dbvars[evar_idx].idx = pvar_idx;
                        dbvars[evar_idx].chunksize = FTI_Data[pvar_idx].size;
                        dbsize += dbvars[evar_idx].chunksize; 
                        evar_idx++;

                        break;

                    case 2:
                        
                        // create data chunk info
                        dbvars = (FTIFF_dbvar*) realloc( dbvars, (evar_idx+1) * sizeof(FTIFF_dbvar) );
                        dbvars[evar_idx].fptr = offset + dbsize;
                        dbvars[evar_idx].dptr = FTI_Data_oldsize[pvar_idx];
                        dbvars[evar_idx].id = FTI_Data[pvar_idx].id;
                        dbvars[evar_idx].idx = pvar_idx;
                        dbvars[evar_idx].chunksize = FTI_Data[pvar_idx].size - FTI_Data_oldsize[pvar_idx];
                        dbsize += dbvars[evar_idx].chunksize; 
                        evar_idx++;

                        break;

                }

            }

            FTIFF_db  *dblock = (FTIFF_db*) malloc( sizeof(FTIFF_db) );
            FTI_Exec->lastdb->next = dblock;
            dblock->previous = FTI_Exec->lastdb;
            dblock->next = NULL;
            dblock->numvars = num_edit_pvars;
            dblock->dbsize = dbsize;
            dblock->dbvars = dbvars;
            FTI_Exec->lastdb = dblock;
        
        }

        FTI_Exec->nbVarStored = FTI_Exec->nbVar;
        
        free(FTI_Data_oldsize);
    
    }

    free(editflags);
    return FTI_SCES;

}

/*-------------------------------------------------------------------------*/
/**
    @brief      Reads datablock structure for FTI File Format from ckpt file.
    @param      FTI_Exec        Execution metadata.
    @return     integer         FTI_SCES if successful.
    
    Builds meta data list from checkpoint file for the FTI File Format

 **/
/*-------------------------------------------------------------------------*/
int FTIFF_ReadDbFTIFF( FTIT_execution *FTI_Exec, FTIT_checkpoint* FTI_Ckpt ) 
{
    char fn[FTI_BUFS]; //Path to the checkpoint file
    char str[FTI_BUFS]; //For console output
	
    int varCnt = 0;

    //Recovering from local for L4 case in FTI_Recover
    if (FTI_Exec->ckptLvel == 4) {
        sprintf(fn, "%s/%s", FTI_Ckpt[1].dir, FTI_Exec->meta[1].ckptFile);
    }
    else {
        sprintf(fn, "%s/%s", FTI_Ckpt[FTI_Exec->ckptLvel].dir, FTI_Exec->meta[FTI_Exec->ckptLvel].ckptFile);
    }

    // get filesize
	struct stat st;
	stat(fn, &st);
	int ferr;
	char strerr[FTI_BUFS];

	// open checkpoint file for read only
	int fd = open( fn, O_RDONLY, 0 );
	if (fd == -1) {
		sprintf( strerr, "FTIFF: Updatedb - could not open '%s' for reading.", fn);
		FTI_Print(strerr, FTI_EROR);
		return FTI_NREC;
	}

	// map file into memory
	char* fmmap = (char*) mmap(0, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
	if (fmmap == MAP_FAILED) {
		sprintf( strerr, "FTIFF: Updatedb - could not map '%s' to memory.", fn);
		FTI_Print(strerr, FTI_EROR);
		close(fd);
		return FTI_NREC;
	}

	// file is mapped, we can close it.
	close(fd);

    // get file meta info
    memcpy( &(FTI_Exec->FTIFFMeta), fmmap, sizeof(FTIFF_metaInfo) );
	
    FTIFF_db *currentdb, *nextdb;
	FTIFF_dbvar *currentdbvar = NULL;
	int dbvar_idx, pvar_idx, dbcounter=0;

	long endoffile = sizeof(FTIFF_metaInfo); // space for timestamp 
    long mdoffset;

	int isnextdb;
	
	currentdb = (FTIFF_db*) malloc( sizeof(FTIFF_db) );
	if (!currentdb) {
		sprintf( strerr, "FTIFF: Updatedb - failed to allocate %ld bytes for 'currentdb'", sizeof(FTIFF_db));
		FTI_Print(strerr, FTI_EROR);
		return FTI_NREC;
	}

	FTI_Exec->firstdb = currentdb;
	FTI_Exec->firstdb->next = NULL;
	FTI_Exec->firstdb->previous = NULL;

	do {

		nextdb = (FTIFF_db*) malloc( sizeof(FTIFF_db) );
		if (!currentdb) {
			sprintf( strerr, "FTIFF: Updatedb - failed to allocate %ld bytes for 'nextdb'", sizeof(FTIFF_db));
			FTI_Print(strerr, FTI_EROR);
			return FTI_NREC;
		}

		isnextdb = 0;

		mdoffset = endoffile;

		memcpy( &(currentdb->numvars), fmmap+mdoffset, sizeof(int) ); 
		mdoffset += sizeof(int);
		memcpy( &(currentdb->dbsize), fmmap+mdoffset, sizeof(long) );
		mdoffset += sizeof(long);

		sprintf(str, "FTIFF: Updatedb - dataBlock:%i, dbsize: %ld, numvars: %i.", 
				dbcounter, currentdb->dbsize, currentdb->numvars);
		FTI_Print(str, FTI_DBUG);

		currentdb->dbvars = (FTIFF_dbvar*) malloc( sizeof(FTIFF_dbvar) * currentdb->numvars );
		if (!currentdb) {
			sprintf( strerr, "FTIFF: Updatedb - failed to allocate %ld bytes for 'currentdb->dbvars'", sizeof(FTIFF_dbvar));
			FTI_Print(strerr, FTI_EROR);
			return FTI_NREC;
		}

		for(dbvar_idx=0;dbvar_idx<currentdb->numvars;dbvar_idx++) {

			currentdbvar = &(currentdb->dbvars[dbvar_idx]);

			memcpy( currentdbvar, fmmap+mdoffset, sizeof(FTIFF_dbvar) );
			mdoffset += sizeof(FTIFF_dbvar);
            
            if ( varCnt == 0 ) { 
                varCnt++;
                FTI_Exec->meta[FTI_Exec->ckptLvel].varID[0] = currentdbvar->id;
                FTI_Exec->meta[FTI_Exec->ckptLvel].varSize[0] = currentdbvar->chunksize;
            } else {
                int i;
                for(i=0; i<varCnt; i++) {
                    if ( FTI_Exec->meta[FTI_Exec->ckptLvel].varID[i] == currentdbvar->id ) {
                        FTI_Exec->meta[FTI_Exec->ckptLvel].varSize[i] += currentdbvar->chunksize;
                        break;
                    }
                }
                if( i == varCnt ) {
                    varCnt++;
                    FTI_Exec->meta[FTI_Exec->ckptLvel].varID[varCnt-1] = currentdbvar->id;
                    FTI_Exec->meta[FTI_Exec->ckptLvel].varSize[varCnt-1] = currentdbvar->chunksize;
                }
            }

			// debug information
			sprintf(str, "FTIFF: Updatedb -  dataBlock:%i/dataBlockVar%i id: %i, idx: %i"
					", destptr: %ld, fptr: %ld, chunksize: %ld.",
					dbcounter, dbvar_idx,  
					currentdbvar->id, currentdbvar->idx, currentdbvar->dptr,
					currentdbvar->fptr, currentdbvar->chunksize);
			FTI_Print(str, FTI_DBUG);

		}

		endoffile += currentdb->dbsize;

		if ( endoffile < FTI_Exec->FTIFFMeta.ckptSize ) {
			memcpy( nextdb, fmmap+endoffile, FTI_dbstructsize );
			currentdb->next = nextdb;
			nextdb->previous = currentdb;
			currentdb = nextdb;
			isnextdb = 1;
		}

		dbcounter++;

	} while( isnextdb );

    FTI_Exec->meta[FTI_Exec->ckptLvel].nbVar[0] = varCnt;
	
    FTI_Exec->lastdb = currentdb;
	FTI_Exec->lastdb->next = NULL;

	// unmap memory
	if ( munmap( fmmap, st.st_size ) == -1 ) {
		FTI_Print("FTIFF: Updatedb - unable to unmap memory", FTI_WARN);
	}

}

/*-------------------------------------------------------------------------*/
/**
    @brief      Writes ckpt to local/PFS using FTIFF.
    @param      FTI_Conf        Configuration metadata.
    @param      FTI_Exec        Execution metadata.
    @param      FTI_Topo        Topology metadata.
    @param      FTI_Ckpt        Checkpoint metadata.
    @param      FTI_Data        Dataset metadata.
    @return     integer         FTI_SCES if successful.

    FTI-FF structure:
    =================
    
    +------+---------+-------------+       +------+---------+-------------+
    |      |         |             |       |      |         |             |
    | db 1 | dbvar 1 | ckpt data 1 | . . . | db n | dbvar n | ckpt data n |
    |      |         |             |       |      |         |             |
    +------+---------+-------------+       +------+---------+-------------+

**/
/*-------------------------------------------------------------------------*/
int FTIFF_WriteFTIFF(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt,
        FTIT_dataset* FTI_Data)
{ 
    FTI_Print("I/O mode: FTI File Format.", FTI_DBUG);
    
    // Update the meta data information -> FTIT_db and FTIT_dbvar
    FTIFF_UpdateDatastructFTIFF( FTI_Exec, FTI_Data );

    // check if metadata exists
    if(!FTI_Exec->firstdb) {
        FTI_Print("No data structure found to write data to file. Discarding checkpoint.", FTI_WARN);
        return FTI_NSCS;
    }

    char str[FTI_BUFS], fn[FTI_BUFS];
    
    //If inline L4 save directly to global directory
    int level = FTI_Exec->ckptLvel;
    if (level == 4 && FTI_Ckpt[4].isInline) { 
        sprintf(fn, "%s/%s", FTI_Conf->gTmpDir, FTI_Exec->meta[0].ckptFile);
    }
    else {
        sprintf(fn, "%s/%s", FTI_Conf->lTmpDir, FTI_Exec->meta[0].ckptFile);
    }

    FILE* fd;

    // If ckpt file does not exist -> open with wb+ (Truncate to zero length or create file for update.)
    if (access(fn,R_OK) != 0) {
        fd = fopen(fn, "wb+");
    } 
    // If file exists -> open with rb+ (Open file for update (reading and writing).)
    else {
        fd = fopen(fn, "rb+");
    }

    if (fd == NULL) {
        sprintf(str, "FTI checkpoint file (%s) could not be opened.", fn);
        FTI_Print(str, FTI_EROR);

        return FTI_NSCS;
    }

    int writeFailed;
   
    // make sure that is never a null ptr. otherwise its to fix.
    assert(FTI_Exec->firstdb);
    FTIFF_db *currentdb = FTI_Exec->firstdb;
    FTIFF_dbvar *currentdbvar = NULL;
    char *dptr;
    int dbvar_idx, pvar_idx, dbcounter=0;
    long mdoffset;
    long endoffile = sizeof(FTIFF_metaInfo); // offset metaInfo FTI-FF
    
    // MD5 context for checksum of data chunks
    MD5_CTX mdContext;
    unsigned char hash[MD5_DIGEST_LENGTH];

    // block size for fwrite buffer in file.
    long membs = 1024*1024*16; // 16 MB
    long cpybuf, cpynow, cpycnt, fptr;

    int isnextdb;
    
    // Write in file with FTI-FF
    do {

        writeFailed = 0;
        isnextdb = 0;

        mdoffset = endoffile;
        
        // write db - datablock meta data
        fseek( fd, mdoffset, SEEK_SET );
        writeFailed += ( fwrite( &(currentdb->numvars), sizeof(int), 1, fd ) == 1 ) ? 0 : 1;
        mdoffset += sizeof(int);
        fseek( fd, mdoffset, SEEK_SET );
        writeFailed += ( fwrite( &(currentdb->dbsize), sizeof(long), 1, fd ) == 1 ) ? 0 : 1;
        mdoffset += sizeof(long);

        // debug information
        sprintf(str, "FTIFF: CKPT(id:%i), dataBlock:%i, dbsize: %ld, numvars: %i, write failed: %i", 
                FTI_Exec->ckptID, dbcounter, currentdb->dbsize, currentdb->numvars, writeFailed);
        FTI_Print(str, FTI_DBUG);

        // write dbvar - datablock variables meta data and 
        // ckpt data
        for(dbvar_idx=0;dbvar_idx<currentdb->numvars;dbvar_idx++) {

            currentdbvar = &(currentdb->dbvars[dbvar_idx]);

            clearerr(fd);
            errno = 0;

            // get source and destination pointer
            dptr = (char*)(FTI_Data[currentdbvar->idx].ptr) + currentdb->dbvars[dbvar_idx].dptr;
            fptr = currentdbvar->fptr;
            
            MD5_Init( &mdContext );
            cpycnt = 0;
            // write ckpt data
            while ( cpycnt < currentdbvar->chunksize ) {
                cpybuf = currentdbvar->chunksize - cpycnt;
                cpynow = ( cpybuf > membs ) ? membs : cpybuf;
                cpycnt += cpynow;
                fseek( fd, fptr, SEEK_SET );
                fwrite( dptr, cpynow, 1, fd );
                // if error for writing the data, print error and exit to calling function.
                if (ferror(fd)) {
                    int fwrite_errno = errno;
         	    	char error_msg[FTI_BUFS];
         	    	error_msg[0] = 0;
         	    	strerror_r(fwrite_errno, error_msg, FTI_BUFS);
         	    	sprintf(str, "Dataset #%d could not be written: %s.", currentdbvar->id, error_msg);
         	    	FTI_Print(str, FTI_EROR);
         	    	fclose(fd);
         	    	return FTI_NSCS;
                }
                MD5_Update( &mdContext, dptr, cpynow );
                dptr += cpynow;
                fptr += cpynow;
            }
            MD5_Final( currentdbvar->hash, &mdContext );
            
            // write datablock variables meta data
            fseek( fd, mdoffset, SEEK_SET );
            writeFailed += ( fwrite( currentdbvar, sizeof(FTIFF_dbvar), 1, fd ) == 1 ) ? 0 : 1;
            mdoffset += sizeof(FTIFF_dbvar);
            
            // debug information
            sprintf(str, "FTIFF: CKPT(id:%i) dataBlock:%i/dataBlockVar%i id: %i, idx: %i"
                    ", dptr: %ld, fptr: %ld, chunksize: %ld, "
                    "base_ptr: 0x%" PRIxPTR " ptr_pos: 0x%" PRIxPTR " write failed: %i", 
                    FTI_Exec->ckptID, dbcounter, dbvar_idx,  
                    currentdbvar->id, currentdbvar->idx, currentdbvar->dptr,
                    currentdbvar->fptr, currentdbvar->chunksize,
                    FTI_Data[currentdbvar->idx].ptr, dptr, writeFailed);
            FTI_Print(str, FTI_DBUG);

        }

        endoffile += currentdb->dbsize;

        if (currentdb->next) {
            currentdb = currentdb->next;
            isnextdb = 1;
        }

        dbcounter++;

    } while( isnextdb );

    FTI_Exec->ckptSize = endoffile;
       
    if( (FTI_Exec->ckptLvel == 2) || (FTI_Exec->ckptLvel == 3) ) { 
        
        long fileSizes[FTI_BUFS], mfs = 0;
        MPI_Allgather(&endoffile, 1, MPI_LONG, fileSizes, 1, MPI_LONG, FTI_Exec->groupComm);
        int ptnerGroupRank, i;
        switch(FTI_Exec->ckptLvel) {

            //update partner file size:
            case 2:

                ptnerGroupRank = (FTI_Topo->groupRank + FTI_Topo->groupSize - 1) % FTI_Topo->groupSize;
                FTI_Exec->FTIFFMeta.ptFs = fileSizes[ptnerGroupRank];
                FTI_Exec->FTIFFMeta.maxFs = -1;
                break;

            //get max file size in group 
            case 3:
                for (i = 0; i < FTI_Topo->groupSize; i++) {
                    if (fileSizes[i] > mfs) {
                        mfs = fileSizes[i]; // Search max. size
                    }
                }

                FTI_Exec->FTIFFMeta.maxFs = mfs;
                FTI_Exec->FTIFFMeta.ptFs = -1;
        }     
    
    } else {
        
        FTI_Exec->FTIFFMeta.ptFs = -1;
        FTI_Exec->FTIFFMeta.maxFs = -1;
    
    }
    
    // write meta data and its hash
    MD5_CTX mdContextTS;
    MD5_Init (&mdContextTS);
    struct timespec ntime;
    clock_gettime(CLOCK_REALTIME, &ntime);
    FTI_Exec->FTIFFMeta.timestamp = ntime.tv_sec*1000000000 + ntime.tv_nsec;
    FTI_Exec->FTIFFMeta.ckptSize = endoffile;
    FTI_Exec->FTIFFMeta.fs = endoffile;
    
    char checksum[MD5_DIGEST_STRING_LENGTH];
    FTI_Checksum( FTI_Exec, FTI_Data, FTI_Conf, checksum );
    strncpy( FTI_Exec->FTIFFMeta.checksum, checksum, MD5_DIGEST_STRING_LENGTH );

    // create checksum of meta data
    MD5_Update( &mdContextTS, FTI_Exec->FTIFFMeta.checksum, MD5_DIGEST_STRING_LENGTH );
    MD5_Update( &mdContextTS, &(FTI_Exec->FTIFFMeta.timestamp), sizeof(long) );
    MD5_Update( &mdContextTS, &(FTI_Exec->FTIFFMeta.ckptSize), sizeof(long) );
    MD5_Update( &mdContextTS, &(FTI_Exec->FTIFFMeta.fs), sizeof(long) );
    MD5_Update( &mdContextTS, &(FTI_Exec->FTIFFMeta.ptFs), sizeof(long) );
    MD5_Update( &mdContextTS, &(FTI_Exec->FTIFFMeta.maxFs), sizeof(long) );
    MD5_Final( FTI_Exec->FTIFFMeta.hashTimestamp, &mdContextTS );
    
    fseek( fd, 0, SEEK_SET );
    
    writeFailed += ( fwrite( &(FTI_Exec->FTIFFMeta), sizeof(FTIFF_metaInfo), 1, fd ) == 1 ) ? 0 : 1;
    
    fclose( fd );

    if (writeFailed) {
        sprintf(str, "FTIFF: An error occured. Discarding checkpoint");
        FTI_Print(str, FTI_WARN);
        return FTI_NSCS;
    }

    return FTI_SCES;

}
