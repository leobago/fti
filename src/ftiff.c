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

void FTI_FreeDbFTIFF(FTIFF_db* last)
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
