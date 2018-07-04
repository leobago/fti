/**
 *  Copyright (c) 2017 Leonardo A. Bautista-Gomez
 *  All rights reserved
 *
 *  FTI - A multi-level checkpointing library for C/C++/Fortran applications
 *
 *  Revision 1.0 : Fault Tolerance Interface (FTI)
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice, this
 *  list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright notice,
 *  this list of conditions and the following disclaimer in the documentation
 *  and/or other materials provided with the distribution.
 *
 *  3. Neither the name of the copyright holder nor the names of its contributors
 *  may be used to endorse or promote products derived from this software without
 *  specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 *  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 *  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 *  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 *  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 *  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *  @author Kai Keller (kellekai@gmx.de)
 *  @file   ftiff.c
 *  @date   October, 2017
 *  @brief  Functions for the FTI File Format (FTI-FF).
 */
#define _GNU_SOURCE

#include "interface.h"

/**  

  +-------------------------------------------------------------------------+
  |   STATIC TYPE DECLARATIONS                                              |
  +-------------------------------------------------------------------------+

 **/

MPI_Datatype FTIFF_MpiTypes[FTIFF_NUM_MPI_TYPES];

/**

  +-------------------------------------------------------------------------+
  |   FUNCTION DEFINITIONS                                                  |
  +-------------------------------------------------------------------------+

 **/

/*-------------------------------------------------------------------------*/
/**
  @brief      Initializes the derived MPI data types used for FTI-FF
 **/
/*-------------------------------------------------------------------------*/
void FTIFF_InitMpiTypes() 
{

    MPI_Aint lb, extent;
    FTIFF_MPITypeInfo MPITypeInfo[FTIFF_NUM_MPI_TYPES];

    // define MPI datatypes

    // headInfo
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
    MPITypeInfo[FTIFF_HEAD_INFO].mbrCnt = headInfo_mbrCnt;
    MPITypeInfo[FTIFF_HEAD_INFO].mbrBlkLen = headInfo_mbrBlkLen;
    MPITypeInfo[FTIFF_HEAD_INFO].mbrTypes = headInfo_mbrTypes;
    MPITypeInfo[FTIFF_HEAD_INFO].mbrDisp = headInfo_mbrDisp;

    // L2Info
    MBR_CNT( L2Info ) =  6;
    MBR_BLK_LEN( L2Info ) = { 1, 1, 1, 1, 1, 1 };
    MBR_TYPES( L2Info ) = { MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_LONG, MPI_LONG };
    MBR_DISP( L2Info ) = {  
        offsetof( FTIFF_L2Info, FileExists), 
        offsetof( FTIFF_L2Info, CopyExists), 
        offsetof( FTIFF_L2Info, ckptID), 
        offsetof( FTIFF_L2Info, rightIdx), 
        offsetof( FTIFF_L2Info, fs), 
        offsetof( FTIFF_L2Info, pfs), 
    };
    MPITypeInfo[FTIFF_L2_INFO].mbrCnt = L2Info_mbrCnt;
    MPITypeInfo[FTIFF_L2_INFO].mbrBlkLen = L2Info_mbrBlkLen;
    MPITypeInfo[FTIFF_L2_INFO].mbrTypes = L2Info_mbrTypes;
    MPITypeInfo[FTIFF_L2_INFO].mbrDisp = L2Info_mbrDisp;

    // L3Info
    MBR_CNT( L3Info ) =  5;
    MBR_BLK_LEN( L3Info ) = { 1, 1, 1, 1, 1 };
    MBR_TYPES( L3Info ) = { MPI_INT, MPI_INT, MPI_INT, MPI_LONG, MPI_LONG };
    MBR_DISP( L3Info ) = {  
        offsetof( FTIFF_L3Info, FileExists), 
        offsetof( FTIFF_L3Info, RSFileExists), 
        offsetof( FTIFF_L3Info, ckptID), 
        offsetof( FTIFF_L3Info, fs), 
        offsetof( FTIFF_L3Info, RSfs), 
    };
    MPITypeInfo[FTIFF_L3_INFO].mbrCnt = L3Info_mbrCnt;
    MPITypeInfo[FTIFF_L3_INFO].mbrBlkLen = L3Info_mbrBlkLen;
    MPITypeInfo[FTIFF_L3_INFO].mbrTypes = L3Info_mbrTypes;
    MPITypeInfo[FTIFF_L3_INFO].mbrDisp = L3Info_mbrDisp;

    // commit MPI types
    int i;
    for(i=0; i<FTIFF_NUM_MPI_TYPES; i++) {
        MPI_Type_create_struct( 
                MPITypeInfo[i].mbrCnt, 
                MPITypeInfo[i].mbrBlkLen, 
                MPITypeInfo[i].mbrDisp, 
                MPITypeInfo[i].mbrTypes, 
                &MPITypeInfo[i].raw );
        MPI_Type_get_extent( 
                MPITypeInfo[i].raw, 
                &lb, 
                &extent );
        MPI_Type_create_resized( 
                MPITypeInfo[i].raw, 
                lb, 
                extent, 
                &FTIFF_MpiTypes[i]);
        MPI_Type_commit( &FTIFF_MpiTypes[i] );
    }


}

/*-------------------------------------------------------------------------*/
/**
  @brief      Reads datablock structure for FTI File Format from ckpt file.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @return     integer         FTI_SCES if successful.

  Builds meta data list from checkpoint file for the FTI File Format

 **/
/*-------------------------------------------------------------------------*/
int FTIFF_ReadDbFTIFF( FTIT_execution *FTI_Exec, FTIT_checkpoint* FTI_Ckpt ) 
{
    char fn[FTI_BUFS]; //Path to the checkpoint file
    char str[FTI_BUFS]; //For console output
    char strerr[FTI_BUFS];

    int varCnt = 0;

    //Recovering from local for L4 case in FTI_Recover
    if (FTI_Exec->ckptLvel == 4) {
        snprintf(fn, FTI_BUFS, "%s/%s", FTI_Ckpt[1].dir, FTI_Exec->meta[1].ckptFile);
    }
    else {
        snprintf(fn, FTI_BUFS, "%s/%s", FTI_Ckpt[FTI_Exec->ckptLvel].dir, FTI_Exec->meta[FTI_Exec->ckptLvel].ckptFile);
    }

    // get filesize
    struct stat st;
    if (stat(fn, &st) == -1) {
        snprintf(strerr, FTI_BUFS, "FTI-FF: ReadDbFTIFF - could not get stats for file: %s", fn); 
        FTI_Print(strerr, FTI_EROR);
        errno = 0;
        return FTI_NSCS;
    }

    // open checkpoint file for read only
    int fd = open( fn, O_RDONLY, 0 );
    if (fd == -1) {
        snprintf( strerr, FTI_BUFS, "FTI-FF: ReadDbFTIFF - could not open '%s' for reading.", fn);
        FTI_Print(strerr, FTI_EROR);
        errno = 0;
        return FTI_NSCS;
    }

    // map file into memory
    char* fmmap = (char*) mmap(0, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (fmmap == MAP_FAILED) {
        snprintf( strerr, FTI_BUFS, "FTI-FF: ReadDbFTIFF - could not map '%s' to memory.", fn);
        FTI_Print(strerr, FTI_EROR);
        close(fd);
        errno = 0;
        return FTI_NSCS;
    }

    // file is mapped, we can close it.
    close(fd);

    long endoffile = 0; // space for timestamp 
    
    // get file meta info
    memcpy( FTI_Exec->FTIFFMeta.checksum, fmmap + endoffile, MD5_DIGEST_STRING_LENGTH );
    endoffile += MD5_DIGEST_STRING_LENGTH;
    memcpy( FTI_Exec->FTIFFMeta.myHash, fmmap + endoffile, MD5_DIGEST_LENGTH );
    endoffile += MD5_DIGEST_LENGTH;
    memcpy( &(FTI_Exec->FTIFFMeta.ckptSize), fmmap + endoffile, sizeof(long) );
    endoffile += sizeof(long);
    memcpy( &(FTI_Exec->FTIFFMeta.fs), fmmap + endoffile, sizeof(long) );
    endoffile += sizeof(long);
    memcpy( &(FTI_Exec->FTIFFMeta.maxFs), fmmap + endoffile, sizeof(long) );
    endoffile += sizeof(long);
    memcpy( &(FTI_Exec->FTIFFMeta.ptFs), fmmap + endoffile, sizeof(long) );
    endoffile += sizeof(long);
    memcpy( &(FTI_Exec->FTIFFMeta.timestamp), fmmap + endoffile, sizeof(long) );
    endoffile += sizeof(long);

    FTIFF_db *currentdb=NULL, *nextdb=NULL;
    FTIFF_dbvar *currentdbvar=NULL;
    int dbvar_idx, dbcounter=0;

    long mdoffset;

    int isnextdb;

    currentdb = (FTIFF_db*) malloc( sizeof(FTIFF_db) );
    if ( currentdb == NULL ) {
        snprintf( strerr, FTI_BUFS, "FTI-FF: ReadDbFTIFF - failed to allocate %ld bytes for 'currentdb'", sizeof(FTIFF_db));
        FTI_Print(strerr, FTI_EROR);
        munmap( fmmap, st.st_size );
        errno = 0;
        return FTI_NSCS;
    }

    FTI_Exec->firstdb = currentdb;
    FTI_Exec->firstdb->next = NULL;
    FTI_Exec->firstdb->previous = NULL;

    do {

        nextdb = (FTIFF_db*) malloc( sizeof(FTIFF_db) );
        if ( nextdb == NULL ) {
            snprintf( strerr, FTI_BUFS, "FTI-FF: ReadDbFTIFF - failed to allocate %ld bytes for 'nextdb'", sizeof(FTIFF_db));
            FTI_Print(strerr, FTI_EROR);
            munmap( fmmap, st.st_size );
            errno = 0;
            return FTI_NSCS;
        }

        isnextdb = 0;

        mdoffset = endoffile;

        memcpy( &(currentdb->numvars), fmmap+mdoffset, sizeof(int) ); 
        mdoffset += sizeof(int);
        memcpy( &(currentdb->dbsize), fmmap+mdoffset, sizeof(long) );
        mdoffset += sizeof(long);

        snprintf(str, FTI_BUFS, "FTI-FF: Updatedb - dataBlock:%i, dbsize: %ld, numvars: %i.", 
                dbcounter, currentdb->dbsize, currentdb->numvars);
        FTI_Print(str, FTI_DBUG);

        currentdb->dbvars = (FTIFF_dbvar*) malloc( sizeof(FTIFF_dbvar) * currentdb->numvars );
        if ( currentdb->dbvars == NULL ) {
            snprintf( strerr, FTI_BUFS, "FTI-FF: Updatedb - failed to allocate %ld bytes for 'currentdb->dbvars'", sizeof(FTIFF_dbvar) * currentdb->numvars);
            FTI_Print(strerr, FTI_EROR);
            munmap( fmmap, st.st_size );
            errno = 0;
            return FTI_NSCS;
        }

        for(dbvar_idx=0;dbvar_idx<currentdb->numvars;dbvar_idx++) {

            currentdbvar = &(currentdb->dbvars[dbvar_idx]);

            //memcpy( currentdbvar, fmmap+mdoffset, sizeof(FTIFF_dbvar) );
            int offset_dbvar = 0;
            memcpy( &(currentdbvar->id), fmmap+mdoffset, sizeof(int));
            offset_dbvar += sizeof(int);
            memcpy( &(currentdbvar->idx), fmmap+mdoffset+offset_dbvar, sizeof(int));
            offset_dbvar += sizeof(int);
            memcpy( &(currentdbvar->containerid), fmmap+mdoffset+offset_dbvar, sizeof(int));
            offset_dbvar += sizeof(int);
            memcpy( &(currentdbvar->hascontent), fmmap+mdoffset+offset_dbvar, sizeof(bool));
            offset_dbvar += sizeof(bool);
            memcpy( &(currentdbvar->hasCkpt), fmmap+mdoffset+offset_dbvar, sizeof(bool));
            offset_dbvar += sizeof(bool);
            memcpy( &(currentdbvar->dptr), fmmap+mdoffset+offset_dbvar, sizeof(uintptr_t));
            offset_dbvar += sizeof(uintptr_t);
            memcpy( &(currentdbvar->fptr), fmmap+mdoffset+offset_dbvar, sizeof(uintptr_t));
            offset_dbvar += sizeof(uintptr_t);
            memcpy( &(currentdbvar->chunksize), fmmap+mdoffset+offset_dbvar, sizeof(long));
            offset_dbvar += sizeof(long);
            memcpy( &(currentdbvar->containersize), fmmap+mdoffset+offset_dbvar, sizeof(long));
            offset_dbvar += sizeof(long);
            memcpy( currentdbvar->hash, fmmap+mdoffset+offset_dbvar, MD5_DIGEST_LENGTH);
            mdoffset += FTI_dbvarstructsize; //sizeof(FTIFF_dbvar);

            currentdbvar->hasCkpt = true;
            
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
            snprintf(str, FTI_BUFS, "FTI-FF: Updatedb -  dataBlock:%i/dataBlockVar%i id: %i, idx: %i"
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
    FTI_Exec->nbVarStored = varCnt;

    FTI_Exec->lastdb = currentdb;
    FTI_Exec->lastdb->next = NULL;

    // unmap memory.
    if ( munmap( fmmap, st.st_size ) == -1 ) {
        FTI_Print("FTI-FF: ReadDbFTIFF - unable to unmap memory", FTI_EROR);
        errno = 0;
        return FTI_NSCS;
    }
    
    return FTI_SCES;

}

/*-------------------------------------------------------------------------*/
/**
  @brief      Determines checksum of chechpoint data.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @return     integer         FTI_SCES if successful.
 **/
/*-------------------------------------------------------------------------*/
int FTIFF_GetFileChecksum( FTIFF_metaInfo *FTIFF_Meta, FTIT_checkpoint* FTI_Ckpt, int fd, unsigned char *hash ) 
{
    char str[FTI_BUFS]; //For console output
    char strerr[FTI_BUFS];

    // map file into memory
    char* fmmap = (char*) mmap(0, FTIFF_Meta->fs, PROT_READ, MAP_SHARED, fd, 0);
    if (fmmap == MAP_FAILED) {
        snprintf( strerr, FTI_BUFS, "FTI-FF: GetFileChecksum - could not map file to memory." );
        FTI_Print(strerr, FTI_EROR);
        errno = 0;
        return FTI_NSCS;
    }

    FTIFF_db *currentdb=NULL, *nextdb=NULL;
    FTIFF_dbvar *currentdbvar=NULL;
    int dbvar_idx, dbcounter=0;

    long endoffile = FTI_filemetastructsize; // space for timestamp 
    long mdoffset;

    int isnextdb;

    currentdb = (FTIFF_db*) malloc( sizeof(FTIFF_db) );
    if ( currentdb == NULL ) {
        snprintf( strerr, FTI_BUFS, "FTI-FF: GetFileChecksum - failed to allocate %ld bytes for 'currentdb'", sizeof(FTIFF_db));
        FTI_Print(strerr, FTI_EROR);
        munmap( fmmap, FTIFF_Meta->fs );
        errno = 0;
        return FTI_NSCS;
    }

    MD5_CTX ctx;
    MD5_Init(&ctx);
    do {

        nextdb = (FTIFF_db*) malloc( sizeof(FTIFF_db) );
        if ( nextdb == NULL ) {
            snprintf( strerr, FTI_BUFS, "FTI-FF: GetFileChecksum - failed to allocate %ld bytes for 'nextdb'", sizeof(FTIFF_db));
            FTI_Print(strerr, FTI_EROR);
            munmap( fmmap, FTIFF_Meta->fs );
            errno = 0;
            return FTI_NSCS;
        }

        isnextdb = 0;

        mdoffset = endoffile;

        memcpy( &(currentdb->numvars), fmmap+mdoffset, sizeof(int) ); 
        mdoffset += sizeof(int);
        memcpy( &(currentdb->dbsize), fmmap+mdoffset, sizeof(long) );
        mdoffset += sizeof(long);

        snprintf(str, FTI_BUFS, "FTI-FF: GetFileChecksum - dataBlock:%i, dbsize: %ld, numvars: %i.", 
                dbcounter, currentdb->dbsize, currentdb->numvars);
        FTI_Print(str, FTI_DBUG);

        currentdb->dbvars = (FTIFF_dbvar*) malloc( sizeof(FTIFF_dbvar) * currentdb->numvars );
        if ( currentdb->dbvars == NULL ) {
            snprintf( strerr, FTI_BUFS, "FTI-FF: GetFileChecksum - failed to allocate %ld bytes for 'currentdb->dbvars'", sizeof(FTIFF_dbvar) * currentdb->numvars);
            FTI_Print(strerr, FTI_EROR);
            munmap( fmmap, FTIFF_Meta->fs );
            errno = 0;
            return FTI_NSCS;
        }

        for(dbvar_idx=0;dbvar_idx<currentdb->numvars;dbvar_idx++) {

            currentdbvar = &(currentdb->dbvars[dbvar_idx]);

            //memcpy( currentdbvar, fmmap+mdoffset, sizeof(FTIFF_dbvar) );
            int offset_dbvar = 0;
            memcpy( &(currentdbvar->id), fmmap+mdoffset, sizeof(int));
            offset_dbvar += sizeof(int);
            memcpy( &(currentdbvar->idx), fmmap+mdoffset+offset_dbvar, sizeof(int));
            offset_dbvar += sizeof(int);
            memcpy( &(currentdbvar->containerid), fmmap+mdoffset+offset_dbvar, sizeof(int));
            offset_dbvar += sizeof(int);
            memcpy( &(currentdbvar->hascontent), fmmap+mdoffset+offset_dbvar, sizeof(bool));
            offset_dbvar += sizeof(bool);
            memcpy( &(currentdbvar->hasCkpt), fmmap+mdoffset+offset_dbvar, sizeof(bool));
            offset_dbvar += sizeof(bool);
            memcpy( &(currentdbvar->dptr), fmmap+mdoffset+offset_dbvar, sizeof(uintptr_t));
            offset_dbvar += sizeof(uintptr_t);
            memcpy( &(currentdbvar->fptr), fmmap+mdoffset+offset_dbvar, sizeof(uintptr_t));
            offset_dbvar += sizeof(uintptr_t);
            memcpy( &(currentdbvar->chunksize), fmmap+mdoffset+offset_dbvar, sizeof(long));
            offset_dbvar += sizeof(long);
            memcpy( &(currentdbvar->containersize), fmmap+mdoffset+offset_dbvar, sizeof(long));
            offset_dbvar += sizeof(long);
            memcpy( currentdbvar->hash, fmmap+mdoffset+offset_dbvar, MD5_DIGEST_LENGTH);
            mdoffset += FTI_dbvarstructsize;//sizeof(FTIFF_dbvar);
            
            DBG_MSG("dptr: %lld | fptr: %lld", -1, currentdbvar->dptr, currentdbvar->fptr);
            // debug information
            snprintf(str, FTI_BUFS, "FTI-FF: GetFileChecksum -  dataBlock:%i/dataBlockVar%i id: %i, idx: %i"
                    ", destptr: %llu, fptr: %llu, chunksize: %ld.",
                    dbcounter, dbvar_idx,  
                    currentdbvar->id, currentdbvar->idx, currentdbvar->dptr,
                    currentdbvar->fptr, currentdbvar->chunksize);
            FTI_Print(str, FTI_DBUG);

            if ( currentdbvar->hascontent ) {
                MD5_Update(&ctx, fmmap+currentdbvar->fptr, currentdbvar->chunksize);
            }
        }

        endoffile += currentdb->dbsize;

        if ( endoffile < FTIFF_Meta->ckptSize ) {
            memcpy( nextdb, fmmap+endoffile, FTI_dbstructsize );
            free(currentdb->dbvars);
            free(currentdb);
            currentdb = nextdb;
            isnextdb = 1;
        }

        dbcounter++;

    } while( isnextdb );

    MD5_Final( hash, &ctx );
    free(currentdb->dbvars);
    free(currentdb);

    // unmap memory.
    if ( munmap( fmmap, FTIFF_Meta->fs ) == -1 ) {
        FTI_Print("FTI-FF: GetFileChecksum - unable to unmap memory", FTI_EROR);
        errno = 0;
        return FTI_NSCS;
    }
    
    return FTI_SCES;

}


/*-------------------------------------------------------------------------*/
/**
  @brief      updates datablock structure for FTI File Format.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Data        Dataset metadata.
  @return     integer         FTI_SCES if successful.

  Updates information about the checkpoint file. Updates file pointers
  in the dbvar structures and updates the db structure.

 **/
/*-------------------------------------------------------------------------*/
int FTIFF_UpdateDatastructFTIFF( FTIT_execution* FTI_Exec, 
        FTIT_dataset* FTI_Data, FTIT_configuration* FTI_Conf )
{

    char str[FTI_BUFS]; 

    if( FTI_Exec->nbVar == 0 ) {
        FTI_Print("FTI-FF - UpdateDatastructFTIFF: No protected Variables, discarding checkpoint!", FTI_WARN);
        return FTI_NSCS;
    }

    char strerr[FTI_BUFS];

    int dbvar_idx, pvar_idx, num_edit_pvars = 0;
    int *editflags = (int*) calloc( FTI_Exec->nbVar, sizeof(int) ); 
    
    // 0 -> nothing to append, 1 -> new pvar, 2 -> size increased
    if ( editflags == NULL ) {
        snprintf( strerr, FTI_BUFS, "FTI-FF: UpdateDatastructFTIFF - failed to allocate %ld bytes for 'editflags'", sizeof(FTIFF_db));
        FTI_Print(strerr, FTI_EROR);
        errno = 0;
        return FTI_NSCS;
    }

    FTIFF_dbvar *dbvars = NULL;
    int isnextdb;
    long offset = FTI_filemetastructsize;
    long dbsize;

    // first call to this function. This means that
    // for all variables only one chunk/container exists.
    if(!FTI_Exec->firstdb) {
        dbsize = FTI_dbstructsize + FTI_dbvarstructsize /*sizeof(FTIFF_dbvar)*/ * FTI_Exec->nbVar;
        
        FTIFF_db *dblock = (FTIFF_db*) malloc( sizeof(FTIFF_db) );
        if ( dblock == NULL ) {
            free(editflags);
            snprintf( strerr, FTI_BUFS, "FTI-FF: UpdateDatastructFTIFF - failed to allocate %ld bytes for 'dblock'", sizeof(FTIFF_db));
            FTI_Print(strerr, FTI_EROR);
            errno = 0;
            return FTI_NSCS;
        }

        dbvars = (FTIFF_dbvar*) malloc( sizeof(FTIFF_dbvar) * FTI_Exec->nbVar );
        if ( dbvars == NULL ) {
            free(editflags);
            snprintf( strerr, FTI_BUFS, "FTI-FF: UpdateDatastructFTIFF - failed to allocate %ld bytes for 'dbvars'", sizeof(FTIFF_dbvar) * FTI_Exec->nbVar );
            FTI_Print(strerr, FTI_EROR);
            errno = 0;
            return FTI_NSCS;
        }

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
            dbvars[dbvar_idx].hascontent = true;
            dbvars[dbvar_idx].hasCkpt = false;
            dbvars[dbvar_idx].containerid = 0;
            dbvars[dbvar_idx].containersize = FTI_Data[dbvar_idx].size;
            // FOR DCP 
            if  ( FTI_Conf->dcpEnabled ) {
                FTI_InitBlockHashArray( &(dbvars[dbvar_idx]), &(FTI_Data[dbvar_idx]) );
                //DBG_MSG("INIT HASH INFO",-1);
            } else {
                dbvars[dbvar_idx].nbHashes = -1;
                dbvars[dbvar_idx].dataDiffHash = NULL;
            }
            dbsize += dbvars[dbvar_idx].containersize; 
            dbvars[dbvar_idx].update = true;
            FTIFF_GetHashdbvar( dbvars[dbvar_idx].myhash, &(dbvars[dbvar_idx]) );
        }
        FTI_Exec->nbVarStored = FTI_Exec->nbVar;
        dblock->dbsize = dbsize;
        
        dblock->update = true;
        FTIFF_GetHashdb( dblock->myhash, dblock );

        // set as first datablock
        FTI_Exec->firstdb = dblock;
        FTI_Exec->lastdb = dblock;

    } else {

        /*
         *  - check if protected variable is in file info
         *  - check if size has changed
         */

        FTI_Exec->lastdb = FTI_Exec->firstdb;

        int* nbContainers = (int*) calloc( FTI_Exec->nbVarStored, sizeof(int) );
        if ( nbContainers == NULL ) {
            free(editflags);
            snprintf( strerr, FTI_BUFS, "FTI-FF: UpdateDatastructFTIFF - failed to allocate %ld bytes for 'nbContainers'", sizeof(int));
            FTI_Print(strerr, FTI_EROR);
            errno = 0;
            return FTI_NSCS;
        }

        long* containerSizesAccu = (long*) calloc( FTI_Exec->nbVarStored, sizeof(long) );
        if ( containerSizesAccu == NULL ) {
            free(editflags);
            snprintf( strerr, FTI_BUFS, "FTI-FF: UpdateDatastructFTIFF - failed to allocate %ld bytes for 'containerSizesAccu'", sizeof(long));
            FTI_Print(strerr, FTI_EROR);
            errno = 0;
            return FTI_NSCS;
        }

        bool* validBlock = (bool*) malloc( FTI_Exec->nbVarStored*sizeof(bool) );
        if ( validBlock == NULL ) {
            free(editflags);
            snprintf( strerr, FTI_BUFS, "FTI-FF: UpdateDatastructFTIFF - failed to allocate %ld bytes for 'validBlock'", FTI_Exec->nbVarStored*sizeof(bool) );
            FTI_Print(strerr, FTI_EROR);
            errno = 0;
            return FTI_NSCS;
        }

        long* overflow = (long*) malloc( FTI_Exec->nbVarStored*sizeof(long) );
        if ( overflow == NULL ) {
            free(editflags);
            snprintf( strerr, FTI_BUFS, "FTI-FF: UpdateDatastructFTIFF - failed to allocate %ld bytes for 'overflow'", FTI_Exec->nbVarStored*sizeof(long) );
            FTI_Print(strerr, FTI_EROR);
            errno = 0;
            return FTI_NSCS;
        }

        // init overflow with the datasizes and validBlock with true.
        for(pvar_idx=0;pvar_idx<FTI_Exec->nbVarStored;pvar_idx++) {
            overflow[pvar_idx] = FTI_Data[pvar_idx].size;
            validBlock[pvar_idx] = true;
        }

        // iterate though datablock list. Current datablock is 'lastdb'.
        // At the beginning of the loop 'lastdb = firstdb'
        do {
            isnextdb = 0;
            for(dbvar_idx=0;dbvar_idx<FTI_Exec->lastdb->numvars;dbvar_idx++) {
                FTIFF_dbvar* dbvar = &(FTI_Exec->lastdb->dbvars[dbvar_idx]);
                for(pvar_idx=0;pvar_idx<FTI_Exec->nbVarStored;pvar_idx++) {
                    FTIT_dataset* data = &FTI_Data[pvar_idx];
                    if( dbvar->id == data->id ) {
                        // collect container info
                        containerSizesAccu[pvar_idx] += dbvar->containersize;
                        nbContainers[pvar_idx]++;
                        // if data was shrinked, invalidate the following blocks (if there are), 
                        // and set their chunksize to 0.
                        if ( !validBlock[pvar_idx] ) {
                            if ( dbvar->hascontent ) {
                                dbvar->hascontent = false;
                                // [FOR DCP] free hash array and hash structure in block
                                if ( ( dbvar->dataDiffHash != NULL ) && FTI_Conf->dcpEnabled ) {
                                    //DBG_MSG("FREE",-1);
                                    free(dbvar->dataDiffHash[0].md5hash);
                                    free(dbvar->dataDiffHash);
                                    dbvar->dataDiffHash = NULL;
                                    dbvar->nbHashes = 0;
                                }
                            }
                            dbvar->chunksize = 0;
                            continue;
                        }
                        // if overflow > containersize, reduce overflow by containersize
                        // set chunksize to containersize and ensure that 'hascontent = true'.
                        if ( overflow[pvar_idx] > dbvar->containersize ) {
                            long chunksizeOld = dbvar->chunksize;
                            dbvar->chunksize = dbvar->containersize;
                            if ( !dbvar->hascontent ) {
                                dbvar->hascontent = true;
                                // [FOR DCP] init hash array for block
                                if ( FTI_Conf->dcpEnabled ) {
                                    FTI_InitBlockHashArray( dbvar, data );
                                    //DBG_MSG("INIT CAUSE DATASET INCREASED",-1);
                                }
                            } else {
                                // [FOR DCP] adjust hash array to new chunksize if chunk size increased
                                if ( FTI_Conf->dcpEnabled ) {
                                    if (  dbvar->chunksize > chunksizeOld ) {
                                        FTI_ExpandBlockHashArray( dbvar, dbvar->containersize, data );
                                        //DBG_MSG("EXPAND",-1);
                                    }
                                    if ( ((FTI_ADDRPTR)(data->ptr + dbvar->dptr)) != dbvar->dataDiffHash[0].ptr ) {
                                        FTI_UpdateBlockHashPtr( dbvar, data );
                                    }
                                }
                            }
                            overflow[pvar_idx] -= dbvar->containersize;
                            continue;
                        }
                        // if overflow <= containersize, set 'validBlock = false' in order to invalidate the
                        // following blocks, set new chunksize to overflow, set afterwards overflow to 0 and 
                        // ensure that 'hascontent = true'. 
                        if ( overflow[pvar_idx] <= dbvar->containersize ) {
                            long chunksizeOld = dbvar->chunksize;
                            dbvar->chunksize = overflow[pvar_idx];
                            if ( !dbvar->hascontent ) {
                                dbvar->hascontent = true;
                                // [FOR DCP] init hash array for block
                                if ( FTI_Conf->dcpEnabled ) {
                                    FTI_InitBlockHashArray( dbvar, data );
                                    //DBG_MSG("INIT TO REACTIVATE BLOCK",-1);
                                }
                            } else {
                                // [FOR DCP] adjust hash array to new chunksize if chunk size decreased
                                if ( FTI_Conf->dcpEnabled ) {
                                    if ( dbvar->chunksize < chunksizeOld ) {
                                        FTI_CollapseBlockHashArray( dbvar, chunksizeOld, data );
                                        //DBG_MSG("COLLAPSE",-1);
                                    }
                                    if ( dbvar->chunksize > chunksizeOld ) {
                                        FTI_ExpandBlockHashArray( dbvar, dbvar->chunksize, data );
                                        //DBG_MSG("EXPAND",-1);
                                    }
                                    if ( ((FTI_ADDRPTR)(data->ptr + dbvar->dptr)) != dbvar->dataDiffHash[0].ptr ) {
                                        FTI_UpdateBlockHashPtr( dbvar, data );
                                    }
                                }
                            }
                            validBlock[pvar_idx] = false;
                            overflow[pvar_idx] = 0;
                            continue;
                        }
                    }
                }
            }
            offset += FTI_Exec->lastdb->dbsize;
            if (FTI_Exec->lastdb->next) {
                FTI_Exec->lastdb = FTI_Exec->lastdb->next;
                isnextdb = 1;
            }
        } while( isnextdb );
        // end of while, 'lastdb' is last datablock.

        // check for new protected variables
        for(pvar_idx=FTI_Exec->nbVarStored;pvar_idx<FTI_Exec->nbVar;pvar_idx++) {
            editflags[pvar_idx] = 1;
            num_edit_pvars++;
        }

        // check if size has increased
        for(pvar_idx=0;pvar_idx<FTI_Exec->nbVarStored;pvar_idx++) {  
            if ( overflow[pvar_idx] > 0 ) {
                editflags[pvar_idx] = 2;
                num_edit_pvars++;
            }
        }

        // if size changed or we have new variables to protect, create new block. 
        dbsize = FTI_dbstructsize + FTI_dbvarstructsize /*sizeof(FTIFF_dbvar)*/ * num_edit_pvars;

        int evar_idx = 0;
        if( num_edit_pvars ) {
            for(pvar_idx=0; pvar_idx<FTI_Exec->nbVar; pvar_idx++) {
                bool callInit = false;
                switch(editflags[pvar_idx]) {

                    case 1:
                        // add new protected variable in next datablock
                        dbvars = (FTIFF_dbvar*) realloc( dbvars, (evar_idx+1) * sizeof(FTIFF_dbvar) );
                        dbvars[evar_idx].fptr = offset + dbsize;
                        dbvars[evar_idx].dptr = 0;
                        dbvars[evar_idx].id = FTI_Data[pvar_idx].id;
                        dbvars[evar_idx].idx = pvar_idx;
                        dbvars[evar_idx].chunksize = FTI_Data[pvar_idx].size;
                        dbvars[evar_idx].hascontent = true;
                        dbvars[evar_idx].hasCkpt = false;
                        dbvars[evar_idx].containerid = 0;
                        dbvars[evar_idx].containersize = FTI_Data[pvar_idx].size;
                        dbsize += dbvars[evar_idx].containersize; 
                        if ( FTI_Conf->dcpEnabled ) {
                            FTI_InitBlockHashArray( &(dbvars[evar_idx]), &(FTI_Data[pvar_idx]) );
                        }
                        evar_idx++;
                        callInit = true;

                        break;

                    case 2:

                        // create data chunk info
                        dbvars = (FTIFF_dbvar*) realloc( dbvars, (evar_idx+1) * sizeof(FTIFF_dbvar) );
                        dbvars[evar_idx].fptr = offset + dbsize;
                        dbvars[evar_idx].dptr = containerSizesAccu[pvar_idx];
                        dbvars[evar_idx].id = FTI_Data[pvar_idx].id;
                        dbvars[evar_idx].idx = pvar_idx;
                        dbvars[evar_idx].chunksize = overflow[pvar_idx];
                        dbvars[evar_idx].hascontent = true;
                        dbvars[evar_idx].hasCkpt = false;
                        dbvars[evar_idx].containerid = nbContainers[pvar_idx];
                        dbvars[evar_idx].containersize = overflow[pvar_idx]; 
                        dbsize += dbvars[evar_idx].containersize; 
                        if ( FTI_Conf->dcpEnabled ) {
                            FTI_InitBlockHashArray( &(dbvars[evar_idx]), &(FTI_Data[pvar_idx]) );
                        }
                        evar_idx++;
                        callInit = true;

                        break;

                }

                dbvars[dbvar_idx].update = true;
                FTIFF_GetHashdbvar( dbvars[evar_idx].myhash, &(dbvars[evar_idx]) );
                // [FOR DCP] init hash array for new block or new protected variable
                //if ( FTI_Conf->dcpEnabled && callInit ) {
                //    FTI_InitBlockHashArray( &(dbvars[evar_idx-1]), &(FTI_Data[pvar_idx]) );
                //    //DBG_MSG("INIT NEW BLOCK/VARIABLE",-1);
                //}

            }

            FTIFF_db  *dblock = (FTIFF_db*) malloc( sizeof(FTIFF_db) );
            if ( dblock == NULL ) {
                free(editflags);
                free(containerSizesAccu);
                free(nbContainers);
                free(overflow);
                free(validBlock);
                snprintf( strerr, FTI_BUFS, "FTI-FF: UpdateDatastructFTIFF - failed to allocate %ld bytes for 'dblock'", sizeof(FTIFF_db));
                FTI_Print(strerr, FTI_EROR);
                errno = 0;
                return FTI_NSCS;
            }

            FTI_Exec->lastdb->next = dblock;
            dblock->previous = FTI_Exec->lastdb;
            dblock->next = NULL;
            dblock->numvars = num_edit_pvars;
            dblock->dbsize = dbsize;
            dblock->dbvars = dbvars;
            FTI_Exec->lastdb = dblock;
            
            dblock->update = true;
            FTIFF_GetHashdb( dblock->myhash, dblock );
        }

        FTI_Exec->nbVarStored = FTI_Exec->nbVar;
        
        free(nbContainers);
        free(containerSizesAccu);
        free(validBlock);
        free(overflow);

    }

    // FOR DEVELOPING
    // FTIFF_PrintDataStructure( 0, FTI_Exec );

    free(editflags);
    return FTI_SCES;

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

  +--------------+ +------------------------+
  |              | |                        |
  | FB           | | VB                     |
  |              | |                        |
  +--------------+ +------------------------+
  
  The FB (file block) holds meta data related to the file whereas the VB 
  (variable block) holds meta and actual data of the variables protected by FTI. 
  
  |<------------------------------------ VB ------------------------------------>|
  #                                                                              #
  |<------------ VCB_1--------------->|      |<------------ VCB_n--------------->|
  #                                   #      #                                   #       
  +-----------------------------------+      +-----------------------------------+
  | +-------++-------+      +-------+ |      | +-------++-------+      +-------+ |
  | |       ||       |      |       | |      | |       ||       |      |       | |
  | | VMB_1 || VC_11 | ---- | VC_1k | | ---- | | VMB_n || VC_n1 | ---- | VC_nl | |
  | |       ||       |      |       | |      | |       ||       |      |       | |
  | +-------++-------+      +-------+ |      | +-------++-------+      +-------+ |
  +-----------------------------------+      +-----------------------------------+

  VMB_i (FTIFF_db + FTIFF_dbvar structures) keeps the data block metadata and 
  VC_ij are the data chunks.

 **/
/*-------------------------------------------------------------------------*/
int FTIFF_WriteFTIFF(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt,
        FTIT_dataset* FTI_Data)
{
    double start_t, end_t;
   
    FTIFF_UpdateDatastructFTIFF( FTI_Exec, FTI_Data, FTI_Conf );
    

    char str[FTI_BUFS], fn[FTI_BUFS], strerr[FTI_BUFS], fnr[FTI_BUFS];
    
    FTI_Print("I/O mode: FTI File Format.", FTI_DBUG);

    // check if metadata exists
    if( FTI_Exec->firstdb == NULL ) {
        FTI_Print("No data structure found to write data to file. Discarding checkpoint.", FTI_WARN);
        return FTI_NSCS;
    }

    //If inline L4 save directly to global directory
    int level = FTI_Exec->ckptLvel;
    if (level == 4 && FTI_Ckpt[4].isInline) { 
        if( FTI_Conf->dcpEnabled && FTI_Ckpt[4].isDcp ) {
            snprintf(fn, FTI_BUFS, "%s/%s", FTI_Ckpt[4].dcpDir, FTI_Ckpt[4].dcpName);
        } else {
            snprintf(fn, FTI_BUFS, "%s/%s", FTI_Conf->gTmpDir, FTI_Exec->meta[0].ckptFile);
        }
    }
    else {
        snprintf(fn, FTI_BUFS, "%s/%s", FTI_Conf->lTmpDir, FTI_Exec->meta[0].ckptFile);
    }

    int fd;

    // for dCP: create if not exists, open if exists
    if ( FTI_Conf->dcpEnabled && FTI_Ckpt[4].isDcp ) {
        if (access(fn,R_OK) != 0) {
            fd = open( fn, O_WRONLY|O_CREAT|O_TRUNC, (mode_t) 0600 ); 
        } 
        else {
            fd = open( fn, O_WRONLY );
        }
    } else {
        fd = open( fn, O_WRONLY|O_CREAT, (mode_t) 0600 ); 
    }

    if (fd == -1) {
        snprintf(strerr, FTI_BUFS, "FTI checkpoint file (%s) could not be opened.", fn);
        FTI_Print(strerr, FTI_EROR);
        return FTI_NSCS;
    }

    // make sure that is never a null ptr. otherwise its to fix.
    assert(FTI_Exec->firstdb);
    FTIFF_db *currentdb = FTI_Exec->firstdb;
    FTIFF_dbvar *currentdbvar = NULL;
    char *dptr;
    int dbvar_idx, dbcounter=0;
    long mdoffset;
    long endoffile = FTI_filemetastructsize;

    // MD5 context for file (only data) checksum
    MD5_CTX mdContext;
    MD5_Init(&mdContext);

    // block size for fwrite buffer in file.
    long membs = 1024*1024*16; // 16 MB
    long cpybuf, cpynow, cpycnt;//, fptr;

    uintptr_t fptr;
    int isnextdb;
    
    int ids[FTI_BUFS];
    int num_ids = 0;

    long diffSize = 0, ckptsize = 0;

    // write FTI-FF meta data
    do {    

        isnextdb = 0;

        mdoffset = endoffile;

        endoffile += currentdb->dbsize;

        if (currentdb->update) {
            
            if ( lseek( fd, mdoffset, SEEK_SET ) == -1 ) {
                snprintf(strerr, FTI_BUFS, "FTI-FF: WriteFTIFF - could not seek in file: %s", fn);
                FTI_Print(strerr, FTI_EROR);
                close(fd);
                errno = 0;
                return FTI_NSCS;
            }

            char * db_ser = (char*) malloc ( FTI_dbstructsize );

            memcpy( db_ser, &(currentdb->numvars), sizeof(int) );

            memcpy( db_ser+sizeof(int), &(currentdb->dbsize), sizeof(long) );
            
            write( fd, db_ser, FTI_dbstructsize );
            
            if ( fd == -1 ) {
                snprintf(strerr, FTI_BUFS, "FTI-FF: WriteFTIFF - could not write metadata in file: %s", fn);
                FTI_Print(strerr, FTI_EROR);
                errno=0;
                close(fd);
                return FTI_NSCS;
            }

            free( db_ser );
        }

        mdoffset += FTI_dbstructsize;

        for(dbvar_idx=0;dbvar_idx<currentdb->numvars;dbvar_idx++) {

            currentdbvar = &(currentdb->dbvars[dbvar_idx]);
            bool hascontent = currentdbvar->hascontent;
            FTI_ADDRVAL cbasePtr = (FTI_ADDRVAL)(FTI_Data[currentdbvar->idx].ptr) + currentdb->dbvars[dbvar_idx].dptr;
            errno = 0;
                
            // create datachunk hash
            if(hascontent) {
                ckptsize += currentdbvar->chunksize;
                MD5_Update( &mdContext, (FTI_ADDRPTR) cbasePtr, currentdbvar->chunksize );
                MD5( (FTI_ADDRPTR) cbasePtr, currentdbvar->chunksize, currentdbvar->hash );  
            }
            char *dbvar_ser = malloc(FTI_dbvarstructsize);
            char *cpy_ptr = dbvar_ser;

            if (currentdbvar->update) {
                
                if ( lseek( fd, mdoffset, SEEK_SET ) == -1 ) {
                    snprintf(strerr, FTI_BUFS, "FTI-FF: WriteFTIFF - could not seek in file: %s", fn);
                    FTI_Print(strerr, FTI_EROR);
                    close(fd);
                    errno = 0;
                    return FTI_NSCS;
                }


                memcpy( cpy_ptr, &(currentdbvar->id), sizeof(int));
                cpy_ptr += sizeof(int);
                memcpy( cpy_ptr, &(currentdbvar->idx), sizeof(int));
                cpy_ptr += sizeof(int);
                memcpy( cpy_ptr, &(currentdbvar->containerid), sizeof(int));
                cpy_ptr += sizeof(int);
                memcpy( cpy_ptr, &(currentdbvar->hascontent), sizeof(bool));
                cpy_ptr += sizeof(bool);
                memcpy( cpy_ptr, &(currentdbvar->hasCkpt), sizeof(bool));
                cpy_ptr += sizeof(bool);
                memcpy( cpy_ptr, &(currentdbvar->dptr), sizeof(uintptr_t));
                cpy_ptr += sizeof(uintptr_t);
                memcpy( cpy_ptr, &(currentdbvar->fptr), sizeof(uintptr_t));
                cpy_ptr += sizeof(uintptr_t);
                memcpy( cpy_ptr, &(currentdbvar->chunksize), sizeof(long));
                cpy_ptr += sizeof(long);
                memcpy( cpy_ptr, &(currentdbvar->containersize), sizeof(long));
                cpy_ptr += sizeof(long);
                memcpy( cpy_ptr, currentdbvar->hash, MD5_DIGEST_LENGTH);
                
                write( fd, dbvar_ser, FTI_dbvarstructsize );
                if ( fd == -1 ) {
                    snprintf(strerr, FTI_BUFS, "FTI-FF: WriteFTIFF - could not write metadata in file: %s", fn);
                    FTI_Print(strerr, FTI_EROR);
                    errno=0;
                    close(fd);
                    return FTI_NSCS;
                }

//                free( dbvar_ser );

            }
            
            off_t off_dptr = 3*sizeof(int) + 2*sizeof(bool);

            DBG_MSG("dptr: %lld | fptr: %lld | _dptr: %llu | _fptr: %llu", -1, 
                    currentdbvar->dptr, currentdbvar->fptr, *(uintptr_t*)(dbvar_ser+off_dptr), *(uintptr_t*)(dbvar_ser+off_dptr+sizeof(uintptr_t)));
           
            mdoffset += FTI_dbvarstructsize;

        }

        if (currentdb->next) {
            currentdb = currentdb->next;
            isnextdb = 1;
        }

        dbcounter++;

    } while( isnextdb );

    // create string of filehash and create other file meta data
    unsigned char fhash[MD5_DIGEST_LENGTH];
    MD5_Final( fhash, &mdContext );
    
    int ii = 0, i;
    for(i = 0; i < MD5_DIGEST_LENGTH; i++) {
        sprintf(&(FTI_Exec->FTIFFMeta.checksum[ii]), "%02x", fhash[i]);
        ii += 2;
    }

    // has to be assigned before FTIFF_CreateMetaData call!
    FTI_Exec->ckptSize = endoffile;
    
    if ( FTI_Try( FTIFF_CreateMetadata( FTI_Exec, FTI_Topo, FTI_Data, FTI_Conf ), "Create FTI-FF meta data" ) != FTI_SCES ) {
        return FTI_NSCS;
    }
    
    DBG_MSG("FTI_Exec->ckptSize: %ld | FTIFF_Meta.ckptSize: %ld",-1, FTI_Exec->ckptSize, FTI_Exec->FTIFFMeta.ckptSize);

    // Write file meta data (serialized)
    char* fmeta_ser = malloc( 
            MD5_DIGEST_STRING_LENGTH +
            MD5_DIGEST_LENGTH +
            5*sizeof(long) );
    
    int pos = 0;
    memcpy( fmeta_ser, FTI_Exec->FTIFFMeta.checksum, MD5_DIGEST_STRING_LENGTH );
    pos += MD5_DIGEST_STRING_LENGTH;
    memcpy( fmeta_ser + pos, FTI_Exec->FTIFFMeta.myHash, MD5_DIGEST_LENGTH );
    pos += MD5_DIGEST_LENGTH;
    memcpy( fmeta_ser + pos, &(FTI_Exec->FTIFFMeta.ckptSize), sizeof(long) );
    pos += sizeof(long);
    memcpy( fmeta_ser + pos, &(FTI_Exec->FTIFFMeta.fs), sizeof(long) );
    pos += sizeof(long);
    memcpy( fmeta_ser + pos, &(FTI_Exec->FTIFFMeta.maxFs), sizeof(long) );
    pos += sizeof(long);
    memcpy( fmeta_ser + pos, &(FTI_Exec->FTIFFMeta.ptFs), sizeof(long) );
    pos += sizeof(long);
    memcpy( fmeta_ser + pos, &(FTI_Exec->FTIFFMeta.timestamp), sizeof(long) );
    pos += sizeof(long);

    write( fd, fmeta_ser, pos );
    if ( fd == -1 ) {
        snprintf(strerr, FTI_BUFS, "FTI-FF: WriteFTIFF - could not write file metadata in file: %s", fn);
        FTI_Print(strerr, FTI_EROR);
        errno=0;
        close(fd);
        return FTI_NSCS;
    }

    free( fmeta_ser );

    // reset db pointer
    currentdb = FTI_Exec->firstdb;
    
    do {    

        isnextdb = 0;

        for(dbvar_idx=0;dbvar_idx<currentdb->numvars;dbvar_idx++) {

            currentdbvar = &(currentdb->dbvars[dbvar_idx]);

            // get source and destination pointer
            dptr = (char*)(FTI_Data[currentdbvar->idx].ptr) + currentdb->dbvars[dbvar_idx].dptr;
            fptr = currentdbvar->fptr;
            uintptr_t chunk_addr, chunk_size, chunk_offset, base;

            uintptr_t chunk_size_dbg, chunk_addr_dbg; 
           
            int chunkid = 0;
            
            while( FTI_ReceiveDataChunk(&chunk_addr, &chunk_size, currentdbvar, FTI_Data) ) {
                chunk_offset = chunk_addr - ((FTI_ADDRVAL)(FTI_Data[currentdbvar->idx].ptr) + currentdbvar->dptr);
                
                dptr += chunk_offset;
                fptr = currentdbvar->fptr + chunk_offset;

                if ( lseek( fd, fptr, SEEK_SET ) == -1 ) {
                    snprintf(strerr, FTI_BUFS, "FTI-FF: WriteFTIFF - could not seek in file: %s", fn);
                    FTI_Print(strerr, FTI_EROR);
                    errno=0;
                    close(fd);
                    return FTI_NSCS;
                }

                cpycnt = 0;
                while ( cpycnt < chunk_size ) {
                    cpybuf = chunk_size - cpycnt;
                    cpynow = ( cpybuf > membs ) ? membs : cpybuf;
                    
                    long WRITTEN = 0;
                    
                    int try = 0; 
                    do {
                        WRITTEN += write( fd, (FTI_ADDRPTR) (chunk_addr+cpycnt), cpynow );
                        if ( fd == -1 ) {
                            snprintf(str, FTI_BUFS, "FTI-FF: WriteFTIFF - Dataset #%d could not be written to file: %s", currentdbvar->id, fn);
                            FTI_Print(str, FTI_EROR);
                            close(fd);
                            errno = 0;
                            return FTI_NSCS;
                        }
                        try++;
                    } while ((WRITTEN < cpynow) && (try < 10));
                    
                    assert( WRITTEN == cpynow );
                    
                    cpycnt += WRITTEN;
                    diffSize += WRITTEN;
                }
                assert(cpycnt == chunk_size);

                chunkid++;

            }

            // debug information
            snprintf(str, FTI_BUFS, "FTIFF: CKPT(id:%i) dataBlock:%i/dataBlockVar%i id: %i, idx: %i"
                    ", dptr: %ld, fptr: %ld, chunksize: %ld, "
                    "base_ptr: 0x%" PRIxPTR " ptr_pos: 0x%" PRIxPTR " ", 
                    FTI_Exec->ckptID, dbcounter, dbvar_idx,  
                    currentdbvar->id, currentdbvar->idx, currentdbvar->dptr,
                    currentdbvar->fptr, currentdbvar->chunksize,
                    (uintptr_t)FTI_Data[currentdbvar->idx].ptr, (uintptr_t)dptr);
            FTI_Print(str, FTI_DBUG);

        }

        if (currentdb->next) {
            currentdb = currentdb->next;
            isnextdb = 1;
        }

        dbcounter++;

    } while( isnextdb );

    long checksize = 0;
    int dataidx;
    for( dataidx=0; dataidx<FTI_Exec->nbVar; ++dataidx ) {
        checksize += FTI_Data[dataidx].size;
    }
    
    long values_total[2], values_local[2];
    values_local[0] = ckptsize;
    values_local[1] = diffSize; 
    MPI_Allreduce(values_local, values_total, 2, MPI_LONG, MPI_SUM, FTI_Exec->globalComm);
    DBG_MSG("share: %.2lf, diffsize: %.4lf MB, ckptsize: %.4lf MB", 
            0, 100.0*(((double)values_total[1])/values_total[0]), 
            (double)values_total[1]/(1024*1024*1024), (double)values_total[0]/(1024*1024*1024));       
 
    close( fd );

    return FTI_SCES;

}

/*-------------------------------------------------------------------------*/
/**
  @brief      Assign meta data to runtime and file meta data types
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Topo        Topology metadata.
  @param      FTI_Data        Dataset metadata.
  @return     integer         FTI_SCES if successful.

  This function gathers information about the checkpoint files in the
  group and stores it in the respective meta data types runtime and
  ckpt file.

 **/
/*-------------------------------------------------------------------------*/
int FTIFF_CreateMetadata( FTIT_execution* FTI_Exec, FTIT_topology* FTI_Topo, 
        FTIT_dataset* FTI_Data, FTIT_configuration* FTI_Conf )
{
    int i;

    // FTI_Exec->ckptSize has to be assigned after successful ckpt write and before this call!
    long fs = FTI_Exec->ckptSize;
    FTI_Exec->FTIFFMeta.ckptSize = fs;
    FTI_Exec->FTIFFMeta.fs = fs;

    // allgather not needed for L1 checkpoint
    if( (FTI_Exec->ckptLvel == 2) || (FTI_Exec->ckptLvel == 3) ) { 

        long fileSizes[FTI_BUFS], mfs = 0;
        MPI_Allgather(&fs, 1, MPI_LONG, fileSizes, 1, MPI_LONG, FTI_Exec->groupComm);
        int ptnerGroupRank, i;
        switch(FTI_Exec->ckptLvel) {

            //get partner file size:
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

    FTI_Exec->meta[0].fs[0] = FTI_Exec->FTIFFMeta.fs;
    FTI_Exec->meta[0].pfs[0] = FTI_Exec->FTIFFMeta.ptFs;
    FTI_Exec->meta[0].maxFs[0] = FTI_Exec->FTIFFMeta.maxFs;

    // write meta data and its hash
    struct timespec ntime;
    if ( clock_gettime(CLOCK_REALTIME, &ntime) == -1 ) {
        FTI_Print("FTI-FF: FTIFF_CreateMetaData - failed to determine time, timestamp set to -1", FTI_WARN);
        FTI_Exec->FTIFFMeta.timestamp = -1;
    } else {
        FTI_Exec->FTIFFMeta.timestamp = ntime.tv_sec*1000000000 + ntime.tv_nsec;
    }

    // create checksum of meta data
    FTIFF_GetHashMetaInfo( FTI_Exec->FTIFFMeta.myHash, &(FTI_Exec->FTIFFMeta) );

    //Flush metadata in case postCkpt done inline
    FTI_Exec->meta[FTI_Exec->ckptLvel].fs[0] = FTI_Exec->meta[0].fs[0];
    FTI_Exec->meta[FTI_Exec->ckptLvel].pfs[0] = FTI_Exec->meta[0].pfs[0];
    FTI_Exec->meta[FTI_Exec->ckptLvel].maxFs[0] = FTI_Exec->meta[0].maxFs[0];
    strncpy(FTI_Exec->meta[FTI_Exec->ckptLvel].ckptFile, FTI_Exec->meta[0].ckptFile, FTI_BUFS);
    for (i = 0; i < FTI_Exec->nbVar; i++) {
        FTI_Exec->meta[0].varID[i] = FTI_Data[i].id;
        FTI_Exec->meta[0].varSize[i] = FTI_Data[i].size;
    }

    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Recovers protected data to the variable pointers for FTI-FF
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @param      FTI_Data        Dataset metadata.
  @return     integer         FTI_SCES if successful.

  This function restores the data of the protected variables to the state
  of the last checkpoint. The function is called by the API function 
  'FTI_Recover'.

 **/
/*-------------------------------------------------------------------------*/
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
        snprintf(str, FTI_BUFS, "Checkpoint has %d protected variables, but FTI protects %d.",
                FTI_Exec->meta[FTI_Exec->ckptLvel].nbVar[0], FTI_Exec->nbVar);
        FTI_Print(str, FTI_WARN);
        return FTI_NREC;
    }
    //Check if sizes of protected variables matches
    int i;
    for (i = 0; i < FTI_Exec->nbVar; i++) {
        if (FTI_Data[i].size != FTI_Exec->meta[FTI_Exec->ckptLvel].varSize[i]) {
            snprintf(str, FTI_BUFS, "Cannot recover %ld bytes to protected variable (ID %d) size: %ld",
                    FTI_Exec->meta[FTI_Exec->ckptLvel].varSize[i], FTI_Exec->meta[FTI_Exec->ckptLvel].varID[i],
                    FTI_Data[i].size);
            FTI_Print(str, FTI_WARN);
            return FTI_NREC;
        }
    }
    
    if (!FTI_Exec->firstdb) {
        FTI_Print( "FTI-FF: FTIFF_Recover - No db meta information. Nothing to recover.", FTI_WARN );
        return FTI_NREC;
    }

    //Recovering from local for L4 case in FTI_Recover
    if (FTI_Exec->ckptLvel == 4) {
        snprintf(fn, FTI_BUFS, "%s/%s", FTI_Ckpt[1].dir, FTI_Exec->meta[1].ckptFile);
    }
    else {
        snprintf(fn, FTI_BUFS, "%s/%s", FTI_Ckpt[FTI_Exec->ckptLvel].dir, FTI_Exec->meta[FTI_Exec->ckptLvel].ckptFile);
    }
    
    char strerr[FTI_BUFS];
    
    // get filesize
    struct stat st;
    if (stat(fn, &st) == -1) {
        snprintf(strerr, FTI_BUFS, "FTI-FF: FTIFF_Recover - could not get stats for file: %s", fn); 
        FTI_Print(strerr, FTI_EROR);
        errno = 0;
        return FTI_NREC;
    }

    // block size for memcpy of pointer.
    long membs = 1024*1024*16; // 16 MB
    long cpybuf, cpynow, cpycnt;

    // open checkpoint file for read only
    int fd = open( fn, O_RDONLY, 0 );
    if (fd == -1) {
        snprintf( strerr, FTI_BUFS, "FTI-FF: FTIFF_Recover - could not open '%s' for reading.", fn);
        FTI_Print(strerr, FTI_EROR);
        return FTI_NREC;
    }

    // map file into memory
    char* fmmap = (char*) mmap(0, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (fmmap == MAP_FAILED) {
        snprintf( strerr, FTI_BUFS, "FTI-FF: FTIFF_Recover - could not map '%s' to memory.", fn);
        FTI_Print(strerr, FTI_EROR);
        close(fd);
        return FTI_NREC;
    }

    // file is mapped, we can close it.
    close(fd);

    FTIFF_db *currentdb;
    FTIFF_dbvar *currentdbvar = NULL;
    char *destptr, *srcptr;
    int dbvar_idx, dbcounter=0;

    // MD5 context for checksum of data chunks
    MD5_CTX mdContext;
    unsigned char hash[MD5_DIGEST_LENGTH];

    int isnextdb;

    currentdb = FTI_Exec->firstdb;

    do {

        isnextdb = 0;

        for(dbvar_idx=0;dbvar_idx<currentdb->numvars;dbvar_idx++) {
            
            currentdbvar = &(currentdb->dbvars[dbvar_idx]);
            
            if(!(currentdbvar->hascontent)) {
                continue;
            }
            // get source and destination pointer
            destptr = (char*) FTI_Data[currentdbvar->idx].ptr + currentdbvar->dptr;
            
            snprintf(str, FTI_BUFS, "[var-id:%d|cont-id:%d] destptr: %p\n", currentdbvar->id, currentdbvar->containerid, (void*) destptr);
            FTI_Print(str, FTI_INFO);

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
            snprintf(str, FTI_BUFS, "FTI-FF: FTIFF_Recover -  dataBlock:%i/dataBlockVar%i id: %i, idx: %i"
                    ", destptr: %ld, fptr: %ld, chunksize: %ld, "
                    "base_ptr: 0x%" PRIxPTR " ptr_pos: 0x%" PRIxPTR ".", 
                    dbcounter, dbvar_idx,  
                    currentdbvar->id, currentdbvar->idx, currentdbvar->dptr,
                    currentdbvar->fptr, currentdbvar->chunksize,
                    (uintptr_t)FTI_Data[currentdbvar->idx].ptr, (uintptr_t)destptr);
            FTI_Print(str, FTI_DBUG);

            MD5_Final( hash, &mdContext );

            // JUST TESTING - print checksum current dataset.
            char checkSum[MD5_DIGEST_STRING_LENGTH];
            int ii = 0, i;
            for(i = 0; i < MD5_DIGEST_LENGTH; i++) {
                sprintf(&checkSum[ii], "%02x", hash[i]);
                ii += 2;
            }
            snprintf(str, FTI_BUFS, "dataset hash id: %d -> %s", currentdbvar->id, checkSum);
            FTI_Print(str, FTI_INFO);

            if ( memcmp( currentdbvar->hash, hash, MD5_DIGEST_LENGTH ) != 0 ) {
                snprintf( strerr, FTI_BUFS, "FTI-FF: FTIFF_Recover - dataset with id:%i|cnt-id:%d has been corrupted! Discard recovery.", currentdbvar->id, currentdbvar->containerid);
                FTI_Print(strerr, FTI_WARN);
                if ( munmap( fmmap, st.st_size ) == -1 ) {
                    FTI_Print("FTIFF: FTIFF_Recover - unable to unmap memory", FTI_EROR);
                    errno = 0;
                }
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
        FTI_Print("FTIFF: FTIFF_Recover - unable to unmap memory", FTI_EROR);
        errno = 0;
        return FTI_NREC;
    }
   
    FTI_Exec->reco = 0;

    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Recovers protected data to the variable pointer with id
  @param      id              Id of protected variable.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Data        Dataset metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @return     integer         FTI_SCES if successful.

  This function restores the data to the protected variable with given id 
  as it was checkpointed during the last checkpoint. 
  The function is called by the API function 'FTI_RecoverVar'.

 **/
/*-------------------------------------------------------------------------*/
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
                snprintf(str, FTI_BUFS, "Cannot recover %ld bytes to protected variable (ID %d) size: %ld",
                        FTI_Exec->meta[FTI_Exec->ckptLvel].varSize[i], FTI_Exec->meta[FTI_Exec->ckptLvel].varID[i],
                        FTI_Data[i].size);
                FTI_Print(str, FTI_WARN);
                return FTI_NREC;
            }
        }
    }

    char fn[FTI_BUFS]; //Path to the checkpoint file

    if (!FTI_Exec->firstdb) {
        FTI_Print( "FTIFF: FTIFF_RecoverVar - No db meta information. Nothing to recover.", FTI_WARN );
        return FTI_NREC;
    }

    //Recovering from local for L4 case in FTI_Recover
    if (FTI_Exec->ckptLvel == 4) {
        snprintf(fn, FTI_BUFS, "%s/%s", FTI_Ckpt[1].dir, FTI_Exec->meta[1].ckptFile);
    }
    else {
        snprintf(fn, FTI_BUFS, "%s/%s", FTI_Ckpt[FTI_Exec->ckptLvel].dir, FTI_Exec->meta[FTI_Exec->ckptLvel].ckptFile);
    }

    char str[FTI_BUFS], strerr[FTI_BUFS];

    snprintf(str, FTI_BUFS, "Trying to load FTI checkpoint file (%s)...", fn);
    FTI_Print(str, FTI_DBUG);

    // get filesize
    struct stat st;
    if (stat(fn, &st) == -1) {
        snprintf(strerr, FTI_BUFS, "FTI-FF: FTIFF_RecoverVar - could not get stats for file: %s", fn); 
        FTI_Print(strerr, FTI_EROR);
        errno = 0;
        return FTI_NREC;
    }

    // block size for memcpy of pointer.
    long membs = 1024*1024*16; // 16 MB
    long cpybuf, cpynow, cpycnt;

    // open checkpoint file for read only
    int fd = open( fn, O_RDONLY, 0 );
    if (fd == -1) {
        snprintf( strerr, FTI_BUFS, "FTIFF: FTIFF_RecoverVar - could not open '%s' for reading.", fn);
        FTI_Print(strerr, FTI_EROR);
        return FTI_NREC;
    }

    // map file into memory
    char* fmmap = (char*) mmap(0, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (fmmap == MAP_FAILED) {
        snprintf( strerr, FTI_BUFS, "FTIFF: FTIFF_RecoverVar - could not map '%s' to memory.", fn);
        FTI_Print(strerr, FTI_EROR);
        close(fd);
        return FTI_NREC;
    }

    // file is mapped, we can close it.
    close(fd);

    FTIFF_db *currentdb;
    FTIFF_dbvar *currentdbvar = NULL;
    char *destptr, *srcptr;
    int dbvar_idx, dbcounter=0;

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
                snprintf(str, FTI_BUFS, "FTIFF: FTIFF_RecoverVar -  dataBlock:%i/dataBlockVar%i id: %i, idx: %i"
                        ", destptr: %ld, fptr: %ld, chunksize: %ld, "
                        "base_ptr: 0x%" PRIxPTR " ptr_pos: 0x%" PRIxPTR ".", 
                        dbcounter, dbvar_idx,  
                        currentdbvar->id, currentdbvar->idx, currentdbvar->dptr,
                        currentdbvar->fptr, currentdbvar->chunksize,
                        (uintptr_t)FTI_Data[currentdbvar->idx].ptr, (uintptr_t)destptr);
                FTI_Print(str, FTI_DBUG);

                MD5_Final( hash, &mdContext );

                if ( memcmp( currentdbvar->hash, hash, MD5_DIGEST_LENGTH ) != 0 ) {
                    snprintf( strerr, FTI_BUFS, "FTIFF: FTIFF_RecoverVar - dataset with id:%i has been corrupted! Discard recovery.", currentdbvar->id);
                    FTI_Print(strerr, FTI_WARN);
                    if ( munmap( fmmap, st.st_size ) == -1 ) {
                        FTI_Print("FTIFF: FTIFF_RecoverVar - unable to unmap memory", FTI_EROR);
                        errno = 0;
                    }
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
        FTI_Print("FTIFF: FTIFF_RecoverVar - unable to unmap memory", FTI_EROR);
        errno = 0;
    }

    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Init of FTI-FF L1 recovery
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Topo        Topology metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @return     integer         FTI_SCES if successful.

  This function initializes the L1 checkpoint recovery. It checks for 
  erasures and loads the required meta data. 
 **/
/*-------------------------------------------------------------------------*/
int FTIFF_CheckL1RecoverInit( FTIT_execution* FTI_Exec, FTIT_topology* FTI_Topo, 
        FTIT_checkpoint* FTI_Ckpt )
{
    char str[FTI_BUFS], tmpfn[FTI_BUFS], strerr[FTI_BUFS];
    int fexist = 0, fileTarget, ckptID, fcount;
    
    struct dirent *entry;
    struct stat ckptFS;
    struct stat ckptDIR;
    
    // File meta-data
    FTIFF_metaInfo *FTIFFMeta = calloc( 1, sizeof(FTIFF_metaInfo) );
    if ( FTIFFMeta == NULL ) {
        snprintf( strerr, FTI_BUFS, "FTI-FF: L1RecoverInit - failed to allocate %ld bytes for 'FTIFFMeta'", sizeof(FTIFF_metaInfo));
        FTI_Print(strerr, FTI_EROR);
        errno = 0;
        goto GATHER_L1INFO;
    }

    MD5_CTX mdContext;
    
    // check if L1 ckpt directory exists
    bool L1CkptDirExists = false;
    if ( stat( FTI_Ckpt[1].dir, &ckptDIR ) == 0 ) {
        if ( S_ISDIR( ckptDIR.st_mode ) != 0 ) {
            L1CkptDirExists = true;
        } else {
            snprintf(strerr, FTI_BUFS, "FTI-FF: L1RecoverInit - (%s) is not a directory.", FTI_Ckpt[1].dir);
            FTI_Print(strerr, FTI_WARN);
            free(FTIFFMeta);
            goto GATHER_L1INFO;
        }
    }

    if(L1CkptDirExists) {
        
        DIR *L1CkptDir = opendir( FTI_Ckpt[1].dir );

        if (L1CkptDir == NULL) {
            snprintf(strerr, FTI_BUFS, "FTI-FF: L1RecoveryInit - checkpoint directory (%s) could not be accessed.", FTI_Ckpt[1].dir);
            FTI_Print(strerr, FTI_EROR);
            errno = 0;
            free(FTIFFMeta);
            goto GATHER_L1INFO;
        }

        while((entry = readdir(L1CkptDir)) != NULL) {
            
            if(strcmp(entry->d_name,".") && strcmp(entry->d_name,"..")) { 
                snprintf(str, FTI_BUFS, "FTI-FF: L1RecoveryInit - found file with name: %s", entry->d_name);
                FTI_Print(str, FTI_DBUG);
                sscanf(entry->d_name, "Ckpt%d-Rank%d.fti", &ckptID, &fileTarget );
                
                // If ranks coincide
                if( fileTarget == FTI_Topo->myRank ) {
                    snprintf(tmpfn, FTI_BUFS, "%s/%s", FTI_Ckpt[1].dir, entry->d_name);
                    
                    if ( stat(tmpfn, &ckptFS) == -1 ) {
                        snprintf( strerr, FTI_BUFS, "FTI-FF: L1RecoveryInit - Problem with stats on file %s", tmpfn );
                        FTI_Print( strerr, FTI_EROR );
                        errno = 0;
                        free(FTIFFMeta);
                        closedir(L1CkptDir);
                        goto GATHER_L1INFO;
                    }

                    if ( S_ISREG(ckptFS.st_mode) == 0 ) {
                        snprintf( strerr, FTI_BUFS, "FTI-FF: L1RecoveryInit - %s is not a regular file", tmpfn );
                        FTI_Print( strerr, FTI_WARN );
                        free(FTIFFMeta);
                        closedir(L1CkptDir);
                        goto GATHER_L1INFO;
                    }
                    
                    // Check for reasonable file size. At least has to contain file meta-data
                    if ( ckptFS.st_size > sizeof(FTIFF_metaInfo ) ) {
                        
                        int fd = open(tmpfn, O_RDONLY);
                        if (fd == -1) {
                            snprintf( strerr, FTI_BUFS, "FTI-FF: L1RecoveryInit - could not open '%s' for reading.", tmpfn);
                            FTI_Print(strerr, FTI_EROR);
                            errno = 0;
                            free(FTIFFMeta);
                            closedir(L1CkptDir);
                            goto GATHER_L1INFO;
                        }

                        if ( lseek(fd, 0, SEEK_SET) == -1 ) {
                            snprintf(strerr, FTI_BUFS, "FTI-FF: L1RecoveryInit - could not seek in file: %s", tmpfn);
                            FTI_Print(strerr, FTI_EROR);
                            errno = 0;
                            free(FTIFFMeta);
                            closedir(L1CkptDir);
                            close(fd);
                            goto GATHER_L1INFO;
                        }
                        
                        // Read in file meta-data
                        if ( 
                                ( read( fd, FTIFFMeta->checksum, MD5_DIGEST_STRING_LENGTH ) == -1 )     ||
                                ( read( fd, FTIFFMeta->myHash, MD5_DIGEST_LENGTH ) == -1 )              ||
                                ( read( fd, &(FTIFFMeta->ckptSize), sizeof(long) ) == -1 )              ||
                                ( read( fd, &(FTIFFMeta->fs), sizeof(long) ) == -1 )                    ||
                                ( read( fd, &(FTIFFMeta->maxFs), sizeof(long) ) == -1 )                 ||
                                ( read( fd, &(FTIFFMeta->ptFs), sizeof(long) ) == -1 )                  ||
                                ( read( fd, &(FTIFFMeta->timestamp), sizeof(long) ) == -1 )                                     
                            ) 
                        {
                            snprintf(strerr, FTI_BUFS, "FTI-FF: L1RecoveryInit - Failed to request file meta data from: %s", tmpfn);
                            FTI_Print(strerr, FTI_EROR);
                            errno=0;
                            free(FTIFFMeta);
                            closedir(L1CkptDir);
                            close(fd);
                            goto GATHER_L1INFO;
                        }

                        unsigned char hash[MD5_DIGEST_LENGTH];
                        FTIFF_GetHashMetaInfo( hash, FTIFFMeta );
                        
                        // Check if hash of file meta-data is consistent
                        if ( memcmp( FTIFFMeta->myHash, hash, MD5_DIGEST_LENGTH ) == 0 ) {
                            
                            unsigned char hash[MD5_DIGEST_LENGTH];
                            FTIFF_GetFileChecksum( FTIFFMeta, FTI_Ckpt, fd, hash ); 
                            
                            int i;
                            char checksum[MD5_DIGEST_STRING_LENGTH];
                            int ii = 0;
                            for(i = 0; i < MD5_DIGEST_LENGTH; i++) {
                                sprintf(&checksum[ii], "%02x", hash[i]);
                                ii += 2;
                            }
                            
                            //if ( 1 ) {
                            if ( strcmp( checksum, FTIFFMeta->checksum ) == 0 ) {
                                FTI_Exec->meta[1].fs[0] = ckptFS.st_size;    
                                FTI_Exec->ckptID = ckptID;
                                strncpy(FTI_Exec->meta[1].ckptFile, entry->d_name, NAME_MAX);
                                fexist = 1;
                            
                            } 
                            else {
                                char str[FTI_BUFS];
                                snprintf(str, FTI_BUFS, "Checksum do not match. \"%s\" file is corrupted. %s != %s",
                                        entry->d_name, checksum, FTIFFMeta->checksum);
                                FTI_Print(str, FTI_WARN);
                                close(fd);
                                free(FTIFFMeta);
                                closedir(L1CkptDir);
                                goto GATHER_L1INFO;
                            }
                        } else {
                            char str[FTI_BUFS];
                            snprintf(str, FTI_BUFS, "Metadata in file \"%s\" is corrupted.",entry->d_name);
                            FTI_Print(str, FTI_WARN);
                            closedir(L1CkptDir);
                            close(fd);
                            goto GATHER_L1INFO;
                        }
                        close(fd);
                        break;
                    } else {
                        char str[FTI_BUFS];
                        snprintf(str, FTI_BUFS, "size %lu of file \"%s\" is smaller then file meta data struct size.", ckptFS.st_size, entry->d_name);
                        FTI_Print(str, FTI_WARN);
                        closedir(L1CkptDir);
                        goto GATHER_L1INFO;
                    }
                }
            }
        }
        closedir(L1CkptDir);
    }

    // collect info from other ranks if file found
    MPI_Allreduce(&fexist, &fcount, 1, MPI_INT, MPI_SUM, FTI_COMM_WORLD);
    int fneeded = FTI_Topo->nbNodes*FTI_Topo->nbApprocs;
    
    int res = (fcount == fneeded) ? FTI_SCES : FTI_NSCS;
    
    free(FTIFFMeta);

    return res;

GATHER_L1INFO:
    
    fexist = 0;
    MPI_Allreduce(&fexist, &fcount, 1, MPI_INT, MPI_SUM, FTI_COMM_WORLD);

    return FTI_NSCS;

}

/*-------------------------------------------------------------------------*/
/**
  @brief      Init of FTI-FF L2 recovery
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Topo        Topology metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @param      exists          Array with info of erased files
  @return     integer         FTI_SCES if successful.

  This function initializes the L2 checkpoint recovery. It checks for 
  erasures and loads the required meta data. 
 **/
/*-------------------------------------------------------------------------*/
int FTIFF_CheckL2RecoverInit( FTIT_execution* FTI_Exec, FTIT_topology* FTI_Topo, 
        FTIT_checkpoint* FTI_Ckpt, int *exists)
{
    char dbgstr[FTI_BUFS], strerr[FTI_BUFS];

    enum {
        LEFT_FILE,  // ckpt file of left partner (on left node)
        MY_FILE,    // my ckpt file (on my node)
        MY_COPY,    // copy of my ckpt file (on right node)
        LEFT_COPY   // copy of ckpt file of my left partner (on my node)
    };

    // determine app rank representation of group ranks left and right
    MPI_Group nodesGroup;
    MPI_Comm_group(FTI_Exec->groupComm, &nodesGroup);
    MPI_Group appProcsGroup;
    MPI_Comm_group(FTI_COMM_WORLD, &appProcsGroup);
    int baseRanks[] = { FTI_Topo->left, FTI_Topo->right };
    int projRanks[2];
    MPI_Group_translate_ranks( nodesGroup, 2, baseRanks, appProcsGroup, projRanks );
    int leftIdx = projRanks[0], rightIdx = projRanks[1];

    int appCommSize = FTI_Topo->nbNodes*FTI_Topo->nbApprocs;
    int fneeded = appCommSize;

    MPI_Group_free(&nodesGroup);
    MPI_Group_free(&appProcsGroup);

    FTIFF_L2Info* appProcsMetaInfo = calloc( appCommSize, sizeof(FTIFF_L2Info) );
    
    FTIFF_L2Info _myMetaInfo;
    FTIFF_L2Info* myMetaInfo = (FTIFF_L2Info*) memset(&_myMetaInfo, 0x0, sizeof(FTIFF_L2Info)); 

    myMetaInfo->rightIdx = rightIdx;

    MD5_CTX mdContext;

    char str[FTI_BUFS], tmpfn[FTI_BUFS];
    int fileTarget, ckptID = -1, fcount = 0, match;
    struct dirent *entry;
    struct stat ckptFS, ckptDIR;

    FTIFF_metaInfo _FTIFFMeta;
    FTIFF_metaInfo *FTIFFMeta = (FTIFF_metaInfo*) memset(&_FTIFFMeta, 0x0, sizeof(FTIFF_metaInfo)); 
    
    // check if L2 ckpt directory exists
    bool L2CkptDirExists = false;
    if ( stat( FTI_Ckpt[2].dir, &ckptDIR ) == 0 ) {
        if ( S_ISDIR( ckptDIR.st_mode ) != 0 ) {
            L2CkptDirExists = true;
        } else {
            snprintf(strerr, FTI_BUFS, "FTI-FF: L2RecoverInit - (%s) is not a directory.", FTI_Ckpt[2].dir);
            FTI_Print(strerr, FTI_WARN);
            goto GATHER_L2INFO;
        }
    }
 
    if(L2CkptDirExists) {
        
        int tmpCkptID;
        
        DIR *L2CkptDir = opendir( FTI_Ckpt[2].dir );

        if (L2CkptDir == NULL) {
            snprintf(strerr, FTI_BUFS, "FTI-FF: L2RecoveryInit - checkpoint directory (%s) could not be accessed.", FTI_Ckpt[2].dir);
            FTI_Print(strerr, FTI_EROR);
            errno = 0;
            goto GATHER_L2INFO;
        }
 
        while((entry = readdir(L2CkptDir)) != NULL) {
            
            if(strcmp(entry->d_name,".") && strcmp(entry->d_name,"..")) { 
                
                snprintf(str, FTI_BUFS, "FTI-FF: L2RecoveryInit - found file with name: %s", entry->d_name);
                FTI_Print(str, FTI_DBUG);
                tmpCkptID = ckptID;
                match = sscanf(entry->d_name, "Ckpt%d-Rank%d.fti", &ckptID, &fileTarget );
                
                if( match == 2 && fileTarget == FTI_Topo->myRank ) {
                    
                    snprintf(tmpfn, FTI_BUFS, "%s/%s", FTI_Ckpt[2].dir, entry->d_name);
                    
                    if ( stat(tmpfn, &ckptFS) == -1 ) {
                        snprintf( strerr, FTI_BUFS, "FTI-FF: L2RecoveryInit - Problem with stats on file %s", tmpfn );
                        FTI_Print( strerr, FTI_EROR );
                        errno = 0;
                        closedir(L2CkptDir);
                        goto GATHER_L2INFO;
                    }

                    if ( S_ISREG(ckptFS.st_mode) == 0 ) {
                        snprintf( strerr, FTI_BUFS, "FTI-FF: L2RecoveryInit - %s is not a regular file", tmpfn );
                        FTI_Print( strerr, FTI_WARN );
                        closedir(L2CkptDir);
                        goto GATHER_L2INFO;
                    }
                       
                    // check if regular file and of reasonable size (at least must contain meta info)
                    if ( ckptFS.st_size > FTI_filemetastructsize ) {
                        int fd = open(tmpfn, O_RDONLY);
                        if (fd == -1) {
                            snprintf( strerr, FTI_BUFS, "FTI-FF: L2RecoveryInit - could not open '%s' for reading.", tmpfn);
                            FTI_Print(strerr, FTI_EROR);
                            errno = 0;
                            closedir(L2CkptDir);
                            goto GATHER_L2INFO;
                        }
 
                        if ( lseek(fd, 0, SEEK_SET) == -1 ) {
                            snprintf(strerr, FTI_BUFS, "FTI-FF: L2RecoveryInit - could not seek in file: %s", tmpfn);
                            FTI_Print(strerr, FTI_EROR);
                            errno = 0;
                            closedir(L2CkptDir);
                            close(fd);
                            goto GATHER_L2INFO;
                        }

                        if ( 
                                ( read( fd, FTIFFMeta->checksum, MD5_DIGEST_STRING_LENGTH ) == -1 )     ||
                                ( read( fd, FTIFFMeta->myHash, MD5_DIGEST_LENGTH ) == -1 )              ||
                                ( read( fd, &(FTIFFMeta->ckptSize), sizeof(long) ) == -1 )              ||
                                ( read( fd, &(FTIFFMeta->fs), sizeof(long) ) == -1 )                    ||
                                ( read( fd, &(FTIFFMeta->maxFs), sizeof(long) ) == -1 )                 ||
                                ( read( fd, &(FTIFFMeta->ptFs), sizeof(long) ) == -1 )                  ||
                                ( read( fd, &(FTIFFMeta->timestamp), sizeof(long) ) == -1 )                                     
                            ) 
                        {
                            snprintf(strerr, FTI_BUFS, "FTI-FF: L2RecoveryInit - Failed to request file meta data from: %s", tmpfn);
                            FTI_Print(strerr, FTI_EROR);
                            errno=0;
                            closedir(L2CkptDir);
                            close(fd);
                            goto GATHER_L2INFO;
                        }

                        unsigned char hash[MD5_DIGEST_LENGTH];
                        FTIFF_GetHashMetaInfo( hash, FTIFFMeta );
                        
                        if ( memcmp( FTIFFMeta->myHash, hash, MD5_DIGEST_LENGTH ) == 0 ) {

                            unsigned char hash[MD5_DIGEST_LENGTH];
                            FTIFF_GetFileChecksum( FTIFFMeta, FTI_Ckpt, fd, hash ); 
                            
                            int i;
                            char checksum[MD5_DIGEST_STRING_LENGTH];
                            int ii = 0;
                            for(i = 0; i < MD5_DIGEST_LENGTH; i++) {
                                sprintf(&checksum[ii], "%02x", hash[i]);
                                ii += 2;
                            }

                            if ( strcmp( checksum, FTIFFMeta->checksum ) == 0 ) {
                                myMetaInfo->fs = FTIFFMeta->fs;    
                                myMetaInfo->ckptID = ckptID;    
                                myMetaInfo->FileExists = 1;
                            } else {
                                char str[FTI_BUFS];
                                snprintf(str, FTI_BUFS, "Checksum do not match. \"%s\" file is corrupted. %s != %s",
                                        entry->d_name, checksum, FTIFFMeta->checksum);
                                FTI_Print(str, FTI_WARN);
                                close(fd);
                                goto GATHER_L2INFO;
                            }
                        } else {
                            char str[FTI_BUFS];
                            snprintf(str, FTI_BUFS, "Metadata in file \"%s\" is corrupted.",entry->d_name);
                            FTI_Print(str, FTI_WARN);
                            closedir(L2CkptDir);
                            close(fd);
                            goto GATHER_L2INFO;
                        }
                        close(fd);
                    } else {
                        char str[FTI_BUFS];
                        snprintf(str, FTI_BUFS, "size %lu of file \"%s\" is smaller then file meta data struct size.", ckptFS.st_size, entry->d_name);
                        FTI_Print(str, FTI_WARN);
                        closedir(L2CkptDir);
                        goto GATHER_L2INFO;
                    }
                } else {
                    ckptID = tmpCkptID;
                }

                tmpCkptID = ckptID;
                
                match = sscanf(entry->d_name, "Ckpt%d-Pcof%d.fti", &ckptID, &fileTarget );
                if( match == 2 && fileTarget == FTI_Topo->myRank ) {
                    
                    snprintf(tmpfn, FTI_BUFS, "%s/%s", FTI_Ckpt[2].dir, entry->d_name);
                    
                    if ( stat(tmpfn, &ckptFS) == -1 ) {
                        snprintf( strerr, FTI_BUFS, "FTI-FF: L2RecoveryInit - Problem with stats on file %s", tmpfn );
                        FTI_Print( strerr, FTI_EROR );
                        errno = 0;
                        closedir(L2CkptDir);
                        goto GATHER_L2INFO;
                    }

                    if ( S_ISREG(ckptFS.st_mode) == 0 ) {
                        snprintf( strerr, FTI_BUFS, "FTI-FF: L2RecoveryInit - %s is not a regular file", tmpfn );
                        FTI_Print( strerr, FTI_WARN );
                        closedir(L2CkptDir);
                        goto GATHER_L2INFO;
                    }
 
                    // check if regular file and of reasonable size (at least must contain meta info)
                    if ( ckptFS.st_size > FTI_filemetastructsize ) {
                        int fd = open(tmpfn, O_RDONLY);
                        if (fd == -1) {
                            snprintf( strerr, FTI_BUFS, "FTI-FF: L2RecoveryInit - could not open '%s' for reading.", tmpfn);
                            FTI_Print(strerr, FTI_EROR);
                            errno = 0;
                            closedir(L2CkptDir);
                            goto GATHER_L2INFO;
                        }
 
                        if ( lseek(fd, 0, SEEK_SET) == -1 ) {
                            snprintf(strerr, FTI_BUFS, "FTI-FF: L2RecoveryInit - could not seek in file: %s", tmpfn);
                            FTI_Print(strerr, FTI_EROR);
                            close(fd);
                            errno = 0;
                            closedir(L2CkptDir);
                            close(fd);
                            goto GATHER_L2INFO;
                        }

                        if ( 
                                ( read( fd, FTIFFMeta->checksum, MD5_DIGEST_STRING_LENGTH ) == -1 )     ||
                                ( read( fd, FTIFFMeta->myHash, MD5_DIGEST_LENGTH ) == -1 )              ||
                                ( read( fd, &(FTIFFMeta->ckptSize), sizeof(long) ) == -1 )              ||
                                ( read( fd, &(FTIFFMeta->fs), sizeof(long) ) == -1 )                    ||
                                ( read( fd, &(FTIFFMeta->maxFs), sizeof(long) ) == -1 )                 ||
                                ( read( fd, &(FTIFFMeta->ptFs), sizeof(long) ) == -1 )                  ||
                                ( read( fd, &(FTIFFMeta->timestamp), sizeof(long) ) == -1 )                                     
                            ) 
                        {
                            snprintf(strerr, FTI_BUFS, "FTI-FF: L2RecoveryInit - Failed to request file meta data from: %s", tmpfn);
                            FTI_Print(strerr, FTI_EROR);
                            errno=0;
                            closedir(L2CkptDir);
                            close(fd);
                            goto GATHER_L2INFO;
                        }
                        
                        unsigned char hash[MD5_DIGEST_LENGTH];
                        FTIFF_GetHashMetaInfo( hash, FTIFFMeta );
                        
                        if ( memcmp( FTIFFMeta->myHash, hash, MD5_DIGEST_LENGTH ) == 0 ) {
                            unsigned char hash[MD5_DIGEST_LENGTH];
                            
                            FTIFF_GetFileChecksum( FTIFFMeta, FTI_Ckpt, fd, hash ); 
                            
                            int i;
                            char checksum[MD5_DIGEST_STRING_LENGTH];
                            int ii = 0;
                            for(i = 0; i < MD5_DIGEST_LENGTH; i++) {
                                sprintf(&checksum[ii], "%02x", hash[i]);
                                ii += 2;
                            }
                            if ( strcmp( checksum, FTIFFMeta->checksum ) == 0 ) {
                                myMetaInfo->pfs = FTIFFMeta->fs;    
                                myMetaInfo->ckptID = ckptID;    
                                myMetaInfo->CopyExists = 1;
                            } else {
                                char str[FTI_BUFS];
                                snprintf(str, FTI_BUFS, "Checksum do not match. \"%s\" file is corrupted. %s != %s",
                                        entry->d_name, checksum, FTIFFMeta->checksum);
                                FTI_Print(str, FTI_WARN);
                                closedir(L2CkptDir);
                                close(fd);
                                goto GATHER_L2INFO;
                            }
                        } else {
                            char str[FTI_BUFS];
                            snprintf(str, FTI_BUFS, "Metadata in file \"%s\" is corrupted.",entry->d_name);
                            FTI_Print(str, FTI_WARN);
                            closedir(L2CkptDir);
                            close(fd);
                            goto GATHER_L2INFO;
                        }
                        close(fd);
                    } else {
                        char str[FTI_BUFS];
                        snprintf(str, FTI_BUFS, "size %lu of file \"%s\" is smaller then file meta data struct size.", ckptFS.st_size, entry->d_name);
                        FTI_Print(str, FTI_WARN);
                        closedir(L2CkptDir);
                        goto GATHER_L2INFO;
                    }
           
                } else {
                    ckptID = tmpCkptID;
                }
            }
            if (myMetaInfo->FileExists && myMetaInfo->CopyExists) {
                break;
            }
        }
        closedir(L2CkptDir);
    }
    
GATHER_L2INFO:

    if(!(myMetaInfo->FileExists) && !(myMetaInfo->CopyExists)) {
        myMetaInfo->ckptID = -1;
    }

    // gather meta info
    MPI_Allgather( myMetaInfo, 1, FTIFF_MpiTypes[FTIFF_L2_INFO], appProcsMetaInfo, 1, FTIFF_MpiTypes[FTIFF_L2_INFO], FTI_COMM_WORLD);

    exists[LEFT_FILE] = appProcsMetaInfo[leftIdx].FileExists;
    exists[MY_FILE] = appProcsMetaInfo[FTI_Topo->splitRank].FileExists;
    exists[MY_COPY] = appProcsMetaInfo[rightIdx].CopyExists;
    exists[LEFT_COPY] = appProcsMetaInfo[FTI_Topo->splitRank].CopyExists;

    // debug Info
    snprintf(dbgstr, FTI_BUFS, "FTI-FF - L2Recovery::FileCheck - CkptFile: %i, CkptCopy: %i", 
            myMetaInfo->FileExists, myMetaInfo->CopyExists);
    FTI_Print(dbgstr, FTI_DBUG);

    // check if recovery possible
    int i, saneCkptID = 0;
    ckptID = 0;
    for(i=0; i<appCommSize; i++) { 
        fcount += ( appProcsMetaInfo[i].FileExists || appProcsMetaInfo[appProcsMetaInfo[i].rightIdx].CopyExists ) ? 1 : 0;
        if (appProcsMetaInfo[i].ckptID > 0) {
            saneCkptID++;
            ckptID += appProcsMetaInfo[i].ckptID;
        }
    }
    int res = (fcount == fneeded) ? FTI_SCES : FTI_NSCS;

    if (res == FTI_SCES) {
        FTI_Exec->ckptID = ckptID/saneCkptID;
        if (myMetaInfo->FileExists) {
            FTI_Exec->meta[2].fs[0] = myMetaInfo->fs;    
        } else {
            FTI_Exec->meta[2].fs[0] = appProcsMetaInfo[rightIdx].pfs;    
        }
        if (myMetaInfo->CopyExists) {
            FTI_Exec->meta[2].pfs[0] = myMetaInfo->pfs;    
        } else {
            FTI_Exec->meta[2].pfs[0] = appProcsMetaInfo[leftIdx].fs;    
        }
    }
    snprintf(dbgstr, FTI_BUFS, "FTI-FF: L2-Recovery - rank: %i, left: %i, right: %i, fs: %ld, pfs: %ld, ckptID: %i",
            FTI_Topo->myRank, leftIdx, rightIdx, FTI_Exec->meta[2].fs[0], FTI_Exec->meta[2].pfs[0], FTI_Exec->ckptID);
    FTI_Print(dbgstr, FTI_DBUG);

    snprintf(FTI_Exec->meta[2].ckptFile, FTI_BUFS, "Ckpt%d-Rank%d.fti", FTI_Exec->ckptID, FTI_Topo->myRank);

    free(appProcsMetaInfo);

    return res;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Init of FTI-FF L3 recovery
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Topo        Topology metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @param      erased          Array with info of erased files
  @return     integer         FTI_SCES if successful.

  This function initializes the L3 checkpoint recovery. It checks for 
  erasures and loads the required meta data. 
 **/
/*-------------------------------------------------------------------------*/
int FTIFF_CheckL3RecoverInit( FTIT_execution* FTI_Exec, FTIT_topology* FTI_Topo, 
        FTIT_checkpoint* FTI_Ckpt, int* erased)
{

    FTIFF_L3Info *groupInfo = calloc( FTI_Topo->groupSize, sizeof(FTIFF_L3Info) );
    FTIFF_L3Info _myInfo;
    FTIFF_L3Info *myInfo = (FTIFF_L3Info*) memset(&_myInfo, 0x0, sizeof(FTIFF_L3Info));

    MD5_CTX mdContext;

    char str[FTI_BUFS], strerr[FTI_BUFS], tmpfn[FTI_BUFS];
    int fileTarget, ckptID = -1, match;
    struct dirent *entry;
    struct stat ckptFS, ckptDIR;

    FTIFF_metaInfo *FTIFFMeta = calloc( 1, sizeof(FTIFF_metaInfo) );
    
    // check if L3 ckpt directory exists
    bool L3CkptDirExists = false;
    if ( stat( FTI_Ckpt[3].dir, &ckptDIR ) == 0 ) {
        if ( S_ISDIR( ckptDIR.st_mode ) != 0 ) {
            L3CkptDirExists = true;
        } else {
            snprintf(strerr, FTI_BUFS, "FTI-FF: L3RecoverInit - (%s) is not a directory.", FTI_Ckpt[3].dir);
            FTI_Print(strerr, FTI_WARN);
            goto GATHER_L3INFO;
        }
    }
 
    if(L3CkptDirExists) {
        
        int tmpCkptID;
        
        DIR *L3CkptDir = opendir( FTI_Ckpt[3].dir );
        
        if (L3CkptDir == NULL) {
            snprintf(strerr, FTI_BUFS, "FTI-FF: L3RecoveryInit - checkpoint directory (%s) could not be accessed.", FTI_Ckpt[3].dir);
            FTI_Print(strerr, FTI_EROR);
            errno = 0;
            goto GATHER_L3INFO;
        }
 
        while((entry = readdir(L3CkptDir)) != NULL) {
            
            if(strcmp(entry->d_name,".") && strcmp(entry->d_name,"..")) { 
                
                snprintf(str, FTI_BUFS, "FTI-FF: L3RecoveryInit - found file with name: %s", entry->d_name);
                FTI_Print(str, FTI_DBUG);
                tmpCkptID = ckptID;
                match = sscanf(entry->d_name, "Ckpt%d-Rank%d.fti", &ckptID, &fileTarget );
                
                if( match == 2 && fileTarget == FTI_Topo->myRank ) {
                    
                    snprintf(tmpfn, FTI_BUFS, "%s/%s", FTI_Ckpt[3].dir, entry->d_name);
                    
                    if ( stat(tmpfn, &ckptFS) == -1 ) {
                        snprintf( strerr, FTI_BUFS, "FTI-FF: L3RecoveryInit - Problem with stats on file %s", tmpfn );
                        FTI_Print( strerr, FTI_EROR );
                        errno = 0;
                        closedir(L3CkptDir);
                        goto GATHER_L3INFO;
                    }

                    if ( S_ISREG(ckptFS.st_mode) == 0 ) {
                        snprintf( strerr, FTI_BUFS, "FTI-FF: L3RecoveryInit - %s is not a regular file", tmpfn );
                        FTI_Print( strerr, FTI_WARN );
                        closedir(L3CkptDir);
                        goto GATHER_L3INFO;
                    }
                       
                    // check if regular file and of reasonable size (at least must contain meta info)
                    if ( ckptFS.st_size > FTI_filemetastructsize ) {
                        int fd = open(tmpfn, O_RDONLY);
                        if (fd == -1) {
                            snprintf( strerr, FTI_BUFS, "FTI-FF: L3RecoveryInit - could not open '%s' for reading.", tmpfn);
                            FTI_Print(strerr, FTI_EROR);
                            errno = 0;
                            closedir(L3CkptDir);
                            goto GATHER_L3INFO;
                        }
 
                        if ( lseek(fd, 0, SEEK_SET) == -1 ) {
                            snprintf(strerr, FTI_BUFS, "FTI-FF: L3RecoveryInit - could not seek in file: %s", tmpfn);
                            FTI_Print(strerr, FTI_EROR);
                            errno = 0;
                            closedir(L3CkptDir);
                            close(fd);
                            goto GATHER_L3INFO;
                        }

                        if ( 
                                ( read( fd, FTIFFMeta->checksum, MD5_DIGEST_STRING_LENGTH ) == -1 )     ||
                                ( read( fd, FTIFFMeta->myHash, MD5_DIGEST_LENGTH ) == -1 )              ||
                                ( read( fd, &(FTIFFMeta->ckptSize), sizeof(long) ) == -1 )              ||
                                ( read( fd, &(FTIFFMeta->fs), sizeof(long) ) == -1 )                    ||
                                ( read( fd, &(FTIFFMeta->maxFs), sizeof(long) ) == -1 )                 ||
                                ( read( fd, &(FTIFFMeta->ptFs), sizeof(long) ) == -1 )                  ||
                                ( read( fd, &(FTIFFMeta->timestamp), sizeof(long) ) == -1 )                                     
                            ) 
                        {
                            snprintf(strerr, FTI_BUFS, "FTI-FF: L3RecoveryInit - Failed to request file meta data from: %s", tmpfn);
                            FTI_Print(strerr, FTI_EROR);
                            errno=0;
                            closedir(L3CkptDir);
                            close(fd);
                            goto GATHER_L3INFO;
                        }

                        unsigned char hash[MD5_DIGEST_LENGTH];
                        FTIFF_GetHashMetaInfo( hash, FTIFFMeta );
                        
                        if ( memcmp( FTIFFMeta->myHash, hash, MD5_DIGEST_LENGTH ) == 0 ) {
                            unsigned char hash[MD5_DIGEST_LENGTH];
                            FTIFF_GetFileChecksum( FTIFFMeta, FTI_Ckpt, fd, hash ); 
                            
                            char checksum[MD5_DIGEST_STRING_LENGTH];
                            int i;
                            int ii = 0;
                            for(i = 0; i < MD5_DIGEST_LENGTH; i++) {
                                sprintf(&checksum[ii], "%02x", hash[i]);
                                ii += 2;
                            }
                            
                            if ( strcmp( checksum, FTIFFMeta->checksum ) == 0 ) {
                                myInfo->fs = FTIFFMeta->fs;    
                                myInfo->ckptID = ckptID;    
                                myInfo->FileExists = 1;
                            } else {
                                char str[FTI_BUFS];
                                snprintf(str, FTI_BUFS, "Checksum do not match. \"%s\" file is corrupted. %s != %s",
                                        entry->d_name, checksum, FTIFFMeta->checksum);
                                FTI_Print(str, FTI_WARN);
                                close(fd);
                                goto GATHER_L3INFO;
                            }
                        } else {
                            char str[FTI_BUFS];
                            snprintf(str, FTI_BUFS, "Metadata in file \"%s\" is corrupted.",entry->d_name);
                            FTI_Print(str, FTI_WARN);
                            closedir(L3CkptDir);
                            close(fd);
                            goto GATHER_L3INFO;
                        }
                        close(fd);
                    } else {
                        char str[FTI_BUFS];
                        snprintf(str, FTI_BUFS, "size %lu of file \"%s\" is smaller then file meta data struct size.", ckptFS.st_size, entry->d_name);
                        FTI_Print(str, FTI_WARN);
                        closedir(L3CkptDir);
                        goto GATHER_L3INFO;
                    }
                } else {
                    ckptID = tmpCkptID;
                }
                
                tmpCkptID = ckptID;
                
                match = sscanf(entry->d_name, "Ckpt%d-RSed%d.fti", &ckptID, &fileTarget );
                if( match == 2 && fileTarget == FTI_Topo->myRank ) {
                    
                    snprintf(tmpfn, FTI_BUFS, "%s/%s", FTI_Ckpt[3].dir, entry->d_name);
                    
                    if ( stat(tmpfn, &ckptFS) == -1 ) {
                        snprintf( strerr, FTI_BUFS, "FTI-FF: L3RecoveryInit - Problem with stats on file %s", tmpfn );
                        FTI_Print( strerr, FTI_EROR );
                        errno = 0;
                        closedir(L3CkptDir);
                        goto GATHER_L3INFO;
                    }

                    if ( S_ISREG(ckptFS.st_mode) == 0 ) {
                        snprintf( strerr, FTI_BUFS, "FTI-FF: L3RecoveryInit - %s is not a regular file", tmpfn );
                        FTI_Print( strerr, FTI_WARN );
                        closedir(L3CkptDir);
                        goto GATHER_L3INFO;
                    }
 
                    // check if regular file and of reasonable size (at least must contain meta info)
                    if ( ckptFS.st_size > FTI_filemetastructsize ) {
                        int fd = open(tmpfn, O_RDONLY);
                        if (fd == -1) {
                            snprintf( strerr, FTI_BUFS, "FTI-FF: L3RecoveryInit - could not open '%s' for reading.", tmpfn);
                            FTI_Print(strerr, FTI_EROR);
                            errno = 0;
                            closedir(L3CkptDir);
                            goto GATHER_L3INFO;
                        }
 
                        if ( lseek(fd, -FTI_filemetastructsize, SEEK_END) == -1 ) {
                            snprintf(strerr, FTI_BUFS, "FTI-FF: L3RecoveryInit - could not seek in file: %s", tmpfn);
                            FTI_Print(strerr, FTI_EROR);
                            errno = 0;
                            closedir(L3CkptDir);
                            close(fd);
                            goto GATHER_L3INFO;
                        }

                        if ( 
                                ( read( fd, FTIFFMeta->checksum, MD5_DIGEST_STRING_LENGTH ) == -1 )     ||
                                ( read( fd, FTIFFMeta->myHash, MD5_DIGEST_LENGTH ) == -1 )              ||
                                ( read( fd, &(FTIFFMeta->ckptSize), sizeof(long) ) == -1 )              ||
                                ( read( fd, &(FTIFFMeta->fs), sizeof(long) ) == -1 )                    ||
                                ( read( fd, &(FTIFFMeta->maxFs), sizeof(long) ) == -1 )                 ||
                                ( read( fd, &(FTIFFMeta->ptFs), sizeof(long) ) == -1 )                  ||
                                ( read( fd, &(FTIFFMeta->timestamp), sizeof(long) ) == -1 )                                     
                            ) 
                        {
                            snprintf(strerr, FTI_BUFS, "FTI-FF: L3RecoveryInit - Failed to request file meta data from: %s", tmpfn);
                            FTI_Print(strerr, FTI_EROR);
                            errno=0;
                            closedir(L3CkptDir);
                            close(fd);
                            goto GATHER_L3INFO;
                        }
                        
                        unsigned char hash[MD5_DIGEST_LENGTH];
                        FTIFF_GetHashMetaInfo( hash, FTIFFMeta );
                        
                        if ( memcmp( FTIFFMeta->myHash, hash, MD5_DIGEST_LENGTH ) == 0 ) {
                            long rcount = 0, toRead, diff;
                            int rbuffer;
                            char buffer[CHUNK_SIZE];
                            MD5_Init (&mdContext);
                            while( rcount < FTIFFMeta->fs ) {
                                if ( lseek( fd, rcount, SEEK_SET ) == -1 ) {
                                    snprintf(strerr, FTI_BUFS, "FTI-FF: L3RecoveryInit - could not seek in file: %s", tmpfn);
                                    FTI_Print(strerr, FTI_EROR);
                                    errno = 0;
                                    closedir(L3CkptDir);
                                    close(fd);
                                    goto GATHER_L3INFO;
                                }

                                diff = FTIFFMeta->fs - rcount;
                                toRead = ( diff < CHUNK_SIZE ) ? diff : CHUNK_SIZE;
                                rbuffer = read( fd, buffer, toRead );
                                if ( rbuffer == -1 ) {
                                    snprintf(strerr, FTI_BUFS, "FTI-FF: L3RecoveryInit - Failed to read %ld bytes from file: %s", toRead, tmpfn);
                                    FTI_Print(strerr, FTI_EROR);
                                    errno=0;
                                    closedir(L3CkptDir);
                                    close(fd);
                                    goto GATHER_L3INFO;
                                }

                                rcount += rbuffer;
                                MD5_Update (&mdContext, buffer, rbuffer);
                            }
                            unsigned char hash[MD5_DIGEST_LENGTH];
                            MD5_Final (hash, &mdContext);
                            int i;
                            char checksum[MD5_DIGEST_STRING_LENGTH];
                            int ii = 0;
                            for(i = 0; i < MD5_DIGEST_LENGTH; i++) {
                                sprintf(&checksum[ii], "%02x", hash[i]);
                                ii += 2;
                            }
                            if ( strcmp( checksum, FTIFFMeta->checksum ) == 0 ) {
                                myInfo->RSfs = FTIFFMeta->fs;    
                                myInfo->ckptID = ckptID;    
                                myInfo->RSFileExists = 1;
                            } else {
                                char str[FTI_BUFS];
                                snprintf(str, FTI_BUFS, "Checksum do not match. \"%s\" file is corrupted. %s != %s",
                                        entry->d_name, checksum, FTIFFMeta->checksum);
                                FTI_Print(str, FTI_WARN);
                                closedir(L3CkptDir);
                                close(fd);
                                goto GATHER_L3INFO;
                            }
                        } else {
                            char str[FTI_BUFS];
                            snprintf(str, FTI_BUFS, "Metadata in file \"%s\" is corrupted.",entry->d_name);
                            FTI_Print(str, FTI_WARN);
                            closedir(L3CkptDir);
                            close(fd);
                            goto GATHER_L3INFO;
                        }
                        close(fd);
                    } else {
                        char str[FTI_BUFS];
                        snprintf(str, FTI_BUFS, "size %lu of file \"%s\" is smaller then file meta data struct size.", ckptFS.st_size, entry->d_name);
                        FTI_Print(str, FTI_WARN);
                        closedir(L3CkptDir);
                        goto GATHER_L3INFO;
                    }
                } else {
                    ckptID = tmpCkptID;
                }
            }
            if (myInfo->FileExists && myInfo->RSFileExists) {
                break;
            }
        }
        closedir(L3CkptDir);
    }

GATHER_L3INFO:

    if(!(myInfo->FileExists) && !(myInfo->RSFileExists)) {
        myInfo->ckptID = -1;
    }

    if(!(myInfo->RSFileExists)) {
        myInfo->RSfs = -1;
    }

    // gather meta info
    MPI_Allgather( myInfo, 1, FTIFF_MpiTypes[FTIFF_L3_INFO], groupInfo, 1, FTIFF_MpiTypes[FTIFF_L3_INFO], FTI_Exec->groupComm);

    // check if recovery possible
    int i, saneCkptID = 0, saneMaxFs = 0, erasures = 0;
    long maxFs = 0;
    ckptID = 0;
    for(i=0; i<FTI_Topo->groupSize; i++) { 
        erased[i]=!groupInfo[i].FileExists;
        erased[i+FTI_Topo->groupSize]=!groupInfo[i].RSFileExists;
        erasures += erased[i] + erased[i+FTI_Topo->groupSize];
        if (groupInfo[i].ckptID > 0) {
            saneCkptID++;
            ckptID += groupInfo[i].ckptID;
        }
        if (groupInfo[i].RSfs > 0) {
            saneMaxFs++;
            maxFs += groupInfo[i].RSfs;
        }
    }
    if( saneCkptID != 0 ) {
        FTI_Exec->ckptID = ckptID/saneCkptID;
    }
    if( saneMaxFs != 0 ) {
        FTI_Exec->meta[3].maxFs[0] = maxFs/saneMaxFs;
    }
    // for the case that all (and only) the encoded files are deleted
    if( saneMaxFs == 0 && !(erasures > FTI_Topo->groupSize) ) {
        MPI_Allreduce( &(FTIFFMeta->maxFs), FTI_Exec->meta[3].maxFs, 1, MPI_LONG, MPI_SUM, FTI_Exec->groupComm );
        FTI_Exec->meta[3].maxFs[0] /= FTI_Topo->groupSize;
    }

    FTI_Exec->meta[3].fs[0] = (myInfo->FileExists) ? myInfo->fs : 0;

    snprintf(FTI_Exec->meta[3].ckptFile, FTI_BUFS, "Ckpt%d-Rank%d.fti", FTI_Exec->ckptID, FTI_Topo->myRank);

    free(groupInfo);

    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Init of FTI-FF L4 recovery
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Topo        Topology metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @param      checksum        Ckpt file checksum
  @return     integer         FTI_SCES if successful.

  This function initializes the L4 checkpoint recovery. It checks for 
  erasures and loads the required meta data. 
 **/
/*-------------------------------------------------------------------------*/
int FTIFF_CheckL4RecoverInit( FTIT_execution* FTI_Exec, FTIT_topology* FTI_Topo, 
        FTIT_checkpoint* FTI_Ckpt)
{
    char str[FTI_BUFS], strerr[FTI_BUFS], tmpfn[FTI_BUFS];
    int fexist = 0, fileTarget, ckptID, fcount;
    
    struct dirent *entry;
    struct stat ckptFS;
    struct stat ckptDIR;
    
    FTIFF_metaInfo _FTIFFMeta;
    FTIFF_metaInfo *FTIFFMeta = (FTIFF_metaInfo*) memset( &_FTIFFMeta, 0x0, sizeof(FTIFF_metaInfo) ); 
   
    char L4DirName[FTI_BUFS];

    if ( FTI_Ckpt[4].isDcp ) {
        strcpy( L4DirName, FTI_Ckpt[4].dcpDir );
    } else {
        strcpy( L4DirName, FTI_Ckpt[4].dir );
    }
    // check if L4 ckpt directory exists
    bool L4CkptDirExists = false;
    if ( stat( L4DirName, &ckptDIR ) == 0 ) {
        if ( S_ISDIR( ckptDIR.st_mode ) != 0 ) {
            L4CkptDirExists = true;
        } else {
            snprintf(strerr, FTI_BUFS, "FTI-FF: L4RecoverInit - (%s) is not a directory.", L4DirName);
            FTI_Print(strerr, FTI_WARN);
            goto GATHER_L4INFO;
        }
    }

    if(L4CkptDirExists) {
        
        DIR *L4CkptDir = opendir( L4DirName );
        
        if (L4CkptDir == NULL) {
            snprintf(strerr, FTI_BUFS, "FTI-FF: L4RecoveryInit - checkpoint directory (%s) could not be accessed.", L4DirName);
            FTI_Print(strerr, FTI_EROR);
            errno = 0;
            goto GATHER_L4INFO;
        }

        while((entry = readdir(L4CkptDir)) != NULL) {
            
            if(strcmp(entry->d_name,".") && strcmp(entry->d_name,"..")) { 
                snprintf(str, FTI_BUFS, "FTI-FF: L4RecoveryInit - found file with name: %s", entry->d_name);
                FTI_Print(str, FTI_DBUG);
                if ( FTI_Ckpt[4].isDcp ) {
                    sscanf(entry->d_name, "dCPFile-Rank%d.fti", &fileTarget );
                } else {
                    sscanf(entry->d_name, "Ckpt%d-Rank%d.fti", &ckptID, &fileTarget );
                }
                if( fileTarget == FTI_Topo->myRank ) {
                    //DBG_MSG("%s %d %d",-1,entry->d_name, fileTarget, FTI_Topo->myRank );
                    snprintf(tmpfn, FTI_BUFS, "%s/%s", L4DirName, entry->d_name);
                    
                    if ( stat(tmpfn, &ckptFS) == -1 ) {
                        snprintf( strerr, FTI_BUFS, "FTI-FF: L4RecoveryInit - Problem with stats on file %s", tmpfn );
                        FTI_Print( strerr, FTI_EROR );
                        errno = 0;
                        closedir(L4CkptDir);
                        goto GATHER_L4INFO;
                    }

                    if ( S_ISREG(ckptFS.st_mode) == 0 ) {
                        snprintf( strerr, FTI_BUFS, "FTI-FF: L4RecoveryInit - %s is not a regular file", tmpfn );
                        FTI_Print( strerr, FTI_WARN );
                        closedir(L4CkptDir);
                        goto GATHER_L4INFO;
                    }
                    
                    // Check for reasonable file size. At least has to contain file meta-data
                    if ( ckptFS.st_size > FTI_filemetastructsize ) {
                        
                        int fd = open(tmpfn, O_RDONLY);
                        if (fd == -1) {
                            snprintf( strerr, FTI_BUFS, "FTI-FF: L4RecoveryInit - could not open '%s' for reading.", tmpfn);
                            FTI_Print(strerr, FTI_EROR);
                            errno = 0;
                            closedir(L4CkptDir);
                            goto GATHER_L4INFO;
                        }

                        if ( lseek(fd, 0, SEEK_SET) == -1 ) {
                            snprintf(strerr, FTI_BUFS, "FTI-FF: L4RecoveryInit - could not seek in file: %s", tmpfn);
                            FTI_Print(strerr, FTI_EROR);
                            errno = 0;
                            closedir(L4CkptDir);
                            close(fd);
                            goto GATHER_L4INFO;
                        }

                        // Read in file meta-data
                        if ( 
                                ( read( fd, FTIFFMeta->checksum, MD5_DIGEST_STRING_LENGTH ) == -1 )     ||
                                ( read( fd, FTIFFMeta->myHash, MD5_DIGEST_LENGTH ) == -1 )              ||
                                ( read( fd, &(FTIFFMeta->ckptSize), sizeof(long) ) == -1 )              ||
                                ( read( fd, &(FTIFFMeta->fs), sizeof(long) ) == -1 )                    ||
                                ( read( fd, &(FTIFFMeta->maxFs), sizeof(long) ) == -1 )                 ||
                                ( read( fd, &(FTIFFMeta->ptFs), sizeof(long) ) == -1 )                  ||
                                ( read( fd, &(FTIFFMeta->timestamp), sizeof(long) ) == -1 )                                     
                            ) 
                        {
                            snprintf(strerr, FTI_BUFS, "FTI-FF: L4RecoveryInit - Failed to request file meta data from: %s", tmpfn);
                            FTI_Print(strerr, FTI_EROR);
                            errno=0;
                            closedir(L4CkptDir);
                            close(fd);
                            goto GATHER_L4INFO;
                        }

                        unsigned char hash[MD5_DIGEST_LENGTH];
                        FTIFF_GetHashMetaInfo( hash, FTIFFMeta );
                        
                        if ( memcmp( FTIFFMeta->myHash, hash, MD5_DIGEST_LENGTH ) == 0 ) {
                            FTI_Exec->meta[4].fs[0] = ckptFS.st_size;    
                            if ( !FTI_Ckpt[4].isDcp ) {
                                FTI_Exec->ckptID = ckptID;
                            }
                            
                            unsigned char hash[MD5_DIGEST_LENGTH];
                            FTIFF_GetFileChecksum( FTIFFMeta, FTI_Ckpt, fd, hash ); 
                            
                            int i;
                            char checksum[MD5_DIGEST_STRING_LENGTH];
                            int ii = 0;
                            for(i = 0; i < MD5_DIGEST_LENGTH; i++) {
                                sprintf(&checksum[ii], "%02x", hash[i]);
                                ii += 2;
                            }
                            
                            if ( strcmp( checksum, FTIFFMeta->checksum ) == 0 ) {
                                if ( !FTI_Ckpt[4].isDcp ) {
                                    strncpy(FTI_Exec->meta[1].ckptFile, entry->d_name, NAME_MAX);
                                } else {
                                    snprintf(FTI_Exec->meta[1].ckptFile, FTI_BUFS, "Ckpt%d-Rank%d.fti", FTI_Exec->ckptID, fileTarget );
                                }
                                strncpy(FTI_Exec->meta[4].ckptFile, entry->d_name, NAME_MAX);
                                fexist = 1;
                            } 
                            else {
                                char str[FTI_BUFS];
                                snprintf(str, FTI_BUFS, "Checksum do not match. \"%s\" file is corrupted. %s != %s",
                                        entry->d_name, checksum, FTIFFMeta->checksum);
                                FTI_Print(str, FTI_WARN);
                                close(fd);
                                free(FTIFFMeta);
                                closedir(L4CkptDir);
                                goto GATHER_L4INFO;
                            }
                        } else {
                            char str[FTI_BUFS];
                            snprintf(str, FTI_BUFS, "Metadata in file \"%s\" is corrupted.",entry->d_name);
                            FTI_Print(str, FTI_WARN);
                            closedir(L4CkptDir);
                            goto GATHER_L4INFO;
                        }
                        close(fd);
                        break;
                    } else {
                        char str[FTI_BUFS];
                        snprintf(str, FTI_BUFS, "size %lu of file \"%s\" is smaller then file meta data struct size.", ckptFS.st_size, entry->d_name);
                        FTI_Print(str, FTI_WARN);
                        closedir(L4CkptDir);
                        goto GATHER_L4INFO;
                    }
                }
            }
        }
        closedir(L4CkptDir);
    }

GATHER_L4INFO:

    MPI_Allreduce(&fexist, &fcount, 1, MPI_INT, MPI_SUM, FTI_COMM_WORLD);
    int fneeded = FTI_Topo->nbNodes*FTI_Topo->nbApprocs;
    int res = (fcount == fneeded) ? FTI_SCES : FTI_NSCS;
    return res;
}

/*-------------------------------------------------------------------------*/
/**
  @brief    Computes has of the ckpt meta data structure   
  @param    hash          hash to compute.
  @param    FTIFFMeta     Ckpt file meta data.
 **/
/*-------------------------------------------------------------------------*/
void FTIFF_GetHashMetaInfo( unsigned char *hash, FTIFF_metaInfo *FTIFFMeta ) 
{
    MD5_CTX md5Ctx;
    MD5_Init (&md5Ctx);
    MD5_Update( &md5Ctx, FTIFFMeta->checksum, MD5_DIGEST_STRING_LENGTH );
    MD5_Update( &md5Ctx, &(FTIFFMeta->timestamp), sizeof(long) );
    MD5_Update( &md5Ctx, &(FTIFFMeta->ckptSize), sizeof(long) );
    MD5_Update( &md5Ctx, &(FTIFFMeta->fs), sizeof(long) );
    MD5_Update( &md5Ctx, &(FTIFFMeta->ptFs), sizeof(long) );
    MD5_Update( &md5Ctx, &(FTIFFMeta->maxFs), sizeof(long) );
    MD5_Final( hash, &md5Ctx );
}

void FTIFF_GetHashdb( unsigned char *hash, FTIFF_db *db ) 
{
    MD5_CTX md5Ctx;
    MD5_Init (&md5Ctx);
    MD5_Update( &md5Ctx, &(db->numvars), sizeof(int) );
    MD5_Update( &md5Ctx, &(db->dbsize), sizeof(long) );
    MD5_Final( hash, &md5Ctx );
}

void FTIFF_GetHashdbvar( unsigned char *hash, FTIFF_dbvar *dbvar ) 
{
    MD5_CTX md5Ctx;
    MD5_Init (&md5Ctx);
    MD5_Update( &md5Ctx, &(dbvar->id), sizeof(int) );
    MD5_Update( &md5Ctx, &(dbvar->idx), sizeof(int) );
    MD5_Update( &md5Ctx, &(dbvar->containerid), sizeof(int) );
    MD5_Update( &md5Ctx, &(dbvar->hascontent), sizeof(bool) );
    MD5_Update( &md5Ctx, &(dbvar->hasCkpt), sizeof(bool) );
    MD5_Update( &md5Ctx, &(dbvar->dptr), sizeof(uintptr_t) );
    MD5_Update( &md5Ctx, &(dbvar->fptr), sizeof(uintptr_t) );
    MD5_Update( &md5Ctx, &(dbvar->chunksize), sizeof(long) );
    MD5_Update( &md5Ctx, &(dbvar->containersize), sizeof(long) );
    MD5_Update( &md5Ctx, dbvar->hash, MD5_DIGEST_LENGTH );
    MD5_Final( hash, &md5Ctx );
}
/*-------------------------------------------------------------------------*/
/**
  @brief      Frees allocated memory for the FTI-FF meta data struct list
  @param      last      Last element in FTI-FF metadata list.
 **/
/*-------------------------------------------------------------------------*/
void FTIFF_FreeDbFTIFF(FTIFF_db* last)
{
    if (last) {
        FTIFF_db *current = last;
        FTIFF_db *previous;
        while( current ) {
            previous = current->previous;
            // make sure there is a dbvar struct allocated.
            assert(current->dbvars);
            free(current->dbvars);
            free(current);
            current = previous;
        }
    }
}

// BEGIN - ONLY FOR DEVELOPPING
/* PRINTING DATA STRUCTURE */
void FTIFF_PrintDataStructure( int rank, FTIT_execution* FTI_Exec, FTIT_dataset* FTI_Data )
{
        int dbrank;
        MPI_Comm_rank(FTI_COMM_WORLD, &dbrank);
        FTIFF_db *dbgdb = FTI_Exec->firstdb;
        if(dbrank == rank) {
            int dbcnt = 0;
printf("------------------- DATASTRUCTURE BEGIN [%d]----------------\n\n", rank);
            do {
printf("    DataBase-id: %d\n", dbcnt);
printf("                 dbsize: %ld\n", dbgdb->dbsize);
printf("                 metasize (offset: %d): %ld\n\n", sizeof(FTIFF_metaInfo), sizeof(int)+sizeof(long)+dbgdb->numvars*FTI_dbvarstructsize);
                dbcnt++;
                int varid=0;
                for(; varid<dbgdb->numvars; ++varid) {
printf("         Var-id: %d\n", varid);
printf("                 id: %d\n"
       "                 idx: %d\n"
       "                 containerid: %d\n"
       "                 hascontent: %s\n"
       "                 hasCkpt: %s\n"
       "                 dptr: %llu\n"
       "                 fptr: %llu\n"
       "                 chunksize: %lu\n"
       "                 containersize: %lu\n\n",/*
       "                 nbHashes: %lu\n"
       "                 diffBlockSize: %d\n"
       "                 addr-hashptr: %p\n"
       "                 addr-dataptr: %p\n"
       "                 lastBlockSize: %d\n\n",*/
                    dbgdb->dbvars[varid].id,
                    dbgdb->dbvars[varid].idx,
                    dbgdb->dbvars[varid].containerid,
                    (dbgdb->dbvars[varid].hascontent) ? "true" : "false",
                    (dbgdb->dbvars[varid].hasCkpt) ? "true" : "false",
                    dbgdb->dbvars[varid].dptr,
                    dbgdb->dbvars[varid].fptr,
                    dbgdb->dbvars[varid].chunksize,
                    dbgdb->dbvars[varid].containersize);/*,
                    dbgdb->dbvars[varid].nbHashes,
                    FTI_GetDiffBlockSize(),
                    dbgdb->dbvars[varid].dataDiffHash[0].ptr,
                    (FTI_ADDRPTR) (FTI_Data[dbgdb->dbvars[varid].idx].ptr+dbgdb->dbvars[varid].dptr),
                    dbgdb->dbvars[varid].dataDiffHash[dbgdb->dbvars[varid].nbHashes-1].blockSize);*/
                }
            } while( (dbgdb = dbgdb->next) );
printf("\n------------------- DATASTRUCTURE END ---------------------\n");
fflush(stdout);
        }
MPI_Barrier(FTI_COMM_WORLD);
}
// END - ONLY FOR DEVELOPPING



