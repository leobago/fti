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
#include "api_cuda.h"

#include "utility.h"

/*  

  +-------------------------------------------------------------------------+
  |   STATIC TYPE DECLARATIONS                                              |
  +-------------------------------------------------------------------------+

*/

MPI_Datatype FTIFF_MpiTypes[FTIFF_NUM_MPI_TYPES];

/*

  +-------------------------------------------------------------------------+
  |   FUNCTION DEFINITIONS                                                  |
  +-------------------------------------------------------------------------+

*/

/*-------------------------------------------------------------------------*/
/**
  @brief      Reads datablock structure for FTI File Format from ckpt file.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @return     integer         FTI_SCES if successful.

  Builds meta data list from checkpoint file for the FTI File Format

 **/
/*-------------------------------------------------------------------------*/
int FTIFF_ReadDbFTIFF( FTIT_configuration *FTI_Conf, FTIT_execution *FTI_Exec, 
        FTIT_checkpoint* FTI_Ckpt ) 
{
    char fn[FTI_BUFS]; //Path to the checkpoint file
    char str[FTI_BUFS]; //For console output
    char strerr[FTI_BUFS];
    unsigned char hash[MD5_DIGEST_LENGTH];
    MD5_CTX ctx;
    MD5_Init( &ctx );

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

    unsigned long fs = st.st_size;

    // open checkpoint file for read only
    int fd = open( fn, O_RDONLY, 0 );
    if (fd == -1) {
        snprintf( strerr, FTI_BUFS, "FTI-FF: ReadDbFTIFF - could not open '%s' for reading.", fn);
        FTI_Print(strerr, FTI_EROR);
        errno = 0;
        return FTI_NSCS;
    }

    // map file into memory
    unsigned char* fmmap = (unsigned char*) mmap(0, fs, PROT_READ, MAP_SHARED, fd, 0);
    if (fmmap == MAP_FAILED) {
        snprintf( strerr, FTI_BUFS, "FTI-FF: ReadDbFTIFF - could not map file to memory.");
        FTI_Print(strerr, FTI_EROR);
        close(fd);
        errno = 0;
        return FTI_NSCS;
    }

    // file is mapped, we can close it.
    close(fd);

    // determine location of file meta data in mapping
    FTI_ADDRPTR seek_ptr = fmmap + (FTI_ADDRVAL) (fs - FTI_filemetastructsize); 
    
    // set end of file to last byte of metadata without file metadata
    FTI_ADDRPTR seek_end = seek_ptr - 1;
    
    if( FTIFF_DeserializeFileMeta( &(FTI_Exec->FTIFFMeta), seek_ptr ) != FTI_SCES ) {
        FTI_Print( "FTI-FF: ReadDbFTIFF - failed to deserialize 'FTI_Exec->FTIFFMeta'", FTI_EROR );
        munmap( fmmap, fs );
        errno = 0;
        return FTI_NSCS;
    }
    DBG_MSG("ckptSize: %lu, dataSize: %lu, metaSize: %lu, fs: %lu",-1,
            FTI_Exec->FTIFFMeta.ckptSize,FTI_Exec->FTIFFMeta.dataSize,FTI_Exec->FTIFFMeta.metaSize,fs);

    int dbcounter=0;

    // set seek_ptr to start of meta data in file
    seek_ptr = fmmap + (FTI_ADDRVAL) FTI_Exec->FTIFFMeta.dataSize;

    int isnextdb;
    
    FTIFF_db *currentdb = (FTIFF_db*) malloc( sizeof(FTIFF_db) );
    if ( currentdb == NULL ) {
        snprintf( strerr, FTI_BUFS, "FTI-FF: ReadDbFTIFF - failed to allocate %ld bytes for 'currentdb'", sizeof(FTIFF_db));
        FTI_Print(strerr, FTI_EROR);
        munmap( fmmap, fs );
        errno = 0;
        return FTI_NSCS;
    }

    FTI_Exec->firstdb = currentdb;
    FTI_Exec->firstdb->finalized = true;
    FTI_Exec->firstdb->next = NULL;
    FTI_Exec->firstdb->previous = NULL;

    do {

        isnextdb = 0;

        if( FTIFF_DeserializeDbMeta( currentdb, seek_ptr ) != FTI_SCES ) {
            FTI_Print( "FTI-FF: ReadDbFTIFF - failed to deserialize 'currentdb'", FTI_EROR );
            munmap( fmmap, fs );
            errno = 0;
            return FTI_NSCS;
        } 

        currentdb->finalized = true;
        // TODO create hash of data base meta data during FTIFF_UpdateDatastruct 
        // and check consistency here to prevent seg faults in case of corruption.
        
        // advance meta data offset
        seek_ptr += (FTI_ADDRVAL) FTI_dbstructsize;

        snprintf(str, FTI_BUFS, "FTI-FF: Updatedb - dataBlock:%i, dbsize: %ld, numvars: %i.", 
                dbcounter, currentdb->dbsize, currentdb->numvars);
        FTI_Print(str, FTI_DBUG);

        currentdb->dbvars = (FTIFF_dbvar*) malloc( sizeof(FTIFF_dbvar) * currentdb->numvars );
        if ( currentdb->dbvars == NULL ) {
            snprintf( strerr, FTI_BUFS, "FTI-FF: Updatedb - failed to allocate %ld bytes for 'currentdb->dbvars'", sizeof(FTIFF_dbvar) * currentdb->numvars);
            FTI_Print(strerr, FTI_EROR);
            munmap( fmmap, fs );
            errno = 0;
            return FTI_NSCS;
        }
        
        int dbvar_idx;
        for(dbvar_idx=0;dbvar_idx<currentdb->numvars;dbvar_idx++) {

            FTIFF_dbvar *currentdbvar = &(currentdb->dbvars[dbvar_idx]);
            
            // get dbvar meta data
            if( FTIFF_DeserializeDbVarMeta( currentdbvar, seek_ptr ) != FTI_SCES ) {
                FTI_Print( "FTI-FF: ReadDbFTIFF - failed to deserialize 'dbvar'", FTI_EROR );
                munmap( fmmap, fs );
                errno = 0;
                return FTI_NSCS;
            } 
            // TODO create hash of data base variable meta data during FTIFF_UpdateDatastruct 
            // and check consistency here to prevent seg faults in case of corruption.

            // if dCP enabled, initialize hash clock structures
            if( FTI_Conf->dcpEnabled ) {
                if( currentdbvar->hascontent ) {
                    FTI_InitBlockHashArray( currentdbvar );
                } else {
                    currentdbvar->dataDiffHash = NULL;
                }
            }

            // advance meta data offset
            seek_ptr += (FTI_ADDRVAL) FTI_dbvarstructsize;

            currentdbvar->hasCkpt = true;

            // compute hash of chunk and file hash
            // (Note: we create the file hash from the chunk hashes due to ICP)
            if( currentdbvar->hascontent ) {
                unsigned char chash[MD5_DIGEST_LENGTH]; 
                MD5( fmmap + currentdbvar->fptr, currentdbvar->chunksize, chash );
                MD5_Update( &ctx, chash, MD5_DIGEST_LENGTH );
            }
            
            // init FTI meta data structure
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

        if ( seek_ptr < seek_end ) {
            FTIFF_db *nextdb = (FTIFF_db*) malloc( sizeof(FTIFF_db) );
            if ( nextdb == NULL ) {
                snprintf( strerr, FTI_BUFS, "FTI-FF: ReadDbFTIFF - failed to allocate %ld bytes for 'nextdb'", sizeof(FTIFF_db));
                FTI_Print(strerr, FTI_EROR);
                munmap( fmmap, fs );
                errno = 0;
                return FTI_NSCS;
            }
            currentdb->next = nextdb;
            nextdb->previous = currentdb;
            currentdb = nextdb;
            isnextdb = 1;
        }

        dbcounter++;

    } while( isnextdb );
   
    MD5_Final( hash, &ctx );

    int i;
    char checksum[MD5_DIGEST_STRING_LENGTH];
    int ii = 0;
    for(i = 0; i < MD5_DIGEST_LENGTH; i++) {
        sprintf(&checksum[ii], "%02x", hash[i]);
        ii += 2;
    }
    
    if ( strcmp( checksum, FTI_Exec->FTIFFMeta.checksum ) != 0 ) {
        char str[FTI_BUFS];
        snprintf(str, FTI_BUFS, "Checksum do not match. file is corrupted. %s != %s",
                checksum, FTI_Exec->FTIFFMeta.checksum);
        FTI_Print(str, FTI_WARN);
        // reset meta data
        FTIFF_FreeDbFTIFF( FTI_Exec->lastdb );
        memset(FTI_Exec->meta,0x0,5*sizeof(FTIT_metadata));
        return FTI_NSCS;
    } 

    FTI_Exec->meta[FTI_Exec->ckptLvel].nbVar[0] = varCnt;
    FTI_Exec->nbVarStored = varCnt;

    FTI_Exec->lastdb = currentdb;
    FTI_Exec->lastdb->next = NULL;

    // unmap memory.
    if ( munmap( fmmap, fs ) == -1 ) {
        FTI_Print("FTI-FF: ReadDbFTIFF - unable to unmap memory", FTI_EROR);
        errno = 0;
        return FTI_NSCS;
    }
    
    return FTI_SCES;

}

/*-------------------------------------------------------------------------*/
/**
  @brief      Determines checksum of checkpoint data.
  @param      FTIFF_Meta      FTI-FF file meta data.
  @param      FTI_Ckpt        Checkpoint metadata.
  @param      fd              file descriptor.
  @param      hash            pointer to MD5 digest container.
  @return     integer         FTI_SCES if successful.

  This function computes the FTI-FF file checksum and places the MD5 digest
  into the 'hash' buffer. The buffer has to be allocated for at least 
  MD5_DIGEST_LENGTH bytes.
 **/
/*-------------------------------------------------------------------------*/
int FTIFF_GetFileChecksum( FTIFF_metaInfo *FTIFFMeta, FTIT_checkpoint* FTI_Ckpt, int fd, char *checksum ) 
{
    char str[FTI_BUFS]; //For console output
    char strerr[FTI_BUFS];
    unsigned char hash[MD5_DIGEST_LENGTH];
    MD5_CTX ctx;
    MD5_Init( &ctx );

    // map file into memory
    unsigned char* fmmap = (unsigned char*) mmap(0, FTIFFMeta->ckptSize, PROT_READ, MAP_SHARED, fd, 0);
    if (fmmap == MAP_FAILED) {
        snprintf( strerr, FTI_BUFS, "FTI-FF: GetFileChecksum - could not map file to memory.");
        FTI_Print(strerr, FTI_EROR);
        errno = 0;
        return FTI_NSCS;
    }

    DBG_MSG("ckptSize: %lu, dataSize: %lu, metaSize: %lu",-1,
            FTIFFMeta->ckptSize,FTIFFMeta->dataSize,FTIFFMeta->metaSize);

    // set seek_ptr to start of meta data in file
    FTI_ADDRPTR seek_ptr = fmmap + (FTI_ADDRVAL) FTIFFMeta->dataSize;
    
    // set end of file to last byte of metadata without file metadata
    FTI_ADDRPTR seek_end = fmmap + (FTI_ADDRVAL) FTIFFMeta->ckptSize - (FTI_ADDRVAL) FTI_filemetastructsize;
    
    FTIFF_db *db = (FTIFF_db*) malloc( sizeof(FTIFF_db) );
    if ( db == NULL ) {
        snprintf( strerr, FTI_BUFS, "FTI-FF: ReadDbFTIFF - failed to allocate %ld bytes for 'db'", sizeof(FTIFF_db));
        FTI_Print(strerr, FTI_EROR);
        munmap( fmmap, FTIFFMeta->ckptSize );
        errno = 0;
        return FTI_NSCS;
    }

    do {

        if( FTIFF_DeserializeDbMeta( db, seek_ptr ) != FTI_SCES ) {
            FTI_Print( "FTI-FF: ReadDbFTIFF - failed to deserialize 'db'", FTI_EROR );
            munmap( fmmap, FTIFFMeta->ckptSize );
            errno = 0;
            return FTI_NSCS;
        } 

        seek_ptr += (FTI_ADDRVAL) FTI_dbstructsize;

        FTIFF_dbvar *dbvars = (FTIFF_dbvar*) malloc( sizeof(FTIFF_dbvar) * db->numvars );
        if ( dbvars == NULL ) {
            snprintf( strerr, FTI_BUFS, "FTI-FF: Updatedb - failed to allocate %ld bytes for 'dbvars'", sizeof(FTIFF_dbvar) * db->numvars);
            FTI_Print(strerr, FTI_EROR);
            munmap( fmmap, FTIFFMeta->ckptSize );
            errno = 0;
            return FTI_NSCS;
        }
        
        int dbvar_idx;
        for(dbvar_idx=0;dbvar_idx<db->numvars;dbvar_idx++) {

            FTIFF_dbvar *dbvar = &(dbvars[dbvar_idx]);
            
            // get dbvar meta data
            if( FTIFF_DeserializeDbVarMeta( dbvar, seek_ptr ) != FTI_SCES ) {
                FTI_Print( "FTI-FF: ReadDbFTIFF - failed to deserialize 'dbvar'", FTI_EROR );
                munmap( fmmap, FTIFFMeta->ckptSize );
                errno = 0;
                return FTI_NSCS;
            } 

            // advance meta data offset
            seek_ptr += (FTI_ADDRVAL) FTI_dbvarstructsize;

            // compute hash of chunk and file hash
            // (Note: we create the file hash from the chunk hashes due to ICP)
            if( dbvar->hascontent ) {
                unsigned char chash[MD5_DIGEST_LENGTH]; 
                MD5( fmmap + dbvar->fptr, dbvar->chunksize, chash );
                MD5_Update( &ctx, chash, MD5_DIGEST_LENGTH );
            }

        }

        free(dbvars);

    } while( seek_ptr < seek_end );
   
    MD5_Final( hash, &ctx );

    int i;
    int ii = 0;
    for(i = 0; i < MD5_DIGEST_LENGTH; i++) {
        sprintf(&checksum[ii], "%02x", hash[i]);
        ii += 2;
    }
    
    // unmap memory.
    if ( munmap( fmmap, FTIFFMeta->ckptSize ) == -1 ) {
        FTI_Print("FTI-FF: ReadDbFTIFF - unable to unmap memory", FTI_EROR);
        errno = 0;
        return FTI_NSCS;
    }

    free(db);
    
    return FTI_SCES;

}

int FTIFF_UpdateDatastructVarFTIFF( FTIT_execution* FTI_Exec, 
        FTIT_dataset* FTI_Data, FTIT_configuration* FTI_Conf, 
        int pvar_idx )
{

    if( FTI_Exec->nbVar == 0 ) {
        FTI_Print("FTI-FF - UpdateDatastructFTIFF: No protected Variables, discarding checkpoint!", FTI_WARN);
        return FTI_NSCS;
    }

    char strerr[FTI_BUFS];
    
    FTIFF_dbvar *dbvars = NULL;

    // first call to this function. This means that
    // for all variables only one chunk/container exists.
    if(!FTI_Exec->firstdb) {
        
        FTIFF_db *dblock = (FTIFF_db*) malloc( sizeof(FTIFF_db) );
        if ( dblock == NULL ) {
            snprintf( strerr, FTI_BUFS, "FTI-FF: UpdateDatastructFTIFF - failed to allocate %ld bytes for 'dblock'", sizeof(FTIFF_db));
            FTI_Print(strerr, FTI_EROR);
            errno = 0;
            return FTI_NSCS;
        }

        dbvars = (FTIFF_dbvar*) malloc( sizeof(FTIFF_dbvar) );
        if ( dbvars == NULL ) {
            snprintf( strerr, FTI_BUFS, "FTI-FF: UpdateDatastructFTIFF - failed to allocate %ld bytes for 'dbvars'", sizeof(FTIFF_dbvar) * FTI_Exec->nbVar );
            FTI_Print(strerr, FTI_EROR);
            errno = 0;
            return FTI_NSCS;
        }

        dblock->previous = NULL;
        dblock->next = NULL;
        dblock->numvars = 1;
        dblock->dbvars = dbvars;
        
        dbvars->fptr = 0;
        dbvars->dptr = 0;
        dbvars->id = FTI_Data[pvar_idx].id;
        dbvars->idx = 0;
        dbvars->chunksize = FTI_Data[pvar_idx].size;
        dbvars->hascontent = true;
        dbvars->hasCkpt = false;
        dbvars->containerid = 0;
        dbvars->containersize = FTI_Data[pvar_idx].size;
        dbvars->cptr = FTI_Data[pvar_idx].ptr;
        // FOR DCP 
        if  ( FTI_Conf->dcpEnabled ) {
            FTI_InitBlockHashArray( dbvars );
        } else {
            dbvars->nbHashes = -1;
            dbvars->dataDiffHash = NULL;
        }
        
        dbvars->update = true;
        
        // TODO this must come later
        //FTIFF_GetHashdbvar( dbvars[dbvar_idx].myhash, &(dbvars[dbvar_idx]) );
        
        // TODO whats that for?
        FTI_Exec->nbVarStored = 1;
        dblock->dbsize = dbvars->containersize;;
        
        dblock->update = true;
        dblock->finalized = false;        
        // TODO this must come later
        // FTIFF_GetHashdb( dblock->myhash, dblock );

        // set as first datablock
        FTI_Exec->firstdb = dblock;
        FTI_Exec->lastdb = dblock;

    } else {

        int dbvar_idx;

        // 0 -> nothing to append, 1 -> new pvar, 2 -> size increased
        int editflags = 0; 
        bool idFound = false; 
        int isnextdb;
        long offset = 0;

        /*
         *  - check if protected variable is in file info
         *  - check if size has changed
         */

        FTI_Exec->lastdb = FTI_Exec->firstdb;

        int nbContainers = 0;
        long containerSizesAccu = 0;

        // init overflow with the datasizes and validBlock with true.
        bool validBlock = true;
        long overflow = FTI_Data[pvar_idx].size;

        // iterate though datablock list. Current datablock is 'lastdb'.
        // At the beginning of the loop 'lastdb = firstdb'
        do {
            isnextdb = 0;
            for(dbvar_idx=0;dbvar_idx<FTI_Exec->lastdb->numvars;dbvar_idx++) {
                FTIFF_dbvar* dbvar = &(FTI_Exec->lastdb->dbvars[dbvar_idx]);
                if( dbvar->id == FTI_Data[pvar_idx].id ) {
                    idFound = true;
                    // collect container info
                    containerSizesAccu += dbvar->containersize;
                    nbContainers++;
                    // if data was shrinked, invalidate the following blocks (if there are), 
                    // and set their chunksize to 0.
                    if ( !validBlock ) {
                        if ( dbvar->hascontent ) {
                            dbvar->hascontent = false;
                            // [FOR DCP] free hash array and hash structure in block
                            if ( ( dbvar->dataDiffHash != NULL ) && FTI_Conf->dcpEnabled ) {
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
                    if ( overflow > dbvar->containersize ) {
                        long chunksizeOld = dbvar->chunksize;
                        dbvar->chunksize = dbvar->containersize;
                        dbvar->cptr = FTI_Data[pvar_idx].ptr + dbvar->dptr;
                        if ( !dbvar->hascontent ) {
                            dbvar->hascontent = true;
                            // [FOR DCP] init hash array for block
                            if ( FTI_Conf->dcpEnabled ) {
                                if( FTI_InitBlockHashArray( dbvar ) != FTI_SCES ) {
                                    FTI_FinalizeDcp( FTI_Conf, FTI_Exec );
                                }
                            }
                        } else {
                            // [FOR DCP] adjust hash array to new chunksize if chunk size increased
                            if ( FTI_Conf->dcpEnabled ) {
                                if (  dbvar->chunksize > chunksizeOld ) {
                                    FTI_ExpandBlockHashArray( dbvar );
                                }
                            }
                        }
                        overflow -= dbvar->containersize;
                        continue;
                    }
                    // if overflow <= containersize, set 'validBlock = false' in order to invalidate the
                    // following blocks, set new chunksize to overflow, set afterwards overflow to 0 and 
                    // ensure that 'hascontent = true'. 
                    if ( overflow <= dbvar->containersize ) {
                        long chunksizeOld = dbvar->chunksize;
                        dbvar->chunksize = overflow;
                        dbvar->cptr = FTI_Data[pvar_idx].ptr + dbvar->dptr;
                        if ( !dbvar->hascontent ) {
                            dbvar->hascontent = true;
                            // [FOR DCP] init hash array for block
                            if ( FTI_Conf->dcpEnabled ) {
                                if( FTI_InitBlockHashArray( dbvar )  != FTI_SCES ) {
                                    FTI_FinalizeDcp( FTI_Conf, FTI_Exec );
                                }

                            }
                        } else {
                            // [FOR DCP] adjust hash array to new chunksize if chunk size decreased
                            if ( FTI_Conf->dcpEnabled ) {
                                if ( dbvar->chunksize < chunksizeOld ) {
                                    FTI_CollapseBlockHashArray( dbvar );
                                }
                                if ( dbvar->chunksize > chunksizeOld ) {
                                    FTI_ExpandBlockHashArray( dbvar );
                                }
                            }
                        }
                        validBlock = false;
                        overflow = 0;
                        continue;
                    }
                }
            }
            offset += FTI_Exec->lastdb->dbsize;
            if ( FTI_Exec->lastdb->next ) {
//                if( FTI_Exec->lastdb->next->finalized ) {
                    FTI_Exec->lastdb = FTI_Exec->lastdb->next;
                    isnextdb = 1;
//                }
            }
        } while( isnextdb );
        // end of while, 'lastdb' is last datablock.

        // check for new protected variables ( editflags == 1 / id not found )
        editflags = !idFound;

        // check if size has increased
        if ( overflow > 0 && idFound ) {
            editflags = 2;
        }

        if( editflags ) {
            FTIFF_db *dblock;
            // if size changed or we have new variables to protect, create new block. 
            if( FTI_Exec->lastdb->next == NULL && FTI_Exec->lastdb->finalized ) {
                dblock = (FTIFF_db*) malloc( sizeof(FTIFF_db) );
                if ( dblock == NULL ) {
                    snprintf( strerr, FTI_BUFS, "FTI-FF: UpdateDatastructFTIFF - failed to allocate %ld bytes for 'dblock'", sizeof(FTIFF_db));
                    FTI_Print(strerr, FTI_EROR);
                    errno = 0;
                    return FTI_NSCS;
                }
                dblock->finalized = false;
                FTI_Exec->lastdb->next = dblock;
                dblock->previous = FTI_Exec->lastdb;
                dblock->next = NULL;
                dblock->numvars = 0;
                dblock->dbsize = 0;
                dblock->dbvars = NULL;
                dblock->update = true;
                FTI_Exec->lastdb = dblock;
            } else {
                dblock = FTI_Exec->lastdb;
            }

            int evar_idx = dblock->numvars;
            long dbsize = dblock->dbsize;
            switch(editflags) {

                case 1:
                    // add new protected variable in next datablock
                    dbvars = (FTIFF_dbvar*) realloc( dblock->dbvars, (evar_idx+1) * sizeof(FTIFF_dbvar) );
                    dbvars[evar_idx].fptr = offset;
                    dbvars[evar_idx].dptr = 0;
                    dbvars[evar_idx].id = FTI_Data[pvar_idx].id;
                    dbvars[evar_idx].idx = pvar_idx;
                    dbvars[evar_idx].chunksize = FTI_Data[pvar_idx].size;
                    dbvars[evar_idx].hascontent = true;
                    dbvars[evar_idx].hasCkpt = false;
                    dbvars[evar_idx].containerid = 0;
                    dbvars[evar_idx].containersize = FTI_Data[pvar_idx].size;
                    dbsize += dbvars[evar_idx].containersize; 
                    dbvars[evar_idx].cptr = FTI_Data[pvar_idx].ptr + dbvars[evar_idx].dptr;
                    if ( FTI_Conf->dcpEnabled ) {
                        if( FTI_InitBlockHashArray( &(dbvars[evar_idx]) ) != FTI_SCES ) {
                            FTI_FinalizeDcp( FTI_Conf, FTI_Exec );
                        }
                    }
                    dbvars[evar_idx].update = true;
                    // TODO create hash of chunk later
                    //FTIFF_GetHashdbvar( dbvars[evar_idx].myhash, &(dbvars[evar_idx]) );
                    FTI_Exec->nbVarStored++;

                    break;

                case 2:

                    // create data chunk info
                    dbvars = (FTIFF_dbvar*) realloc( dblock->dbvars, (evar_idx+1) * sizeof(FTIFF_dbvar) );
                    dbvars[evar_idx].fptr = offset;
                    dbvars[evar_idx].dptr = containerSizesAccu;
                    dbvars[evar_idx].id = FTI_Data[pvar_idx].id;
                    dbvars[evar_idx].idx = pvar_idx;
                    dbvars[evar_idx].chunksize = overflow;
                    dbvars[evar_idx].hascontent = true;
                    dbvars[evar_idx].hasCkpt = false;
                    dbvars[evar_idx].containerid = nbContainers;
                    dbvars[evar_idx].containersize = overflow; 
                    dbsize += dbvars[evar_idx].containersize; 
                    dbvars[evar_idx].cptr = FTI_Data[pvar_idx].ptr + dbvars[evar_idx].dptr;
                    if ( FTI_Conf->dcpEnabled ) {
                        if( FTI_InitBlockHashArray( &(dbvars[evar_idx]) ) != FTI_SCES ) {
                            FTI_FinalizeDcp( FTI_Conf, FTI_Exec );
                        }
                    }
                    dbvars[evar_idx].update = true;
                    // TODO create hash of chunk later
                    //FTIFF_GetHashdbvar( dbvars[evar_idx].myhash, &(dbvars[evar_idx]) );

                    break;


            }

            //dblock->previous = FTI_Exec->lastdb;
            //dblock->next = NULL;
            dblock->numvars++;
            dblock->dbsize = dbsize;
            dblock->dbvars = dbvars;
            //FTI_Exec->lastdb = dblock;
            
            dblock->update = true;
            // TODO this has to come later
            //FTIFF_GetHashdb( dblock->myhash, dblock );
        }

    }

    // FOR DEVELOPING
    // FTIFF_PrintDataStructure( 0, FTI_Exec, FTI_Data );
    
    return FTI_SCES;

}

// creates hashes of the dbvar-struct's
int FTIFF_createHashesDbVarFTIFF( FTIT_execution* FTI_Exec, FTIT_dataset* FTI_Data )
{
    int isnextdb;
    FTIFF_db *db = FTI_Exec->firstdb;
    FTIFF_dbvar *dbvar;

    do {
    
        isnextdb = 0;
        
        dbvar = db->dbvars;

        int dbvar_idx;
        for( dbvar_idx=0; dbvar_idx<db->numvars; dbvar_idx++ ) {
            // create hash for data chunk and (!!) afterwards (!!) hash of meta data
            // TODO for GPU this will be done during the write...
            // FTIFF_SetHashChunk( &(dbvar[dbvar_idx]), FTI_Data );
            FTIFF_GetHashdbvar( dbvar[dbvar_idx].myhash, &(dbvar[dbvar_idx]) );
        }
        if (db->next) {
            db = db->next;
            isnextdb = 1;
        }

    } while( isnextdb );

}

// creates hashes of the dbvar-struct's (for a ceratin ID)
int FTIFF_createHashesDbVarIdFTIFF( FTIT_execution* FTI_Exec, FTIT_dataset* FTI_Data, int idx )
{
    int isnextdb;
    FTIFF_db *db = FTI_Exec->firstdb;
    FTIFF_dbvar *dbvar;

    do {
    
        isnextdb = 0;

        dbvar = db->dbvars;

        int dbvar_idx;
        for( dbvar_idx=0; dbvar_idx<db->numvars; dbvar_idx++ ) {
            if( dbvar[dbvar_idx].id == FTI_Data[idx].id ) {
                // create hash for data chunk and (!!) afterwards (!!) hash of meta data
                // TODO for GPU this will be done during the write...
                // FTIFF_SetHashChunk( &(dbvar[dbvar_idx]), FTI_Data );
                FTIFF_GetHashdbvar( dbvar[dbvar_idx].myhash, &(dbvar[dbvar_idx]) );
            }
        }
        if (db->next) {
            db = db->next;
            isnextdb = 1;
        }

    } while( isnextdb );

}

// finalizes datablocks (finalized=true), creates hash of db-struct's and computes and assigns metaSize to FTIFFMeta.metaSize 
//TODO remove not finalized db when checkpoint failed
int FTIFF_finalizeDatastructFTIFF( FTIT_execution* FTI_Exec, FTIT_dataset* FTI_Data )
{
    int isnextdb;
    FTIFF_db *currentdb = FTI_Exec->firstdb;
    
    unsigned long metaSize = FTI_filemetastructsize;

    do {
    
        isnextdb = 0;
        
        currentdb->finalized = true;
        
        // create hash of block metadata 
        FTIFF_GetHashdb( currentdb->myhash, currentdb );

        metaSize += FTI_dbstructsize + currentdb->numvars * FTI_dbvarstructsize;

        if (currentdb->next) {
            currentdb = currentdb->next;
            isnextdb = 1;
        }

    } while( isnextdb );

    FTI_Exec->FTIFFMeta.metaSize = metaSize;

}

// computes CP-Data hash (computes hash from the chunk hashes),creates file meta data hash,  
// serializes metadata into buffer and appends buffer to file.
int FTIFF_writeMetaDataFTIFF( FTIT_execution* FTI_Exec, int fd )
{
    char strerr[FTI_BUFS];
    int isnextdb;
    FTIFF_db *db = FTI_Exec->firstdb;
    FTIFF_dbvar *dbvar;
    
    FTI_ADDRPTR mbuf = malloc( FTI_Exec->FTIFFMeta.metaSize );
    FTI_ADDRVAL mbuf_pos = (FTI_ADDRVAL) mbuf;

    if( mbuf == NULL ) {
        return FTI_NSCS;
    }

    MD5_CTX ctx;
    MD5_Init( &ctx );

    do {
    
        isnextdb = 0;

        FTIFF_SerializeDbMeta( db, (FTI_ADDRPTR) mbuf_pos );
        mbuf_pos += FTI_dbstructsize;

        dbvar = db->dbvars;

        int dbvar_idx=0;
        for(; dbvar_idx<db->numvars; dbvar_idx++) {
            FTIFF_SerializeDbVarMeta( &dbvar[dbvar_idx], (FTI_ADDRPTR) mbuf_pos );
            // compute CP hash from chunk hashes 
            MD5_Update( &ctx, dbvar[dbvar_idx].hash, MD5_DIGEST_LENGTH );
            mbuf_pos += FTI_dbvarstructsize;
        }

        if (db->next) {
            db = db->next;
            isnextdb = 1;
        }


    } while( isnextdb );

    unsigned char fhash[MD5_DIGEST_LENGTH];
    MD5_Final( fhash, &ctx );
    
    int ii = 0, i;
    for(i = 0; i < MD5_DIGEST_LENGTH; i++) {
        sprintf(&(FTI_Exec->FTIFFMeta.checksum[ii]), "%02x", fhash[i]);
        ii += 2;
    }

    // compute and set hash of file meta data
    FTIFF_GetHashMetaInfo( FTI_Exec->FTIFFMeta.myHash, &(FTI_Exec->FTIFFMeta) ); 
    FTIFF_SerializeFileMeta( &FTI_Exec->FTIFFMeta, (FTI_ADDRPTR) mbuf_pos );
    
    if ( lseek( fd, FTI_Exec->FTIFFMeta.dataSize, SEEK_SET ) == -1 ) {
        snprintf(strerr, FTI_BUFS, "FTI-FF: WriteMetaDataFTIFF - could not seek in file");
        FTI_Print(strerr, FTI_EROR);
        close(fd);
        errno = 0;
        return FTI_NSCS;
    }
    
    write( fd, mbuf, FTI_Exec->FTIFFMeta.metaSize );
    if ( fd == -1 ) {
        snprintf(strerr, FTI_BUFS, "FTI-FF: WriteMetaDataFTIFF - could not write metadata in file");
        FTI_Print(strerr, FTI_EROR);
        errno=0;
        close(fd);
        return FTI_NSCS;
    }
    
    free( mbuf );

}

/*-------------------------------------------------------------------------*/
/**
  @brief      updates datablock structure for FTI File Format.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Data        Dataset metadata.
  @param      FTI_Conf        Configuration metadata.
  @return     integer         FTI_SCES if successful.

  Updates information about the checkpoint file. Updates file pointers
  in the dbvar structures and updates the db structure.

 **/
/*-------------------------------------------------------------------------*/
int FTIFF_UpdateDatastructFTIFF( FTIT_execution* FTI_Exec, 
        FTIT_dataset* FTI_Data, FTIT_configuration* FTI_Conf )
{

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
            dbvars[dbvar_idx].cptr = FTI_Data[dbvar_idx].ptr + dbvars[dbvar_idx].dptr;
            // FOR DCP 
            if  ( FTI_Conf->dcpEnabled ) {
                FTI_InitBlockHashArray( &(dbvars[dbvar_idx]) );
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
                            dbvar->cptr = data->ptr + dbvar->dptr;
                            if ( !dbvar->hascontent ) {
                                dbvar->hascontent = true;
                                // [FOR DCP] init hash array for block
                                if ( FTI_Conf->dcpEnabled ) {
                                    if( FTI_InitBlockHashArray( dbvar ) != FTI_SCES ) {
                                        FTI_FinalizeDcp( FTI_Conf, FTI_Exec );
                                    }
                                }
                            } else {
                                // [FOR DCP] adjust hash array to new chunksize if chunk size increased
                                if ( FTI_Conf->dcpEnabled ) {
                                    if (  dbvar->chunksize > chunksizeOld ) {
                                        FTI_ExpandBlockHashArray( dbvar );
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
                            dbvar->cptr = data->ptr + dbvar->dptr;
                            if ( !dbvar->hascontent ) {
                                dbvar->hascontent = true;
                                // [FOR DCP] init hash array for block
                                if ( FTI_Conf->dcpEnabled ) {
                                    if( FTI_InitBlockHashArray( dbvar )  != FTI_SCES ) {
                                        FTI_FinalizeDcp( FTI_Conf, FTI_Exec );
                                    }

                                }
                            } else {
                                // [FOR DCP] adjust hash array to new chunksize if chunk size decreased
                                if ( FTI_Conf->dcpEnabled ) {
                                    if ( dbvar->chunksize < chunksizeOld ) {
                                        FTI_CollapseBlockHashArray( dbvar );
                                    }
                                    if ( dbvar->chunksize > chunksizeOld ) {
                                        FTI_ExpandBlockHashArray( dbvar );
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
                        dbvars[evar_idx].cptr = FTI_Data[pvar_idx].ptr + dbvars[evar_idx].dptr;
                        if ( FTI_Conf->dcpEnabled ) {
                            if( FTI_InitBlockHashArray( &(dbvars[evar_idx]) ) != FTI_SCES ) {
                                FTI_FinalizeDcp( FTI_Conf, FTI_Exec );
                            }
                        }
                        dbvars[evar_idx].update = true;
                        FTIFF_GetHashdbvar( dbvars[evar_idx].myhash, &(dbvars[evar_idx]) );
                        evar_idx++;

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
                        dbvars[evar_idx].cptr = FTI_Data[pvar_idx].ptr + dbvars[evar_idx].dptr;
                        if ( FTI_Conf->dcpEnabled ) {
                            if( FTI_InitBlockHashArray( &(dbvars[evar_idx]) ) != FTI_SCES ) {
                                FTI_FinalizeDcp( FTI_Conf, FTI_Exec );
                            }
                        }
                        dbvars[evar_idx].update = true;
                        FTIFF_GetHashdbvar( dbvars[evar_idx].myhash, &(dbvars[evar_idx]) );
                        evar_idx++;

                        break;

                }

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
    int pvar_idx;
    for(pvar_idx=0; pvar_idx<FTI_Exec->nbVar; pvar_idx++) {
        FTIFF_UpdateDatastructVarFTIFF( FTI_Exec, FTI_Data, FTI_Conf, pvar_idx );
    }
    //FTIFF_UpdateDatastructFTIFF( FTI_Exec, FTI_Data, FTI_Conf );
    
    //FOR DEVELOPING 
    // FTIFF_PrintDataStructure( 0, FTI_Exec, FTI_Data );

    char str[FTI_BUFS], fn[FTI_BUFS], strerr[FTI_BUFS];
    
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
    } else if ( level == 4 && !FTI_Ckpt[4].isInline )
        if( FTI_Conf->dcpEnabled && FTI_Ckpt[4].isDcp ) {
            snprintf(fn, FTI_BUFS, "%s/%s", FTI_Ckpt[1].dcpDir, FTI_Ckpt[4].dcpName);
        } else {
            snprintf(fn, FTI_BUFS, "%s/%s", FTI_Conf->lTmpDir, FTI_Exec->meta[0].ckptFile);
        }
    else {
        snprintf(fn, FTI_BUFS, "%s/%s", FTI_Conf->lTmpDir, FTI_Exec->meta[0].ckptFile);
    }

    int fd;

    // for dCP: create if not exists, open if exists
    if ( FTI_Conf->dcpEnabled && FTI_Ckpt[4].isDcp ) {
        if (access(fn,R_OK) != 0) {
            fd = open( fn, O_WRONLY|O_CREAT, (mode_t) 0600 ); 
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

    // TODO remove assert.
    //assert(FTI_Exec->firstdb);
    FTIFF_db *currentdb;
    FTIFF_dbvar *currentdbvar;
    char *dptr;
    int dbvar_idx, dbcounter=0;

    // block size for fwrite buffer in file.
    long membs = 1024*1024*16; // 16 MB
    long cpybuf, cpynow, cpycnt;//, fptr;

    uintptr_t fptr;
    int isnextdb;
    
    long dcpSize = 0, dataSize = 0;

    // write FTI-FF meta data
    // 
#ifdef GPUSUPPORT
    copyDataFromDevive( FTI_Exec, FTI_Data );
#endif    

    currentdb = FTI_Exec->firstdb;
    
    do {    

        isnextdb = 0;

        for(dbvar_idx=0;dbvar_idx<currentdb->numvars;dbvar_idx++) {

            currentdbvar = &(currentdb->dbvars[dbvar_idx]);
            dataSize += currentdbvar->chunksize;
                
            DBG_MSG("id: %d, datasize: %lu, chunksize: %lu",0, currentdbvar->id, FTI_Data[currentdbvar->idx].size, currentdbvar->chunksize );

            // get source and destination pointer
            dptr = (char*)(FTI_Data[currentdbvar->idx].ptr) + currentdb->dbvars[dbvar_idx].dptr;
            fptr = currentdbvar->fptr;
            uintptr_t chunk_addr, chunk_size, chunk_offset;

            int chunkid = 0;
            
            while( FTI_ReceiveDataChunk(&chunk_addr, &chunk_size, currentdbvar, FTI_Data) ) {
                chunk_offset = chunk_addr - ((FTI_ADDRVAL)(FTI_Data[currentdbvar->idx].ptr) + currentdbvar->dptr);
                
                dptr += chunk_offset;
                fptr = currentdbvar->fptr + chunk_offset;
                
                unsigned long DBG_fpos; 
                
                if ( ( DBG_fpos = lseek( fd, fptr, SEEK_SET ) ) == -1 ) {
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
                        int returnVal;
                        FTI_FI_WRITE( returnVal, fd, (FTI_ADDRPTR) (chunk_addr+cpycnt), cpynow, fn );
                        if ( returnVal == -1 ) {
                            snprintf(str, FTI_BUFS, "FTI-FF: WriteFTIFF - Dataset #%d could not be written to file: %s", currentdbvar->id, fn);
                            FTI_Print(str, FTI_EROR);
                            close(fd);
                            errno = 0;
                            return FTI_NSCS;
                        }
                        WRITTEN += returnVal;
                        try++;
                    } while ((WRITTEN < cpynow) && (try < 10));
                    
                    assert( WRITTEN == cpynow );
                    
                    cpycnt += WRITTEN;
                    dcpSize += WRITTEN;
                }
                assert(cpycnt == chunk_size);

                chunkid++;

            }
            
            // create hash for datachunk
            FTIFF_SetHashChunk( currentdbvar, FTI_Data );

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
 
    // only for printout of dCP share in FTI_Checkpoint
    FTI_Exec->FTIFFMeta.dcpSize = dcpSize;
    FTI_Exec->FTIFFMeta.dataSize = dataSize;
     
    FTIFF_finalizeDatastructFTIFF( FTI_Exec, FTI_Data );
    
    if ( FTI_Try( FTIFF_CreateMetadata( FTI_Exec, FTI_Topo, FTI_Data, FTI_Conf ), "Create FTI-FF meta data" ) != FTI_SCES ) {
        return FTI_NSCS;
    }
    
    FTIFF_createHashesDbVarFTIFF( FTI_Exec, FTI_Data );
    FTIFF_writeMetaDataFTIFF( FTI_Exec, fd );

    fdatasync( fd );
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

    FTI_Exec->ckptSize = FTI_Exec->FTIFFMeta.metaSize + FTI_Exec->FTIFFMeta.dataSize;
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

                // append additionally file meta data at the end to recover 
                // original filesize if Rank file gets lost
                mfs += FTI_filemetastructsize;

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
    // FTIFF_GetHashMetaInfo( FTI_Exec->FTIFFMeta.myHash, &(FTI_Exec->FTIFFMeta) );

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
    //FTIFF_PrintDataStructure( 0, FTI_Exec, FTI_Data );
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
            FTI_Print(str, FTI_DBUG);

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
            char checkSum_struct[MD5_DIGEST_STRING_LENGTH];
            ii = 0;
            for(i = 0; i < MD5_DIGEST_LENGTH; i++) {
                sprintf(&checkSum_struct[ii], "%02x", currentdbvar->hash[i]);
                ii += 2;
            }
            snprintf(str, FTI_BUFS, "dataset hash id: %d -> %s", currentdbvar->id, checkSum);
            FTI_Print(str, FTI_DBUG);

            if ( memcmp( currentdbvar->hash, hash, MD5_DIGEST_LENGTH ) != 0 ) {
                snprintf( strerr, FTI_BUFS, "FTI-FF: FTIFF_Recover - dataset with id:%i|cnt-id:%d has been corrupted! Discard recovery (%s!=%s).", currentdbvar->id, currentdbvar->containerid,checkSum,checkSum_struct );
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
        FTIT_checkpoint* FTI_Ckpt, FTIT_configuration *FTI_Conf )
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
            snprintf(strerr, FTI_BUFS, "FTI-FF: L1RecoveryInit - checkpoint directory (%s) (%s) could not be accessed.",FTI_Ckpt[1].dir, FTI_Ckpt[1].dir);
            FTI_Print(strerr, FTI_EROR);
            errno = 0;
            free(FTIFFMeta);
            goto GATHER_L1INFO;
        }

        while((entry = readdir(L1CkptDir)) != NULL) {
            
            if(strcmp(entry->d_name,".") && strcmp(entry->d_name,"..")) { 
                snprintf(str, FTI_BUFS, "FTI-FF: L1RecoveryInit - found file with name: %s %s",FTI_Ckpt[1].dir, entry->d_name);
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

                        if ( lseek(fd, -FTI_filemetastructsize, SEEK_END) == -1 ) {
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
                                ( read( fd, &(FTIFFMeta->metaSize), sizeof(long) ) == -1 )              ||
                                ( read( fd, &(FTIFFMeta->dataSize), sizeof(long) ) == -1 )              ||
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
                            
                            char checksum[MD5_DIGEST_STRING_LENGTH];
                            FTIFF_GetFileChecksum( FTIFFMeta, FTI_Ckpt, fd, checksum ); 
                            
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
        FTIT_checkpoint* FTI_Ckpt, FTIT_configuration* FTI_Conf, int *exists)
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
 
                        if ( lseek(fd, -FTI_filemetastructsize, SEEK_END) == -1 ) {
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
                                ( read( fd, &(FTIFFMeta->metaSize), sizeof(long) ) == -1 )              ||
                                ( read( fd, &(FTIFFMeta->dataSize), sizeof(long) ) == -1 )              ||
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

                            char checksum[MD5_DIGEST_STRING_LENGTH];
                            FTIFF_GetFileChecksum( FTIFFMeta, FTI_Ckpt, fd, checksum ); 
                            
                            if ( strcmp( checksum, FTIFFMeta->checksum ) == 0 ) {
                                myMetaInfo->fs = FTIFFMeta->fs;    
                                myMetaInfo->ckptID = ckptID;    
                                myMetaInfo->FileExists = 1;
                            } else {
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
 
                        if ( lseek(fd, -FTI_filemetastructsize, SEEK_END) == -1 ) {
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
                                ( read( fd, &(FTIFFMeta->metaSize), sizeof(long) ) == -1 )              ||
                                ( read( fd, &(FTIFFMeta->dataSize), sizeof(long) ) == -1 )              ||
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
                            
                            DBG_MSG("ckptSize: %lu, dataSize: %lu, metaSize: %lu",-1,
                                    FTIFFMeta->ckptSize,FTIFFMeta->dataSize,FTIFFMeta->metaSize);
                            
                            char checksum[MD5_DIGEST_STRING_LENGTH];
                            FTIFF_GetFileChecksum( FTIFFMeta, FTI_Ckpt, fd, checksum ); 
                            
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
                                ( read( fd, &(FTIFFMeta->metaSize), sizeof(long) ) == -1 )              ||
                                ( read( fd, &(FTIFFMeta->dataSize), sizeof(long) ) == -1 )              ||
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
                            
                            char checksum[MD5_DIGEST_STRING_LENGTH];
                            FTIFF_GetFileChecksum( FTIFFMeta, FTI_Ckpt, fd, checksum ); 
                            
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
                                ( read( fd, &(FTIFFMeta->metaSize), sizeof(long) ) == -1 )              ||
                                ( read( fd, &(FTIFFMeta->dataSize), sizeof(long) ) == -1 )              ||
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
                if ( FTI_Ckpt[4].isDcp ) {
                    sscanf(entry->d_name, "dCPFile-Rank%d.fti", &fileTarget );
                } else {
                    sscanf(entry->d_name, "Ckpt%d-Rank%d.fti", &ckptID, &fileTarget );
                }
                if( fileTarget == FTI_Topo->myRank ) {
                    snprintf(str, FTI_BUFS, "FTI-FF: L4RecoveryInit - found file with name: %s", entry->d_name);
                    FTI_Print(str, FTI_DBUG);
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

void FTIFF_SetHashChunk( FTIFF_dbvar *dbvar, FTIT_dataset* FTI_Data ) 
{
    if( dbvar->hascontent ) {
        void * ptr = FTI_Data[dbvar->idx].ptr + dbvar->dptr;
        unsigned long size = dbvar->chunksize;
        MD5( ptr, size, dbvar->hash );
    }
}

/*-------------------------------------------------------------------------*/
/**
  @brief    Computes hash of the FTI-FF file meta data structure   
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
    MD5_Update( &md5Ctx, &(FTIFFMeta->metaSize), sizeof(long) );
    MD5_Update( &md5Ctx, &(FTIFFMeta->dataSize), sizeof(long) );
    MD5_Update( &md5Ctx, &(FTIFFMeta->fs), sizeof(long) );
    MD5_Update( &md5Ctx, &(FTIFFMeta->ptFs), sizeof(long) );
    MD5_Update( &md5Ctx, &(FTIFFMeta->maxFs), sizeof(long) );
    MD5_Final( hash, &md5Ctx );
}

/*-------------------------------------------------------------------------*/
/**
  @brief    Computes hash of the FTI-FF file data block meta data structure   
  @param    hash          hash to compute.
  @param    FTIFFMeta     file data block meta data.
 **/
/*-------------------------------------------------------------------------*/
void FTIFF_GetHashdb( unsigned char *hash, FTIFF_db *db ) 
{
    MD5_CTX md5Ctx;
    MD5_Init (&md5Ctx);
    MD5_Update( &md5Ctx, &(db->numvars), sizeof(int) );
    MD5_Update( &md5Ctx, &(db->dbsize), sizeof(long) );
    MD5_Final( hash, &md5Ctx );
}

/*-------------------------------------------------------------------------*/
/**
  @brief    Computes hash of the FTI-FF data chunk meta data structure   
  @param    hash          hash to compute.
  @param    dbvar         data chunk meta data.
 **/
/*-------------------------------------------------------------------------*/
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
  @brief      Initializes the derived MPI data types used for FTI-FF
 **/
/*-------------------------------------------------------------------------*/
void FTIFF_InitMpiTypes() 
{

    MPI_Aint lb, extent;
    FTIFF_MPITypeInfo MPITypeInfo[FTIFF_NUM_MPI_TYPES];

    // define MPI datatypes

    // headInfo
    MBR_CNT( headInfo ) =  7;
    MBR_BLK_LEN( headInfo ) = { 1, 1, FTI_BUFS, 1, 1, 1, 1 };
    MBR_TYPES( headInfo ) = { MPI_INT, MPI_INT, MPI_CHAR, MPI_LONG, MPI_LONG, MPI_LONG, MPI_INT };
    MBR_DISP( headInfo ) = {  
        offsetof( FTIFF_headInfo, exists), 
        offsetof( FTIFF_headInfo, nbVar), 
        offsetof( FTIFF_headInfo, ckptFile), 
        offsetof( FTIFF_headInfo, maxFs), 
        offsetof( FTIFF_headInfo, fs), 
        offsetof( FTIFF_headInfo, pfs), 
        offsetof( FTIFF_headInfo, isDcp) 
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
  @brief    deserializes FTI-FF file meta data   
  @param    meta          FTI-FF file meta data.
  @param    buffer_ser    serialized file meta data.
 **/
/*-------------------------------------------------------------------------*/
int FTIFF_DeserializeFileMeta( FTIFF_metaInfo* meta, char* buffer_ser )
{
    
    if ( (buffer_ser == NULL) || (meta == NULL) ) {
        FTI_Print("nullptr passed to 'FTIFF_DeserializeFilemeta!", FTI_WARN);
        return FTI_NSCS;
    }

    int pos = 0;
    memcpy( meta->checksum        , buffer_ser + pos, MD5_DIGEST_STRING_LENGTH );
    pos += MD5_DIGEST_STRING_LENGTH;
    memcpy( meta->myHash          , buffer_ser + pos, MD5_DIGEST_LENGTH );
    pos += MD5_DIGEST_LENGTH;
    memcpy( &(meta->ckptSize)     , buffer_ser + pos, sizeof(long) );
    pos += sizeof(long);
    memcpy( &(meta->metaSize)     , buffer_ser + pos, sizeof(long) );
    pos += sizeof(long);
    memcpy( &(meta->dataSize)     , buffer_ser + pos, sizeof(long) );
    pos += sizeof(long);
    memcpy( &(meta->fs)           , buffer_ser + pos, sizeof(long) );
    pos += sizeof(long);
    memcpy( &(meta->maxFs)        , buffer_ser + pos, sizeof(long) );
    pos += sizeof(long);
    memcpy( &(meta->ptFs)         , buffer_ser + pos, sizeof(long) );
    pos += sizeof(long);
    memcpy( &(meta->timestamp)    , buffer_ser + pos, sizeof(long) );

    return FTI_SCES;

}

/*-------------------------------------------------------------------------*/
/**
  @brief    deserializes FTI-FF file data block meta data   
  @param    db            FTI-FF file data block meta data.
  @param    buffer_ser    serialized file data block meta data.
 **/
/*-------------------------------------------------------------------------*/
int FTIFF_DeserializeDbMeta( FTIFF_db* db, char* buffer_ser )
{
    
    if ( (buffer_ser == NULL) || (db == NULL) ) {
        FTI_Print("nullptr passed to 'FTIFF_DeserializeFileMeta!", FTI_WARN);
        return FTI_NSCS;
    }

    int pos = 0;
    memcpy( &(db->numvars)    , buffer_ser + pos, sizeof(int) ); 
    pos += sizeof(int);
    memcpy( &(db->dbsize)     , buffer_ser + pos, sizeof(long) );

    return FTI_SCES;

}

/*-------------------------------------------------------------------------*/
/**
  @brief    deserializes FTI-FF data chunk meta data   
  @param    dbvar         FTI-FF data chunk meta data.
  @param    buffer_ser    serialized data chunk meta data.
 **/
/*-------------------------------------------------------------------------*/
int FTIFF_DeserializeDbVarMeta( FTIFF_dbvar* dbvar, char* buffer_ser )
{
    
    if ( (buffer_ser == NULL) || (dbvar == NULL) ) {
        FTI_Print("nullptr passed to 'FTIFF_DeserializeFileMeta!", FTI_WARN);
        return FTI_NSCS;
    }

    int pos = 0;
    memcpy( &(dbvar->id)              , buffer_ser + pos, sizeof(int));
    pos += sizeof(int);
    memcpy( &(dbvar->idx)             , buffer_ser + pos, sizeof(int));
    pos += sizeof(int);
    memcpy( &(dbvar->containerid)     , buffer_ser + pos, sizeof(int));
    pos += sizeof(int);
    memcpy( &(dbvar->hascontent)      , buffer_ser + pos, sizeof(bool));
    pos += sizeof(bool);
    memcpy( &(dbvar->hasCkpt)         , buffer_ser + pos, sizeof(bool));
    pos += sizeof(bool);
    memcpy( &(dbvar->dptr)            , buffer_ser + pos, sizeof(uintptr_t));
    pos += sizeof(uintptr_t);
    memcpy( &(dbvar->fptr)            , buffer_ser + pos, sizeof(uintptr_t));
    pos += sizeof(uintptr_t);
    memcpy( &(dbvar->chunksize)       , buffer_ser + pos, sizeof(long));
    pos += sizeof(long);
    memcpy( &(dbvar->containersize)   , buffer_ser + pos, sizeof(long));
    pos += sizeof(long);
    memcpy( dbvar->hash               , buffer_ser + pos, MD5_DIGEST_LENGTH);

    return FTI_SCES;

}

/*-------------------------------------------------------------------------*/
/**
  @brief    serializes FTI-FF file meta data   
  @param    meta          FTI-FF file meta data.
  @param    buffer_ser    serialized file meta data.
 **/
/*-------------------------------------------------------------------------*/
int FTIFF_SerializeFileMeta( FTIFF_metaInfo* meta, char* buffer_ser )
{
    
    if ( (buffer_ser == NULL) || (meta == NULL) ) {
        FTI_Print("nullptr passed to 'FTIFF_DeserializeFileMeta!", FTI_WARN);
        return FTI_NSCS;
    }

    int pos = 0;
    memcpy( buffer_ser + pos, meta->checksum        , MD5_DIGEST_STRING_LENGTH );
    pos += MD5_DIGEST_STRING_LENGTH;
    memcpy( buffer_ser + pos, meta->myHash          , MD5_DIGEST_LENGTH );
    pos += MD5_DIGEST_LENGTH;
    memcpy( buffer_ser + pos, &(meta->ckptSize)     , sizeof(long) );
    pos += sizeof(long);
    memcpy( buffer_ser + pos, &(meta->metaSize)     , sizeof(long) );
    pos += sizeof(long);
    memcpy( buffer_ser + pos, &(meta->dataSize)     , sizeof(long) );
    pos += sizeof(long);
    memcpy( buffer_ser + pos, &(meta->fs)           , sizeof(long) );
    pos += sizeof(long);
    memcpy( buffer_ser + pos, &(meta->maxFs)        , sizeof(long) );
    pos += sizeof(long);
    memcpy( buffer_ser + pos, &(meta->ptFs)         , sizeof(long) );
    pos += sizeof(long);
    memcpy( buffer_ser + pos, &(meta->timestamp)    , sizeof(long) );

    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief    serializes FTI-FF file data block meta data   
  @param    db            FTI-FF file data block meta data.
  @param    buffer_ser    serialized file data block meta data.
 **/
/*-------------------------------------------------------------------------*/
int FTIFF_SerializeDbMeta( FTIFF_db* db, char* buffer_ser )
{
    
    if ( (buffer_ser == NULL) || (db == NULL) ) {
        FTI_Print("nullptr passed to 'FTIFF_DeserializeFileMeta!", FTI_WARN);
        return FTI_NSCS;
    }

    int pos = 0;
    memcpy( buffer_ser + pos, &(db->numvars)    , sizeof(int) ); 
    pos += sizeof(int);
    memcpy( buffer_ser + pos, &(db->dbsize)     , sizeof(long) );

    return FTI_SCES;

}

/*-------------------------------------------------------------------------*/
/**
  @brief    serializes FTI-FF data chunk meta data   
  @param    dbvar         FTI-FF data chunk meta data.
  @param    buffer_ser    serialized data chunk meta data.
 **/
/*-------------------------------------------------------------------------*/
int FTIFF_SerializeDbVarMeta( FTIFF_dbvar* dbvar, char* buffer_ser )
{
    
    if ( (buffer_ser == NULL) || (dbvar == NULL) ) {
        FTI_Print("nullptr passed to 'FTIFF_DeserializeFileMeta!", FTI_WARN);
        return FTI_NSCS;
    }

    int pos = 0;
    memcpy( buffer_ser + pos, &(dbvar->id)              , sizeof(int));
    pos += sizeof(int);
    memcpy( buffer_ser + pos, &(dbvar->idx)             , sizeof(int));
    pos += sizeof(int);
    memcpy( buffer_ser + pos, &(dbvar->containerid)     , sizeof(int));
    pos += sizeof(int);
    memcpy( buffer_ser + pos, &(dbvar->hascontent)      , sizeof(bool));
    pos += sizeof(bool);
    memcpy( buffer_ser + pos, &(dbvar->hasCkpt)         , sizeof(bool));
    pos += sizeof(bool);
    memcpy( buffer_ser + pos, &(dbvar->dptr)            , sizeof(uintptr_t));
    pos += sizeof(uintptr_t);
    memcpy( buffer_ser + pos, &(dbvar->fptr)            , sizeof(uintptr_t));
    pos += sizeof(uintptr_t);
    memcpy( buffer_ser + pos, &(dbvar->chunksize)       , sizeof(long));
    pos += sizeof(long);
    memcpy( buffer_ser + pos, &(dbvar->containersize)   , sizeof(long));
    pos += sizeof(long);
    memcpy( buffer_ser + pos, dbvar->hash               , MD5_DIGEST_LENGTH);

    return FTI_SCES;

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
printf("                 metasize (offset: %d): %d\n\n", FTI_filemetastructsize, FTI_dbstructsize+dbgdb->numvars*FTI_dbvarstructsize);
                dbcnt++;
                int varid=0;
                for(; varid<dbgdb->numvars; ++varid) {
printf("         Var-id: %d\n", varid);
printf("                 id: %d\n"
       "                 idx: %d\n"
       "                 containerid: %d\n"
       "                 hascontent: %s\n"
       "                 hasCkpt: %s\n"
       "                 dptr: %lu\n"
       "                 fptr: %lu\n"
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



