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

#include <sys/mman.h>
#include <time.h>
#include <sys/types.h>
#include <dirent.h>
#include <inttypes.h>

#include "interface.h"
#include "ftiff.h"

char *filemmap;
struct stat filestats;

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
int FTIFF_ReadDbFTIFF(FTIT_configuration *FTI_Conf, FTIT_execution *FTI_Exec,
        FTIT_checkpoint* FTI_Ckpt, FTIT_keymap* FTI_Data) {
    char fn[FTI_BUFS];   // Path to the checkpoint file
    char str[FTI_BUFS];   // For console output
    char strerr[FTI_BUFS];

    int *varsFound = NULL;
    int varCnt = 0;

    // Recovering from local for L4 case in FTI_Recover
    if (FTI_Exec->ckptLvel == 4) {
        snprintf(fn, FTI_BUFS, "%s/%s", FTI_Ckpt[1].dir,
         FTI_Exec->ckptMeta.ckptFile);
    } else {
        snprintf(fn, FTI_BUFS, "%s/%s", FTI_Ckpt[FTI_Exec->ckptLvel].dir,
         FTI_Exec->ckptMeta.ckptFile);
    }

    // get filesize
    struct stat st;
    if (stat(fn, &st) == -1) {
        snprintf(strerr, FTI_BUFS, "FTI-FF: ReadDbFTIFF - could not get"
        " stats for file: %s", fn);
        FTI_Print(strerr, FTI_EROR);
        errno = 0;
        return FTI_NSCS;
    }

    uint64_t fs = st.st_size;

    // open checkpoint file for read only
    int fd = open(fn, O_RDONLY, 0);
    if (fd == -1) {
        snprintf(strerr, FTI_BUFS, "FTI-FF: ReadDbFTIFF - could not open"
        " '%s' for reading.", fn);
        FTI_Print(strerr, FTI_EROR);
        errno = 0;
        return FTI_NSCS;
    }

    // map file into memory
    unsigned char* fmmap = (unsigned char*) mmap(0, fs, PROT_READ,
     MAP_SHARED, fd, 0);
    if (fmmap == MAP_FAILED) {
        snprintf(strerr, FTI_BUFS, "FTI-FF: ReadDbFTIFF - could not map"
        " file to memory.");
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

    if (FTIFF_DeserializeFileMeta(&(FTI_Exec->FTIFFMeta),
     seek_ptr) != FTI_SCES) {
        FTI_Print("FTI-FF: ReadDbFTIFF - failed to deserialize"
        " 'FTI_Exec->FTIFFMeta'", FTI_EROR);
        munmap(fmmap, fs);
        errno = 0;
        return FTI_NSCS;
    }

    int dbcounter = 0;

    // set seek_ptr to start of meta data in file
    seek_ptr = fmmap + (FTI_ADDRVAL) FTI_Exec->FTIFFMeta.dataSize;

    int isnextdb;

    FTIFF_db *currentdb = talloc(FTIFF_db, 1);
    if (currentdb == NULL) {
        snprintf(strerr, FTI_BUFS, "FTI-FF: ReadDbFTIFF - failed to allocate"
        " %ld bytes for 'currentdb'", sizeof(FTIFF_db));
        FTI_Print(strerr, FTI_EROR);
        munmap(fmmap, fs);
        errno = 0;
        return FTI_NSCS;
    }

    FTI_Exec->firstdb = currentdb;
    FTI_Exec->firstdb->finalized = true;
    FTI_Exec->firstdb->next = NULL;
    FTI_Exec->firstdb->previous = NULL;

    do {
        isnextdb = 0;
        if (FTIFF_DeserializeDbMeta(currentdb, seek_ptr) != FTI_SCES) {
            FTI_Print("FTI-FF: ReadDbFTIFF - failed to deserialize "
                "'currentdb'", FTI_EROR);
            munmap(fmmap, fs);
            errno = 0;
            return FTI_NSCS;
        }

        currentdb->finalized = true;
        // TODO(leobago) create hash of data base meta data during
        // FTIFF_UpdateDatastruct and check consistency here to
        // prevent seg faults in case of corruption.

        // advance meta data offset
        seek_ptr += (FTI_ADDRVAL) FTI_dbstructsize;

        snprintf(str, FTI_BUFS, "FTI-FF: Updatedb - dataBlock:%i, dbsize:"
            " %ld, numvars: %i.", dbcounter, currentdb->dbsize,
             currentdb->numvars);
        FTI_Print(str, FTI_DBUG);

        currentdb->dbvars = talloc(FTIFF_dbvar, currentdb->numvars);
        if (currentdb->dbvars == NULL) {
            snprintf(strerr, FTI_BUFS, "FTI-FF: Updatedb - failed to allocate"
            " %ld bytes for 'currentdb->dbvars'",
            sizeof(FTIFF_dbvar) * currentdb->numvars);
            FTI_Print(strerr, FTI_EROR);
            munmap(fmmap, fs);
            errno = 0;
            return FTI_NSCS;
        }

        int dbvar_idx;
        for (dbvar_idx = 0; dbvar_idx < currentdb->numvars; dbvar_idx++) {
            FTIFF_dbvar *currentdbvar = &(currentdb->dbvars[dbvar_idx]);

            // get dbvar meta data
            if (FTIFF_DeserializeDbVarMeta(currentdbvar,
             seek_ptr) != FTI_SCES) {
                FTI_Print("FTI-FF: ReadDbFTIFF - failed to deserialize"
                " 'dbvar'", FTI_EROR);
                munmap(fmmap, fs);
                errno = 0;
                return FTI_NSCS;
            }
            // TODO(lebago) create hash of data base variable meta data during
            // FTIFF_UpdateDatastruct
            // and check consistency here to prevent seg faults
            // in case of corruption.

            // if dCP enabled, initialize hash clock structures
            if (FTI_Conf->dcpFtiff) {
                if (currentdbvar->hascontent) {
                    FTI_InitBlockHashArray(currentdbvar);
                } else {
                    currentdbvar->dataDiffHash = NULL;
                }
            }

            // advance meta data offset
            seek_ptr += (FTI_ADDRVAL) FTI_dbvarstructsize;

            currentdbvar->hasCkpt = true;

            FTIT_dataset* data;
            if (FTI_Data->get(&data, currentdbvar->id) != FTI_SCES)
                return FTI_NSCS;

            if (!data) {
                FTIT_dataset dataNew;
                FTI_InitDataset(FTI_Exec, &dataNew, currentdbvar->id);
                dataNew.sizeStored = currentdbvar->chunksize;
                dataNew.recovered = true;
                FTI_Data->push_back(&dataNew, currentdbvar->id);
            } else {
                data->sizeStored += currentdbvar->chunksize;
            }

            if (varCnt == 0) {
                varsFound = realloc(varsFound, sizeof(int) * (varCnt+1));
                varsFound[varCnt++] = currentdbvar->id;
            } else {
                int i;
                for (i = 0; i < varCnt; i++) {
                    if (varsFound[i] == currentdbvar->id) {
                        break;
                    }
                }
                if (i == varCnt) {
                    varsFound = realloc(varsFound, sizeof(int) * (varCnt+1));
                    varsFound[varCnt++] = currentdbvar->id;
                }
            }

            // debug information
            snprintf(str, FTI_BUFS, "FTI-FF: Updatedb -  dataBlock:%i/"
                "dataBlockVar%i id: %i, destptr: %ld, fptr: %ld, "
                "chunksize: %ld.", dbcounter, dbvar_idx,
                    currentdbvar->id, currentdbvar->dptr,
                    currentdbvar->fptr, currentdbvar->chunksize);
            FTI_Print(str, FTI_DBUG);
        }

        if (seek_ptr < seek_end) {
            FTIFF_db *nextdb = talloc(FTIFF_db, 1);
            if (nextdb == NULL) {
                snprintf(strerr, FTI_BUFS, "FTI-FF: ReadDbFTIFF - failed to "
                    "allocate %ld bytes for 'nextdb'", sizeof(FTIFF_db));
                FTI_Print(strerr, FTI_EROR);
                munmap(fmmap, fs);
                errno = 0;
                return FTI_NSCS;
            }
            currentdb->next = nextdb;
            nextdb->previous = currentdb;
            currentdb = nextdb;
            isnextdb = 1;
        }

        dbcounter++;
    } while (isnextdb);

    free(varsFound);

    FTI_Exec->nbVarStored = varCnt;

    FTI_Exec->lastdb = currentdb;
    FTI_Exec->lastdb->next = NULL;

    // unmap memory.
    if (munmap(fmmap, fs) == -1) {
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
int FTIFF_GetFileChecksum(FTIFF_metaInfo *FTIFFMeta, int fd, char *checksum) {
    char strerr[FTI_BUFS];
    unsigned char hash[MD5_DIGEST_LENGTH];
    MD5_CTX ctx;
    MD5_Init(&ctx);

    // map file into memory
    unsigned char* fmmap = (unsigned char*) mmap(0, FTIFFMeta->ckptSize,
     PROT_READ, MAP_SHARED, fd, 0);
    if (fmmap == MAP_FAILED) {
        snprintf(strerr, FTI_BUFS, "FTI-FF: GetFileChecksum - could not "
            "map file to memory.");
        FTI_Print(strerr, FTI_EROR);
        errno = 0;
        return FTI_NSCS;
    }

    // set seek_ptr to start of meta data in file
    FTI_ADDRPTR seek_ptr = fmmap + (FTI_ADDRVAL) FTIFFMeta->dataSize;

    // set end of file to last byte of metadata without file metadata
    FTI_ADDRPTR seek_end = fmmap + (FTI_ADDRVAL) FTIFFMeta->ckptSize -
     (FTI_ADDRVAL) FTI_filemetastructsize;

    FTIFF_db *db = talloc(FTIFF_db, 1);
    if (db == NULL) {
        snprintf(strerr, FTI_BUFS, "FTI-FF: ReadDbFTIFF - failed to allocate "
            "%ld bytes for 'db'", sizeof(FTIFF_db));
        FTI_Print(strerr, FTI_EROR);
        munmap(fmmap, FTIFFMeta->ckptSize);
        errno = 0;
        return FTI_NSCS;
    }

    do {
        if (FTIFF_DeserializeDbMeta(db, seek_ptr) != FTI_SCES) {
            FTI_Print("FTI-FF: ReadDbFTIFF - failed to deserialize 'db'",
             FTI_EROR);
            munmap(fmmap, FTIFFMeta->ckptSize);
            errno = 0;
            return FTI_NSCS;
        }

        seek_ptr += (FTI_ADDRVAL) FTI_dbstructsize;

        FTIFF_dbvar *dbvars = talloc(FTIFF_dbvar, db->numvars);
        if (dbvars == NULL) {
            snprintf(strerr, FTI_BUFS, "FTI-FF: Updatedb - failed to allocate"
            " %ld bytes for 'dbvars'", sizeof(FTIFF_dbvar) * db->numvars);
            FTI_Print(strerr, FTI_EROR);
            munmap(fmmap, FTIFFMeta->ckptSize);
            errno = 0;
            return FTI_NSCS;
        }

        int dbvar_idx;
        for (dbvar_idx = 0; dbvar_idx < db->numvars; dbvar_idx++) {
            FTIFF_dbvar *dbvar = &(dbvars[dbvar_idx]);

            // get dbvar meta data
            if (FTIFF_DeserializeDbVarMeta(dbvar, seek_ptr) != FTI_SCES) {
                FTI_Print("FTI-FF: ReadDbFTIFF - failed to deserialize "
                    "'dbvar'", FTI_EROR);
                munmap(fmmap, FTIFFMeta->ckptSize);
                errno = 0;
                return FTI_NSCS;
            }

            // advance meta data offset
            seek_ptr += (FTI_ADDRVAL) FTI_dbvarstructsize;

            // compute hash of chunk and file hash
            // (Note: we create the file hash from the chunk hashes due to ICP)
            if (dbvar->hascontent) {
                unsigned char chash[MD5_DIGEST_LENGTH];
                
		MD5_CTX dbvar_ctx;
		int64_t remaining = dbvar->chunksize;
		int block_size = CHUNK_SIZE;
		MD5_Init(&dbvar_ctx);
		while (remaining > 0){
		    int len_update = (remaining > block_size) ?
			    block_size : remaining;
		    int64_t offset = dbvar->chunksize - remaining;
		    MD5_Update(&dbvar_ctx, fmmap + dbvar->fptr + offset, len_update);
		    remaining -= len_update;
		}
		MD5_Final(chash, &dbvar_ctx);
		MD5_Update(&ctx, chash, MD5_DIGEST_LENGTH);
            }
        }

        free(dbvars);
    } while (seek_ptr < seek_end);

    MD5_Final(hash, &ctx);

    int i;
    int ii = 0;
    for (i = 0; i < MD5_DIGEST_LENGTH; i++) {
        snprintf(&checksum[ii], sizeof(char[3]), "%02x", hash[i]);
        ii += 2;
    }

    // unmap memory.
    if (munmap(fmmap, FTIFFMeta->ckptSize) == -1) {
        FTI_Print("FTI-FF: ReadDbFTIFF - unable to unmap memory", FTI_EROR);
        errno = 0;
        return FTI_NSCS;
    }

    free(db);

    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      updates datablock structure for FTI File Format.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Data        Dataset metadata.
  @param      FTI_Conf        Configuration metadata.
  @return     integer         FTI_SCES if successful.

  Updates information about the checkpoint file for id = FTI_Data[pvar_idx]. 
  Updates file pointers in the dbvar structures and updates the db structure.

 **/
/*-------------------------------------------------------------------------*/
int FTIFF_UpdateDatastructVarFTIFF(FTIT_execution* FTI_Exec,
        FTIT_dataset* data, FTIT_configuration* FTI_Conf) {
    if (FTI_Exec->nbVar == 0) {
        FTI_Print("FTI-FF - UpdateDatastructFTIFF: No protected Variables, "
            "discarding checkpoint!", FTI_WARN);
        return FTI_NSCS;
    }

    char strerr[FTI_BUFS];

    FTIFF_dbvar *dbvars = NULL;

    // first call to this function. This means that
    // for all variables only one chunk/container exists.
    if (!FTI_Exec->firstdb) {
        FTIFF_db *dblock = talloc(FTIFF_db, 1);
        if (dblock == NULL) {
            snprintf(strerr, FTI_BUFS, "FTI-FF: UpdateDatastructFTIFF - "
                "failed to allocate %ld bytes for 'dblock'", sizeof(FTIFF_db));
            FTI_Print(strerr, FTI_EROR);
            errno = 0;
            return FTI_NSCS;
        }

        dbvars = talloc(FTIFF_dbvar, 1);
        if (dbvars == NULL) {
            snprintf(strerr, FTI_BUFS, "FTI-FF: UpdateDatastructFTIFF - "
                "failed to allocate %ld bytes for 'dbvars'",
                 sizeof(FTIFF_dbvar) * FTI_Exec->nbVar);
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
        dbvars->id = data->id;
        dbvars->chunksize = data->size;
        dbvars->hascontent = true;
        dbvars->hasCkpt = false;
        dbvars->containerid = 0;
        dbvars->containersize = data->size;
        dbvars->cptr = data->ptr;
        // FOR DCP
        if  (FTI_Conf->dcpFtiff) {
            FTI_InitBlockHashArray(dbvars);
        } else {
            dbvars->dataDiffHash = NULL;
        }

        dbvars->update = true;

        FTI_Exec->nbVarStored = 1;
        dblock->dbsize = dbvars->containersize;

        dblock->update = true;
        dblock->finalized = false;

        // set as first datablock
        FTI_Exec->firstdb = dblock;
        FTI_Exec->lastdb = dblock;

    } else {
        int dbvar_idx;

        // 0 -> nothing to append, 1 -> new pvar, 2 -> size increased
        int editflags = 0;
        bool idFound = false;
        int isnextdb;
        int64_t offset = 0;

        /*
         *  - check if protected variable is in file info
         *  - check if size has changed
         */

        FTI_Exec->lastdb = FTI_Exec->firstdb;

        int nbContainers = 0;
        int64_t containerSizesAccu = 0;

        // init overflow with the datasizes and validBlock with true.
        bool validBlock = true;
        int64_t overflow = data->size;

        // iterate though datablock list. Current datablock is 'lastdb'.
        // At the beginning of the loop 'lastdb = firstdb'
        do {
            isnextdb = 0;
            for (dbvar_idx = 0; dbvar_idx < FTI_Exec->lastdb->numvars;
             dbvar_idx++) {
                FTIFF_dbvar* dbvar = &(FTI_Exec->lastdb->dbvars[dbvar_idx]);
                if (dbvar->id == data->id) {
                    idFound = true;
                    // collect container info
                    containerSizesAccu += dbvar->containersize;
                    nbContainers++;
                    // if data was shrinked, invalidate the following
                    // blocks (if there are),
                    // and set their chunksize to 0.
                    if (!validBlock) {
                        if (dbvar->hascontent) {
                            dbvar->hascontent = false;
                            // [FOR DCP] free hash array and
                            // hash structure in block
                            if ((dbvar->dataDiffHash != NULL) &&
                             FTI_Conf->dcpFtiff) {
                                FTI_FreeDataDiff(dbvar->dataDiffHash);
                                free(dbvar->dataDiffHash);
                                dbvar->dataDiffHash = NULL;
                            }
                        }
                        dbvar->chunksize = 0;
                        continue;
                    }
                    // if overflow > containersize, reduce overflow
                    // by containersize
                    // set chunksize to containersize and ensure that
                    // 'hascontent = true'.
                    if (overflow > dbvar->containersize) {
                        int64_t chunksizeOld = dbvar->chunksize;
                        dbvar->chunksize = dbvar->containersize;
                        dbvar->cptr = data->ptr + dbvar->dptr;
                        if (!dbvar->hascontent) {
                            dbvar->hascontent = true;
                            // [FOR DCP] init hash array for block
                            if (FTI_Conf->dcpFtiff) {
                                if (FTI_InitBlockHashArray(dbvar) != FTI_SCES) {
                                    FTI_FinalizeDcp(FTI_Conf, FTI_Exec);
                                }
                            }
                        } else {
                            // [FOR DCP] adjust hash array to new chunksize
                            // if chunk size increased
                            if (FTI_Conf->dcpFtiff) {
                                if (dbvar->chunksize > chunksizeOld) {
                                    FTI_ExpandBlockHashArray(dbvar->
                                        dataDiffHash, dbvar->chunksize);
                                }
                            }
                        }
                        overflow -= dbvar->containersize;
                        continue;
                    }
                    // if overflow <= containersize, set 'validBlock = false'
                    // in order to invalidate the
                    // following blocks, set new chunksize to overflow, set
                    // afterwards overflow to 0 and
                    // ensure that 'hascontent = true'.
                    if (overflow <= dbvar->containersize) {
                        int64_t chunksizeOld = dbvar->chunksize;
                        dbvar->chunksize = overflow;
                        dbvar->cptr = data->ptr + dbvar->dptr;
                        if (!dbvar->hascontent) {
                            dbvar->hascontent = true;
                            // [FOR DCP] init hash array for block
                            if (FTI_Conf->dcpFtiff) {
                                if (FTI_InitBlockHashArray(dbvar) != FTI_SCES) {
                                    FTI_FinalizeDcp(FTI_Conf, FTI_Exec);
                                }
                            }
                        } else {
                            // [FOR DCP] adjust hash array to new chunksize
                            // if chunk size decreased
                            if (FTI_Conf->dcpFtiff) {
                                if (dbvar->chunksize < chunksizeOld) {
                                    FTI_CollapseBlockHashArray(dbvar->
                                        dataDiffHash, dbvar->chunksize);
                                }
                                if (dbvar->chunksize > chunksizeOld) {
                                    FTI_ExpandBlockHashArray(dbvar->
                                        dataDiffHash, dbvar->chunksize);
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
            if (FTI_Exec->lastdb->next) {
                FTI_Exec->lastdb = FTI_Exec->lastdb->next;
                isnextdb = 1;
            }
        } while (isnextdb);
        // end of while, 'lastdb' is last datablock.

        // check for new protected variables (editflags == 1 / id not found)
        editflags = !idFound;

        // check if size has increased
        if (overflow > 0 && idFound) {
            editflags = 2;
        }

        if (editflags) {
            FTIFF_db *dblock;
            // if size changed or we have new variables to
            // protect, create new block.
            if (FTI_Exec->lastdb->next == NULL &&
             FTI_Exec->lastdb->finalized) {
                dblock = talloc(FTIFF_db, 1);
                if (dblock == NULL) {
                    snprintf(strerr, FTI_BUFS, "FTI-FF: UpdateDatastructFTIFF"
                    " - failed to allocate %ld bytes for 'dblock'",
                    sizeof(FTIFF_db));
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
            int64_t dbsize = dblock->dbsize;
            switch (editflags) {
                case 1:
                    // add new protected variable in next datablock
                    dbvars = (FTIFF_dbvar*) realloc(dblock->dbvars,
                     (evar_idx+1) * sizeof(FTIFF_dbvar));
                    dbvars[evar_idx].fptr = offset;
                    dbvars[evar_idx].dptr = 0;
                    dbvars[evar_idx].id = data->id;
                    dbvars[evar_idx].chunksize = data->size;
                    dbvars[evar_idx].hascontent = true;
                    dbvars[evar_idx].hasCkpt = false;
                    dbvars[evar_idx].containerid = 0;
                    dbvars[evar_idx].containersize = data->size;
                    dbsize += dbvars[evar_idx].containersize;
                    dbvars[evar_idx].cptr = data->ptr + dbvars[evar_idx].dptr;
                    if (FTI_Conf->dcpFtiff) {
                        if (FTI_InitBlockHashArray(&(dbvars[evar_idx])) !=
                         FTI_SCES) {
                            FTI_FinalizeDcp(FTI_Conf, FTI_Exec);
                        }
                    }
                    dbvars[evar_idx].update = true;
                    FTI_Exec->nbVarStored++;

                    break;

                case 2:

                    // create data chunk info
                    dbvars = (FTIFF_dbvar*) realloc(dblock->dbvars,
                     (evar_idx+1) * sizeof(FTIFF_dbvar));
                    dbvars[evar_idx].fptr = offset;
                    dbvars[evar_idx].dptr = containerSizesAccu;
                    dbvars[evar_idx].id = data->id;
                    dbvars[evar_idx].chunksize = overflow;
                    dbvars[evar_idx].hascontent = true;
                    dbvars[evar_idx].hasCkpt = false;
                    dbvars[evar_idx].containerid = nbContainers;
                    dbvars[evar_idx].containersize = overflow;
                    dbsize += dbvars[evar_idx].containersize;
                    dbvars[evar_idx].cptr = data->ptr + dbvars[evar_idx].dptr;
                    if (FTI_Conf->dcpFtiff) {
                        if (FTI_InitBlockHashArray(&(dbvars[evar_idx])) !=
                         FTI_SCES) {
                            FTI_FinalizeDcp(FTI_Conf, FTI_Exec);
                        }
                    }
                    dbvars[evar_idx].update = true;

                    break;
            }

            dblock->numvars++;
            dblock->dbsize = dbsize;
            dblock->dbvars = dbvars;

            dblock->update = true;
        }
    }

    // FOR DEVELOPING
    // FTIFF_PrintDataStructure(0, FTI_Exec, data);

    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Writes the data of memory chunk to the appropriate file location. 
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Data        Dataset metadata.
  @param      FTIFF_dbvar     dbVar to be written to the checkpoint file
  @param      dptr            pointer to the data to be stored      
  @param      currentOffset   offset from the file point where we should write the data 
  @param      fetchedBytes    number of the bytes fetched 
  @param      dcpSize         number of changed bytes 
  @param      fd              file descriptor
  @param      fn              file name

  This function writes a subset of the data of a dbvar on the checkpointed file. If the 
  data are in the CPU memory the subset is equal to the size of the dbvar otherwise
  we process smaller chunks of memory (usually equal to 32Mb).
 **/
/*-------------------------------------------------------------------------*/
int FTI_WriteMemFTIFFChunk(FTIT_execution *FTI_Exec, FTIFF_dbvar *currentdbvar,
        unsigned char *dptr, size_t currentOffset, size_t fetchedBytes,
        int64_t *dcpSize, WriteFTIFFInfo_t *fd) {
    unsigned char *chunk_addr = NULL;
    size_t chunk_size, chunk_offset;
    size_t remainingBytes = fetchedBytes;
    chunk_size = 0;
    chunk_offset = 0;

    int64_t membs = 1024*1024*16;  // 16 MB
    int64_t cpybuf, cpynow, cpycnt;  //, fptr;

    uintptr_t fptr = currentdbvar-> fptr + currentOffset;
    uintptr_t fptrTemp = fptr;
    size_t prevRemBytes = remainingBytes;

    while (FTI_ReceiveDataChunk(&chunk_addr, &chunk_size, currentdbvar,
     dptr, &remainingBytes)) {
        chunk_offset = chunk_addr - dptr;
        fptr = fptrTemp + chunk_offset;
        // #warning handle error of seek
        FTI_PosixSeek(fptr, fd);

        cpycnt = 0;
        while (cpycnt < chunk_size) {
            cpybuf = chunk_size - cpycnt;
            cpynow = (cpybuf > membs) ? membs : cpybuf;
            // #warning We need to also fault inject the writes etc.
            FTI_PosixWrite(&chunk_addr[cpycnt], cpynow, fd);
            cpycnt += cpynow;
        }
        (*dcpSize) += chunk_size;
        assert(cpycnt == chunk_size);
        dptr += (prevRemBytes-remainingBytes);
        prevRemBytes = remainingBytes;
        fptrTemp += chunk_offset + chunk_size;
    }
    assert(remainingBytes == 0);
    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Writes the data of a single dbVar to the checkpoint file.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Conf        Configuration metadata.
  @param      FTIFF_dbvar     dbVar to be written to the checkpoint file
  @param      FTI_Data        Dataset metadata.
  @param      hashchk         On return it contains the checksum of this dataset
  @param      fd              File descriptor of the checkpoint file 
  @param      fn              Name of the checkpoint file
  @param      dcpSize         On return it will store the number of bytes actually written.
  @param      dptr            Memory location of the processed data (for debugging prints)
  @return     integer         FTI_SCES if successful.

  This function writes the FTIFF datachunk to the checkpoint file. In the case of dcp
  it only stores the data that have changed up to now. 
 **/
/*-------------------------------------------------------------------------*/
int FTI_ProcessDBVar(FTIT_execution *FTI_Exec, FTIT_configuration *FTI_Conf,
    FTIFF_dbvar *currentdbvar, FTIT_dataset *data, unsigned char *hashchk,
     WriteFTIFFInfo_t *fd, int64_t *dcpSize, unsigned char **dptr) {
    bool hascontent = currentdbvar->hascontent;
    unsigned char *cbasePtr = NULL;
    errno = 0;


    size_t totalBytes;
    MD5_CTX dbContext;

    if (hascontent) {
        // Now I allocate the New hash tables

        FTI_InitNextHashData(currentdbvar->dataDiffHash);

        size_t offset = 0;
        totalBytes = 0;
        FTIT_data_prefetch prefetcher;
        MD5_Init(&dbContext);

        // DCP_BLOCK_SIZE is 1 if dcp is disabled.
        // I nitialize prefetcher to get data from device

#ifdef GPUSUPPORT
        size_t DCP_BLOCK_SIZE = FTI_GetDiffBlockSize();
        prefetcher.fetchSize = ((FTI_Conf->cHostBufSize) /
         DCP_BLOCK_SIZE) * DCP_BLOCK_SIZE;
#else
	prefetcher.fetchSize = 1024*1024*1024; // 1GB
#endif

        prefetcher.totalBytesToFetch = currentdbvar->chunksize;
        prefetcher.isDevice = data->isDevicePtr;

        if (prefetcher.isDevice) {
            prefetcher.dptr = (unsigned char *)((FTI_ADDRVAL)(data->devicePtr) +
             currentdbvar->dptr);
            *dptr = prefetcher.dptr;
        } else {
            prefetcher.dptr = (unsigned char *)((FTI_ADDRVAL)(data->ptr) +
             currentdbvar->dptr);
            *dptr =  prefetcher.dptr;
        }

        FTI_InitPrefetcher(&prefetcher);

        if (FTI_Try(FTI_getPrefetchedData (&prefetcher, &totalBytes,
         &cbasePtr), " Fetching Next Memory block from memory") != FTI_SCES) {
            return FTI_NSCS;
        }

        while (cbasePtr) {
	    int64_t remaining = totalBytes;
            int block_size = CHUNK_SIZE;
            while (remaining > 0){
                int len_update = (remaining > block_size) ?
                    block_size : remaining;
                int64_t offset = totalBytes - remaining;
                MD5_Update(&dbContext, cbasePtr + offset, len_update);
                remaining -= len_update;
            }
            FTI_WriteMemFTIFFChunk(FTI_Exec, currentdbvar, cbasePtr, offset,
             totalBytes, dcpSize, fd);
            offset+=totalBytes;
            if (FTI_Try(FTI_getPrefetchedData (&prefetcher, &totalBytes,
             &cbasePtr),
             " Fetching Next Memory block from memory") != FTI_SCES) {
                return FTI_NSCS;
            }
        }
        MD5_Final(hashchk, &dbContext);
    }

    printf("Ends.\n"); fflush(stdout);

    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Initializes ckpt to local/PFS using FTIFF.
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Topo        Topology metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @param      FTI_Data        Dataset metadata.
  @return     void*           Pointer to FTI-FF file descriptor.

  FTI-FF structure:
  =================

  +------------------------++--------------+
  |                        ||              |
  | VB                     || MB           |
  |                        ||              |
  +------------------------++--------------+

  The MB (Meta block) holds meta data related to the file and the size and 
  location of the variable chunks whereas the VB (variable block) holds the
  actual data of the variables protected by FTI. 

  |<------------------------------------ VB ------------------------------------>|
  #                                                                              #
  |<------------ VCB_1--------------->|      |<------------ VCB_n--------------->|
  #                                   #      #                                   #       
  +-----------------------------------+      +-----------------------------------+
  | +-------++-------+      +-------+ |      | +-------++-------+      +-------+ |
  | |       ||       |      |       | |      | |       ||       |      |       | |
  | | VC_11 || VC_12 | ---- | VC_1k | | ---- | | VC_n1 || VC_n2 | ---- | VC_nl | |
  | |       ||       |      |       | |      | |       ||       |      |       | |
  | +-------++-------+      +-------+ |      | +-------++-------+      +-------+ |
  +-----------------------------------+      +-----------------------------------+
  
  VCB_i - short for 'Variable Chunk Block' 
  VC_ij - are the data chunks.
  
  |<------------------------------------ MB ---------------------------------->|
  #                                                                            #
  |<------------ MD_1------------->|    |<------------ MD_n------------->|
  #                                #    #                                #       
  +--------------------------------+    +--------------------------------++----+
  | +------++-------+    +-------+ |    | +------++-------+    +-------+ ||    |
  | |      ||       |    |       | |    | |      ||       |    |       | ||    |
  | | MI_1 || MV_11 | -- | MV_1k | | -- | | MI_1 || MV_n1 | -- | MV_nl | || MF |
  | |      ||       |    |       | |    | |      ||       |    |       | ||    |
  | +------++-------+    +-------+ |    | +------++-------+    +-------+ ||    |
  +--------------------------------+    +--------------------------------++----+
  
  MD_i  - Meta data block
  MI_i  - Meta data for block (number variables blocksize)
  MV_ij - Variable chunk meta data (id, location in file, ptr offset, etc...)
  MF    - File meta data

 **/
/*-------------------------------------------------------------------------*/
void* FTI_InitFtiff(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt,
        FTIT_keymap* FTI_Data) {
    char fn[FTI_BUFS];
    WriteFTIFFInfo_t *write_info = talloc(WriteFTIFFInfo_t, 1);
    FTI_Print("I/O mode: FTI File Format.", FTI_DBUG);
    // only for printout of dCP share in FTI_Checkpoint
    FTI_Exec->FTIFFMeta.dcpSize = 0;
    // important for reading and writing operations
    FTI_Exec->FTIFFMeta.dataSize = 0;
    FTI_Exec->FTIFFMeta.pureDataSize = 0;

    // update ckpt file name
    snprintf(FTI_Exec->ckptMeta.ckptFile, FTI_BUFS, "Ckpt%d-Rank%d.%s",
     FTI_Exec->ckptMeta.ckptId, FTI_Topo->myRank, FTI_Conf->suffix);

    // If inline L4 save directly to global directory
    int level = FTI_Exec->ckptMeta.level;
    if (level == 4 && FTI_Ckpt[4].isInline) {
        if (FTI_Conf->dcpFtiff&& FTI_Ckpt[4].isDcp) {
            snprintf(fn, FTI_BUFS, "%s/%s", FTI_Ckpt[4].dcpDir,
             FTI_Ckpt[4].dcpName);
        } else {
            snprintf(fn, FTI_BUFS, "%s/%s", FTI_Conf->gTmpDir,
             FTI_Exec->ckptMeta.ckptFile);
        }
    } else if (level == 4 && !FTI_Ckpt[4].isInline) {
        if (FTI_Conf->dcpFtiff && FTI_Ckpt[4].isDcp) {
            snprintf(fn, FTI_BUFS, "%s/%s", FTI_Ckpt[1].dcpDir,
             FTI_Ckpt[4].dcpName);
        } else {
            snprintf(fn, FTI_BUFS, "%s/%s", FTI_Conf->lTmpDir,
             FTI_Exec->ckptMeta.ckptFile);
        }
    } else {
        snprintf(fn, FTI_BUFS, "%s/%s", FTI_Conf->lTmpDir,
        FTI_Exec->ckptMeta.ckptFile);
    }

    // for dCP: create if not exists, open if exists
    if (FTI_Conf->dcpFtiff && FTI_Ckpt[4].isDcp) {
        if (access(fn, R_OK) != 0) {
            write_info->flag = 'w';
        } else {
            write_info->flag = 'e';  // e means extend file
        }
    } else {
        write_info->flag = 'w';
    }
    write_info->offset = 0;
    FTI_PosixOpen(fn, write_info);
    write_info->FTI_Conf = FTI_Conf;
    write_info->FTI_Exec = FTI_Exec;
    write_info->FTI_Topo = FTI_Topo;
    write_info->FTI_Ckpt = FTI_Ckpt;
    write_info->FTI_Data = FTI_Data;
    return write_info;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Writes protected buffer to ckpt file.
  @param      FTI_Data        Dataset metadata.
  @param      fd              Pointer to filedescriptor.
  @return     integer         FTI_SCES if successful.

  FTI_Data here is a particular element from the global protected dataset 
  array FTI_Data.
 **/
/*-------------------------------------------------------------------------*/
int FTI_WriteFtiffData(FTIT_dataset* data, void *fd) {
    WriteFTIFFInfo_t *write_info = (WriteFTIFFInfo_t*) fd;

    FTIFF_db *db = write_info->FTI_Exec->firstdb;
    FTIFF_dbvar *dbvar = NULL;
    unsigned char *dptr;
    int dbvar_idx, dbcounter = 0;
    int isnextdb;
    int64_t dcpSize = 0;
    int64_t dataSize = 0;
    int64_t pureDataSize = 0;

    FTIFF_UpdateDatastructVarFTIFF(write_info->FTI_Exec, data,
     write_info->FTI_Conf);

    // check if metadata exists
    if (write_info->FTI_Exec->firstdb == NULL) {
        FTI_Print("No data structure found to write data to file. "
            "Discarding checkpoint.", FTI_WARN);
        return FTI_NSCS;
    }

    db = write_info->FTI_Exec->firstdb;

    do {
        isnextdb = 0;

        for (dbvar_idx = 0; dbvar_idx < db->numvars; dbvar_idx++) {
            dbvar = &(db->dbvars[dbvar_idx]);

            if (dbvar->id == data->id) {
                unsigned char hashchk[MD5_DIGEST_LENGTH];
                // important for dCP!
                // TODO(leobago) check if we can use:
                // 'dataSize += dbvar->chunksize'
                // for dCP disabled
                dataSize += dbvar->containersize;
                if (dbvar->hascontent)
                    pureDataSize += dbvar->chunksize;

                FTI_ProcessDBVar(write_info->FTI_Exec, write_info->FTI_Conf,
                 dbvar , data, hashchk, fd, &dcpSize, &dptr);
                // create hash for datachunk and assign to member 'hash'
                if (dbvar->hascontent) {
                    memcpy(dbvar->hash, hashchk, MD5_DIGEST_LENGTH);
                }
            }
        }

        if (db->next) {
            db = db->next;
            isnextdb = 1;
        }

        dbcounter++;
    } while (isnextdb);

    // only for printout of dCP share in FTI_Checkpoint
    write_info->FTI_Exec->FTIFFMeta.dcpSize += dcpSize;
    write_info->FTI_Exec->FTIFFMeta.pureDataSize += pureDataSize;

    // important for reading and writing operations
    write_info->FTI_Exec->FTIFFMeta.dataSize += dataSize;

    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Finalizes the ckpt file.
  @param      fd              Pointer to filedescriptor.
  @return     integer         FTI_SCES if successful.
 **/
/*-------------------------------------------------------------------------*/
int FTI_FinalizeFtiff(void *fd) {
    WriteFTIFFInfo_t *write_info = (WriteFTIFFInfo_t*) fd;

    if (FTI_Try(FTIFF_CreateMetadata(write_info->FTI_Exec,
     write_info->FTI_Topo, write_info->FTI_Conf),
     "Create FTI-FF meta data") != FTI_SCES) {
        return FTI_NSCS;
    }

    write_info->FTI_Exec->FTIFFMeta.ckptId = write_info->FTI_Exec->
    ckptMeta.ckptId;

    FTIFF_writeMetaDataFTIFF(write_info->FTI_Exec, write_info);

    FTI_PosixSync(write_info);
    FTI_PosixClose(write_info);

    // set hasCkpt flags true
    if (write_info->FTI_Ckpt[4].isDcp) {
        FTIFF_db* currentDB = write_info->FTI_Exec->firstdb;
        currentDB->update = false;
        do {
            int varIdx;
            for (varIdx = 0; varIdx < currentDB->numvars; ++varIdx) {
                FTIFF_dbvar* currentdbVar = &(currentDB->dbvars[varIdx]);
                currentdbVar->hasCkpt = true;
                currentdbVar->update = false;
            }
        } while ((currentDB = currentDB->next) != NULL);


        FTI_UpdateDcpChanges(write_info->FTI_Exec);
        write_info->FTI_Ckpt[4].hasDcp = true;
    }
    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      needed for consistency of ckpt methodology.
  @return     integer         0.
 **/
/*-------------------------------------------------------------------------*/
size_t FTI_DummyFilePos(void *ignore) {
    return 0;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      creates hashes of chunk meta data.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Data        Dataset metadata.
  @return     integer         FTI_SCES if successful.

  iterates through FTI-FF meta data structure, creates the hashes of the
  variable chunks meta data and stores them into 'myHash'. It is important
  that this function is called AFTER all the members are properly updated.
 **/
/*-------------------------------------------------------------------------*/
int FTIFF_createHashesDbVarFTIFF(FTIT_execution* FTI_Exec) {
    FTIFF_db *db = FTI_Exec->firstdb;
    FTIFF_dbvar *dbvar;

    if (db == NULL) {
        FTI_Print("FTIFF_createHashesDbVarFTIFF: no meta data available"
        " (FTI_Exec->firstdb == NULL)", FTI_WARN);
        return FTI_NSCS;
    }

    do {
        dbvar = db->dbvars;

        if (dbvar == NULL) {
            FTI_Print("FTIFF_createHashesDbVarFTIFF: no variable chunk meta"
            " data available (db->dbvars == NULL)", FTI_WARN);
            return FTI_NSCS;
        }

        int dbvar_idx;
        for (dbvar_idx = 0; dbvar_idx < db->numvars; dbvar_idx++) {
            FTIFF_GetHashdbvar(dbvar[dbvar_idx].myhash, &(dbvar[dbvar_idx]));
        }
    } while ((db = db->next));

    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      finalizes meta data blocks and determines meta data size.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Data        Dataset metadata.
  @return     integer         FTI_SCES if successful.

  iterates through FTI-FF meta data structure, determines the meta data size 
  and finalizes the meta data blocks. The finalize member is important 
  for integration of iCP. That is, we need to update the meta
  data blocks during the iCP phase iteratively for each protected variable.
  Newly created blocks are not finalized and may be extended during the
  iCP phase.
 **/
/*-------------------------------------------------------------------------*/
int FTIFF_finalizeDatastructFTIFF(FTIT_execution* FTI_Exec) {
    FTIFF_db *db = FTI_Exec->firstdb;

    if (db == NULL) {
        FTI_Print("FTIFF_finalizeDatastructFTIFF: no meta data available"
        " (FTI_Exec->firstdb == NULL)", FTI_WARN);
        return FTI_NSCS;
    }

    uint64_t metaSize = FTI_filemetastructsize;

    do {
        db->finalized = true;

        // create hash of block metadata
        FTIFF_GetHashdb(db->myhash, db);

        metaSize += FTI_dbstructsize + db->numvars * FTI_dbvarstructsize;
    } while ((db = db->next));

    FTI_Exec->FTIFFMeta.metaSize = metaSize;

    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      creates file hash and appends meta data to Ckpt file
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Data        Dataset metadata.
  @return     integer         FTI_SCES if successful.

  The file hash is created out of the hashes for the variable chunks. We
  have to do this in order to have a consistent implementation for both, 
  conventional and incremental checkpointing.
 **/
/*-------------------------------------------------------------------------*/
int FTIFF_writeMetaDataFTIFF(FTIT_execution* FTI_Exec, WriteFTIFFInfo_t *fd) {
    FTIFF_db *db = FTI_Exec->firstdb;
    FTIFF_dbvar *dbvar;

    FTI_ADDRPTR mbuf = malloc(FTI_Exec->FTIFFMeta.metaSize);
    if (mbuf == NULL) {
        FTI_Print("FTIFF_writeMetaDataFTIFF: failed to allocate memory"
        " for 'mbuf'!", FTI_EROR);
        return FTI_NSCS;
    }

    FTI_ADDRVAL mbuf_pos = (FTI_ADDRVAL) mbuf;

    MD5_CTX ctx;
    MD5_Init(&ctx);

    do {
        FTIFF_SerializeDbMeta(db, (FTI_ADDRPTR) mbuf_pos);
        mbuf_pos += FTI_dbstructsize;

        dbvar = db->dbvars;

        int dbvar_idx = 0;
        for (; dbvar_idx < db->numvars; dbvar_idx++) {
            FTIFF_SerializeDbVarMeta(&dbvar[dbvar_idx],
             (FTI_ADDRPTR) mbuf_pos);
            // compute CP hash from chunk hashes
            MD5_Update(&ctx, dbvar[dbvar_idx].hash, MD5_DIGEST_LENGTH);
            mbuf_pos += FTI_dbvarstructsize;
        }
    } while ((db = db->next));

    unsigned char fhash[MD5_DIGEST_LENGTH];
    MD5_Final(fhash, &ctx);

    int ii = 0, i;
    for (i = 0; i < MD5_DIGEST_LENGTH; i++) {
        snprintf(&(FTI_Exec->FTIFFMeta.checksum[ii]), sizeof(char[3]),
                "%02x", fhash[i]);
        ii += 2;
    }

    // compute and set hash of file meta data
    FTIFF_GetHashMetaInfo(FTI_Exec->FTIFFMeta.myHash, &(FTI_Exec->FTIFFMeta));
    FTIFF_SerializeFileMeta(&FTI_Exec->FTIFFMeta, (FTI_ADDRPTR) mbuf_pos);

    // #warning handle error of seek
    FTI_PosixSeek(FTI_Exec->FTIFFMeta.dataSize, fd);

    // #warning handle error of write
    FTI_PosixWrite(mbuf, FTI_Exec->FTIFFMeta.metaSize, fd);

    free(mbuf);

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
int FTIFF_CreateMetadata(FTIT_execution* FTI_Exec, FTIT_topology* FTI_Topo,
        FTIT_configuration* FTI_Conf) {
    // determine meta data size and finalize meta data blocks
    FTIFF_finalizeDatastructFTIFF(FTI_Exec);

    FTI_Exec->ckptSize = FTI_Exec->FTIFFMeta.metaSize +
     FTI_Exec->FTIFFMeta.dataSize;
    int64_t fs = FTI_Exec->ckptSize;
    FTI_Exec->FTIFFMeta.ckptSize = fs;
    FTI_Exec->FTIFFMeta.fs = fs;

    // allgather not needed for L1 checkpoint
    if ((FTI_Exec->ckptMeta.level == 2) || (FTI_Exec->ckptMeta.level == 3)) {
        int64_t fileSizes[FTI_BUFS], mfs = 0;
        MPI_Allgather(&fs, 1, MPI_INT32_T, fileSizes, 1, MPI_INT32_T,
         FTI_Exec->groupComm);
        int ptnerGroupRank, i;
        switch (FTI_Exec->ckptMeta.level) {
            // get partner file size:
            case 2:

                ptnerGroupRank = (FTI_Topo->groupRank +
                 FTI_Topo->groupSize - 1) % FTI_Topo->groupSize;
                FTI_Exec->FTIFFMeta.ptFs = fileSizes[ptnerGroupRank];
                FTI_Exec->FTIFFMeta.maxFs = -1;
                break;

                // get max file size in group
            case 3:
                for (i = 0; i < FTI_Topo->groupSize; i++) {
                    if (fileSizes[i] > mfs) {
                        mfs = fileSizes[i];  // Search max. size
                    }
                }

                // increase maxFs in order to append additionally file
                // meta data at the end to recover original filesize if Rank
                // file gets lost
                // [Important for FTI_RSenc after file truncation to maxFs]
                mfs += sizeof(off_t);

                FTI_Exec->FTIFFMeta.maxFs = mfs;
                FTI_Exec->FTIFFMeta.ptFs = -1;
        }

    } else {
        FTI_Exec->FTIFFMeta.ptFs = -1;
        FTI_Exec->FTIFFMeta.maxFs = -1;
    }

    FTI_Exec->ckptMeta.fs = FTI_Exec->FTIFFMeta.fs;
    FTI_Exec->ckptMeta.pfs = FTI_Exec->FTIFFMeta.ptFs;
    FTI_Exec->ckptMeta.maxFs = FTI_Exec->FTIFFMeta.maxFs;

    // write meta data and its hash
    struct timespec ntime;
    if (clock_gettime(CLOCK_REALTIME, &ntime) == -1) {
        FTI_Print("FTI-FF: FTIFF_CreateMetaData - failed to determine time,"
            " timestamp set to -1", FTI_WARN);
        FTI_Exec->FTIFFMeta.timestamp = -1;
    } else {
        FTI_Exec->FTIFFMeta.timestamp = ntime.tv_sec*1000000000 + ntime.tv_nsec;
    }

    // create hashes of chunks meta data
    FTIFF_createHashesDbVarFTIFF(FTI_Exec);

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
int FTIFF_Recover(FTIT_execution *FTI_Exec, FTIT_keymap *FTI_Data,
 FTIT_checkpoint *FTI_Ckpt) {
    // FTIFF_PrintDataStructure(0, FTI_Exec, FTI_Data);
    if (FTI_Exec->initSCES == 0) {
        FTI_Print("FTI is not initialized.", FTI_WARN);
        return FTI_NSCS;
    }
    if (FTI_Exec->initSCES == 2) {
        FTI_Print("No checkpoint files to make recovery.", FTI_WARN);
        return FTI_NSCS;
    }

    char fn[FTI_BUFS];  // Path to the checkpoint file
    char str[FTI_BUFS];  // For console output

    // Check if nubmer of protected variables matches
    if (FTI_Exec->nbVar != FTI_Exec->nbVarStored) {
        snprintf(str, FTI_BUFS, "Checkpoint has %d protected variables,"
                " but FTI protects %d.", FTI_Exec->nbVarStored,
                 FTI_Exec->nbVar);
        FTI_Print(str, FTI_WARN);
        return FTI_NREC;
    }

    // Check if sizes of protected variables matches
    FTIT_dataset* data;
    if (FTI_Data->data(&data, FTI_Exec->nbVar) != FTI_SCES) return FTI_NSCS;

    int i = 0; for (; i < FTI_Exec->nbVar; i++) {
        if (data[i].size != data[i].sizeStored) {
            snprintf(str, FTI_BUFS, "Cannot recover %ld bytes to protected"
                    " variable (ID %d) size: %ld",
                    data[i].sizeStored, data[i].id,
                    data[i].size);
            FTI_Print(str, FTI_WARN);
            return FTI_NREC;
        }
    }

    if (!FTI_Exec->firstdb) {
        FTI_Print("FTI-FF: FTIFF_Recover - No db meta information. "
            "Nothing to recover.", FTI_WARN);
        return FTI_NREC;
    }

    // Recovering from local for L4 case in FTI_Recover
    if (FTI_Exec->ckptLvel == 4) {
        snprintf(fn, FTI_BUFS, "%s/%s",
         FTI_Ckpt[1].dir, FTI_Exec->ckptMeta.ckptFile);
    } else {
        snprintf(fn, FTI_BUFS, "%s/%s",
         FTI_Ckpt[FTI_Exec->ckptLvel].dir, FTI_Exec->ckptMeta.ckptFile);
    }

    char strerr[FTI_BUFS];

    // get filesize
    struct stat st;
    if (stat(fn, &st) == -1) {
        snprintf(strerr, FTI_BUFS,
         "FTI-FF: FTIFF_Recover - could not get stats for file: %s", fn);
        FTI_Print(strerr, FTI_EROR);
        errno = 0;
        return FTI_NREC;
    }

    // block size for memcpy of pointer.
    int64_t membs = 1024*1024*16;  // 16 MB
    int64_t cpybuf, cpynow, cpycnt;

    // open checkpoint file for read only
    int fd = open(fn, O_RDONLY, 0);
    if (fd == -1) {
        snprintf(strerr, FTI_BUFS,
         "FTI-FF: FTIFF_Recover - could not open '%s' for reading.", fn);
        FTI_Print(strerr, FTI_EROR);
        return FTI_NREC;
    }

    // map file into memory
    char* fmmap = (char*) mmap(0, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (fmmap == MAP_FAILED) {
        snprintf(strerr, FTI_BUFS,
         "FTI-FF: FTIFF_Recover - could not map '%s' to memory.", fn);
        FTI_Print(strerr, FTI_EROR);
        close(fd);
        return FTI_NREC;
    }

    // file is mapped, we can close it.
    close(fd);

    FTIFF_db *currentdb;
    FTIFF_dbvar *currentdbvar = NULL;
    char *destptr, *srcptr;
    int dbvar_idx, dbcounter = 0;

    // MD5 context for checksum of data chunks
    MD5_CTX mdContext;
    unsigned char hash[MD5_DIGEST_LENGTH];

    int isnextdb;

    currentdb = FTI_Exec->firstdb;

    do {
        isnextdb = 0;

        for (dbvar_idx = 0; dbvar_idx < currentdb->numvars; dbvar_idx++) {
            currentdbvar = &(currentdb->dbvars[dbvar_idx]);

            if (FTI_Data->get(&data, currentdbvar->id) != FTI_SCES)
                return FTI_NSCS;

            if (!data) {
                snprintf(str, FTI_BUFS, "id '%d' does not exist!",
                 currentdbvar->id);
                FTI_Print(str, FTI_EROR);
                return FTI_NSCS;
            }

            if (!(currentdbvar->hascontent)) {
                continue;
            }
#ifdef GPUSUPPORT
            bool  isDevice = data->isDevicePtr;

            if (isDevice) {
                destptr = (char*) data->devicePtr+ currentdbvar->dptr;
            } else {
                destptr = (char*) data->ptr + currentdbvar->dptr;
            }
#else
            destptr = (char*) data->ptr + currentdbvar->dptr;
#endif

            snprintf(str, FTI_BUFS, "[var-id:%d|cont-id:%d] destptr: %p\n",
             currentdbvar->id, currentdbvar->containerid, (void*) destptr);
            FTI_Print(str, FTI_DBUG);

            srcptr = (char*) fmmap + currentdbvar->fptr;

            MD5_Init(&mdContext);
            cpycnt = 0;
            while (cpycnt < currentdbvar->chunksize) {
                cpybuf = currentdbvar->chunksize - cpycnt;
                cpynow = (cpybuf > membs) ? membs : cpybuf;
                cpycnt += cpynow;
#ifdef GPUSUPPORT
                if (isDevice)
                    FTI_copy_to_device_async(destptr, srcptr, cpynow);
                else
                    memcpy(destptr, srcptr, cpynow);
#else
                memcpy(destptr, srcptr, cpynow);
#endif
                MD5_Update(&mdContext, srcptr , cpynow);
                destptr += cpynow;
                srcptr += cpynow;
            }

            // debug information
            snprintf(str, FTI_BUFS, "FTI-FF: FTIFF_Recover -  "
                    "dataBlock:%i/dataBlockVar%i id: %i"
                    ", destptr: %ld, fptr: %ld, chunksize: %ld, "
                    "base_ptr: 0x%" PRIxPTR " ptr_pos: 0x%" PRIxPTR ".",
                    dbcounter, dbvar_idx,
                    currentdbvar->id, currentdbvar->dptr,
                    currentdbvar->fptr, currentdbvar->chunksize,
                    (uintptr_t)data->ptr, (uintptr_t)destptr);
            FTI_Print(str, FTI_DBUG);

            MD5_Final(hash, &mdContext);

            // JUST TESTING - print checksum current dataset.
            char checkSum[MD5_DIGEST_STRING_LENGTH];
            int ii = 0, i;
            for (i = 0; i < MD5_DIGEST_LENGTH; i++) {
                snprintf(&checkSum[ii], sizeof(char[3]), "%02x", hash[i]);
                ii += 2;
            }
            char checkSum_struct[MD5_DIGEST_STRING_LENGTH];
            ii = 0;
            for (i = 0; i < MD5_DIGEST_LENGTH; i++) {
                snprintf(&checkSum_struct[ii], sizeof(char[3]),
                        "%02x", currentdbvar->hash[i]);
                ii += 2;
            }
            snprintf(str, FTI_BUFS, "dataset hash id: %d -> %s",
             currentdbvar->id, checkSum);
            FTI_Print(str, FTI_DBUG);

            if (memcmp(currentdbvar->hash, hash, MD5_DIGEST_LENGTH) != 0) {
                snprintf(strerr, FTI_BUFS, "FTI-FF: FTIFF_Recover - dataset"
                " with id:%i|cnt-id:%d has been corrupted! Discard recovery"
                " (%s!=%s).", currentdbvar->id, currentdbvar->containerid,
                 checkSum, checkSum_struct);
                FTI_Print(strerr, FTI_WARN);
                if (munmap(fmmap, st.st_size) == -1) {
                    FTI_Print("FTIFF: FTIFF_Recover - unable to unmap memory",
                     FTI_EROR);
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
    } while (isnextdb);

    FTI_device_sync();
    // unmap memory
    if (munmap(fmmap, st.st_size) == -1) {
        FTI_Print("FTIFF: FTIFF_Recover - unable to unmap memory", FTI_EROR);
        errno = 0;
        return FTI_NREC;
    }

    FTI_Exec->reco = 0;

    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Initializes variable recovery for FTIFF I/O mode
  @param      fn                     ckpt file           
  @return     Integer                FTI_SCES if successful 
                                        
 **/
/*-------------------------------------------------------------------------*/
int FTIFF_RecoverVarInit(char* fn) {
    int res = FTI_SCES;

    if (stat(fn, &filestats) == -1) {
        // could not open file
        res = FTI_NREC;
    }

    // open checkpoint file for read only
    int fd = open(fn, O_RDONLY, 0);
    if (fd == -1) {
        res = FTI_NREC;
    }
    // map file into memory
    filemmap = (char*) mmap(0, filestats.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (filemmap == MAP_FAILED) {
        res = FTI_NREC;
    }
    // file is mapped, we can close it.
    close(fd);
    return res;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Recovers variable for FTIFF mode
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Topo        Topology metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @param      FTI_Data        Dataset metadata.
  @param      id                     variable id           
  @return     Integer                FTI_SCES if successful 
                                        
 **/
/*-------------------------------------------------------------------------*/
int FTIFF_RecoverVar(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
    FTIT_topology* FTI_Topo, FTIT_checkpoint *FTI_Ckpt,
    FTIT_keymap *FTI_Data, int id) {
    int res = FTI_NREC;
    FTIFF_db *currentdb;
    FTIFF_dbvar *currentdbvar = NULL;
    char *destptr, *srcptr;
    int dbvar_idx, dbcounter = 0;

    // block size for memcpy of pointer.
    int64_t membs = 1024*1024*16;  // 16 MB
    int64_t cpybuf, cpynow, cpycnt;

    // MD5 context for checksum of data chunks
    MD5_CTX mdContext;
    unsigned char hash[MD5_DIGEST_LENGTH];

    int isnextdb;

    char str[FTI_BUFS], strerr[FTI_BUFS];

    currentdb = FTI_Exec->firstdb;
    do {
        isnextdb = 0;

        for (dbvar_idx = 0; dbvar_idx < currentdb->numvars; dbvar_idx++) {
            currentdbvar = &(currentdb->dbvars[dbvar_idx]);

            if (currentdbvar->id == id) {
                // get source and destination pointer

                FTIT_dataset* data;

                if (FTI_Data->get(&data, id) != FTI_SCES) return FTI_NSCS;

                if (!data) {
                    snprintf(str, FTI_BUFS, "id '%d' does not exist!", id);
                    FTI_Print(str, FTI_EROR);
                    return FTI_NSCS;
                }
                if (data->size != data->sizeStored) {
                    snprintf(str, sizeof(str), "Cannot recover %ld bytes to "
                        "protected variable (ID %d) size: %ld",
                        data->sizeStored, data->id, data->size);
                    FTI_Print(str, FTI_WARN);
                    return FTI_NREC;
                }

                destptr = (char*) data->ptr + currentdbvar->dptr;
                srcptr = (char*) filemmap + currentdbvar->fptr;

                MD5_Init(&mdContext);
                cpycnt = 0;
                while (cpycnt < currentdbvar->chunksize) {
                    cpybuf = currentdbvar->chunksize - cpycnt;
                    cpynow = (cpybuf > membs) ? membs : cpybuf;
                    cpycnt += cpynow;
                    memcpy(destptr, srcptr, cpynow);
                    MD5_Update(&mdContext, destptr, cpynow);
                    destptr += cpynow;
                    srcptr += cpynow;
                }

                // debug information
                snprintf(str, FTI_BUFS, "FTIFF: FTIFF_RecoverVar -"
                        "  dataBlock:%i/dataBlockVar%i id: %i"
                        ", destptr: %ld, fptr: %ld, chunksize: %ld, "
                        "base_ptr: 0x%" PRIxPTR " ptr_pos: 0x%" PRIxPTR ".",
                        dbcounter, dbvar_idx,
                        currentdbvar->id, currentdbvar->dptr,
                        currentdbvar->fptr, currentdbvar->chunksize,
                        (uintptr_t)data->ptr, (uintptr_t)destptr);
                FTI_Print(str, FTI_DBUG);

                MD5_Final(hash, &mdContext);

                if (memcmp(currentdbvar->hash, hash, MD5_DIGEST_LENGTH) != 0) {
                    snprintf(strerr, FTI_BUFS,
                     "FTIFF: FTIFF_RecoverVar - dataset with id:%i has been "
                     "corrupted! Discard recovery.", currentdbvar->id);
                    FTI_Print(strerr, FTI_WARN);
                    if (munmap(filemmap, filestats.st_size) == -1) {
                        FTI_Print("FTIFF: FTIFF_RecoverVar - unable to"
                        " unmap memory", FTI_EROR);
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
    } while (isnextdb);
    return res;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Finalizes variable recovery for FTIFF mode
  @return     Integer                FTI_SCES if successful 
                                        
 **/
/*-------------------------------------------------------------------------*/
int FTIFF_RecoverVarFinalize() {
    int res = FTI_NREC;
    if (munmap(filemmap, filestats.st_size) == -1) {
        FTI_Print("Could not close FTI checkpoint file.", FTI_EROR);
    } else {
        res = FTI_SCES;
    }
    return res;
}


/*-------------------------------------------------------------------------*/
/**
  @brief      Sets filename for the recovery
  @param      dir             Directory of recovery for 'level'.
  @param      rank            Process rank of global communicator.
  @param      level           Recovery level.
  @param      dcp             1 if recovery is recovery from dcp.
  @param      backup          1 if file is partner or encoded file.
  @param      fn              Filename found for recovery.
  @return     integer         FTI_SCES if successful.
 **/
/*-------------------------------------------------------------------------*/
int FTIFF_RequestFileName(char* dir, int rank, int level, int dcp,
 int backup, char* fn) {
    char str[FTI_BUFS], strerr[FTI_BUFS];
    int fileTarget, match, ckptId;
    struct dirent *entry;
    struct stat ckptDIR;

    if ((level == 4) && dcp) {
        snprintf(fn, FTI_BUFS, "dCPFile-Rank%d.fti", rank);
        return FTI_SCES;
    }
    // check if ckpt directory exists
    bool dirExists = false;
    if (stat(dir, &ckptDIR) == 0) {
        if (S_ISDIR(ckptDIR.st_mode) != 0) {
            dirExists = true;
        } else {
            snprintf(strerr, FTI_BUFS, "FTI-FF: L%dRecoverInit - (%s) is not"
            " a directory.", level, dir);
            FTI_Print(strerr, FTI_WARN);
            return FTI_NSCS;
        }
    }

    if (dirExists) {
        DIR *ckptDir = opendir(dir);

        if (ckptDir == NULL) {
            snprintf(strerr, FTI_BUFS, "FTI-FF: L%dRecoveryInit - checkpoint"
            " directory (%s) could not be accessed.", level, dir);
            FTI_Print(strerr, FTI_EROR);
            errno = 0;
            return FTI_NSCS;
        }

        while ((entry = readdir(ckptDir)) != NULL) {
            if (strcmp(entry->d_name, ".") && strcmp(entry->d_name, "..")) {
                match = sscanf(entry->d_name, CKPT_FN_FORMAT(level, backup),
                 &ckptId, &fileTarget);

                if (match == 2 && fileTarget == rank) {
                    snprintf(str, FTI_BUFS, "FTI-FF: L%dRecoveryInit - found "
                        "file with name: %s", level, entry->d_name);
                    FTI_Print(str, FTI_DBUG);
                    snprintf(fn, FTI_BUFS, CKPT_FN_FORMAT(level, backup),
                     ckptId, rank);
                    return FTI_SCES;
                }
            }
        }
    }

    return FTI_NSCS;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Loads file meta data for filedescriptor 'fd'
  @param      fd              File descriptor of ckpt file.
  @param      fm              Pointer to file meta data instance.
  @return     integer         FTI_SCES if successful.
 **/
/*-------------------------------------------------------------------------*/
int FTIFF_LoadFileMeta(int fd, FTIFF_metaInfo* fm) {
    if (lseek(fd, -FTI_filemetastructsize, SEEK_END) == -1) {
        FTI_Print("unable to seek in file.", FTI_EROR);
        return FTI_NSCS;
    }
    void* buffer = malloc(FTI_filemetastructsize);
    if (buffer == NULL) {
        FTI_Print("unable to allocate memory.", FTI_EROR);
        return FTI_NSCS;
    }
    if (read(fd, buffer, FTI_filemetastructsize) == -1) {
        FTI_Print("unable to read in file.", FTI_EROR);
        return FTI_NSCS;
    }

    int res = FTIFF_DeserializeFileMeta(fm, buffer);
    free(buffer);
    return res;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Opens and checks ckpt/backup file
  @param      fn              Ckpt/backup file name.
  @param      oflag           Access mode to open the file.
  @return     integer         FTI_SCES if successful.
 **/
/*-------------------------------------------------------------------------*/
int FTIFF_OpenCkptFile(char* fn, int oflag) {
    char strerr[FTI_BUFS];
    int fd = -1;
    if (access(fn, F_OK) == 0) {
        struct stat fileStatus;
        if (stat(fn, &fileStatus) == 0) {
            if (fileStatus.st_size >= sizeof(FTIFF_metaInfo)) {
                fd = open(fn, oflag);
                if (fd < 0) {
                    snprintf(strerr, FTI_BUFS, "file '%s' cannot be opened.",
                     fn);
                    FTI_Print(strerr, FTI_EROR);
                    errno = 0;
                }
            }
        }
    }
    return fd;
}

int FTIFF_LoadMetaPostprocessing(FTIT_execution* FTI_Exec,
 FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt,
 FTIT_configuration* FTI_Conf, int proc) {
    char strerr[FTI_BUFS], path[FTI_BUFS], file[FTI_BUFS], dir[FTI_BUFS];

    int level = FTI_Exec->ckptMeta.level;

    if (FTI_Ckpt[level].isDcp)
        strncpy(dir, FTI_Ckpt[1].dcpDir, FTI_BUFS);
    else
        strncpy(dir, FTI_Conf->lTmpDir, FTI_BUFS);

    if (FTIFF_RequestFileName(dir, FTI_Topo->body[proc-1], level,
     FTI_Ckpt[level].isDcp, 0, file) != FTI_SCES) {
        return FTI_NSCS;
    }

    FTIFF_metaInfo *FTIFFMeta = calloc(1, sizeof(FTIFF_metaInfo));
    if (FTIFFMeta == NULL) {
        snprintf(strerr, FTI_BUFS, "failed to allocate %ld bytes",
         sizeof(FTIFF_metaInfo));
        FTI_Print(strerr, FTI_EROR);
        errno = 0;
        return FTI_NSCS;
    }

    snprintf(path, FTI_BUFS, "%s/%s", dir, file);
    int fd = FTIFF_OpenCkptFile(path, O_RDONLY);
    if (fd == -1) {
        free(FTIFFMeta);
        return FTI_NSCS;
    }

    if (FTIFF_LoadFileMeta(fd, FTIFFMeta) == FTI_NSCS) {
        FTI_Print("unable to load file meta data.", FTI_WARN);
        close(fd);
        free(FTIFFMeta);
        return FTI_NSCS;
    }

    /**< TRUE if metadata exists               */
    FTI_Exec->ckptMeta.level = level;
    /**< Maximum file size.                    */
    FTI_Exec->ckptMeta.maxFs = FTIFFMeta->maxFs;
    /**< File size.                            */
    FTI_Exec->ckptMeta.fs = FTIFFMeta->fs;
    /**< Partner file size.                    */
    FTI_Exec->ckptMeta.pfs = FTIFFMeta->ptFs;
    /**< Partner file size.                    */
    FTI_Exec->ckptMeta.ckptId = FTIFFMeta->ckptId;
    /**< Ckpt file name. [FTI_BUFS]            */
    strncpy(FTI_Exec->ckptMeta.ckptFile, file, FTI_BUFS);

    return FTI_SCES;
}
/*-------------------------------------------------------------------------*/
/**
  @brief      Computes checksum for encoded file
  @param      FTIFFMeta       Pointer to file meta data instance.
  @param      fd              File descriptor of ckpt file.
  @param      checksum        Checksum of encoded file.
  @return     integer         FTI_SCES if successful.
 **/
/*-------------------------------------------------------------------------*/
int FTIFF_GetEncodedFileChecksum(FTIFF_metaInfo *FTIFFMeta, int fd,
 char *checksum) {
    int64_t rcount = 0, toRead, diff;
    int rbuffer;
    char buffer[CHUNK_SIZE], strerr[FTI_BUFS];
    MD5_CTX mdContext;
    MD5_Init(&mdContext);
    while (rcount < FTIFFMeta->fs) {
        if (lseek(fd, rcount, SEEK_SET) == -1) {
            FTI_Print("FTI-FF: L3RecoveryInit - could not seek in file",
             FTI_EROR);
            errno = 0;
            return FTI_NSCS;
        }

        diff = FTIFFMeta->fs - rcount;
        toRead = (diff < CHUNK_SIZE) ? diff : CHUNK_SIZE;
        rbuffer = read(fd, buffer, toRead);
        if (rbuffer == -1) {
            snprintf(strerr, FTI_BUFS, "FTI-FF: L3RecoveryInit - Failed to"
            " read %ld bytes from file", toRead);
            FTI_Print(strerr, FTI_EROR);
            errno = 0;
            return FTI_NSCS;
        }

        rcount += rbuffer;
        MD5_Update(&mdContext, buffer, rbuffer);
    }
    unsigned char hash[MD5_DIGEST_LENGTH];
    MD5_Final(hash, &mdContext);
    int i;
    int ii = 0;
    for (i = 0; i < MD5_DIGEST_LENGTH; i++) {
        snprintf(&checksum[ii], sizeof(char[3]), "%02x", hash[i]);
        ii += 2;
    }
    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Determines recovery information
  @param      info            Recovery information.
  @param      dir             Directory of recovery for 'level'.
  @param      rank            Process rank of global communicator.
  @param      level           Recovery level.
  @param      dcp             1 if recovery is recovery from dcp.
  @param      backup          1 if file is partner or encoded file.
  @return     integer         FTI_SCES if successful.

  This function collects information, necessary to determine if the recovery
  can be performed. The information is set into 'info'.
 **/
/*-------------------------------------------------------------------------*/
int FTIFF_RequestRecoveryInfo(FTIFF_RecoveryInfo* info, char* dir, int rank,
 int level, bool dcp, bool backup) {
    char strerr[FTI_BUFS], path[FTI_BUFS], file[FTI_BUFS];

    if (FTIFF_RequestFileName(dir, rank, level, dcp, backup,
     file) != FTI_SCES) {
        return FTI_NSCS;
    }

    FTIFF_metaInfo *FTIFFMeta = calloc(1, sizeof(FTIFF_metaInfo));
    if (FTIFFMeta == NULL) {
        snprintf(strerr, FTI_BUFS, "failed to allocate %ld bytes",
         sizeof(FTIFF_metaInfo));
        FTI_Print(strerr, FTI_EROR);
        errno = 0;
        return FTI_NSCS;
    }

    snprintf(path, FTI_BUFS, "%s/%s", dir, file);

    int fd = FTIFF_OpenCkptFile(path, O_RDONLY);
    if (fd == -1) {
        free(FTIFFMeta);
        return FTI_NSCS;
    }

    if (FTIFF_LoadFileMeta(fd, FTIFFMeta) == FTI_NSCS) {
        FTI_Print("unable to load file meta data.", FTI_WARN);
        close(fd);
        free(FTIFFMeta);
        return FTI_NSCS;
    }


    unsigned char hash[MD5_DIGEST_LENGTH];
    FTIFF_GetHashMetaInfo(hash, FTIFFMeta);

    // Check if hash of file meta-data is consistent
    if (memcmp(FTIFFMeta->myHash, hash, MD5_DIGEST_LENGTH) == 0) {
        char checksum[MD5_DIGEST_STRING_LENGTH];

        if ((level == 3) && backup) {
            FTIFF_GetEncodedFileChecksum(FTIFFMeta, fd, checksum);
        } else {
            FTIFF_GetFileChecksum(FTIFFMeta, fd, checksum);
        }

        if (strcmp(checksum, FTIFFMeta->checksum) == 0) {
            if (backup) {
                info->bfs = FTIFFMeta->fs;
                info->BackupExists = 1;
            } else {
                info->fs = FTIFFMeta->fs;
                info->FileExists = 1;
            }
            info->ckptId = FTIFFMeta->ckptId;
            info->maxFs = FTIFFMeta->maxFs;
        }  else {
            char str[FTI_BUFS];
            snprintf(str, FTI_BUFS, "Checksum do not match. \"%s\" file is"
            " corrupted. %s != %s", file, checksum, FTIFFMeta->checksum);
            FTI_Print(str, FTI_WARN);
            close(fd);
            free(FTIFFMeta);
            return FTI_NSCS;
        }
    } else {
        char str[FTI_BUFS];
        snprintf(str, FTI_BUFS, "Metadata in file \"%s\" is corrupted.", file);
        FTI_Print(str, FTI_WARN);
        close(fd);
        free(FTIFFMeta);
        return FTI_NSCS;
    }

    close(fd);
    free(FTIFFMeta);
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
int FTIFF_CheckL1RecoverInit(FTIT_execution* FTI_Exec, FTIT_topology* FTI_Topo,
        FTIT_checkpoint* FTI_Ckpt, FTIT_configuration *FTI_Conf) {
    int fcount, fneeded;

    FTIFF_RecoveryInfo info;

    FTIFF_RequestRecoveryInfo(&info, FTI_Ckpt[1].dir, FTI_Topo->myRank,
     1, 0, 0);

    MPI_Allreduce(&info.FileExists, &fcount, 1, MPI_INT, MPI_SUM,
     FTI_COMM_WORLD);

    fneeded = FTI_Topo->nbNodes*FTI_Topo->nbApprocs;

    if (fcount == fneeded) {
        FTI_Exec->ckptMeta.fs = info.fs;
        FTI_Exec->ckptId = info.ckptId;
        snprintf(FTI_Exec->ckptMeta.ckptFile, FTI_BUFS, "Ckpt%d-Rank%d.fti",
         FTI_Exec->ckptId, FTI_Topo->myRank);
        return FTI_SCES;
    } else {
        return FTI_NSCS;
    }
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
int FTIFF_CheckL2RecoverInit(FTIT_execution* FTI_Exec, FTIT_topology* FTI_Topo,
        FTIT_checkpoint* FTI_Ckpt, FTIT_configuration* FTI_Conf, int *exists) {
    char dbgstr[FTI_BUFS];

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
    MPI_Group_translate_ranks(nodesGroup, 2, baseRanks,
     appProcsGroup, projRanks);
    int leftIdx = projRanks[0], rightIdx = projRanks[1];

    int appCommSize = FTI_Topo->nbNodes*FTI_Topo->nbApprocs;
    int fneeded = appCommSize;

    MPI_Group_free(&nodesGroup);
    MPI_Group_free(&appProcsGroup);

    FTIFF_RecoveryInfo* appProcsMetaInfo = calloc(appCommSize,
     sizeof(FTIFF_RecoveryInfo));

    int ckptId = -1, fcount = 0;

    FTIFF_RecoveryInfo info = {0};

    info.rightIdx = rightIdx;

    FTIFF_RequestRecoveryInfo(&info, FTI_Ckpt[2].dir,
     FTI_Topo->myRank, 2, 0, 0);

    FTIFF_RequestRecoveryInfo(&info, FTI_Ckpt[2].dir,
     FTI_Topo->myRank, 2, 0, 1);

    if (!(info.FileExists) && !(info.BackupExists)) {
        info.ckptId = -1;
    }

    // gather meta info
    MPI_Allgather(&info, 1, FTIFF_MpiTypes[FTIFF_RECO_INFO], appProcsMetaInfo,
     1, FTIFF_MpiTypes[FTIFF_RECO_INFO], FTI_COMM_WORLD);

    exists[LEFT_FILE] = appProcsMetaInfo[leftIdx].FileExists;
    exists[MY_FILE] = appProcsMetaInfo[FTI_Topo->splitRank].FileExists;
    exists[MY_COPY] = appProcsMetaInfo[rightIdx].BackupExists;
    exists[LEFT_COPY] = appProcsMetaInfo[FTI_Topo->splitRank].BackupExists;

    // debug Info
    snprintf(dbgstr, FTI_BUFS, "FTI-FF - L2Recovery::FileCheck - CkptFile:"
        " %i, CkptCopy: %i", info.FileExists, info.BackupExists);
    FTI_Print(dbgstr, FTI_DBUG);

    // check if recovery possible
    int i, saneCkptID = 0;
    ckptId = 0;
    for (i = 0; i < appCommSize; i++) {
        fcount += (appProcsMetaInfo[i].FileExists ||
         appProcsMetaInfo[appProcsMetaInfo[i].rightIdx].BackupExists) ? 1 : 0;
        if (appProcsMetaInfo[i].ckptId > 0) {
            saneCkptID++;
            ckptId += appProcsMetaInfo[i].ckptId;
        }
    }
    int res = (fcount == fneeded) ? FTI_SCES : FTI_NSCS;

    if (res == FTI_SCES) {
        FTI_Exec->ckptId = ckptId/saneCkptID;
        if (info.FileExists) {
            FTI_Exec->ckptMeta.fs = info.fs;
        } else {
            FTI_Exec->ckptMeta.fs = appProcsMetaInfo[rightIdx].bfs;
        }
        if (info.BackupExists) {
            FTI_Exec->ckptMeta.pfs = info.bfs;
        } else {
            FTI_Exec->ckptMeta.pfs = appProcsMetaInfo[leftIdx].fs;
        }
    }
    snprintf(dbgstr, FTI_BUFS, "FTI-FF: L2-Recovery - rank: %i, left: %i,"
        " right: %i, fs: %ld, pfs: %ld, ckptId: %i",
            FTI_Topo->myRank, leftIdx, rightIdx, FTI_Exec->ckptMeta.fs,
             FTI_Exec->ckptMeta.pfs, FTI_Exec->ckptId);
    FTI_Print(dbgstr, FTI_DBUG);

    snprintf(FTI_Exec->ckptMeta.ckptFile, FTI_BUFS, "Ckpt%d-Rank%d.fti",
     FTI_Exec->ckptId, FTI_Topo->myRank);

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
int FTIFF_CheckL3RecoverInit(FTIT_execution* FTI_Exec, FTIT_topology* FTI_Topo,
        FTIT_checkpoint* FTI_Ckpt, int* erased) {
    int ckptId;

    FTIFF_RecoveryInfo *groupInfo = calloc(FTI_Topo->groupSize,
     sizeof(FTIFF_RecoveryInfo));
    FTIFF_RecoveryInfo info = {0};

    FTIFF_RequestRecoveryInfo(&info, FTI_Ckpt[3].dir, FTI_Topo->myRank,
     3, 0, 0);

    FTIFF_RequestRecoveryInfo(&info, FTI_Ckpt[3].dir, FTI_Topo->myRank,
     3, 0, 1);

    if (!(info.FileExists) && !(info.BackupExists)) {
        info.ckptId = -1;
    }

    if (!(info.BackupExists)) {
        info.bfs = -1;
    }

    // gather meta info
    MPI_Allgather(&info, 1, FTIFF_MpiTypes[FTIFF_RECO_INFO], groupInfo, 1,
     FTIFF_MpiTypes[FTIFF_RECO_INFO], FTI_Exec->groupComm);

    // check if recovery possible
    int i, saneCkptID = 0, saneMaxFs = 0, erasures = 0;
    int64_t maxFs = 0;
    ckptId = 0;
    for (i = 0; i < FTI_Topo->groupSize; i++) {
        erased[i]=!groupInfo[i].FileExists;
        erased[i+FTI_Topo->groupSize]=!groupInfo[i].BackupExists;
        erasures += erased[i] + erased[i+FTI_Topo->groupSize];
        if (groupInfo[i].ckptId > 0) {
            saneCkptID++;
            ckptId += groupInfo[i].ckptId;
        }
        if (groupInfo[i].bfs > 0) {
            saneMaxFs++;
            maxFs += groupInfo[i].bfs;
        }
    }
    if (saneCkptID != 0) {
        FTI_Exec->ckptId = ckptId/saneCkptID;
    }
    if (saneMaxFs != 0) {
        FTI_Exec->ckptMeta.maxFs = maxFs/saneMaxFs;
    }
    // for the case that all (and only) the encoded files are deleted
    if (saneMaxFs == 0 && !(erasures > FTI_Topo->groupSize)) {
        MPI_Allreduce(&(info.maxFs), &FTI_Exec->ckptMeta.maxFs, 1, MPI_INT32_T,
         MPI_SUM, FTI_Exec->groupComm);
        FTI_Exec->ckptMeta.maxFs /= FTI_Topo->groupSize;
    }

    FTI_Exec->ckptMeta.fs = (info.FileExists) ? info.fs : 0;

    snprintf(FTI_Exec->ckptMeta.ckptFile, FTI_BUFS, "Ckpt%d-Rank%d.fti",
     FTI_Exec->ckptId, FTI_Topo->myRank);

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
int FTIFF_CheckL4RecoverInit(FTIT_execution* FTI_Exec, FTIT_topology* FTI_Topo,
        FTIT_checkpoint* FTI_Ckpt) {
    char fn[FTI_BUFS];
    int fcount, fneeded;

    FTIFF_RecoveryInfo info;

    if (FTI_Ckpt[4].recoIsDcp) {
        FTIFF_RequestRecoveryInfo(&info, FTI_Ckpt[4].dcpDir, FTI_Topo->myRank,
         4, 1, 0);
        snprintf(fn, FTI_BUFS, "dCPFile-Rank%d.fti", FTI_Topo->myRank);
    } else {
        FTIFF_RequestRecoveryInfo(&info, FTI_Ckpt[4].dir, FTI_Topo->myRank,
         4, 0, 0);
        snprintf(fn, FTI_BUFS, "Ckpt%d-Rank%d.fti", info.ckptId,
         FTI_Topo->myRank);
    }

    MPI_Allreduce(&info.FileExists, &fcount, 1, MPI_INT, MPI_SUM,
     FTI_COMM_WORLD);

    fneeded = FTI_Topo->nbNodes*FTI_Topo->nbApprocs;

    if (fcount == fneeded) {
        FTI_Exec->ckptMeta.fs = info.fs;
        FTI_Exec->ckptId = info.ckptId;
        snprintf(FTI_Exec->ckptMeta.ckptFile, FTI_BUFS, "Ckpt%d-Rank%d.fti",
         FTI_Exec->ckptId, FTI_Topo->myRank);
        strncpy(FTI_Exec->ckptMeta.ckptFile, fn, NAME_MAX);
        return FTI_SCES;
    }

    return FTI_NSCS;
}

void FTIFF_SetHashChunk(FTIFF_dbvar *dbvar, FTIT_keymap* FTI_Data) {
    if (dbvar->hascontent) {
        FTIT_dataset* data;

        if (FTI_Data->get(&data, dbvar->id) != FTI_SCES) return;

        if (!data) {
            char str[FTI_BUFS];
            snprintf(str, FTI_BUFS, "id '%d' does not exist!", dbvar->id);
            FTI_Print(str, FTI_EROR);
            return;
        }

	MD5_CTX dbvar_ctx;
	void * ptr = data->ptr + dbvar->dptr;
        int64_t remaining = dbvar->chunksize;
        int block_size = CHUNK_SIZE;
        MD5_Init(&dbvar_ctx);
        while (remaining > 0){
            int len_update = (remaining > block_size) ?
                block_size : remaining;
            int64_t offset = dbvar->chunksize - remaining;
            MD5_Update(&dbvar_ctx, ptr + offset, len_update);
            remaining -= len_update;
        }
        MD5_Final(dbvar->hash, &dbvar_ctx);
    }
}

/*-------------------------------------------------------------------------*/
/**
  @brief    Computes hash of the FTI-FF file meta data structure   
  @param    hash          hash to compute.
  @param    FTIFFMeta     Ckpt file meta data.
 **/
/*-------------------------------------------------------------------------*/
void FTIFF_GetHashMetaInfo(unsigned char *hash, FTIFF_metaInfo *FTIFFMeta) {
    MD5_CTX md5Ctx;
    MD5_Init(&md5Ctx);
    MD5_Update(&md5Ctx, FTIFFMeta->checksum, MD5_DIGEST_STRING_LENGTH);
    MD5_Update(&md5Ctx, &(FTIFFMeta->timestamp), sizeof(int64_t));
    MD5_Update(&md5Ctx, &(FTIFFMeta->ckptSize), sizeof(int64_t));
    MD5_Update(&md5Ctx, &(FTIFFMeta->metaSize), sizeof(int64_t));
    MD5_Update(&md5Ctx, &(FTIFFMeta->dataSize), sizeof(int64_t));
    MD5_Update(&md5Ctx, &(FTIFFMeta->fs), sizeof(int64_t));
    MD5_Update(&md5Ctx, &(FTIFFMeta->ptFs), sizeof(int64_t));
    MD5_Update(&md5Ctx, &(FTIFFMeta->maxFs), sizeof(int64_t));
    MD5_Final(hash, &md5Ctx);
}

/*-------------------------------------------------------------------------*/
/**
  @brief    Computes hash of the FTI-FF file data block meta data structure   
  @param    hash          hash to compute.
  @param    FTIFFMeta     file data block meta data.
 **/
/*-------------------------------------------------------------------------*/
void FTIFF_GetHashdb(unsigned char *hash, FTIFF_db *db) {
    MD5_CTX md5Ctx;
    MD5_Init(&md5Ctx);
    MD5_Update(&md5Ctx, &(db->numvars), sizeof(int));
    MD5_Update(&md5Ctx, &(db->dbsize), sizeof(int64_t));
    MD5_Final(hash, &md5Ctx);
}

/*-------------------------------------------------------------------------*/
/**
  @brief    Computes hash of the FTI-FF data chunk meta data structure   
  @param    hash          hash to compute.
  @param    dbvar         data chunk meta data.
 **/
/*-------------------------------------------------------------------------*/
void FTIFF_GetHashdbvar(unsigned char *hash, FTIFF_dbvar *dbvar)  {
    MD5_CTX md5Ctx;
    MD5_Init(&md5Ctx);
    MD5_Update(&md5Ctx, &(dbvar->id), sizeof(int));
    MD5_Update(&md5Ctx, &(dbvar->containerid), sizeof(int));
    MD5_Update(&md5Ctx, &(dbvar->hascontent), sizeof(bool));
    MD5_Update(&md5Ctx, &(dbvar->hasCkpt), sizeof(bool));
    MD5_Update(&md5Ctx, &(dbvar->dptr), sizeof(uintptr_t));
    MD5_Update(&md5Ctx, &(dbvar->fptr), sizeof(uintptr_t));
    MD5_Update(&md5Ctx, &(dbvar->chunksize), sizeof(int64_t));
    MD5_Update(&md5Ctx, &(dbvar->containersize), sizeof(int64_t));
    MD5_Update(&md5Ctx, dbvar->hash, MD5_DIGEST_LENGTH);
    MD5_Final(hash, &md5Ctx);
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Initializes the derived MPI data types used for FTI-FF
 **/
/*-------------------------------------------------------------------------*/
void FTIFF_InitMpiTypes() {
    MPI_Aint lb, extent;
    FTIFF_MPITypeInfo MPITypeInfo[FTIFF_NUM_MPI_TYPES];

    // define MPI datatypes

    // headInfo
    MBR_CNT(headInfo) =  7;
    MBR_BLK_LEN(headInfo) = { 1, 1, FTI_BUFS, 1, 1, 1, 1 };
    MBR_TYPES(headInfo) = { MPI_INT, MPI_INT, MPI_CHAR,
     MPI_INT32_T, MPI_INT32_T, MPI_INT32_T, MPI_INT };
    MBR_DISP(headInfo) = {
        offsetof(FTIFF_headInfo, exists),
        offsetof(FTIFF_headInfo, nbVar),
        offsetof(FTIFF_headInfo, ckptFile),
        offsetof(FTIFF_headInfo, maxFs),
        offsetof(FTIFF_headInfo, fs),
        offsetof(FTIFF_headInfo, pfs),
        offsetof(FTIFF_headInfo, isDcp)
    };
    MPITypeInfo[FTIFF_HEAD_INFO].mbrCnt = headInfo_mbrCnt;
    MPITypeInfo[FTIFF_HEAD_INFO].mbrBlkLen = headInfo_mbrBlkLen;
    MPITypeInfo[FTIFF_HEAD_INFO].mbrTypes = headInfo_mbrTypes;
    MPITypeInfo[FTIFF_HEAD_INFO].mbrDisp = headInfo_mbrDisp;

    // L2Info
    MBR_CNT(RecoInfo) = 6;
    MBR_BLK_LEN(RecoInfo) = { 1, 1, 1, 1, 1, 1 };
    MBR_TYPES(RecoInfo) = { MPI_INT, MPI_INT, MPI_INT,
     MPI_INT, MPI_INT32_T, MPI_INT32_T };
    MBR_DISP(RecoInfo) = {
        offsetof(FTIFF_RecoveryInfo, FileExists),
        offsetof(FTIFF_RecoveryInfo, BackupExists),
        offsetof(FTIFF_RecoveryInfo, ckptId),
        offsetof(FTIFF_RecoveryInfo, rightIdx),
        offsetof(FTIFF_RecoveryInfo, fs),
        offsetof(FTIFF_RecoveryInfo, bfs),
    };
    MPITypeInfo[FTIFF_RECO_INFO].mbrCnt = RecoInfo_mbrCnt;
    MPITypeInfo[FTIFF_RECO_INFO].mbrBlkLen = RecoInfo_mbrBlkLen;
    MPITypeInfo[FTIFF_RECO_INFO].mbrTypes = RecoInfo_mbrTypes;
    MPITypeInfo[FTIFF_RECO_INFO].mbrDisp = RecoInfo_mbrDisp;

    // commit MPI types
    int i;
    for (i = 0; i < FTIFF_NUM_MPI_TYPES; i++) {
        MPI_Type_create_struct(
                MPITypeInfo[i].mbrCnt,
                MPITypeInfo[i].mbrBlkLen,
                MPITypeInfo[i].mbrDisp,
                MPITypeInfo[i].mbrTypes,
                &MPITypeInfo[i].raw);
        MPI_Type_get_extent(
                MPITypeInfo[i].raw,
                &lb,
                &extent);
        MPI_Type_create_resized(
                MPITypeInfo[i].raw,
                lb,
                extent,
                &FTIFF_MpiTypes[i]);
        MPI_Type_commit(&FTIFF_MpiTypes[i]);
    }
}

/*-------------------------------------------------------------------------*/
/**
  @brief    deserializes FTI-FF file meta data   
  @param    meta          FTI-FF file meta data.
  @param    buffer_ser    serialized file meta data.
 **/
/*-------------------------------------------------------------------------*/
int FTIFF_DeserializeFileMeta(FTIFF_metaInfo* meta, char* buffer_ser) {
    if ((buffer_ser == NULL) || (meta == NULL)) {
        FTI_Print("nullptr passed to 'FTIFF_DeserializeFilemeta!", FTI_WARN);
        return FTI_NSCS;
    }

    int64_t pos = 0;
    memcpy(meta->checksum        , buffer_ser + pos, MD5_DIGEST_STRING_LENGTH);
    pos += MD5_DIGEST_STRING_LENGTH;
    memcpy(meta->myHash          , buffer_ser + pos, MD5_DIGEST_LENGTH);
    pos += MD5_DIGEST_LENGTH;
    memcpy(&(meta->ckptId)       , buffer_ser + pos, sizeof(int));
    pos += sizeof(int);
    memcpy(&(meta->ckptSize)     , buffer_ser + pos, sizeof(int64_t));
    pos += sizeof(int64_t);
    memcpy(&(meta->metaSize)     , buffer_ser + pos, sizeof(int64_t));
    pos += sizeof(int64_t);
    memcpy(&(meta->dataSize)     , buffer_ser + pos, sizeof(int64_t));
    pos += sizeof(int64_t);
    memcpy(&(meta->fs)           , buffer_ser + pos, sizeof(int64_t));
    pos += sizeof(int64_t);
    memcpy(&(meta->maxFs)        , buffer_ser + pos, sizeof(int64_t));
    pos += sizeof(int64_t);
    memcpy(&(meta->ptFs)         , buffer_ser + pos, sizeof(int64_t));
    pos += sizeof(int64_t);
    memcpy(&(meta->timestamp)    , buffer_ser + pos, sizeof(int64_t));

    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief    deserializes FTI-FF file data block meta data   
  @param    db            FTI-FF file data block meta data.
  @param    buffer_ser    serialized file data block meta data.
 **/
/*-------------------------------------------------------------------------*/
int FTIFF_DeserializeDbMeta(FTIFF_db* db, char* buffer_ser) {
    if ((buffer_ser == NULL) || (db == NULL)) {
        FTI_Print("nullptr passed to 'FTIFF_DeserializeFileMeta!", FTI_WARN);
        return FTI_NSCS;
    }

    int64_t pos = 0;
    memcpy(&(db->numvars)    , buffer_ser + pos, sizeof(int));
    pos += sizeof(int);
    memcpy(&(db->dbsize)     , buffer_ser + pos, sizeof(int64_t));

    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief    deserializes FTI-FF data chunk meta data   
  @param    dbvar         FTI-FF data chunk meta data.
  @param    buffer_ser    serialized data chunk meta data.
 **/
/*-------------------------------------------------------------------------*/
int FTIFF_DeserializeDbVarMeta(FTIFF_dbvar* dbvar, char* buffer_ser) {
    if ((buffer_ser == NULL) || (dbvar == NULL)) {
        FTI_Print("nullptr passed to 'FTIFF_DeserializeFileMeta!", FTI_WARN);
        return FTI_NSCS;
    }

    int64_t pos = 0;
    memcpy(&(dbvar->id)              , buffer_ser + pos, sizeof(int));
    pos += sizeof(int);
    memcpy(&(dbvar->containerid)     , buffer_ser + pos, sizeof(int));
    pos += sizeof(int);
    memcpy(&(dbvar->hascontent)      , buffer_ser + pos, sizeof(bool));
    pos += sizeof(bool);
    memcpy(&(dbvar->hasCkpt)         , buffer_ser + pos, sizeof(bool));
    pos += sizeof(bool);
    memcpy(&(dbvar->dptr)            , buffer_ser + pos, sizeof(uintptr_t));
    pos += sizeof(uintptr_t);
    memcpy(&(dbvar->fptr)            , buffer_ser + pos, sizeof(uintptr_t));
    pos += sizeof(uintptr_t);
    memcpy(&(dbvar->chunksize)       , buffer_ser + pos, sizeof(int64_t));
    pos += sizeof(int64_t);
    memcpy(&(dbvar->containersize)   , buffer_ser + pos, sizeof(int64_t));
    pos += sizeof(int64_t);
    memcpy(dbvar->hash               , buffer_ser + pos, MD5_DIGEST_LENGTH);

    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief    serializes FTI-FF file meta data   
  @param    meta          FTI-FF file meta data.
  @param    buffer_ser    serialized file meta data.
 **/
/*-------------------------------------------------------------------------*/
int FTIFF_SerializeFileMeta(FTIFF_metaInfo* meta, char* buffer_ser) {
    if ((buffer_ser == NULL) || (meta == NULL)) {
        FTI_Print("nullptr passed to 'FTIFF_DeserializeFileMeta!", FTI_WARN);
        return FTI_NSCS;
    }

    int64_t pos = 0;
    memcpy(buffer_ser + pos, meta->checksum        , MD5_DIGEST_STRING_LENGTH);
    pos += MD5_DIGEST_STRING_LENGTH;
    memcpy(buffer_ser + pos, meta->myHash          , MD5_DIGEST_LENGTH);
    pos += MD5_DIGEST_LENGTH;
    memcpy(buffer_ser + pos, &(meta->ckptId)       , sizeof(int));
    pos += sizeof(int);
    memcpy(buffer_ser + pos, &(meta->ckptSize)     , sizeof(int64_t));
    pos += sizeof(int64_t);
    memcpy(buffer_ser + pos, &(meta->metaSize)     , sizeof(int64_t));
    pos += sizeof(int64_t);
    memcpy(buffer_ser + pos, &(meta->dataSize)     , sizeof(int64_t));
    pos += sizeof(int64_t);
    memcpy(buffer_ser + pos, &(meta->fs)           , sizeof(int64_t));
    pos += sizeof(int64_t);
    memcpy(buffer_ser + pos, &(meta->maxFs)        , sizeof(int64_t));
    pos += sizeof(int64_t);
    memcpy(buffer_ser + pos, &(meta->ptFs)         , sizeof(int64_t));
    pos += sizeof(int64_t);
    memcpy(buffer_ser + pos, &(meta->timestamp)    , sizeof(int64_t));

    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief    serializes FTI-FF file data block meta data   
  @param    db            FTI-FF file data block meta data.
  @param    buffer_ser    serialized file data block meta data.
 **/
/*-------------------------------------------------------------------------*/
int FTIFF_SerializeDbMeta(FTIFF_db* db, char* buffer_ser) {
    if ((buffer_ser == NULL) || (db == NULL)) {
        FTI_Print("nullptr passed to 'FTIFF_DeserializeFileMeta!", FTI_WARN);
        return FTI_NSCS;
    }

    int64_t pos = 0;
    memcpy(buffer_ser + pos, &(db->numvars)    , sizeof(int));
    pos += sizeof(int);
    memcpy(buffer_ser + pos, &(db->dbsize)     , sizeof(int64_t));

    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief    serializes FTI-FF data chunk meta data   
  @param    dbvar         FTI-FF data chunk meta data.
  @param    buffer_ser    serialized data chunk meta data.
 **/
/*-------------------------------------------------------------------------*/
int FTIFF_SerializeDbVarMeta(FTIFF_dbvar* dbvar, char* buffer_ser) {
    if ((buffer_ser == NULL) || (dbvar == NULL)) {
        FTI_Print("nullptr passed to 'FTIFF_DeserializeFileMeta!", FTI_WARN);
        return FTI_NSCS;
    }

    int64_t pos = 0;
    memcpy(buffer_ser + pos, &(dbvar->id)              , sizeof(int));
    pos += sizeof(int);
    memcpy(buffer_ser + pos, &(dbvar->containerid)     , sizeof(int));
    pos += sizeof(int);
    memcpy(buffer_ser + pos, &(dbvar->hascontent)      , sizeof(bool));
    pos += sizeof(bool);
    memcpy(buffer_ser + pos, &(dbvar->hasCkpt)         , sizeof(bool));
    pos += sizeof(bool);
    memcpy(buffer_ser + pos, &(dbvar->dptr)            , sizeof(uintptr_t));
    pos += sizeof(uintptr_t);
    memcpy(buffer_ser + pos, &(dbvar->fptr)            , sizeof(uintptr_t));
    pos += sizeof(uintptr_t);
    memcpy(buffer_ser + pos, &(dbvar->chunksize)       , sizeof(int64_t));
    pos += sizeof(int64_t);
    memcpy(buffer_ser + pos, &(dbvar->containersize)   , sizeof(int64_t));
    pos += sizeof(int64_t);
    memcpy(buffer_ser + pos, dbvar->hash               , MD5_DIGEST_LENGTH);

    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Frees allocated memory for the FTI-FF meta data struct list
  @param      last      Last element in FTI-FF metadata list.
 **/
/*-------------------------------------------------------------------------*/
void FTIFF_FreeDbFTIFF(FTIFF_db* last) {
    if (last) {
        FTIFF_db *current = last;
        FTIFF_db *previous;
        while (current) {
            previous = current->previous;
            free(current->dbvars);
            free(current);
            current = previous;
        }
    }
}

// BEGIN - ONLY FOR DEVELOPPING
/* PRINTING DATA STRUCTURE */
void FTIFF_PrintDataStructure(int rank, FTIT_execution* FTI_Exec) {
    int dbrank;
    MPI_Comm_rank(FTI_COMM_WORLD, &dbrank);
    FTIFF_db *dbgdb = FTI_Exec->firstdb;
    if (dbrank == rank) {
        int dbcnt = 0;
        printf("------------------- DATASTRUCTURE BEGIN"
        " [%d]----------------\n\n", rank);
        do {
            printf("    DataBase-id: %d\n", dbcnt);
            printf("                 dbsize: %ld\n", dbgdb->dbsize);
            printf("                 metasize (offset: %d): %d\n\n",
             FTI_filemetastructsize,
             FTI_dbstructsize+dbgdb->numvars*FTI_dbvarstructsize);
            dbcnt++;
            int varid = 0;
            for (; varid < dbgdb->numvars; ++varid) {
                printf("         Var-id: %d\n", varid);
                printf("                 id: %d\n"
                        "                 containerid: %d\n"
                        "                 hascontent: %s\n"
                        "                 hasCkpt: %s\n"
                        "                 dptr: %lu\n"
                        "                 fptr: %lu\n"
                        "                 chunksize: %ld\n"
                        "                 containersize: %ld\n\n",
                        /*
                         "                 nbHashes: %lu\n"
                         "                 diffBlockSize: %d\n"
                         "                 addr-hashptr: %p\n"
                         "                 addr-dataptr: %p\n"
                         "                 lastBlockSize: %d\n\n",*/
                        dbgdb->dbvars[varid].id,
                        dbgdb->dbvars[varid].containerid,
                        (dbgdb->dbvars[varid].hascontent) ? "true" : "false",
                        (dbgdb->dbvars[varid].hasCkpt) ? "true" : "false",
                        dbgdb->dbvars[varid].dptr,
                        dbgdb->dbvars[varid].fptr,
                        dbgdb->dbvars[varid].chunksize,
                        dbgdb->dbvars[varid].containersize);
                /*,
                  dbgdb->dbvars[varid].nbHashes,
                  FTI_GetDiffBlockSize(),
                  dbgdb->dbvars[varid].dataDiffHash[0].ptr,
                  (FTI_ADDRPTR) (FTI_Data[dbgdb->dbvars[varid].idx].ptr+dbgdb->dbvars[varid].dptr),
                  dbgdb->dbvars[varid].dataDiffHash[dbgdb->dbvars[varid].nbHashes-1].blockSize);*/
            }
        } while ((dbgdb = dbgdb->next));
        printf("\n------------------- DATASTRUCTURE END "
            "---------------------\n");
        fflush(stdout);
    }
    MPI_Barrier(FTI_COMM_WORLD);
}
// END - ONLY FOR DEVELOPPING



