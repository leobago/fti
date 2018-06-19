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
 *  @file   diff-checkpoint.c
 *  @date   February, 2018
 *  @brief  Differential checkpointing routines.
 */

#define _DEFAULT_SOURCE
#define _BSD_SOURCE

#include "interface.h"

/**                                                                                     */
/** Static Global Variables                                                             */

static int                  HASH_MODE;
static int                  DIFF_BLOCK_SIZE;

/** File Local Variables                                                                */

static bool enableDiffCkpt;
static int diffMode;

/** Function Definitions                                                                */

int FTI_FinalizeDcp() 
{
}

int FTI_InitDcp( FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec, FTIT_dataset* FTI_Data )
{
    
    int rank;
    MPI_Comm_rank(FTI_COMM_WORLD, &rank);
    if( getenv("FTI_HASH_MODE") != 0 ) {
        HASH_MODE = atoi(getenv("FTI_HASH_MODE"));
    } else {
        HASH_MODE = FTI_DCP_MODE_MD5;
    }
    if( getenv("FTI_DIFF_BLOCK_SIZE") != 0 ) {
        DIFF_BLOCK_SIZE = atoi(getenv("FTI_DIFF_BLOCK_SIZE"));
    } else {
        DIFF_BLOCK_SIZE = 2048;
    }

    if(rank == 0) {
        switch (HASH_MODE) {
            case -1:
                printf("[ " BLU "FTI  DIFFCKPT" RESET " ] : HASH MODE IS -> OFF\n");
                break;
            case 0:
                printf("[ " BLU "FTI  DIFFCKPT" RESET " ] : HASH MODE IS -> MD5\n");
                break;
            case 1:
                printf("[ " BLU "FTI  DIFFCKPT" RESET " ] : HASH MODE IS -> CRC32\n");
                break;
        }
        printf("[ " BLU "FTI  DIFFCKPT" RESET " ] : DIFF_BLOCK_SIZE IS -> %d\n", DIFF_BLOCK_SIZE);
    }

    enableDiffCkpt = FTI_Conf->enableDiffCkpt;
    
    diffMode = FTI_Conf->diffMode;
    return FTI_SCES;
}

int FTI_GetDiffBlockSize() 
{
    return DIFF_BLOCK_SIZE;
}

int FTI_GetDcpMode() 
{
    return HASH_MODE;
}

int FTI_InitBlockHashArray( FTIFF_dbvar* dbvar, FTIT_dataset* FTI_Data ) 
{
    dbvar->nbHashes = FTI_CalcNumHashes( dbvar->chunksize );
    dbvar->dataDiffHash = (FTIT_DataDiffHash*) malloc ( sizeof(FTIT_DataDiffHash) * dbvar->nbHashes );
    assert( dbvar->dataDiffHash != NULL );
    FTIT_DataDiffHash* hashes = dbvar->dataDiffHash;
    FTI_ADDRVAL ptr = (FTI_ADDRVAL) FTI_Data->ptr + (FTI_ADDRVAL) dbvar->dptr;
    FTI_ADDRVAL end = ptr + (FTI_ADDRVAL) dbvar->chunksize;
    int hashIdx;
    if ( FTI_GetDcpMode() == FTI_DCP_MODE_MD5 ) {
        // we want the hash array to be dense
        hashes[0].md5hash = (unsigned char*) malloc( MD5_DIGEST_LENGTH * dbvar->nbHashes );
    }
    int diffBlockSize = FTI_GetDiffBlockSize();
    for(hashIdx = 0; hashIdx<dbvar->nbHashes; ++hashIdx) {
        int hashBlockSize = ( (end - ptr) > diffBlockSize ) ? diffBlockSize : end-ptr;
        if ( FTI_GetDcpMode() == FTI_DCP_MODE_MD5 ) {
            hashes[hashIdx].md5hash = (unsigned char*) &(hashes[0].md5hash) + hashIdx * MD5_DIGEST_LENGTH;
        } else {
            hashes[hashIdx].md5hash = NULL;
        }
        hashes[hashIdx].isValid = false;
        hashes[hashIdx].ptr = (FTI_ADDRPTR) ptr;
        hashes[hashIdx].blockSize = hashBlockSize;
        ptr += hashBlockSize;
    }
}

int FTI_CollapseBlockHashArray( FTIFF_dbvar* dbvar, long new_size, FTIT_dataset* FTI_Data ) 
{
    assert( new_size <= dbvar->containersize );

    bool changeSize = true;

    long nbHashesOld = dbvar->nbHashes;

    // update to new number of hashes (which might be actually unchanged)
    dbvar->nbHashes = FTI_CalcNumHashes( new_size );
 
    if ( dbvar->nbHashes == nbHashesOld ) {
        changeSize = false;
    }

    // reallocate hash array to new size if changed
    if ( changeSize ) {
        assert( dbvar->dataDiffHash != NULL );
        dbvar->dataDiffHash = (FTIT_DataDiffHash*) realloc ( dbvar->dataDiffHash, sizeof(FTIT_DataDiffHash) * dbvar->nbHashes );
        assert( dbvar->dataDiffHash != NULL );
    }
    
    FTIT_DataDiffHash* hashes = dbvar->dataDiffHash;
    
    // we need this pointer since data was most likely re allocated 
    // and thus might have a new memory location
    FTI_ADDRVAL ptr = (FTI_ADDRVAL) FTI_Data->ptr + (FTI_ADDRVAL) dbvar->dptr;
    FTI_ADDRVAL end = ptr + (FTI_ADDRVAL) new_size;
    int hashIdx;
    if ( (FTI_GetDcpMode() == FTI_DCP_MODE_MD5) && changeSize ) {
        // we want the hash array to be dense
        hashes[0].md5hash = (unsigned char*) realloc( hashes[0].md5hash, MD5_DIGEST_LENGTH * dbvar->nbHashes );
    }
    int lastIdx = dbvar->nbHashes-1;
    int diffBlockSize = FTI_GetDiffBlockSize();
    for(hashIdx = 0; hashIdx<dbvar->nbHashes; ++hashIdx) {
        int hashBlockSize = ( (end - ptr) > diffBlockSize ) ? diffBlockSize : end-ptr;
        // keep track of new memory locations for dense hash array
        if ( changeSize ) {
            if ( FTI_GetDcpMode() == FTI_DCP_MODE_MD5 ) {
                hashes[hashIdx].md5hash = (unsigned char*) &(hashes[0].md5hash) + hashIdx * MD5_DIGEST_LENGTH;
            } else {
                hashes[hashIdx].md5hash = NULL;
            }
        }
    
        // invalidate last hash in (almost) any case. If number of hashes remain the same, 
        // the last block changed size and with that data changed content.
        // if number decreased and the blocksize of new last block is less the DIFF_BLOCK_SIZE
        // the hash is invalid too.
        if ( hashIdx == lastIdx ) {
            hashes[hashIdx].blockSize = hashBlockSize;
            if ( ( hashes[lastIdx].blockSize < DIFF_BLOCK_SIZE ) && changeSize ) {
                hashes[lastIdx].isValid = false;
            } else if ( !changeSize ) {
                hashes[lastIdx].isValid = false;
            }
        }
        hashes[hashIdx].ptr = (FTI_ADDRPTR) ptr;
        ptr += hashBlockSize;
    }


    return FTI_SCES;    
}

int FTI_ExpandBlockHashArray( FTIFF_dbvar* dbvar, long new_size, FTIT_dataset* FTI_Data ) 
{
    assert( new_size <= dbvar->containersize );

    bool changeSize = true;

    long nbHashesOld = dbvar->nbHashes;
    // current last hash is invalid in any case. 
    // If number of blocks remain the same, the size of the last block changed to 'new_size - old_size', 
    // thus also the data that is contained in it. 
    // If the nuber of blocks increased, the blocksize is changed for the current 
    // last block as well, in fact to DIFF_BLOCK_SIZE. 
    // This is taken care of in the for loop (after comment 'invalidate new hashes...').
    
    // update to new number of hashes (which might be actually unchanged)
    dbvar->nbHashes = FTI_CalcNumHashes( new_size );
 
    if ( dbvar->nbHashes == nbHashesOld ) {
        changeSize = false;
    }

    // reallocate hash array to new size if changed
    if ( changeSize ) {
        assert( dbvar->dataDiffHash != NULL );
        dbvar->dataDiffHash = (FTIT_DataDiffHash*) realloc ( dbvar->dataDiffHash, sizeof(FTIT_DataDiffHash) * dbvar->nbHashes );
        assert( dbvar->dataDiffHash != NULL );
    }
    
    FTIT_DataDiffHash* hashes = dbvar->dataDiffHash;
    
    // we need this pointer since data was most likely re allocated 
    // and thus might have a new memory location
    FTI_ADDRVAL ptr = (FTI_ADDRVAL) FTI_Data->ptr + (FTI_ADDRVAL) dbvar->dptr;
    FTI_ADDRVAL end = ptr + (FTI_ADDRVAL) new_size;
    int hashIdx;
    if ( (FTI_GetDcpMode() == FTI_DCP_MODE_MD5) && changeSize ) {
        // we want the hash array to be dense
        hashes[0].md5hash = (unsigned char*) realloc( hashes[0].md5hash, MD5_DIGEST_LENGTH * dbvar->nbHashes );
    }
    int diffBlockSize = FTI_GetDiffBlockSize();
    for(hashIdx = 0; hashIdx<dbvar->nbHashes; ++hashIdx) {
        int hashBlockSize = ( (end - ptr) > diffBlockSize ) ? diffBlockSize : end-ptr;
        // keep track of new memory locations for dense hash array
        if ( changeSize ) {
            if ( FTI_GetDcpMode() == FTI_DCP_MODE_MD5 ) {
                hashes[hashIdx].md5hash = (unsigned char*) &(hashes[0].md5hash) + hashIdx * MD5_DIGEST_LENGTH;
            } else {
                if ( hashIdx >= nbHashesOld ) {
                    hashes[hashIdx].md5hash = NULL;
                }
            }
        }
        // invalidate new hashes and former last hash and set block to new size
        if ( hashIdx >= (nbHashesOld-1) ) {
            hashes[hashIdx].isValid = false;
            hashes[hashIdx].blockSize = hashBlockSize;
        }
        hashes[hashIdx].ptr = (FTI_ADDRPTR) ptr;
        ptr += hashBlockSize;
    }
    return FTI_SCES;    
}

long FTI_CalcNumHashes( long chunkSize ) 
{
    if ( (chunkSize%DIFF_BLOCK_SIZE) == 0 ) {
        return chunkSize/DIFF_BLOCK_SIZE;
    } else {
        return chunkSize/DIFF_BLOCK_SIZE + 1;
    }
}

int FTI_HashCmp( long hashIdx, FTIFF_dbvar* dbvar )
{
    
    // if out of range return -1
    if ( hashIdx == dbvar->nbHashes ) {
        return -1;
    } else {
        unsigned char md5hashNow[MD5_DIGEST_LENGTH];
        uint32_t bit32hashNow;
        FTIT_DataDiffHash* hashInfo = &(dbvar->dataDiffHash[hashIdx]);
        bool clean;
        switch ( HASH_MODE ) {
            case FTI_DCP_MODE_MD5:
                MD5(hashInfo->ptr, hashInfo->blockSize, md5hashNow);
                clean = memcmp(md5hashNow, hashInfo->md5hash, MD5_DIGEST_LENGTH) == 0;
            case FTI_DCP_MODE_CRC32:
                bit32hashNow = crc_32( hashInfo->ptr, hashInfo->blockSize );
                clean = bit32hashNow == hashInfo->bit32hash;
                break;
        }
        // set clean if unchanged
        if ( clean ) {
            hashInfo->dirty = false;
            return 0;
        // set dirty if changed
        } else {
            hashInfo->dirty = true;
            return 1;
        }
    } 
}

int FTI_UpdateDcpChanges(FTIT_dataset* FTI_Data, FTIT_execution* FTI_Exec) 
{
    FTIFF_db *db = FTI_Exec->firstdb;
    FTIFF_dbvar *dbvar;
    char *dptr;
    int dbvar_idx, dbcounter=0;

    int isnextdb;

    do {

        isnextdb = 0;

        for(dbvar_idx=0;dbvar_idx<db->numvars;dbvar_idx++) {

            dbvar = &(db->dbvars[dbvar_idx]);
            FTIT_DataDiffHash* hashInfo = dbvar->dataDiffHash;

            int hashIdx;
            for(hashIdx=0; hashIdx<dbvar->nbHashes; ++hashIdx) {
                hashInfo[hashIdx].isValid = true;
                if (hashInfo[hashIdx].dirty) {
                    switch ( HASH_MODE ) {
                        case FTI_DCP_MODE_MD5:
                            MD5(hashInfo->ptr, hashInfo->blockSize, hashInfo[hashIdx].md5hash);
                            break;
                        case FTI_DCP_MODE_CRC32:
                            hashInfo[hashIdx].bit32hash = crc_32( hashInfo->ptr, hashInfo->blockSize );
                            break;
                    }
                }
            }
        }

        if (db->next) {
            db = db->next;
            isnextdb = 1;
        }

        dbcounter++;

    } while( isnextdb );

}

int FTI_ReceiveDataChunk(FTI_ADDRVAL* buffer_offset, FTI_ADDRVAL* buffer_size, FTIFF_dbvar* dbvar, FTIT_dataset* FTI_Data) 
{
    
    static bool init = true;
    static bool reset;
    static long hashIdx;
    char strdbg[FTI_BUFS];
    
    if ( init ) {
        hashIdx = 0;
        reset = false;
        init = false;
    }
    
    // reset function and return not found
    if ( reset ) {
        init = true;
        return 0;
    }
   
    // if differential ckpt is disabled, return whole chunk and finalize call
    if ( !enableDiffCkpt ) {
        reset = true;
        *buffer_offset = (FTI_ADDRVAL) FTI_Data[dbvar->idx].ptr;
        *buffer_size = dbvar->chunksize;
        return 1;
    }

    // advance *buffer_offset for clean regions
    bool clean = FTI_HashCmp( hashIdx, dbvar ) == 0;
    clean &= dbvar->dataDiffHash[hashIdx].isValid;
    while( clean ) {
        hashIdx++;
        clean = FTI_HashCmp( hashIdx, dbvar ) == 0;
        clean &= dbvar->dataDiffHash[hashIdx].isValid;
    }

    /* if at call pointer to dirty region then data_ptr unchanged */
    *buffer_offset = (FTI_ADDRVAL) dbvar->dataDiffHash[hashIdx].ptr;
    *buffer_size = 0;

    // advance *buffer_size for dirty regions
    bool dirty = FTI_HashCmp( hashIdx, dbvar ) == 1;
    dirty |= !(dbvar->dataDiffHash[hashIdx].isValid);
    while( dirty ) {
        *buffer_size += dbvar->dataDiffHash[hashIdx].blockSize;
        hashIdx++;
        dirty = FTI_HashCmp( hashIdx, dbvar ) == 1;
        dirty |= !(dbvar->dataDiffHash[hashIdx].isValid);
    }

    // check if we are at the end of the data region
    if ( hashIdx == dbvar->nbHashes ) {
        if ( *buffer_size != 0 ) {
            reset = true;
            return 1;
        } else {
            init = true;
            return 0;
        }
    }
    return 1;
}
