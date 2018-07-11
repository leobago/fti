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
 *  @file   diff-checkpoint.c
 *  @date   February, 2018
 *  @brief  Differential checkpointing routines.
 */

#define _DEFAULT_SOURCE
#define _BSD_SOURCE

#include "interface.h"

/** File Local Variables                                                                */

static bool* dcpEnabled;
static int                  DCP_MODE;
static dcpBLK_t             DCP_BLOCK_SIZE;

/** Function Definitions                                                                */

int FTI_FinalizeDcp( FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec ) 
{
    // deallocate memory in dcp structures
    FTIFF_db* currentDB = FTI_Exec->firstdb;
    do {    
        int varIdx;
        for(varIdx=0; varIdx<currentDB->numvars; ++varIdx) {
            FTIFF_dbvar* currentdbVar = &(currentDB->dbvars[varIdx]);
            if( currentdbVar->dataDiffHash != NULL ) {
                if( DCP_MODE == FTI_DCP_MODE_MD5 ) {
                    free( currentdbVar->dataDiffHash[0].md5hash );
                }
                free( currentdbVar->dataDiffHash );
            }
        }
    }
    while ( (currentDB = currentDB->next) != NULL );

    // disable dCP
    FTI_Conf->dcpEnabled = false;

    return FTI_SCES;
}

int FTI_InitDcp( FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec, FTIT_dataset* FTI_Data )
{
    char str[FTI_BUFS];

    if ( !FTI_Conf->dcpEnabled ) {
        return FTI_SCES;
    }
    
    int rank;
    MPI_Comm_rank(FTI_COMM_WORLD, &rank);
    if( getenv("FTI_DCP_HASH_MODE") != 0 ) {
        DCP_MODE = atoi(getenv("FTI_DCP_HASH_MODE")) + FTI_DCP_MODE_OFFSET;
        if ( (DCP_MODE < FTI_DCP_MODE_MD5) || (DCP_MODE > FTI_DCP_MODE_CRC32) ) {
            FTI_Print("dCP mode ('Basic:dcp_mode') must be either 1 (MD5) or 2 (CRC32), dCP disabled.", FTI_WARN);
            FTI_Conf->dcpEnabled = false;
            return FTI_NSCS;
        }
    } else {
        // check if dcpMode correct in 'conf.c'
        DCP_MODE = FTI_Conf->dcpMode;
    }
    if( getenv("FTI_DCP_BLOCK_SIZE") != 0 ) {
        int chk_size = atoi(getenv("FTI_DCP_BLOCK_SIZE"));
        if( (chk_size < USHRT_MAX) && (chk_size > 512) ) {
            DCP_BLOCK_SIZE = (dcpBLK_t) chk_size;
        } else {
            snprintf( str, FTI_BUFS, "dCP block size ('Basic:dcp_block_size') must be between 512 and %d bytes, dCP disabled", USHRT_MAX );
            FTI_Print( str, FTI_WARN );
            FTI_Conf->dcpEnabled = false;
            return FTI_NSCS;
        }
    } else {
        // check if dcpBlockSize is in range in 'conf.c'
        DCP_BLOCK_SIZE = (dcpBLK_t) FTI_Conf->dcpBlockSize;
    }

    switch (DCP_MODE) {
        case FTI_DCP_MODE_MD5:
            FTI_Print( "Hash algorithm in use is MD5.", FTI_IDCP );
            break;
        case FTI_DCP_MODE_CRC32:
            FTI_Print( "Hash algorithm in use is CRC32.", FTI_IDCP );
            break;
        default:
            FTI_Print("Hash mode not recognized, dCP disabled!", FTI_WARN);
            FTI_Conf->dcpEnabled = false;
            return FTI_NSCS;
    }
    snprintf( str, FTI_BUFS, "dCP hash block size is %d bytes.", DCP_BLOCK_SIZE);
    FTI_Print( str, FTI_IDCP ); 

    dcpEnabled = &(FTI_Conf->dcpEnabled);

    return FTI_SCES;
}

dcpBLK_t FTI_GetDiffBlockSize() 
{
    return DCP_BLOCK_SIZE;
}

int FTI_GetDcpMode() 
{
    return DCP_MODE;
}

int FTI_InitBlockHashArray( FTIFF_dbvar* dbvar ) 
{   
    dbvar->nbHashes = FTI_CalcNumHashes( dbvar->chunksize );
    dbvar->dataDiffHash = (FTIT_DataDiffHash*) malloc ( sizeof(FTIT_DataDiffHash) * dbvar->nbHashes );
    if( dbvar->dataDiffHash == NULL ) {
        FTI_Print( "FTI_InitBlockHashArray - Unable to allocate memory for dcp meta info, disable dCP...", FTI_WARN );
        return FTI_NSCS;
    }
    FTIT_DataDiffHash* hashes = dbvar->dataDiffHash;
    long pos = 0; 
    long end = dbvar->chunksize;
    int hashIdx;
    if ( FTI_GetDcpMode() == FTI_DCP_MODE_MD5 ) {
        // we want the hash array to be dense
        hashes[0].md5hash = (unsigned char*) malloc( MD5_DIGEST_LENGTH * dbvar->nbHashes );
        if( hashes[0].md5hash == NULL ) {
            FTI_Print( "FTI_InitBlockHashArray - Unable to allocate memory for dcp meta info, disable dCP...", FTI_WARN );
            free(dbvar->dataDiffHash);
            dbvar->dataDiffHash = NULL;
            return FTI_NSCS;
        }
    }
    dcpBLK_t diffBlockSize = FTI_GetDiffBlockSize();
    for(hashIdx = 0; hashIdx<dbvar->nbHashes; ++hashIdx) {
        dcpBLK_t hashBlockSize = ( (end - pos) > diffBlockSize ) ? diffBlockSize : (dcpBLK_t) end-pos;
        
        if ( FTI_GetDcpMode() == FTI_DCP_MODE_MD5 ) {
            hashes[hashIdx].md5hash = (unsigned char*) hashes[0].md5hash + hashIdx * MD5_DIGEST_LENGTH;
        } else {
            hashes[hashIdx].md5hash = NULL;
        }
        hashes[hashIdx].isValid = false;
        hashes[hashIdx].dirty = false;
        hashes[hashIdx].blockSize = hashBlockSize;
        pos += hashBlockSize;
    }

    return FTI_SCES;
}

int FTI_CollapseBlockHashArray( FTIFF_dbvar* dbvar ) 
{
    bool changeSize = true;

    long nbHashesOld = dbvar->nbHashes;

    // update to new number of hashes (which might be actually unchanged)
    dbvar->nbHashes = FTI_CalcNumHashes( dbvar->chunksize );
 
    assert( nbHashesOld >= dbvar->nbHashes );
    if ( dbvar->nbHashes == nbHashesOld ) {
        changeSize = false;
    }

    // reallocate hash array to new size if changed
    if ( changeSize ) {
        unsigned char* hashPtr;
        if ( FTI_GetDcpMode() == FTI_DCP_MODE_MD5 ) {
            // we want the hash array to be dense
            assert( dbvar->dataDiffHash[0].md5hash != NULL );
            hashPtr = (unsigned char*) realloc( dbvar->dataDiffHash[0].md5hash, MD5_DIGEST_LENGTH * dbvar->nbHashes );
            if( hashPtr == NULL ) {
                FTI_Print( "FTI_CollapseBlockHashArray - Unable to allocate memory for dcp meta info, disable dCP...", FTI_WARN );
                free(dbvar->dataDiffHash);
                dbvar->dataDiffHash = NULL;
                return FTI_NSCS;
            }
        }
        assert( dbvar->dataDiffHash != NULL );
        dbvar->dataDiffHash = (FTIT_DataDiffHash*) realloc ( dbvar->dataDiffHash, sizeof(FTIT_DataDiffHash) * dbvar->nbHashes );
        if( dbvar->dataDiffHash == NULL ) {
            FTI_Print( "FTI_CollapseBlockHashArray - Unable to allocate memory for dcp meta info, disable dCP...", FTI_WARN );
            free( hashPtr );
            return FTI_NSCS;
        }
        dbvar->dataDiffHash[0].md5hash = hashPtr;
    }

    
    FTIT_DataDiffHash* hashes = dbvar->dataDiffHash;
    
    // we need this pointer since data was most likely re allocated 
    // and thus might have a new memory location
    long pos = 0;
    long end = dbvar->chunksize;
    int hashIdx;
    int lastIdx = dbvar->nbHashes-1;
    dcpBLK_t diffBlockSize = FTI_GetDiffBlockSize();
    for(hashIdx = 0; hashIdx<dbvar->nbHashes; ++hashIdx) {
        dcpBLK_t hashBlockSize = ( (end - pos) > diffBlockSize ) ? diffBlockSize : end-pos;
        // keep track of new memory locations for dense hash array
        if ( changeSize ) {
            if ( FTI_GetDcpMode() == FTI_DCP_MODE_MD5 ) {
                hashes[hashIdx].md5hash = (unsigned char*) hashes[0].md5hash + hashIdx * MD5_DIGEST_LENGTH;
            } else {
                hashes[hashIdx].md5hash = NULL;
            }
        }
    
        // invalidate last hash in (almost) any case. If number of hashes remain the same, 
        // the last block changed size and with that data changed content.
        // if number decreased and the blocksize of new last block is less the DCP_BLOCK_SIZE
        // the hash is invalid too.
        if ( hashIdx == lastIdx ) {
            hashes[hashIdx].blockSize = hashBlockSize;
            if ( ( hashes[lastIdx].blockSize < DCP_BLOCK_SIZE ) && changeSize ) {
                hashes[lastIdx].isValid = false;
                hashes[lastIdx].dirty = false;
            } else if ( !changeSize ) {
                hashes[lastIdx].isValid = false;
                hashes[lastIdx].dirty = false;
            }
        }
        pos += hashBlockSize;
    }

    return FTI_SCES;    
}

int FTI_ExpandBlockHashArray( FTIFF_dbvar* dbvar ) 
{
    bool changeSize = true;

    long nbHashesOld = dbvar->nbHashes;
    // current last hash is invalid in any case. 
    // If number of blocks remain the same, the size of the last block changed to 'new_size - old_size', 
    // thus also the data that is contained in it. 
    // If the nuber of blocks increased, the blocksize is changed for the current 
    // last block as well, in fact to DCP_BLOCK_SIZE. 
    // This is taken care of in the for loop (after comment 'invalidate new hashes...').
    
    // update to new number of hashes (which might be actually unchanged)
    dbvar->nbHashes = FTI_CalcNumHashes( dbvar->chunksize );
 
    assert( nbHashesOld <= dbvar->nbHashes );
    if ( dbvar->nbHashes == nbHashesOld ) {
        changeSize = false;
    }

    // reallocate hash array to new size if changed
    if ( changeSize ) {
        unsigned char* hashPtr;
        if ( FTI_GetDcpMode() == FTI_DCP_MODE_MD5 ) {
            // we want the hash array to be dense
            assert( dbvar->dataDiffHash[0].md5hash != NULL );
            hashPtr = (unsigned char*) realloc( dbvar->dataDiffHash[0].md5hash, MD5_DIGEST_LENGTH * dbvar->nbHashes );
            if( hashPtr == NULL ) {
                FTI_Print( "FTI_ExpandBlockHashArray - Unable to allocate memory for dcp meta info, disable dCP...", FTI_WARN );
                free(dbvar->dataDiffHash);
                dbvar->dataDiffHash = NULL;
                return FTI_NSCS;
            }
        }
        assert( dbvar->dataDiffHash != NULL );
        dbvar->dataDiffHash = (FTIT_DataDiffHash*) realloc ( dbvar->dataDiffHash, sizeof(FTIT_DataDiffHash) * dbvar->nbHashes );
        if( dbvar->dataDiffHash == NULL ) {
            FTI_Print( "FTI_ExpandBlockHashArray - Unable to allocate memory for dcp meta info, disable dCP...", FTI_WARN );
            free( hashPtr );
            return FTI_NSCS;
        }
        dbvar->dataDiffHash[0].md5hash = hashPtr;
    }
    
    FTIT_DataDiffHash* hashes = dbvar->dataDiffHash;
    
    // we need this pointer since data was most likely re allocated 
    // and thus might have a new memory location
    long pos = 0;
    long end = dbvar->chunksize;
    int hashIdx;
    dcpBLK_t diffBlockSize = FTI_GetDiffBlockSize();
    for(hashIdx = 0; hashIdx<dbvar->nbHashes; ++hashIdx) {
        dcpBLK_t hashBlockSize = ( (end - pos) > diffBlockSize ) ? diffBlockSize : end-pos;
        // keep track of new memory locations for dense hash array
        if ( changeSize ) {
            if ( FTI_GetDcpMode() == FTI_DCP_MODE_MD5 ) {
                hashes[hashIdx].md5hash = (unsigned char*) hashes[0].md5hash + hashIdx * MD5_DIGEST_LENGTH;
            } else {
                if ( hashIdx >= nbHashesOld ) {
                    hashes[hashIdx].md5hash = NULL;
                }
            }
        }
        // invalidate new hashes and former last hash and set block to new size
        if ( hashIdx >= (nbHashesOld-1) ) {
            hashes[hashIdx].isValid = false;
            hashes[hashIdx].dirty = false;
            hashes[hashIdx].blockSize = hashBlockSize;
        }
        pos += hashBlockSize;
    }
    return FTI_SCES;    
}

long FTI_CalcNumHashes( long chunkSize ) 
{
    if ( (chunkSize%((unsigned long)DCP_BLOCK_SIZE)) == 0 ) {
        return chunkSize/DCP_BLOCK_SIZE;
    } else {
        return chunkSize/DCP_BLOCK_SIZE + 1;
    }
}

int FTI_HashCmp( long hashIdx, FTIFF_dbvar* dbvar )
{
    
    // if out of range return -1
    bool clean = true;
    assert( !(hashIdx > dbvar->nbHashes) );
    if ( hashIdx == dbvar->nbHashes ) {
        return -1;
    } else if ( !(dbvar->dataDiffHash[hashIdx].isValid) ){
        return 1;
    } else {
        unsigned char* ptr = (unsigned char*) dbvar->cptr + hashIdx * DCP_BLOCK_SIZE;
        unsigned char md5hashNow[MD5_DIGEST_LENGTH];
        uint32_t bit32hashNow;
        FTIT_DataDiffHash* hashInfo = &(dbvar->dataDiffHash[hashIdx]);
        switch ( DCP_MODE ) {
            case FTI_DCP_MODE_MD5:
                assert((hashInfo->blockSize>0)&&(hashInfo->blockSize<=DCP_BLOCK_SIZE));
                MD5( ptr, hashInfo->blockSize, md5hashNow);
                clean = memcmp(md5hashNow, hashInfo->md5hash, MD5_DIGEST_LENGTH) == 0;
                break;
            case FTI_DCP_MODE_CRC32:
                bit32hashNow = crc_32( ptr, hashInfo->blockSize );
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
    int dbvar_idx, dbcounter=0;

    int isnextdb;

    do {

        isnextdb = 0;

        for(dbvar_idx=0;dbvar_idx<db->numvars;dbvar_idx++) {

            dbvar = &(db->dbvars[dbvar_idx]);
            FTIT_DataDiffHash* hashInfo = dbvar->dataDiffHash;
        
            int hashIdx;
            for(hashIdx=0; hashIdx<dbvar->nbHashes; ++hashIdx) {
                if (hashInfo[hashIdx].dirty || !hashInfo[hashIdx].isValid) {
                    unsigned char* ptr = (unsigned char*) dbvar->cptr + hashIdx * DCP_BLOCK_SIZE;
                    switch ( DCP_MODE ) {
                        case FTI_DCP_MODE_MD5:
                            MD5( ptr, hashInfo[hashIdx].blockSize, hashInfo[hashIdx].md5hash);
                            break;
                        case FTI_DCP_MODE_CRC32:
                            hashInfo[hashIdx].bit32hash = crc_32( ptr, hashInfo[hashIdx].blockSize );
                            break;
                    }
                    if(dbvar->hascontent) {
                        hashInfo[hashIdx].isValid = true;
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

    return FTI_SCES;
}

int FTI_ReceiveDataChunk(FTI_ADDRVAL* buffer_addr, FTI_ADDRVAL* buffer_size, FTIFF_dbvar* dbvar, FTIT_dataset* FTI_Data) 
{

    int rank;
    MPI_Comm_rank(FTI_COMM_WORLD, &rank);

    static bool init = true;
    static bool reset;
    static long hashIdx;
    
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
    if ( !dcpEnabled ) {
        reset = true;
        *buffer_addr = (FTI_ADDRVAL) FTI_Data[dbvar->idx].ptr + dbvar->dptr;
        *buffer_size = dbvar->chunksize;
        return 1;
    }

    // advance *buffer_offset for clean regions
    bool clean = FTI_HashCmp( hashIdx, dbvar ) == 0;
    while( clean ) {
        hashIdx++;
        clean = FTI_HashCmp( hashIdx, dbvar ) == 0;
    }

    // check if region clean until end
    if ( hashIdx == dbvar->nbHashes ) {
        init = true;
        return 0;
    }

    /* if at call pointer to dirty region then data_ptr unchanged */
    *buffer_addr = (FTI_ADDRVAL) dbvar->cptr + hashIdx * DCP_BLOCK_SIZE;
    *buffer_size = 0;

    // advance *buffer_size for dirty regions
    bool dirty = FTI_HashCmp( hashIdx, dbvar ) == 1;
    while( dirty ) {
        *buffer_size += dbvar->dataDiffHash[hashIdx].blockSize;
        hashIdx++;
        dirty = FTI_HashCmp( hashIdx, dbvar ) == 1;
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
