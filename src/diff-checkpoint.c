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

#include "diff-checkpoint.h"

/**                                                                                     */
/** Static Global Variables                                                             */

static int                  HASH_MODE;
static int                  DIFF_BLOCK_SIZE;

static FTIT_DataDiffInfoHash      FTI_HashDiffInfo;   /**< container for diff of datasets     */

/** File Local Variables                                                                */

static bool enableDiffCkpt;
static int diffMode;

/** Function Definitions                                                                */

int FTI_InitDiffCkpt( FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec, FTIT_dataset* FTI_Data )
{
    
    int rank;
    MPI_Comm_rank(FTI_COMM_WORLD, &rank);
    if( getenv("FTI_HASH_MODE") != 0 ) {
        HASH_MODE = atoi(getenv("FTI_HASH_MODE"));
    } else {
        HASH_MODE = 0;
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
            case 2:
                printf("[ " BLU "FTI  DIFFCKPT" RESET " ] : HASH MODE IS -> ADLER32\n");
                break;
            case 3:
                printf("[ " BLU "FTI  DIFFCKPT" RESET " ] : HASH MODE IS -> FLETCHER32\n");
                break;
        }
        printf("[ " BLU "FTI  DIFFCKPT" RESET " ] : DIFF_BLOCK_SIZE IS -> %d\n", DIFF_BLOCK_SIZE);
    }

    enableDiffCkpt = FTI_Conf->enableDiffCkpt;
    
    diffMode = FTI_Conf->diffMode;
    if( enableDiffCkpt && FTI_Conf->diffMode == 0 ) {
        FTI_HashDiffInfo.dataDiff = NULL;
        FTI_HashDiffInfo.nbProtVar = 0;
        return FTI_SCES;
    } else {
        return FTI_SCES;
    }
}

int FTI_FinalizeDiffCkpt()
{
    int res = 0;
    if( enableDiffCkpt ) {
    }
    return ( res == 0 ) ? FTI_SCES : FTI_NSCS;
}

int FTI_RegisterProtections(int idx, FTIT_dataset* FTI_Data, FTIT_execution* FTI_Exec) 
{   
    if( diffMode == 0 ) {
        return FTI_GenerateHashBlocks( idx, FTI_Data, FTI_Exec );
    } else {
        return FTI_SCES;
    }

}

int FTI_UpdateProtections(int idx, FTIT_dataset* FTI_Data, FTIT_execution* FTI_Exec) 
{   
    if( diffMode == 0 ) {
        return FTI_UpdateHashBlocks( idx, FTI_Data, FTI_Exec );
    } else {
        return FTI_SCES;
    }

}

int FTI_UpdateHashBlocks(int idx, FTIT_dataset* FTI_Data, FTIT_execution* FTI_Exec) 
{
    FTI_ADDRVAL data_ptr = (FTI_ADDRVAL) FTI_Data[idx].ptr;
    FTI_ADDRVAL data_end = (FTI_ADDRVAL) FTI_Data[idx].ptr + (FTI_ADDRVAL) FTI_Data[idx].size;
    FTI_ADDRVAL data_size = (FTI_ADDRVAL) FTI_Data[idx].size;

    FTI_HashDiffInfo.dataDiff[idx].basePtr = data_ptr; 
    long newNbBlocks = data_size/DIFF_BLOCK_SIZE;
    long oldNbBlocks;
    newNbBlocks += ((data_size%DIFF_BLOCK_SIZE) == 0) ? 0 : 1;
    oldNbBlocks = FTI_HashDiffInfo.dataDiff[idx].nbBlocks;
    
    assert(oldNbBlocks > 0);
    
    FTI_HashDiffInfo.dataDiff[idx].nbBlocks = newNbBlocks;
    FTI_HashDiffInfo.dataDiff[idx].totalSize = data_size;

    // if number of blocks decreased
    if ( newNbBlocks < oldNbBlocks ) {
        
        // reduce hash array
        FTI_HashDiffInfo.dataDiff[idx].hashBlocks = (FTIT_HashBlock*) realloc(FTI_HashDiffInfo.dataDiff[idx].hashBlocks, sizeof(FTIT_HashBlock)*(newNbBlocks));
        if ( HASH_MODE == 0 ) {
            FTI_HashDiffInfo.dataDiff[idx].hashBlocks[0].md5hash = (unsigned char*) realloc( FTI_HashDiffInfo.dataDiff[idx].hashBlocks[0].md5hash, (MD5_DIGEST_LENGTH)*(newNbBlocks) );
            int hashIdx;
            for(hashIdx = 0; hashIdx<newNbBlocks; ++hashIdx) {
                FTI_HashDiffInfo.dataDiff[idx].hashBlocks[hashIdx].md5hash = FTI_HashDiffInfo.dataDiff[idx].hashBlocks[0].md5hash + (hashIdx) * MD5_DIGEST_LENGTH;
            }
        }

    // if number of blocks increased
    } else if ( newNbBlocks > oldNbBlocks ) {
        
        // extend hash array
        FTI_HashDiffInfo.dataDiff[idx].hashBlocks = (FTIT_HashBlock*) realloc(FTI_HashDiffInfo.dataDiff[idx].hashBlocks, sizeof(FTIT_HashBlock)*(newNbBlocks));    
        int hashIdx;
        if ( HASH_MODE == 0 ) {
            FTI_HashDiffInfo.dataDiff[idx].hashBlocks[0].md5hash = (unsigned char*) realloc( FTI_HashDiffInfo.dataDiff[idx].hashBlocks[0].md5hash, (MD5_DIGEST_LENGTH) * newNbBlocks );
            for(hashIdx = 0; hashIdx<newNbBlocks; ++hashIdx) {
                FTI_HashDiffInfo.dataDiff[idx].hashBlocks[hashIdx].md5hash = FTI_HashDiffInfo.dataDiff[idx].hashBlocks[0].md5hash + (hashIdx) * MD5_DIGEST_LENGTH;
            }
        }
        data_ptr += oldNbBlocks * DIFF_BLOCK_SIZE;
        // set new hash values
        for(hashIdx = oldNbBlocks; hashIdx<newNbBlocks; ++hashIdx) {
            FTI_HashDiffInfo.dataDiff[idx].hashBlocks[hashIdx].dirty = true; 
            FTI_HashDiffInfo.dataDiff[idx].hashBlocks[hashIdx].isValid = false; 
        }
    
    }
    return 0;
}

int FTI_GenerateHashBlocks( int idx, FTIT_dataset* FTI_Data, FTIT_execution* FTI_Exec ) 
{
   
    FTI_HashDiffInfo.dataDiff = (FTIT_DataDiffHash*) realloc( FTI_HashDiffInfo.dataDiff, (FTI_HashDiffInfo.nbProtVar+1) * sizeof(FTIT_DataDiffHash));
    assert( FTI_HashDiffInfo.dataDiff != NULL );
    FTI_ADDRVAL basePtr = (FTI_ADDRVAL) FTI_Data[idx].ptr;
    FTI_ADDRVAL ptr = (FTI_ADDRVAL) FTI_Data[idx].ptr;
    FTI_ADDRVAL end = (FTI_ADDRVAL) FTI_Data[idx].ptr + (FTI_ADDRVAL) FTI_Data[idx].size;
    long nbHashBlocks = FTI_Data[idx].size/DIFF_BLOCK_SIZE;
    nbHashBlocks += ( (FTI_Data[idx].size%DIFF_BLOCK_SIZE) == 0 ) ? 0 : 1; 
    FTI_HashDiffInfo.dataDiff[FTI_HashDiffInfo.nbProtVar].nbBlocks = nbHashBlocks;
    FTIT_HashBlock* hashBlocks = (FTIT_HashBlock*) malloc( sizeof(FTIT_HashBlock) * nbHashBlocks );
    assert( hashBlocks != NULL );
    // keep hashblocks array dense
    if ( HASH_MODE == 0 ) {
        hashBlocks[0].md5hash = (unsigned char*) malloc( (MD5_DIGEST_LENGTH) * nbHashBlocks );
        assert( hashBlocks[0].md5hash != NULL );
    }
    long cnt = 0;
    while( ptr < end ) {
        int hashBlockSize = ( (end - ptr) > DIFF_BLOCK_SIZE ) ? DIFF_BLOCK_SIZE : end-ptr;
        if ( HASH_MODE == 0 ) {
            hashBlocks[cnt].md5hash = hashBlocks[0].md5hash + cnt*MD5_DIGEST_LENGTH;
        }
        hashBlocks[cnt].dirty = true;
        hashBlocks[cnt].isValid = false;
        cnt++;
        ptr+=hashBlockSize;
    }
    assert( nbHashBlocks == cnt );
    FTI_HashDiffInfo.dataDiff[FTI_HashDiffInfo.nbProtVar].hashBlocks    = hashBlocks;
    FTI_HashDiffInfo.dataDiff[FTI_HashDiffInfo.nbProtVar].basePtr       = basePtr;
    FTI_HashDiffInfo.dataDiff[FTI_HashDiffInfo.nbProtVar].totalSize     = end - basePtr;
    FTI_HashDiffInfo.dataDiff[FTI_HashDiffInfo.nbProtVar].id            = FTI_Data[idx].id;
    FTI_HashDiffInfo.nbProtVar++;
    return FTI_SCES;
}

int FTI_HashCmp( int varIdx, long hashIdx, FTI_ADDRPTR ptr, int hashBlockSize ) 
{
    
    // check if in range
    if ( hashIdx < FTI_HashDiffInfo.dataDiff[varIdx].nbBlocks ) {
        unsigned char md5hashNow[MD5_DIGEST_LENGTH];
        uint32_t bit32hashNow;
        bool clean;
        if ( HASH_MODE == 0 ) {
            MD5(ptr, hashBlockSize, md5hashNow);
            clean = memcmp(md5hashNow, FTI_HashDiffInfo.dataDiff[varIdx].hashBlocks[hashIdx].md5hash, MD5_DIGEST_LENGTH) == 0;
        } else {
            switch ( HASH_MODE ) {
                case 1:
                    bit32hashNow = crc_32( ptr, hashBlockSize );
                    break;
                case 2:
                    bit32hashNow = adler32( ptr, hashBlockSize );
                    break;
                case 3:
                    bit32hashNow = fletcher32( ptr, hashBlockSize );
                    break;
            }
            clean = bit32hashNow == FTI_HashDiffInfo.dataDiff[varIdx].hashBlocks[hashIdx].bit32hash;
        }
        // set clean if unchanged
        if ( clean ) {
            FTI_HashDiffInfo.dataDiff[varIdx].hashBlocks[hashIdx].dirty = false;
            return 0;
        // set dirty if changed
        } else {
            FTI_HashDiffInfo.dataDiff[varIdx].hashBlocks[hashIdx].dirty = true;
            return 1;
        }
    // return -1 if end
    } else {
        return -1;
    }
}

int FTI_UpdateChanges(FTIT_dataset* FTI_Data) 
{
    if( diffMode == 0 ) {
        return FTI_UpdateHashChanges( FTI_Data );
    } else {
        return FTI_SCES;
    }
}

int FTI_UpdateHashChanges(FTIT_dataset* FTI_Data) 
{

    struct timespec t1;
    clock_gettime( CLOCK_REALTIME, &t1 );
    double start_t = MPI_Wtime();
    int varIdx;
    int nbProtVar = FTI_HashDiffInfo.nbProtVar;
    long memuse = 0;
    long totalmemprot = 0;
    for(varIdx=0; varIdx<nbProtVar; ++varIdx) {
        assert(FTI_Data[varIdx].size == FTI_HashDiffInfo.dataDiff[varIdx].totalSize);
        totalmemprot += FTI_Data[varIdx].size;
        FTI_ADDRPTR ptr = FTI_Data[varIdx].ptr;
        long pos = 0;
        int width = 0;
        int blockIdx;
        int nbBlocks = FTI_HashDiffInfo.dataDiff[varIdx].nbBlocks;
        for(blockIdx=0; blockIdx<nbBlocks; ++blockIdx) {
            if ( HASH_MODE == 0 ) {
                memuse += MD5_DIGEST_LENGTH;
            } else {
                memuse += sizeof(uint32_t); 
            }
            if ( !FTI_HashDiffInfo.dataDiff[varIdx].hashBlocks[blockIdx].isValid || FTI_HashDiffInfo.dataDiff[varIdx].hashBlocks[blockIdx].dirty ) {
                width = ( (FTI_HashDiffInfo.dataDiff[varIdx].totalSize - pos) > DIFF_BLOCK_SIZE ) ? DIFF_BLOCK_SIZE : (FTI_HashDiffInfo.dataDiff[varIdx].totalSize - pos);
                switch ( HASH_MODE ) {
                    case 0:
                        MD5(ptr, width, FTI_HashDiffInfo.dataDiff[varIdx].hashBlocks[blockIdx].md5hash);
                        break;
                    case 1:
                        FTI_HashDiffInfo.dataDiff[varIdx].hashBlocks[blockIdx].bit32hash = crc_32( ptr, width );
                        break;
                    case 2:
                        FTI_HashDiffInfo.dataDiff[varIdx].hashBlocks[blockIdx].bit32hash = adler32( ptr, width );
                        break;
                    case 3:
                        FTI_HashDiffInfo.dataDiff[varIdx].hashBlocks[blockIdx].bit32hash = fletcher32( ptr, width );
                        break;
                }
                FTI_HashDiffInfo.dataDiff[varIdx].hashBlocks[blockIdx].dirty = false;
                FTI_HashDiffInfo.dataDiff[varIdx].hashBlocks[blockIdx].isValid = true;
            }
            ptr += (FTI_ADDRVAL) width;
            pos += width;
        }
    }
    char strout[FTI_BUFS];
    int rank;
}

int FTI_ReceiveDiffChunk(int id, FTI_ADDRVAL data_offset, FTI_ADDRVAL data_size, FTI_ADDRVAL* buffer_offset, FTI_ADDRVAL* buffer_size, FTIT_execution* FTI_Exec, FTIFF_dbvar* dbvar) 
{
    
    static bool init = true;
    static long pos;
    static FTI_ADDRVAL data_ptr;
    static FTI_ADDRVAL data_end;
    static long hash_ptr;
    char strdbg[FTI_BUFS];
    if ( init ) {
        hash_ptr = dbvar->dptr;
        pos = 0;
        data_ptr = data_offset;
        data_end = data_offset + data_size;
        init = false;
    }
    
    int idx;
    long i;
    bool flag;
    // reset function and return not found
    if ( pos == -1 ) {
        init = true;
        return 0;
    }
   
    // if differential ckpt is disabled, return whole chunk and finalize call
    if ( !enableDiffCkpt ) {
        pos = -1;
        *buffer_offset = data_ptr;
        *buffer_size = data_size;
        return 1;
    }

    if ( diffMode == 0 ) {
        for(idx=0; (flag = FTI_HashDiffInfo.dataDiff[idx].id != id) && (idx < FTI_HashDiffInfo.nbProtVar); ++idx);
        if ( !flag ) {
            
            int hashBlockSize = ( (data_end - data_ptr) > DIFF_BLOCK_SIZE ) ? DIFF_BLOCK_SIZE : data_end - data_ptr;
            long hashIdx = hash_ptr/DIFF_BLOCK_SIZE;
            
            /* TODO CREATE HASH BLOCKS BELONGING TO CONTAINERS INSTEAD OF WHOLE DATASET */
            if ( (hash_ptr%DIFF_BLOCK_SIZE) == data_size ) {
                printf("WARNING: manual correcting applied!\n");
                *buffer_offset = data_ptr;
                *buffer_size = data_size;
                pos = -1;
                return 1;
            }
            
            // advance *buffer_offset for clean regions
            bool clean = FTI_HashCmp( idx, hashIdx, (FTI_ADDRPTR) data_ptr, hashBlockSize ) == 0;
            clean &= FTI_HashDiffInfo.dataDiff[idx].hashBlocks[hashIdx].isValid;
            int clean_cnt=0;
            while( clean ) {
                clean_cnt++;
                data_ptr += hashBlockSize;
                hash_ptr += hashBlockSize;
                hashIdx = hash_ptr/DIFF_BLOCK_SIZE;
                if (hashIdx >= FTI_HashDiffInfo.dataDiff[idx].nbBlocks) {
                    hashIdx--;
                    break; 
                }
                hashBlockSize = ( (data_end - data_ptr) > DIFF_BLOCK_SIZE ) ? DIFF_BLOCK_SIZE : data_end - data_ptr;
                clean = FTI_HashCmp( idx, hashIdx, (FTI_ADDRPTR) data_ptr, hashBlockSize ) == 0;
                clean &= FTI_HashDiffInfo.dataDiff[idx].hashBlocks[hashIdx].isValid;
            }
            
            /* if at call pointer to dirty region then data_ptr unchanged */
            *buffer_offset = data_ptr;
            *buffer_size = 0;
            
            // advance *buffer_size for dirty regions
            bool dirty = FTI_HashCmp( idx, hashIdx, (FTI_ADDRPTR) data_ptr, hashBlockSize ) == 1;
            dirty |= !(FTI_HashDiffInfo.dataDiff[idx].hashBlocks[hashIdx].isValid);
            bool inRange = data_ptr < data_end;
            int dirty_cnt = 0;
            while( dirty && inRange ) {
                dirty_cnt++;
                *buffer_size += hashBlockSize;
                data_ptr += hashBlockSize;
                hash_ptr += hashBlockSize;
                hashIdx = hash_ptr/DIFF_BLOCK_SIZE;
                if (hashIdx >= FTI_HashDiffInfo.dataDiff[idx].nbBlocks) {
                    hashIdx--;
                    break; 
                }
                hashBlockSize = ( (data_end - data_ptr) > DIFF_BLOCK_SIZE ) ? DIFF_BLOCK_SIZE : data_end - data_ptr;
                dirty = FTI_HashCmp( idx, hashIdx, (FTI_ADDRPTR) data_ptr, hashBlockSize ) == 1;
                dirty |= !(FTI_HashDiffInfo.dataDiff[idx].hashBlocks[hashIdx].isValid);
                inRange = data_ptr < data_end;
            }
             
            // check if we are at the end of the data region
            if ( data_ptr == data_end ) {
                if ( *buffer_size != 0 ) {
                    pos = -1;
                    return 1;
                } else {
                    init = true;
                    return 0;
                }
            }
            pos = hashIdx;
            return 1;
        }
    }
    
    // nothing to return -> function reset
    init = true;
    return 0;
}
