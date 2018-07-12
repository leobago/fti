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


#ifdef FTI_NOZLIB
const uint32_t crc32_tab[] = {
    0x00000000, 0x77073096, 0xee0e612c, 0x990951ba, 0x076dc419, 0x706af48f,
    0xe963a535, 0x9e6495a3, 0x0edb8832, 0x79dcb8a4, 0xe0d5e91e, 0x97d2d988,
    0x09b64c2b, 0x7eb17cbd, 0xe7b82d07, 0x90bf1d91, 0x1db71064, 0x6ab020f2,
    0xf3b97148, 0x84be41de, 0x1adad47d, 0x6ddde4eb, 0xf4d4b551, 0x83d385c7,
    0x136c9856, 0x646ba8c0, 0xfd62f97a, 0x8a65c9ec, 0x14015c4f, 0x63066cd9,
    0xfa0f3d63, 0x8d080df5, 0x3b6e20c8, 0x4c69105e, 0xd56041e4, 0xa2677172,
    0x3c03e4d1, 0x4b04d447, 0xd20d85fd, 0xa50ab56b, 0x35b5a8fa, 0x42b2986c,
    0xdbbbc9d6, 0xacbcf940, 0x32d86ce3, 0x45df5c75, 0xdcd60dcf, 0xabd13d59,
    0x26d930ac, 0x51de003a, 0xc8d75180, 0xbfd06116, 0x21b4f4b5, 0x56b3c423,
    0xcfba9599, 0xb8bda50f, 0x2802b89e, 0x5f058808, 0xc60cd9b2, 0xb10be924,
    0x2f6f7c87, 0x58684c11, 0xc1611dab, 0xb6662d3d, 0x76dc4190, 0x01db7106,
    0x98d220bc, 0xefd5102a, 0x71b18589, 0x06b6b51f, 0x9fbfe4a5, 0xe8b8d433,
    0x7807c9a2, 0x0f00f934, 0x9609a88e, 0xe10e9818, 0x7f6a0dbb, 0x086d3d2d,
    0x91646c97, 0xe6635c01, 0x6b6b51f4, 0x1c6c6162, 0x856530d8, 0xf262004e,
    0x6c0695ed, 0x1b01a57b, 0x8208f4c1, 0xf50fc457, 0x65b0d9c6, 0x12b7e950,
    0x8bbeb8ea, 0xfcb9887c, 0x62dd1ddf, 0x15da2d49, 0x8cd37cf3, 0xfbd44c65,
    0x4db26158, 0x3ab551ce, 0xa3bc0074, 0xd4bb30e2, 0x4adfa541, 0x3dd895d7,
    0xa4d1c46d, 0xd3d6f4fb, 0x4369e96a, 0x346ed9fc, 0xad678846, 0xda60b8d0,
    0x44042d73, 0x33031de5, 0xaa0a4c5f, 0xdd0d7cc9, 0x5005713c, 0x270241aa,
    0xbe0b1010, 0xc90c2086, 0x5768b525, 0x206f85b3, 0xb966d409, 0xce61e49f,
    0x5edef90e, 0x29d9c998, 0xb0d09822, 0xc7d7a8b4, 0x59b33d17, 0x2eb40d81,
    0xb7bd5c3b, 0xc0ba6cad, 0xedb88320, 0x9abfb3b6, 0x03b6e20c, 0x74b1d29a,
    0xead54739, 0x9dd277af, 0x04db2615, 0x73dc1683, 0xe3630b12, 0x94643b84,
    0x0d6d6a3e, 0x7a6a5aa8, 0xe40ecf0b, 0x9309ff9d, 0x0a00ae27, 0x7d079eb1,
    0xf00f9344, 0x8708a3d2, 0x1e01f268, 0x6906c2fe, 0xf762575d, 0x806567cb,
    0x196c3671, 0x6e6b06e7, 0xfed41b76, 0x89d32be0, 0x10da7a5a, 0x67dd4acc,
    0xf9b9df6f, 0x8ebeeff9, 0x17b7be43, 0x60b08ed5, 0xd6d6a3e8, 0xa1d1937e,
    0x38d8c2c4, 0x4fdff252, 0xd1bb67f1, 0xa6bc5767, 0x3fb506dd, 0x48b2364b,
    0xd80d2bda, 0xaf0a1b4c, 0x36034af6, 0x41047a60, 0xdf60efc3, 0xa867df55,
    0x316e8eef, 0x4669be79, 0xcb61b38c, 0xbc66831a, 0x256fd2a0, 0x5268e236,
    0xcc0c7795, 0xbb0b4703, 0x220216b9, 0x5505262f, 0xc5ba3bbe, 0xb2bd0b28,
    0x2bb45a92, 0x5cb36a04, 0xc2d7ffa7, 0xb5d0cf31, 0x2cd99e8b, 0x5bdeae1d,
    0x9b64c2b0, 0xec63f226, 0x756aa39c, 0x026d930a, 0x9c0906a9, 0xeb0e363f,
    0x72076785, 0x05005713, 0x95bf4a82, 0xe2b87a14, 0x7bb12bae, 0x0cb61b38,
    0x92d28e9b, 0xe5d5be0d, 0x7cdcefb7, 0x0bdbdf21, 0x86d3d2d4, 0xf1d4e242,
    0x68ddb3f8, 0x1fda836e, 0x81be16cd, 0xf6b9265b, 0x6fb077e1, 0x18b74777,
    0x88085ae6, 0xff0f6a70, 0x66063bca, 0x11010b5c, 0x8f659eff, 0xf862ae69,
    0x616bffd3, 0x166ccf45, 0xa00ae278, 0xd70dd2ee, 0x4e048354, 0x3903b3c2,
    0xa7672661, 0xd06016f7, 0x4969474d, 0x3e6e77db, 0xaed16a4a, 0xd9d65adc,
    0x40df0b66, 0x37d83bf0, 0xa9bcae53, 0xdebb9ec5, 0x47b2cf7f, 0x30b5ffe9,
    0xbdbdf21c, 0xcabac28a, 0x53b39330, 0x24b4a3a6, 0xbad03605, 0xcdd70693,
    0x54de5729, 0x23d967bf, 0xb3667a2e, 0xc4614ab8, 0x5d681b02, 0x2a6f2b94,
    0xb40bbe37, 0xc30c8ea1, 0x5a05df1b, 0x2d02ef8d
};
#endif

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
#ifdef FTI_NOZLIB
                bit32hashNow = crc32( ptr, hashInfo->blockSize );
#else
                bit32hashNow = crc32( 0L, Z_NULL, 0 );
                bit32hashNow = crc32( bit32hashNow, ptr, hashInfo->blockSize );
#endif
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
#ifdef FTI_NOZLIB
                            DBG_MSG("NO ZLIB!",-1);
                            hashInfo[hashIdx].bit32hash = crc32( ptr, hashInfo[hashIdx].blockSize );
#else
                            hashInfo[hashIdx].bit32hash = crc32( 0L, Z_NULL, 0 ); 
                            hashInfo[hashIdx].bit32hash = crc32( hashInfo[hashIdx].bit32hash, ptr, hashInfo[hashIdx].blockSize );
#endif                            
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
