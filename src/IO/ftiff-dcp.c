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

#include "../interface.h"
#include "ftiff-dcp.h"

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

/** File Local Variables                                                */

static bool* dcpEnabled = NULL;
static int                  DCP_MODE = 0;
static dcpBLK_t             DCP_BLOCK_SIZE = 1;

const char* hashType[] = {
    "NEW HASH",
    "REALLOCED DECREASED SIZE",
    "REALLOC INCREASED SIZE",
    "DELETED HASH"
};


#define CURRENT(var) (var->currentId)
#define NEXT(var)    ((var->currentId + 1)%2)

#define NEWHASH 0
#define HASHREALLOC_DEC 1
#define HASHREALLOC_INC 2
#define HASHREALLOC_DEL 3



/** Function Definitions                                                   */

/*-------------------------------------------------------------------------*/
/**
  @brief      Allocates and initializes the next data of a new hash table.
  @param      FTIT_DataDiffHash hashes data which need to be initialized.
  @return     integer         FTI_SCES if successful.

  This function allocates the structures for the next hash which will be created
  during this checkpoint round
 **/
/*-------------------------------------------------------------------------*/


int FTI_InitNextHashData(FTIT_DataDiffHash *hashes) {
    if (!dcpEnabled)
        return FTI_SCES;

    if (!dcpEnabled)
        return FTI_SCES;

    if (hashes->nbHashes == 0) {
        FTI_Print("THIS SHOULD NEVER HAPPEN", FTI_EROR);
    }


    if (FTI_GetDcpMode() == FTI_DCP_MODE_MD5) {
        if (hashes->md5hash[NEXT(hashes)] != NULL) {
            FTI_Print("The next hash table should be always NULL"
            " before initializing it", FTI_EROR);
        }

        hashes->md5hash[NEXT(hashes)] = (unsigned char *)
        malloc(sizeof(unsigned char) * MD5_DIGEST_LENGTH * hashes->nbHashes);

        if (!hashes->md5hash[NEXT(hashes)]) {
            FTI_Print("Could Not Allocate memory for hashes", FTI_EROR);
            return FTI_NSCS;
        }
        return FTI_SCES;
    } else {
        if (hashes->bit32hash[NEXT(hashes)] != NULL) {
            FTI_Print("The next hash table should be always NULL"
            " before initializing it", FTI_EROR);
        }

        hashes->bit32hash[NEXT(hashes)] = talloc(uint32_t, hashes->nbHashes);
        if (!hashes->bit32hash[NEXT(hashes)]) {
            FTI_Print("Could Not Allocate memory for hashes", FTI_EROR);
            return FTI_NSCS;
        }

        return FTI_SCES;
    }
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Deallocates all data regarding the hash codes .
  @param      FTIT_DataDiffHash hashes data which need to freed.
  @return     integer         FTI_SCES if successful.

  This function allocates the structures for the next hash which will be created
  during this checkpoint round
 **/
/*-------------------------------------------------------------------------*/

int FTI_FreeDataDiff(FTIT_DataDiffHash *dhash) {
    dhash->currentId = 0;
    dhash->creationType = HASHREALLOC_DEL;

    if (FTI_GetDcpMode() == FTI_DCP_MODE_MD5) {
        if (dhash->md5hash[0]) {
            free(dhash->md5hash[0]);
            dhash->md5hash[0] = NULL;
        }
        if (dhash->md5hash[1]) {
            free(dhash->md5hash[1]);
            dhash->md5hash[1] = NULL;
        }
    } else {
        if (dhash->bit32hash[0]) {
            free(dhash->bit32hash[0]);
            dhash->bit32hash[0] = NULL;
        }
        if (dhash->bit32hash[1]) {
            free(dhash->bit32hash[1]);
            dhash->bit32hash[1] = NULL;
        }
    }

    free(dhash->blockSize);
    dhash->blockSize = NULL;
    free(dhash->isValid);
    dhash->isValid = NULL;
    return FTI_SCES;
}


/*-------------------------------------------------------------------------*/
/**
  @brief      Finalizes dCP
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @return     integer         FTI_SCES if successful.

  This function deallocates structures used for dCP and exposes the 
  status dcp disabled to FTI. It is also called for failures during
  dCP creation.
 **/
/*-------------------------------------------------------------------------*/
int FTI_FinalizeDcp(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec) {
    // nothing to do, no ckpt was taken.
    if (FTI_Exec->firstdb == NULL) {
        FTI_Conf->dcpFtiff = false;
        return FTI_SCES;
    }

    // deallocate memory in dcp structures
    FTIFF_db* currentDB = FTI_Exec->firstdb;
    do {
        int varIdx;
        for (varIdx = 0; varIdx < currentDB->numvars; ++varIdx) {
            FTIFF_dbvar* currentdbVar = &(currentDB->dbvars[varIdx]);
            if (currentdbVar->dataDiffHash != NULL) {
                FTI_FreeDataDiff(currentdbVar->dataDiffHash);
                free(currentdbVar->dataDiffHash);
                currentdbVar->dataDiffHash = NULL;
            }
        }
    } while ((currentDB = currentDB->next) != NULL);

    // disable dCP
    FTI_Conf->dcpFtiff = false;

    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Initializes dCP 
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Data        Dataset metadata.
  @return     integer         FTI_SCES if successful.

  This function looks for environment variables set for the dCP mode and dCP
  block size and overwrites, if found, the values from the configuration file.

  It also initializes the file local variables 'dcpEnabled', 'DCP_MODE' and 
  'DCP_BLOCK_SIZE'.
 **/
/*-------------------------------------------------------------------------*/
int FTI_InitDcp(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec) {
    char str[FTI_BUFS];
    if (getenv("FTI_DCP_HASH_MODE") != 0) {
        DCP_MODE = atoi(getenv("FTI_DCP_HASH_MODE")) + FTI_DCP_MODE_OFFSET;
        if ((DCP_MODE < FTI_DCP_MODE_MD5) || (DCP_MODE > FTI_DCP_MODE_CRC32)) {
            FTI_Print("dCP mode ('Basic:dcp_mode') must be either 1 (MD5)"
                " or 2 (CRC32), dCP disabled.", FTI_WARN);
            FTI_Conf->dcpFtiff = false;
            return FTI_NSCS;
        }
    } else {
        // check if dcpMode correct in 'conf.c'
        DCP_MODE = FTI_Conf->dcpMode;
    }
    if (getenv("FTI_DCP_BLOCK_SIZE") != 0) {
        int chk_size = atoi(getenv("FTI_DCP_BLOCK_SIZE"));
        if ((chk_size < USHRT_MAX) && (chk_size > 512)) {
            DCP_BLOCK_SIZE = (dcpBLK_t) chk_size;
        } else {
            snprintf(str, FTI_BUFS, "dCP block size ('Basic:dcp_block_size')"
                " must be between 512 and %d bytes, dCP disabled", USHRT_MAX);
            FTI_Print(str, FTI_WARN);
            FTI_Conf->dcpFtiff = false;
            return FTI_NSCS;
        }
    } else {
        // check if dcpBlockSize is in range in 'conf.c'
        DCP_BLOCK_SIZE = (dcpBLK_t) FTI_Conf->dcpBlockSize;
    }

    switch (DCP_MODE) {
        case FTI_DCP_MODE_MD5:
            FTI_Print("Hash algorithm in use is MD5.", FTI_IDCP);
            break;
        case FTI_DCP_MODE_CRC32:
            FTI_Print("Hash algorithm in use is CRC32.", FTI_IDCP);
            break;
        default:
            FTI_Print("Hash mode not recognized, dCP disabled!", FTI_WARN);
            FTI_Conf->dcpFtiff = false;
            return FTI_NSCS;
    }
    snprintf(str, FTI_BUFS, "dCP hash block size is %d bytes.",
     DCP_BLOCK_SIZE);
    FTI_Print(str, FTI_IDCP);

    dcpEnabled = &(FTI_Conf->dcpFtiff);

    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Returns the dCP block size
 **/
/*-------------------------------------------------------------------------*/
dcpBLK_t FTI_GetDiffBlockSize() {
    return DCP_BLOCK_SIZE;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Returns the dCP mode.
 **/
/*-------------------------------------------------------------------------*/
int FTI_GetDcpMode() {
    return DCP_MODE;
}


/*-------------------------------------------------------------------------*/
/**
  @brief      Reallocate  meta data related to dCP 
  @param      FTIT_DataDiffHash   metadata to be reallocated.
  @param      int32_t                number of hashes that i need to reallocate
  @return     integer             FTI_SCES if successful.
  This function reallocates all the the metadata related to the dCP (isValid & blockSize);
 **/
/*-------------------------------------------------------------------------*/

int FTI_ReallocateDataDiff(FTIT_DataDiffHash *dhash, int32_t nbHashes) {
    if (!dcpEnabled)
        return FTI_SCES;

    if (!(*dcpEnabled))
        return FTI_SCES;

    void *check = NULL;
    assert(dhash != NULL);
    assert(dhash->isValid!= NULL);
    assert(dhash->blockSize != NULL);

    if (nbHashes == 0) {
        FTI_Print("I AM GOING TO REDUSE SIZE TO 0", FTI_EROR);
    }

    // Re Allocate isValid | blockSize arrays
    check =  realloc (dhash->isValid, sizeof(bool) * nbHashes);
    if (check == NULL) {
        // PrintDataHashInfo(dhash, -1, nbHashes);
        FTI_Print("FTI_ReallocateDataDiff :: Unable to reallocate"
        " memory for dcp meta info", FTI_EROR);
        return FTI_NSCS;
    } else {
        dhash->isValid = (bool*) check;
    }
    check = realloc(dhash->blockSize, sizeof(int16_t) * nbHashes);

    if (!check) {
        // PrintDataHashInfo(dhash, -1, -1);
        FTI_Print("FTI_ReallocateDataDiff blockSize- Unable to allocate "
            "memory for dcp meta info, disable dCP...", FTI_WARN);
        return FTI_NSCS;
    } else {
        dhash->blockSize = (uint16_t*) check;
    }
    return FTI_SCES;
}



/*-------------------------------------------------------------------------*/
/**
  @brief      Initializes a new hash meta data structure for data chunk
  @param      dbvar           Datchunk metadata.
  @return     integer         FTI_SCES if successful.

  This function allocates memory for the 'dataDiffHash' member of the 
  'FTIFF_dbvar' structure and if dCP mode is MD5 also for the MD5 digest
  array placed in the member 'md5hash' of the 'dataDiffHash' structure.

  It also initializes the other members of the 'dataDiffHash' structure.
 **/
/*-------------------------------------------------------------------------*/
int FTI_InitBlockHashArray(FTIFF_dbvar* dbvar) {
    if (!dcpEnabled)
        return FTI_SCES;

    if (!(*dcpEnabled))
        return FTI_SCES;


    dbvar->dataDiffHash = talloc(FTIT_DataDiffHash, 1);
    if (dbvar->dataDiffHash == NULL) {
        FTI_Print("FTI_InitBlockHashArray - Unable to allocate memory "
            "for dcp meta info, disable dCP...", FTI_WARN);
        return FTI_NSCS;
    }

    FTIT_DataDiffHash* hashes = dbvar->dataDiffHash;
    hashes->creationType = NEWHASH;
    hashes->lifetime = 0;
    int nbHashes = FTI_CalcNumHashes(dbvar->chunksize);
    hashes->currentId = 0;
    hashes->nbHashes = nbHashes;
    int hashIdx;

    // I dont need to allocate memory for the hash codes.
    // This will be done when I compute the hashvalues
    hashes->md5hash[0] = NULL;
    hashes->md5hash[1] = NULL;
    hashes->bit32hash[0] = NULL;
    hashes->bit32hash[1] = NULL;

    // This will be done when start calculate the hash codes themselfes.
    // I only allocate memory for the data regarding the status of the dataset.
    hashes->isValid = talloc(bool, nbHashes);


    if (!hashes->isValid) {
        FTI_Print("FTI_InitBlockHashArray - Unable to allocate memory for dcp"
        " meta info, disable dCP...", FTI_WARN);
        free(dbvar->dataDiffHash);
        dbvar->dataDiffHash = NULL;
        return FTI_NSCS;
    }

    hashes->blockSize = talloc(uint16_t, nbHashes);

    if (!hashes->blockSize) {
        FTI_Print("FTI_InitBlockHashArray - Unable to allocate memory for dcp"
        " meta info, disable dCP...", FTI_WARN);
        free(dbvar->dataDiffHash);
        dbvar->dataDiffHash = NULL;
        return FTI_NSCS;
    }

    dcpBLK_t diffBlockSize = FTI_GetDiffBlockSize();
    for (hashIdx = 0; hashIdx < nbHashes - 1; ++hashIdx) {
        hashes->isValid[hashIdx] = false;
        hashes->blockSize[hashIdx] = diffBlockSize;
    }

    hashes->isValid[hashIdx] = false;
    hashes->blockSize[hashIdx] = dbvar->chunksize -
     (nbHashes - 1) * diffBlockSize;

    if (hashes->blockSize[hashIdx] > diffBlockSize) {
        FTI_Print("FTI_InitBlockHashArray :: The blockSize computed is larger"
        " than the supported one", FTI_EROR);
        return FTI_NSCS;
    }


    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Shrinks an existing hash meta data structure for data chunk
  @param      dataHash        The hash meta that need to be expanded.
  @param      chunkSize       The new Size of the data.
  @return     integer         FTI_SCES if successful.

  This function re-allocates memory for the 'dataDiffHash' inValid and blockSize member
  of the  data hash structure and if dCP 
CAUTION: This function does not reallocate the actual hashes of the data struct.
When the checkpoint will terminate the current hash will be freed, whereas the next 
will be used as current. Keep in mind that the next has the correct size
 **/
/*-------------------------------------------------------------------------*/
int FTI_CollapseBlockHashArray(FTIT_DataDiffHash* hashes, int64_t chunkSize) {
    if (!dcpEnabled)
        return FTI_SCES;

    if (!(*dcpEnabled))
        return FTI_SCES;

    bool changeSize = true;

    int32_t nbHashesOld = hashes->nbHashes;
    int32_t newNumber = FTI_CalcNumHashes(chunkSize);

    // update to new number of hashes (which might be actually unchanged)
    hashes->nbHashes = newNumber;

    assert(nbHashesOld >= newNumber);

    if (newNumber == nbHashesOld) {
        changeSize = false;
    }

    hashes->creationType = HASHREALLOC_DEC;

    // reallocate hash array to new size if changed
    // CAUTION:
    if (changeSize) {
        FTI_ReallocateDataDiff(hashes, newNumber);
    }

    int lastIdx = newNumber -1;
    dcpBLK_t diffBlockSize = FTI_GetDiffBlockSize();
    uint16_t lastBlockSize = (uint16_t) (chunkSize -
     ((chunkSize)/DCP_BLOCK_SIZE)*DCP_BLOCK_SIZE);

    hashes->blockSize[lastIdx] = lastBlockSize;
    if ((hashes->blockSize[lastIdx] < DCP_BLOCK_SIZE) && changeSize) {
        hashes->isValid[lastIdx] = false;
    } else if (!changeSize) {
        hashes->isValid[lastIdx] = false;
    }

    if (hashes->blockSize[lastIdx] > diffBlockSize) {
        FTI_Print("FTI_CollapseBlockHashArray:: The blockSize computed "
            "is larger than the supported one", FTI_EROR);
        return FTI_NSCS;
    }

    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Expands an existing hash meta data structure for data chunk
  @param      dataHash        The hash meta that need to be expanded.
  @param      chunkSize       The new Size of the data.
  @return     integer         FTI_SCES if successful.

  This function re-allocates memory for the 'dataDiffHash' inValid and blockSize member
  of the  data hash structure and if dCP 

CAUTION: This function does not reallocate the actual hashes of the data struct.
When the checkpoint will terminate the current hash will be freed, whereas the next 
will be used as current. Keep in mind that the next has the correct size
 **/
/*-------------------------------------------------------------------------*/
int FTI_ExpandBlockHashArray(FTIT_DataDiffHash* dataHash, int64_t chunkSize) {
    if (!dcpEnabled)
        return FTI_SCES;

    if (!(*dcpEnabled))
        return FTI_SCES;

    bool changeSize = true;
    int32_t nbHashesOld = dataHash->nbHashes;

    //
    int32_t newNumber =  FTI_CalcNumHashes(chunkSize);

    assert(nbHashesOld <= newNumber);
    if (newNumber == nbHashesOld) {
        changeSize = false;
    }

    dataHash->creationType = HASHREALLOC_INC;

    // reallocate hash array to new size if changed
    if (changeSize) {
        FTI_ReallocateDataDiff(dataHash, newNumber);
    }

    int hashIdx;
    dcpBLK_t diffBlockSize = FTI_GetDiffBlockSize();
    dataHash->nbHashes = newNumber;

    /* current last hash is invalid in any case. 
       If number of blocks remain the same, the size of the last block changed to 'new_size - old_size', 
       thus also the data that is contained in it. 
       If the nuber of blocks increased, the blocksize is changed for the current 
       last block as well, in fact to DCP_BLOCK_SIZE. */

    for (hashIdx = (nbHashesOld-1); hashIdx < newNumber-1; hashIdx++) {
        dataHash->isValid[hashIdx] = false;
        dataHash->blockSize[hashIdx] = diffBlockSize;
    }

    dataHash->blockSize[hashIdx] = chunkSize - (newNumber-1) * diffBlockSize;
    dataHash->isValid[hashIdx] = false;

    if (dataHash->blockSize[hashIdx] > diffBlockSize) {
        FTI_Print("FTI_ExpandBlockHashArray:: The blockSize computed is"
        " larger than the supported one", FTI_EROR);
        return FTI_NSCS;
    }
    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Computes number of hashblocks for chunk size.
  @param      chunkSize       chunk size of data chunk
  @return     int32_t         FTI_SCES if successful.

  This function computes the number of hash blocks according to the set dCP
  block size corresponding to chunkSize.
 **/
/*-------------------------------------------------------------------------*/
int32_t FTI_CalcNumHashes(int64_t chunkSize) {
    if ((chunkSize%((uint32_t)DCP_BLOCK_SIZE)) == 0) {
        return chunkSize/DCP_BLOCK_SIZE;
    } else {
        return chunkSize/DCP_BLOCK_SIZE + 1;
    }
}

void PrintDataHashInfo(FTIT_DataDiffHash* dataHash, int32_t chunkSize, int id) {
    char str[FTI_BUFS];
    FTI_Print("+++++++++++++++ INFO IS  +++++++++++++++", FTI_INFO);
    snprintf(str, sizeof(str), "I want to access index of the following id %d",
     id);
    FTI_Print(str, FTI_INFO);
    FTI_Print("+++++++++++++++ Data Hash INFO +++++++++++++++", FTI_INFO);
    snprintf(str, sizeof(str), "Num Hashes are %d", dataHash->nbHashes);
    FTI_Print(str, FTI_INFO);
    snprintf(str, sizeof(str), "Type of hash is %d", dataHash->creationType);
    FTI_Print(str, FTI_INFO);
    snprintf(str, sizeof(str), "Pointer of Hash is [%d] Lifetime %d",
     dataHash->currentId, dataHash->lifetime);
    FTI_Print(str, FTI_INFO);

    snprintf(str, sizeof(str), "Pointer of CURRENT Hash Table %p",
     dataHash->md5hash[CURRENT(dataHash)]);
    FTI_Print(str, FTI_INFO);

    snprintf(str, sizeof(str), "Pointer of NEXT Hash Table %p",
     dataHash->md5hash[NEXT(dataHash)]);
    FTI_Print(str, FTI_INFO);

    snprintf(str, sizeof(str), "Pointer of BLOCK SIZE Table %p",
     dataHash->blockSize);
    FTI_Print(str, FTI_INFO);

    snprintf(str, sizeof(str), "Pointer of ISVALIDE  Table %p",
     dataHash->isValid);
    FTI_Print(str, FTI_INFO);

    snprintf(str, sizeof(str), "Last Block Size is  %d",
     dataHash->blockSize[dataHash->nbHashes-1]);
    FTI_Print(str, FTI_INFO);
    snprintf(str, sizeof(str),
     "Total Block size is %d, Computed Block Size is %d", chunkSize,
      (dataHash->nbHashes-1)*FTI_GetDiffBlockSize() +
      dataHash->blockSize[dataHash->nbHashes-1]);
    FTI_Print(str, FTI_INFO);
}


/*-------------------------------------------------------------------------*/
/**
  @brief      Checks if data block is dirty, clean or invalid.
  @param      hashIdx         index for hash meta data in data chunk 
  meta data.
  @param      dbvar           Data chunk meta data.
  @return     integer         0 if data block is clean.
  @return     integer         1 if data block is dirty or invalid.
  @return     integer         -1 if hashIdx not in range.

  This function checks if data block corresponding to the hash meta data 
  element is clean, dirty or invalid.  

  It returns -1 if hashIdx is out of range.
 **/
/*-------------------------------------------------------------------------*/
int FTI_HashCmp(int32_t hashIdx, FTIFF_dbvar* dbvar, unsigned char *ptr) {
    bool clean = true;
    uint32_t bit32hashNow = 0;
    unsigned char *prevHash = NULL;
    unsigned char *nextHash = NULL;

    FTIT_DataDiffHash* hashes = dbvar->dataDiffHash;
    // unsigned char* ptr = (unsigned char*) dbvar->cptr +
    // hashIdx * DCP_BLOCK_SIZE;

    assert(!(hashIdx > hashes->nbHashes));


    if (hashIdx == hashes->nbHashes) {
        return -1;
    }

    // I Compute the hash code for the upcoming checkpoint On the Next status
    if (FTI_GetDcpMode() == FTI_DCP_MODE_MD5) {
        MD5(ptr, hashes->blockSize[hashIdx],
         &(hashes->md5hash[NEXT(hashes)][MD5_DIGEST_LENGTH * hashIdx]));
    } else {
#ifdef FTI_NOZLIB
        bit32hashNow = crc32(ptr, hashes->blockSize[hashIdx]);
#else
        bit32hashNow = crc32(0L, Z_NULL, 0);
        bit32hashNow = crc32(bit32hashNow, ptr, hashes->blockSize[hashIdx]);
#endif
        hashes->bit32hash[NEXT(hashes)][hashIdx] = bit32hashNow;
    }

    clean = 0;
    if (!(hashes->isValid[hashIdx])) {
        return 1;
    } else {
        switch (DCP_MODE) {
            case FTI_DCP_MODE_MD5:
                prevHash = &(hashes->md5hash[CURRENT(hashes)]
                    [MD5_DIGEST_LENGTH * hashIdx]);
                nextHash = &(hashes->md5hash[NEXT(hashes)][MD5_DIGEST_LENGTH *
                 hashIdx]);
                clean = memcmp(nextHash , prevHash , MD5_DIGEST_LENGTH) == 0;
                break;
            case FTI_DCP_MODE_CRC32:
                clean = (bit32hashNow == hashes->bit32hash[CURRENT(hashes)]
                    [hashIdx]);
                break;
        }
        // isValid is false, in the case in which I dont manage to update
        // the checkpoint, this memory region will be marked as invalid and
        // therefore it will be checkpointed on the next checkpoint.
        // set clean if unchanged
        if (clean) {
            return 0;
        } else {
            return 1;
        }
    }
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Updates data chunk hash meta data.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Data        Dataset metadata.
  @return     integer         FTI_SCES if successful.

  This function updates the hashes of data blocks that were identified as
  dirty and initializes the hashes for data blocks that are invalid.
 **/
/*-------------------------------------------------------------------------*/
int FTI_UpdateDcpChanges(FTIT_execution* FTI_Exec) {
    FTIFF_db *db = FTI_Exec->firstdb;
    FTIFF_dbvar *dbvar;
    int dbvar_idx, dbcounter = 0;
    int isnextdb;
    do {
        isnextdb = 0;
        for (dbvar_idx = 0; dbvar_idx < db->numvars; dbvar_idx++) {
            dbvar = &(db->dbvars[dbvar_idx]);
            FTIT_DataDiffHash* hashInfo = dbvar->dataDiffHash;
            if (dbvar->hascontent) {
                memset(hashInfo->isValid, true, hashInfo->nbHashes);
                // I need to free current hash table
                if (FTI_GetDcpMode() == FTI_DCP_MODE_MD5) {
                    if (hashInfo->md5hash[CURRENT(hashInfo)] != NULL) {
                        free(hashInfo->md5hash[CURRENT(hashInfo)]);
                        hashInfo->md5hash[CURRENT(hashInfo)] = NULL;
                    }
                } else {
                    if (hashInfo->bit32hash[CURRENT(hashInfo)] != NULL) {
                        free(hashInfo->bit32hash[CURRENT(hashInfo)]);
                        hashInfo->bit32hash[CURRENT(hashInfo)] = NULL;
                    }
                }
                hashInfo->currentId = (hashInfo->currentId +1)%2;
                hashInfo->lifetime++;
            }
        }
        if (db->next) {
            db = db->next;
            isnextdb = 1;
        }
        dbcounter++;
    } while (isnextdb);
    return FTI_SCES;
}




/*-------------------------------------------------------------------------*/
/**
  @brief      Returns pointer and size of buffer to write during checkpoint
  @param      buffer_addr        Pointer to buffer.
  @param      buffer_size        Size of buffer.
  @param      dbvar              Data chunk meta data.
  @param      FTI_Data           Dataset metadata.
  @return     integer            1 if buffer holds data to write.
  @return     integer            0 if nothing to write.

  This function is called repeatedly for each data chunk. If it returns 1,
  'buffer_addr' holds the pointer to a memory region inside the data chunk
  and 'buffer_size' holds the size of the region. 
  For dCP disabled, this region is the whole data chunk. For dCP enabled, 
  the function returns a pointer to contiguous dirty regions until no further 
  dirty regions are found in which case 0 is returned.
 **/
/*-------------------------------------------------------------------------*/
int FTI_ReceiveDataChunk(unsigned char** buffer_addr, size_t* buffer_size,
 FTIFF_dbvar* dbvar, unsigned char *startAddr, size_t *totalBytes) {
    static bool init = true;
    static bool reset;
    static int32_t hashIdx;
    unsigned char *ptr = startAddr;
    static int called = 0;

    if (init) {
        called = 0;
        hashIdx = 0;
        reset = false;
        init = false;
    }

    // reset function and return not found
    if (reset) {
        init = true;
        return 0;
    }

    called++;

    // if differential ckpt is disabled, return whole chunk and finalize call
    if (!dcpEnabled) {
        reset = true;
        *buffer_addr = ptr;
        *buffer_size = *totalBytes;
        *totalBytes = 0;
        return 1;
    }

    int maxNumHashes = hashIdx + ((*totalBytes)/DCP_BLOCK_SIZE) +
     (((*totalBytes) % DCP_BLOCK_SIZE) != 0);
    // advance *buffer_offset for clean regions
    unsigned char clean = 1;
    int cleanIdx = hashIdx;
    while (hashIdx < maxNumHashes && clean) {
        clean = FTI_HashCmp(hashIdx, dbvar, ptr) == 0;
        ptr += (clean) * (dbvar->dataDiffHash->blockSize[hashIdx]);
        (*totalBytes) -= (clean) * (dbvar->dataDiffHash->blockSize[hashIdx]);
        hashIdx += (clean) *1;
    }
    memset(&(dbvar->dataDiffHash->isValid[cleanIdx]), false,
     (hashIdx-cleanIdx));

    // check if region clean until end
    if (hashIdx == dbvar->dataDiffHash->nbHashes) {
        init = true;
        return 0;
    }
    // check if I have processed all fetched memory up to current byte;
    if (hashIdx == maxNumHashes) {
        return 0;
    }

    *buffer_addr = ptr;
    *buffer_size = 0;
    unsigned dirty = 1;
    int dirtyIdx = hashIdx;

    while (hashIdx < maxNumHashes && dirty) {
        dirty = FTI_HashCmp(hashIdx, dbvar, ptr);
        ptr += (dirty) * (dbvar->dataDiffHash->blockSize[hashIdx]);
        *buffer_size += (dirty) * (dbvar->dataDiffHash->blockSize[hashIdx]);
        (*totalBytes) -= (dirty) * (dbvar->dataDiffHash->blockSize[hashIdx]);
        hashIdx += (dirty) *  1;
    }

    memset(&(dbvar->dataDiffHash->isValid[dirtyIdx]), false,
     (hashIdx-dirtyIdx));

    // check if we are at the end of the data region
    if (hashIdx == dbvar->dataDiffHash->nbHashes) {
        if (*buffer_size != 0) {
            reset = true;
            return 1;
        } else {
            init = true;
            return 0;
        }
    }
    return 1;
}
