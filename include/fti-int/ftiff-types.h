#ifndef FTIFF_H
#define FTIFF_H

#include <fti-int/defs.h>

/** @typedef    FTIT_DataDiffHash
 *  @brief      dCP information about data block.
 *  
 *  Holds information for each data block relevant for the dCP mechanism.
 *  This structure is a member of FTIFF_dbvar. It is stored as an array
 *  with n elements, where n corresponds to the number of data blocks in 
 *  that the data chunk is partitioned (depending on the dCP block size).
 */
typedef struct              FTIT_DataDiffHash
{
    unsigned char*          md5hash[2];    /**< MD5 digest                       */
    uint32_t*               bit32hash[2];  /**< CRC32 digest                     */
    unsigned short*         blockSize;  /**< data block size                  */
    bool*                   isValid;    /**< indicates if data block is valid */
    long                    nbHashes;     /**< holds the number of hashes for the data chunk                    */ 
    int                     currentId;
    int                     creationType;
    int                     lifetime;
}FTIT_DataDiffHash;

/** @typedef    FTIFF_metaInfo
 *  @brief      Meta Information about file.
 *
 *  (For FTI-FF only)
 *  Keeps information about the file. 'checksum' is the hash of the file
 *  excluding the file meta data. 'myHash' is the hash of the file meta data.
 *
 */
typedef struct FTIFF_metaInfo {
    char checksum[MD5_DIGEST_STRING_LENGTH]; /**< hash of file without meta */
    unsigned char myHash[MD5_DIGEST_LENGTH]; /**< hash of this struct       */
    long metaSize;  /**< size of ckpt data                                  */
    long ckptSize;  /**< also file size TODO remove                         */
    long dataSize;  /**< total size of protected data (excluding meta data) */
    long pureDataSize;  /**< total size of protected data (excluding meta data) */
    long fs;        /**< file size                                          */
    long maxFs;     /**< maximum file size in group                         */
    long ptFs;      /**< partner copy file size                             */
    long timestamp; /**< time when ckpt was created in ns (CLOCK_REALTIME)  */
    long dcpSize;   /**< how much actually written by rank                  */
} FTIFF_metaInfo;

/** @typedef    FTIFF_dbvar
 *  @brief      Information about protected variable in datablock.
 *
 *  (For FTI-FF only)
 *  Keeps information about the chunk of the protected variable with id
 *  stored in the current datablock. 'idx' is the index for the array
 *  element of 'FTIT_dataset* FTI_Data', that contains variable with 'id'.
 *
 */
typedef struct FTIFF_dbvar {
    int id;             /**< id of protected variable                         */
    int idx;            /**< index to corresponding id in pvar array          */
    int containerid;    /**< container index (first container -> 0)           */
    bool hascontent;    /**< indicates if container holds ckpt data           */
    bool hasCkpt;       /**< indicates if container is stored in ckpt         */
    uintptr_t dptr;     /**< data pointer offset				              */
    uintptr_t fptr;     /**< file pointer offset                              */
    long chunksize;     /**< chunk size stored aof prot. var. in this block   */
    long containersize; /**< chunk size stored aof prot. var. in this block   */
    unsigned char hash[MD5_DIGEST_LENGTH];  /**< hash of variable chunk       */
    unsigned char myhash[MD5_DIGEST_LENGTH];  /**< hash of this structure     */
    bool update;        /**< TRUE if struct needs to be updated in ckpt file  */
    FTIT_DataDiffHash* dataDiffHash; /**< dCP meta data for data chunk        */
    char *cptr;         /**< pointer to memory address of container origin    */
} FTIFF_dbvar;

/** @typedef    FTIFF_db
 *  @brief      Information about current datablock.
 *
 *  (For FTI-FF only)
 *  Keeps information about the current datablock in file
 *
 */

typedef struct FTIFF_db {
    int numvars;            /**< number of protected variables in datablock   */
    long dbsize;            /**< size of metadata + data for block in bytes   */
    unsigned char myhash[MD5_DIGEST_LENGTH];  /**< hash of variable chunk     */
    bool update;        /**< TRUE if struct needs to be updated in ckpt file  */
    bool finalized;        /**< TRUE if block is stored in cp file            */
    FTIFF_dbvar *dbvars;    /**< pointer to related dbvar array               */
    struct FTIFF_db *previous;  /**< link to previous datablock               */
    struct FTIFF_db *next;      /**< link to next datablock                   */
} FTIFF_db;

/** @typedef    dcpBLK_t
 *  @brief      unsigned short (0 - 65535).
 *  
 *  Type that keeps the block sizes inside the hash meta data. 
 *  unsigned short is a trade off between memory occupation and block 
 *  size range.
 */
typedef unsigned short dcpBLK_t;

/** @typedef    FTIFF_headInfo
 *  @brief      Runtime meta info for the heads.
 *
 *  keeps all necessary meta data information for the heads, in order to
 *  perform the checkpoint.
 *
 */
typedef struct FTIFF_headInfo {
    int exists;
    int nbVar;
    char ckptFile[FTI_BUFS];
    long maxFs;
    long fs;
    long pfs;
    int isDcp;
} FTIFF_headInfo;

/** @typedef    FTIFF_L2Info
 *  @brief      Meta data for L2 recovery.
 *
 *  keeps meta data information that needs to be exchanged between the ranks.
 *
 */
typedef struct FTIFF_L2Info {
    int FileExists;
    int CopyExists;
    int ckptID;
    int rightIdx;
    long fs;
    long pfs;
} FTIFF_L2Info;

/** @typedef    FTIFF_L3Info
 *  @brief      Meta data for L3 recovery.
 *
 *  keeps meta data information that needs to be exchanged between the ranks.
 *
 */
typedef struct FTIFF_L3Info {
    int FileExists;
    int RSFileExists;
    int ckptID;
    long fs;
    long RSfs;  // maxFs
} FTIFF_L3Info;

/**

  +-------------------------------------------------------------------------+
  |   MPI DERIVED DATA TYPES                                                |
  +-------------------------------------------------------------------------+

 **/

// ID MPI types
enum {
    FTIFF_HEAD_INFO,
    FTIFF_L2_INFO,
    FTIFF_L3_INFO,
    FTIFF_NUM_MPI_TYPES
};

// declare MPI datatypes
extern MPI_Datatype FTIFF_MpiTypes[FTIFF_NUM_MPI_TYPES];

typedef struct FTIFF_MPITypeInfo {
    MPI_Datatype        raw;
    int                 mbrCnt;
    int*                mbrBlkLen;
    MPI_Datatype*       mbrTypes;
    MPI_Aint*           mbrDisp;
} FTIFF_MPITypeInfo;

#endif // FTIFF_H
