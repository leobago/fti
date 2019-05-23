#ifndef _TYPES_H_
#define _TYPES_H_

#include <mpi.h>
#ifdef GPUSUPPORT
#include <cuda_runtime_api.h>
#endif

#ifdef ENABLE_HDF5
#include "hdf5.h"
#include "hdf5_hl.h"
#endif
#include <stdio.h>

/*---------------------------------------------------------------------------
  ICP TYPES
  ---------------------------------------------------------------------------*/

#define MD5_DIGEST_LENGTH 16
#define MD5_DIGEST_STRING_LENGTH 33

#define FTI_ICP_NINI 0
#define FTI_ICP_ACTV 1
#define FTI_ICP_FAIL 2

#define FTI_GT(NUM1, NUM2) ((NUM1) > (NUM2)) ? NUM1 : NUM2
#define FTI_PO_FH FILE*
#define FTI_FF_FH int
#define FTI_MI_FH MPI_File
#ifdef ENABLE_SIONLIB // --> If SIONlib is installed
#   define FTI_SL_FH int
#endif
#ifdef ENABLE_HDF5 // --> If HDF5 is desired
#   define FTI_H5_FH hid_t
#endif

#if !defined (ENABLE_SIONLIB) && !defined (ENABLE_HDF5) 
#   define FTI_ICP_FH_SIZE FTI_GT( sizeof(FTI_PO_FH), FTI_GT( sizeof(FTI_FF_FH), sizeof(FTI_MI_FH) ) )
#elif !defined (ENABLE_SIONLIB) && defined (ENABLE_HDF5)
#   define FTI_ICP_FH_SIZE FTI_GT( \
        FTI_GT( sizeof(FTI_PO_FH), sizeof(FTI_H5_FH) ), \
        FTI_GT( sizeof(FTI_FF_FH), sizeof(FTI_MI_FH) ) )
#elif defined (ENABLE_SIONLIB) && !defined (ENABLE_HDF5)
#   define FTI_ICP_FH_SIZE FTI_GT( \
        FTI_GT( sizeof(FTI_PO_FH), sizeof(FTI_SL_FH) ), \
        FTI_GT( sizeof(FTI_FF_FH), sizeof(FTI_MI_FH) ) )
#elif defined (ENABLE_SIONLIB) && defined (ENABLE_HDF5)
#   define FTI_ICP_FH_SIZE FTI_GT( \
        FTI_GT( sizeof(FTI_PO_FH), sizeof(FTI_H5_FH) ), \
        FTI_GT( sizeof(FTI_FF_FH),\
        FTI_GT( sizeof(FTI_SL_FH), sizeof(FTI_MI_FH) ) ) )
#endif

/** @typedef    FTIT_iCPInfo
 *  @brief      Meta Information needed for iCP.
 *  
 *  The member fh is a generic file handle container large enough to hold any
 *  file handle type of I/O modes that are used within FTI.
 */
typedef struct FTIT_iCPInfo {
    bool isFirstCp;             /**< TRUE if first cp in run                */
    short status;               /**< holds status (active,failed) of iCP    */
    int  result;                /**< holds result of I/O specific write     */
    int lastCkptLvel;           /**< holds last successful cp level         */
    int lastCkptID;             /**< holds last successful cp ID            */
    int countVar;               /**< counts datasets written                */
    int isWritten[FTI_BUFS];    /**< holds IDs of datasets in cp file       */
    double t0;                  /**< timing for CP statistics               */
    double t1;                  /**< timing for CP statistics               */
    char fh[FTI_ICP_FH_SIZE];   /**< generic fh container                   */
    char fn[FTI_BUFS];          /**< Name of the checkpoint file            */
    unsigned long long offset;  /**< file offset (for MPI-IO only)          */
} FTIT_iCPInfo;

/*---------------------------------------------------------------------------
  FTIFF TYPES
  ---------------------------------------------------------------------------*/
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

/*---------------------------------------------------------------------------
  STAGE TYPES
  ---------------------------------------------------------------------------*/

//#include <fti-int/stage-types.h>
/** @typedef    FTIT_StageInfo
 *  @brief      Staging meta info.
 *  
 *  The request pointer is void in order to allow the structure to
 *  keep the head rank staging info if used by a head process or the
 *  application rank staging info otherwise. The cast is performed
 *  via the macros 'FTI_SI_HPTR( ptr )' for the head processes and
 *  'FTI_SI_APTR( ptr )' for the application processes.
 */
typedef struct FTIT_StageInfo {
    int nbRequest;  /**< Number of allocated request info structures        */
    void *request;  /**< pointer to request meta info array                 */
} FTIT_StageInfo;

/** @typedef    FTIT_StatusField
 *  @brief      valid fields of 'status'.
 * 
 *  enum that keeps the particular field identifiers for the 'status'
 *  field.
 */
typedef enum {
    FTI_SIF_AVL = 0,
    FTI_SIF_VAL,
} FTIT_StatusField;

/** @typedef    FTIT_RequestField
 *  @brief      valid fields of 'idxRequest'.
 * 
 *  enum that keeps the particular field identifiers for the
 *  'idxRequest' field.
 */
typedef enum {
    FTI_SIF_ALL = 0,
    FTI_SIF_IDX
} FTIT_RequestField;

/** @typedef    FTIT_StageHeadInfo
 *  @brief      Head rank staging meta info.
 */
typedef struct FTIT_StageHeadInfo {
    char lpath[FTI_BUFS];           /**< file path                      */
    char rpath[FTI_BUFS];           /**< file name                      */
    size_t offset;                  /**< current offset of file pointer */
    size_t size;                    /**< file size                      */
    int ID;                         /**< ID of request                  */
} FTIT_StageHeadInfo;

/** @typedef    FTIT_StageAppInfo
 *  @brief      Application rank staging meta info.
 */
typedef struct FTIT_StageAppInfo {
    void *sendBuf;                  /**< send buffer of MPI_Isend       */
    MPI_Request mpiReq;             /**< MPI_Request of MPI_Isend       */
    int ID;                         /**< ID of request                  */
} FTIT_StageAppInfo;

/*---------------------------------------------------------------------------
  HDF5 TYPES
  ---------------------------------------------------------------------------*/

/** @typedef    FTIT_complexType
 *  @brief      Type that consists of other FTI types
 *
 *  This type allows creating complex datatypes.
 */
typedef struct FTIT_H5Group {
    int                 id;                     /**< ID of the group.               */
    char                name[FTI_BUFS];         /**< Name of the group.             */
    int                 childrenNo;             /**< Number of children             */
    int                 childrenID[FTI_BUFS];   /**< IDs of the children groups     */
#ifdef ENABLE_HDF5
    hid_t               h5groupID;              /**< Group hid_t.                   */
#endif
} FTIT_H5Group;

typedef struct FTIT_globalDataset {
    bool                        initialized;    /**< Dataset is initialized         */
    int                         rank;           /**< Rank of dataset                */
    int                         id;             /**< ID of dataset.                 */
    int                         numSubSets;     /**< Number of assigned sub-sets    */
    int*                        varIdx;         /**< FTI_Data index of subset var   */
    FTIT_H5Group*               location;       /**< Dataset location in file.      */
#ifdef ENABLE_HDF5
    hid_t                       hid;            /**< HDF5 id datset.                */
    hid_t                       fileSpace;      /**< HDF5 id dataset filespace      */
    hid_t                       hdf5TypeId;     /**< HDF5 id of assigned FTI type   */
    hsize_t*                    dimension;      /**< num of elements for each dim.  */
#endif
    struct FTIT_globalDataset*  next;           /**< Pointer to next dataset        */
    struct FTIT_type*           type;           /**< corresponding FTI type.        */
    char                        name[FTI_BUFS]; /**< Dataset name.                  */
} FTIT_globalDataset;

typedef struct FTIT_sharedData {
    FTIT_globalDataset* dataset;                /**< Pointer to global dataset.     */
#ifdef ENABLE_HDF5
    hsize_t*            count;                  /**< num of elem in each dim.       */
    hsize_t*            offset;                 /**< coord origin of sub-set.       */
#endif
} FTIT_sharedData;

/** @typedef    FTIT_typeField
 *  @brief      Holds info about field in complex type
 *
 *  This type simplify creating complex datatypes.
 */
typedef struct FTIT_typeField {
    int                 typeID;                 /**< FTI type ID of the field.          */
    int                 offset;                 /**< Offset of the field in structure.  */
    int                 rank;                   /**< Field rank (max. 32)               */
    int                 dimLength[32];          /**< Lenght of each dimention           */
    char                name[FTI_BUFS];         /**< Name of the field                  */
} FTIT_typeField;

/** @typedef    FTIT_complexType
 *  @brief      Type that consists of other FTI types
 *
 *  This type allows creating complex datatypes.
 */
typedef struct FTIT_complexType {
    char                name[FTI_BUFS];         /**< Name of the complex type.          */
    int                 length;                 /**< Number of types in complex type.   */
    FTIT_typeField      field[FTI_BUFS];        /**< Fields of the complex type.        */
} FTIT_complexType;

/*---------------------------------------------------------------------------
  KERNEL TYPES
  ---------------------------------------------------------------------------*/

/** @typedef    FTIT_level
 *  @brief      holds the level id.
 */
typedef enum {
    FTI_L1 = 1,
    FTI_L2,
    FTI_L3,
    FTI_L4,
    FTI_L1_DCP,
    FTI_L2_DCP,
    FTI_L3_DCP,
    FTI_L4_DCP,
    FTI_L4_H5_SINGLE,
    FTI_MIN_LEVEL_ID = FTI_L1,
    FTI_MAX_LEVEL_ID = FTI_L4_H5_SINGLE
} FTIT_level;

/** @typedef    FTIT_double
 *  @brief      Double mapped as two integers to allow bit-wise operations.
 *
 *  Double mapped as integer and byte array to allow bit-wise operators so
 *  that we can inject failures on it.
 */
typedef union FTIT_double {
    double          value;              /**< Double floating point value.   */
    float           floatval[2];        /**< Float mapped to do bit edits.  */
    int             intval[2];          /**< Integer mapped to do bit edits.*/
    char            byte[8];            /**< Byte array for coarser control.*/
} FTIT_double;

/** @typedef    FTIT_float
 *  @brief      Float mapped as integer to allow bit-wise operations.
 *
 *  Float mapped as integer and byte array to allow bit-wise operators so
 *  that we can inject failures on it.
 */
typedef union FTIT_float {
    float           value;              /**< Floating point value.          */
    int             intval;             /**< Integer mapped to do bit edits.*/
    char            byte[4];            /**< Byte array for coarser control.*/
} FTIT_float;
  
/** @typedef    FTIT_type
 *  @brief      Type recognized by FTI.
 *
 *  This type allows handling data structures.
 */
typedef struct FTIT_type {
    int                 id;                     /**< ID of the data type.           */
    int                 size;                   /**< Size of the data type.         */
    FTIT_complexType*   structure;              /**< Logical structure for HDF5.    */
    FTIT_H5Group*       h5group;                /**< Group of this datatype.        */
#ifdef ENABLE_HDF5
    hid_t               h5datatype;             /**< HDF5 datatype.                 */
#endif
} FTIT_type;

/** @typedef    FTIT_dataset
 *  @brief      Dataset metadata.
 *
 *  This type stores the metadata related with a dataset.
 */
typedef struct FTIT_dataset {
    int                 id;                 /**< ID to search/update dataset.                   */
    void                *ptr;               /**< Pointer to the dataset.                        */
    long                count;              /**< Number of elements in dataset.                 */
    struct FTIT_type*   type;               /**< Data type for the dataset.                     */
    int                 eleSize;            /**< Element size for the dataset.                  */
    long                size;               /**< Total size of the dataset.                     */
    int                 rank;               /**< Rank of dataset (for HDF5).                    */
    int                 dimLength[32];      /**< Lenght of each dimention.                      */
    char                name[FTI_BUFS];     /**< Name of the dataset.                           */
    FTIT_H5Group*       h5group;            /**< Group of this dataset                          */
    bool                isDevicePtr;        /**< True if this data are stored in a device memory*/
    void                *devicePtr;         /**< Pointer to data in the device                  */
    FTIT_sharedData     sharedData;         /**< Info if dataset is sub-set (VPR)               */
} FTIT_dataset;

/** @typedef    FTIT_metadata
 *  @brief      Metadata for restart.
 *
 *  This type stores all the metadata necessary for the restart.
 */
typedef struct FTIT_metadata {
    int*             exists;             /**< TRUE if metadata exists               */
    long*            maxFs;              /**< Maximum file size.                    */
    long*            fs;                 /**< File size.                            */
    long*            pfs;                /**< Partner file size.                    */
    char*            ckptFile;           /**< Ckpt file name. [FTI_BUFS]            */
    char*            currentL4CkptFile;  /**< Current Ckpt file name. [FTI_BUFS]    */        
    int*             nbVar;              /**< Number of variables. [FTI_BUFS]       */
    int*             varID;              /**< Variable id for size.[FTI_BUFS]       */
    long*            varSize;            /**< Variable size. [FTI_BUFS]             */
} FTIT_metadata;

/** @typedef    FTIT_execution
 *  @brief      Execution metadata.
 *
 *  This type stores all the dynamic metadata related to the current execution
 */
typedef struct FTIT_execution {
    char            id[FTI_BUFS];       /**< Execution ID.                  */
    int             ckpt;               /**< Checkpoint flag.               */
    int             reco;               /**< Recovery flag.                 */
    int             ckptLvel;           /**< Checkpoint level.              */
    int             ckptIntv;           /**< Ckpt. interval in minutes.     */
    int             lastCkptLvel;       /**< Last checkpoint level.         */
    int             wasLastOffline;     /**< TRUE if last ckpt. offline.    */
    double          iterTime;           /**< Current wall time.             */
    double          lastIterTime;       /**< Time spent in the last iter.   */
    double          meanIterTime;       /**< Mean iteration time.           */
    double          globMeanIter;       /**< Global mean iteration time.    */
    double          totalIterTime;      /**< Total main loop time spent.    */
    unsigned int    syncIter;           /**< To check mean iter. time.      */
    int             syncIterMax;        /**< Maximal synch. intervall.      */
    unsigned int    minuteCnt;          /**< Checkpoint minute counter.     */
    bool            hasCkpt;            /**< Indicator that ckpt exists     */
    bool            h5SingleFile;       /**< Indicator if HDF5 single file  */
    unsigned int    ckptCnt;            /**< Checkpoint number counter.     */
    unsigned int    ckptIcnt;           /**< Iteration loop counter.        */
    unsigned int    ckptID;             /**< Checkpoint ID.                 */
    unsigned int    ckptNext;           /**< Iteration for next checkpoint. */
    unsigned int    ckptLast;           /**< Iteration for last checkpoint. */
    long            ckptSize;           /**< Checkpoint size.               */
    unsigned int    nbVar;              /**< Number of protected variables. */
    unsigned int    nbVarStored;        /**< Nr. prot. var. stored in file  */
    unsigned int    nbType;             /**< Number of data types.          */
    int             nbGroup;            /**< Number of protected groups.    */
    int             metaAlloc;          /**< TRUE if meta allocated.        */
    int             initSCES;           /**< TRUE if FTI initialized.       */
    char    h5SingleFileLast[FTI_BUFS]; /**< Last HDF5 single file name     */
    FTIT_metadata   meta[5];            /**< Metadata for each ckpt level   */
    FTIFF_db         *firstdb;          /**< Pointer to first datablock     */
    FTIFF_db         *lastdb;           /**< Pointer to first datablock     */
    FTIFF_metaInfo  FTIFFMeta;          /**< File meta data for FTI-FF      */
    FTIT_type**     FTI_Type;           /**< Pointer to FTI_Types           */
    FTIT_H5Group**  H5groups;           /**< HDF5 root group.               */
    FTIT_globalDataset* globalDatasets; /**< Pointer to first global dataset*/
    FTIT_StageInfo* stageInfo;          /**< root of staging requests       */
    FTIT_iCPInfo    iCPInfo;            /**< meta info iCP                  */
    MPI_Comm        globalComm;         /**< Global communicator.           */
    MPI_Comm        groupComm;          /**< Group communicator.            */
    MPI_Comm        nodeComm;
} FTIT_execution;

/** @typedef    FTIT_configuration
 *  @brief      Configuration metadata.
 *
 *  This type stores the general configuration metadata.
 */
typedef struct FTIT_configuration {
    bool            stagingEnabled;
    bool            dcpEnabled;         /**< Enable differential ckpt.      */
    bool            keepL4Ckpt;         /**< TRUE if l4 ckpts to keep       */        
    bool            keepHeadsAlive;     /**< TRUE if heads return           */
    int             dcpMode;            /**< dCP mode.                      */
    int             dcpBlockSize;       /**< Block size for dCP hash        */
    char            cfgFile[FTI_BUFS];  /**< Configuration file name.       */
    int             saveLastCkpt;       /**< TRUE to save last checkpoint.  */
    int             verbosity;          /**< Verbosity level.               */
    int             blockSize;          /**< Communication block size.      */
    int             transferSize;       /**< Transfer size local to PFS     */
#ifdef LUSTRE
    int             stripeUnit;         /**< Striping Unit for Lustre FS    */
    int             stripeOffset;       /**< Striping Offset for Lustre FS  */
    int             stripeFactor;       /**< Striping Factor for Lustre FS  */
#endif
    int             ckptTag;            /**< MPI tag for ckpt requests.         */
    int             stageTag;           /**< MPI tag for staging comm.          */
    int             finalTag;           /**< MPI tag for finalize comm.         */
    int             generalTag;         /**< MPI tag for general comm.          */
    int             test;               /**< TRUE if local test.                */
    int             l3WordSize;         /**< RS encoding word size.             */
    int             ioMode;             /**< IO mode for L4 ckpt.               */
    bool            h5SingleFileEnable; /**< TRUE if VPR enabled                */
    bool            h5SingleFileKeep;   /**< TRUE if VPR files to keep          */
    char            h5SingleFileDir[FTI_BUFS]; /**< HDF5 single file dir        */
    char            h5SingleFilePrefix[FTI_BUFS]; /**< HDF5 single file prefix  */
    char            stageDir[FTI_BUFS]; /**< Staging directory.                 */
    char            localDir[FTI_BUFS]; /**< Local directory.                   */
    char            glbalDir[FTI_BUFS]; /**< Global directory.                  */
    char            metadDir[FTI_BUFS]; /**< Metadata directory.                */
    char            lTmpDir[FTI_BUFS];  /**< Local temporary directory.         */
    char            gTmpDir[FTI_BUFS];  /**< Global temporary directory.        */
    char            mTmpDir[FTI_BUFS];  /**< Metadata temporary directory.      */
    size_t          cHostBufSize;       /**< Host buffer size for GPU data. */

} FTIT_configuration;

/** @typedef    FTIT_topology
 *  @brief      Topology metadata.
 *
 *  This type stores the topology metadata.
 */
typedef struct FTIT_topology {
    int             nbProc;             /**< Total global number of proc.   */
    int             nbNodes;            /**< Total global number of nodes.  */
    int             myRank;             /**< My rank on the global comm.    */
    int             splitRank;          /**< My rank on the FTI comm.       */
    int             nodeSize;           /**< Total number of pro. per node. */
    int             nbHeads;            /**< Number of FTI proc. per node.  */
    int             nbApprocs;          /**< Number of app. proc. per node. */
    int             groupSize;          /**< Group size for L2 and L3.      */
    int             sectorID;           /**< Sector ID in the system.       */
    int             nodeID;             /**< Node ID in the system.         */
    int             groupID;            /**< Group ID in the node.          */
    int             amIaHead;           /**< TRUE if FTI process.           */
    int             headRank;           /**< Rank of the head in this node. */
    int             headRankNode;       /**< Rank of the head in node comm. */
    int             nodeRank;           /**< Rank of the node.              */
    int             groupRank;          /**< My rank in the group comm.     */
    int             right;              /**< Proc. on the right of the ring.*/
    int             left;               /**< Proc. on the left of the ring. */
    int             body[FTI_BUFS];     /**< List of app. proc. in the node.*/
} FTIT_topology;


/** @typedef    FTIT_checkpoint
 *  @brief      Checkpoint metadata.
 *
 *  This type stores all the checkpoint metadata.
 */
typedef struct FTIT_checkpoint {
    char            dir[FTI_BUFS];      /**< Checkpoint directory.                  */
    char            dcpDir[FTI_BUFS];   /**< dCP directory.                         */
    char            archDir[FTI_BUFS];  /**< Checkpoint directory.                  */        
    char            metaDir[FTI_BUFS];  /**< Metadata directory.                    */
    char            dcpName[FTI_BUFS];  /**< dCP file name.                         */
    bool            isDcp;              /**< TRUE if dCP requested                  */
    bool            hasDcp;             /**< TRUE if execution has already a dCP    */
    bool            hasCkpt;            /**< TRUE if level has ckpt                 */        
    int             isInline;           /**< TRUE if work is inline.                */
    int             ckptIntv;           /**< Checkpoint interval.                   */
    int             ckptCnt;            /**< Checkpoint counter.                    */
    int             ckptDcpIntv;        /**< Checkpoint interval.                   */
    int             ckptDcpCnt;         /**< Checkpoint counter.                    */

} FTIT_checkpoint;

/** @typedef    FTIT_injection
 *  @brief      Type to describe failure injections in FTI.
 *
 *  This type allows users to describe a SDC failure injection model.
 */
typedef struct FTIT_injection {
    int             rank;               /**< Rank of proc. that injects     */
    int             index;              /**< Array index of the bit-flip.   */
    int             position;           /**< Bit position of the bit-flip.  */
    int             number;             /**< Number of bit-flips to inject. */
    int             frequency;          /**< Injection frequency (in min.)  */
    int             counter;            /**< Injection counter.             */
    double          timer;              /**< Timer to measure frequency     */
} FTIT_injection;

extern MPI_Comm FTI_COMM_WORLD;

/** FTI data type for chars.                                               */
extern FTIT_type FTI_CHAR;
/** FTI data type for short integers.                                      */
extern FTIT_type FTI_SHRT;
/** FTI data type for integers.                                            */
extern FTIT_type FTI_INTG;
/** FTI data type for long integers.                                       */
extern FTIT_type FTI_LONG;
/** FTI data type for unsigned chars.                                      */
extern FTIT_type FTI_UCHR;
/** FTI data type for unsigned short integers.                             */
extern FTIT_type FTI_USHT;
/** FTI data type for unsigned integers.                                   */
extern FTIT_type FTI_UINT;
/** FTI data type for unsigned long integers.                              */
extern FTIT_type FTI_ULNG;
/** FTI data type for single floating point.                               */
extern FTIT_type FTI_SFLT;
/** FTI data type for double floating point.                               */
extern FTIT_type FTI_DBLE;
/** FTI data type for long doble floating point.                           */
extern FTIT_type FTI_LDBE;

typedef struct
{
  FTIT_configuration* FTI_Conf;
  MPI_File pfh;
  MPI_Offset offset;
  int err;
} WriteMPIInfo_t;

#endif // _TYPES_H_
