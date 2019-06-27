#ifndef __FTI_TYPES_H__
#define __FTI_TYPES_H__

#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <errno.h>
#include <stdint.h>
#include <unistd.h>
#include <stdbool.h>
#include <mpi.h>

#ifdef ENABLE_HDF5 // --> If HDF5 is installed
#include "hdf5.h"
#else
typedef size_t 	hsize_t;
#endif

#ifdef GPUSUPPORT
#include <cuda_runtime_api.h>
#endif

#ifdef ENABLE_HDF5 // --> If HDF5 is installed
#include "hdf5.h"
#else
typedef size_t 	hsize_t;
#endif

/** Malloc macro.                                                          */
#define talloc(type, num) (type *)malloc(sizeof(type) * (num))

typedef uintptr_t           FTI_ADDRVAL;        /**< for ptr manipulation       */
typedef void*               FTI_ADDRPTR;        /**< void ptr type              */ 

#define LOCAL 0
#define GLOBAL 1

#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

#define DBG_MSG(MSG,RANK,...) do { \
    int rank; \
    MPI_Comm_rank(MPI_COMM_WORLD,&rank); \
    if ( rank == RANK ) \
        printf( "%s:%d[DEBUG-%d] " MSG "\n", __FILENAME__,__LINE__,rank, ##__VA_ARGS__); \
    if ( RANK == -1 ) \
        printf( "%s:%d[DEBUG-%d] " MSG "\n", __FILENAME__,__LINE__,rank, ##__VA_ARGS__); \
} while (0)

/** MD5-hash: unsigned char digest length.                                 */
#define MD5_DIGEST_LENGTH 16
/** MD5-hash: hex converted char digest length.                            */
#define MD5_DIGEST_STRING_LENGTH 33

/** Define RED color for FTI output.                                       */
#define FTI_COLOR_RED   "\x1B[31m"
/** Define ORANGE color for FTI output.                                    */
#define FTI_COLOR_ORG   "\x1B[38;5;202m"
/** Define GREEN color for FTI output.                                     */
#define FTI_COLOR_GRN   "\x1B[32m"
/** Define BLUE color for FTI output.                                       */
#define FTI_COLOR_BLU   "\x1B[34m"
/** Define color RESET for FTI output.                                     */
#define FTI_COLOR_RESET "\x1B[0m"

/** Verbosity level to print only errors.                                  */
#define FTI_EROR 4
/** Verbosity level to print only warning and errors.                      */
#define FTI_WARN 3
/** Verbosity level to print main information.                             */
#define FTI_IDCP 5
/** Verbosity level to print debug messages.                               */
#define FTI_INFO 2
/** Verbosity level to print debug messages.                               */
#define FTI_DBUG 1

/** Token for checkpoint Baseline.                                         */
#define FTI_BASE 990
/** Token for checkpoint Level 1.                                          */
#define FTI_CKTW 991
/** Token for checkpoint Level 2.                                          */
#define FTI_XORW 992
/** Token for checkpoint Level 3.                                          */
#define FTI_RSEW 993
/** Token for checkpoint Level 4.                                          */
#define FTI_PFSW 994
/** Token for end of the execution.                                        */
#define FTI_ENDW 995
/** Token to reject checkpoint.                                            */
#define FTI_REJW 996
/** Token for IO mode Posix.                                               */
#define FTI_IO_POSIX 1001
/** Token for IO mode MPI.                                                 */
#define FTI_IO_MPI 1002
/** Token for IO mode FTI-FF.                                              */
#define FTI_IO_FTIFF 1003
/** Token for IO mode HDF5.                                         */
#define FTI_IO_HDF5 1005
#ifdef ENABLE_SIONLIB // --> If SIONlib is installed
/** Token for IO mode SIONlib.                                             */
#define FTI_IO_SIONLIB 1004
#endif

#define MAX_STACK_SIZE 10

#ifdef __cplusplus
extern "C" {
#endif

    typedef struct FTIT_datasetInfo {
        int varID;
        unsigned long varSize;
    } FTIT_datasetInfo;

    typedef struct FTIT_dcpConfigurationPosix 
    {
        unsigned int digestWidth;
        unsigned char* (*hashFunc)( const unsigned char *data, unsigned long nBytes, unsigned char *hash );
        unsigned int StackSize;
        unsigned long BlockSize;

    } FTIT_dcpConfigurationPosix;

    typedef struct FTIT_dcpExecutionPosix
    {
        int nbLayerReco;
        int nbVarReco;
        unsigned int Counter;
        unsigned long FileSize;
        unsigned long dataSize;
        unsigned long dcpSize;
        unsigned long LayerSize[MAX_STACK_SIZE];
        FTIT_datasetInfo datasetInfo[MAX_STACK_SIZE][FTI_BUFS];
        char LayerHash[MAX_STACK_SIZE*MD5_DIGEST_STRING_LENGTH];

    } FTIT_dcpExecutionPosix;

    typedef struct FTIT_dcpDatasetPosix
    {
        unsigned long hashDataSize;
        unsigned char* hashArray;
        unsigned char* hashArrayTmp;

    } FTIT_dcpDatasetPosix;

    typedef struct blockMetaInfo_t
    {
        unsigned long varId : 18;
        unsigned long blockId : 30;

    } blockMetaInfo_t;

    /*---------------------------------------------------------------------------
      FTI-FF types
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
        int lastCkptID;             /**< holds last successful cp ID            */
        int countVar;               /**< counts datasets written                */
        int isWritten[FTI_BUFS];    /**< holds IDs of datasets in cp file       */
        double t0;                  /**< timing for CP statistics               */
        double t1;                  /**< timing for CP statistics               */
        char fn[FTI_BUFS];          /**< Name of the checkpoint file            */
        void *fd;  
    } FTIT_iCPInfo;

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
        int ckptID;     /**< Checkpoint ID                                      */
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

    /*---------------------------------------------------------------------------
      New types
      ---------------------------------------------------------------------------*/

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

    /** @typedef    FTIT_complexType
     *  @brief      Type that consists of other FTI types
     *
     *  This type allows creating complex datatypes.
     */
    typedef struct FTIT_complexType FTIT_complexType;

    typedef struct FTIT_H5Group FTIT_H5Group;

    typedef struct FTIT_H5Group {
        int                 id;                     /**< ID of the group.               */
        char                name[FTI_BUFS];         /**< Name of the group.             */
        char                fullName[FTI_BUFS];     /**< Holds full 'path' of group     */
        int                 childrenNo;             /**< Number of children             */
        int                 childrenID[FTI_BUFS];   /**< IDs of the children groups     */
#ifdef ENABLE_HDF5
        hid_t               h5groupID;              /**< Group hid_t.                   */
#endif
    } FTIT_H5Group;

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
        FTIT_type                   type;           /**< corresponding FTI type.        */
        char                        name[FTI_BUFS]; /**< Dataset name.                  */
        char                        fullName[FTI_BUFS]; /**< full 'path' of dataset     */
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

    /** @typedef    FTIT_dataset
     *  @brief      Dataset metadata.
     *
     *  This type stores the metadata related with a dataset.
     */
    typedef struct FTIT_dataset {
        int                 id;                 /**< ID to search/update dataset.                   */
        void                *ptr;               /**< Pointer to the dataset.                        */
        long                count;              /**< Number of elements in dataset.                 */
        FTIT_type*          type;               /**< Data type for the dataset.                     */
        int                 eleSize;            /**< Element size for the dataset.                  */
        long                size;               /**< Total size of the dataset.                     */
        int                 rank;               /**< Rank of dataset (for HDF5).                    */
        int                 dimLength[32];      /**< Lenght of each dimention.                      */
        char                name[FTI_BUFS];     /**< Name of the dataset.                           */
        FTIT_H5Group*       h5group;            /**< Group of this dataset                          */
        bool                isDevicePtr;        /**< True if this data are stored in a device memory*/
        void                *devicePtr;         /**< Pointer to data in the device                  */
        FTIT_sharedData     sharedData;         /**< Info if dataset is sub-set (VPR)               */
        FTIT_dcpDatasetPosix dcpInfoPosix;      /**< dCP info for posix I/O                         */
        size_t				filePos; 
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
        long*            filePos;            /**< File Postion of each variable			*/
    } FTIT_metadata;

    /** @typedef    FTIT_configuration
     *  @brief      Configuration metadata.
     *
     *  This type stores the general configuration metadata.
     */
    typedef struct FTIT_configuration {
        bool            stagingEnabled;
        bool            dcpFtiff;         /**< Enable differential ckpt.      */
        bool            dcpPosix;         /**< Enable differential ckpt.      */
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
        char 	 		suffix[4];			/** Suffix of the checkpoint files		*/
        FTIT_dcpConfigurationPosix dcpInfoPosix;      /**< dCP info for posix I/O   */


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
        bool            recoIsDcp;          /**< TRUE if dCP requested                  */
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

    typedef struct FTIT_execution FTIT_execution;


    /** @typedef    FTIT_IO 
     *  @brief      Pointer to functions which are used to write the checkpoint file.
     *
     * This is a general description of what different file formats need to implement in 
     * order to checkpoint.
     */

    typedef struct ftit_io{
        void*  (*initCKPT) (	FTIT_configuration* , 
                FTIT_execution*  ,
                FTIT_topology*   ,
                FTIT_checkpoint* , 
                FTIT_dataset *);

        int (*WriteData) (	FTIT_dataset * ,
                void *write_info);
        int (*finCKPT)	(void *fileDesc);
        size_t (*getPos) (void *fileDesc);
        void (*finIntegrity) (unsigned char *, void*);


    }FTIT_IO;


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
        char    h5SingleFileReco[FTI_BUFS]; /**< HDF5 single fn from recovery   */
        unsigned char 	integrity[MD5_DIGEST_LENGTH];
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
        FTIT_dcpExecutionPosix dcpInfoPosix;      /**< dCP info for posix I/O   */
        int (*ckptFunc[2]) 					/** A function pointer pointing to  */									
            (FTIT_configuration* , 		/** the function which actually 	*/
             struct FTIT_execution* ,	/** the checkpoint file. Noticeably	*/ 
             FTIT_topology* ,			/** We need 2 function pointers,	*/ 
             FTIT_checkpoint* , 			/** One for the Level 4 checkpoint  */
             FTIT_dataset*,				/** And one for the remaining cases	*/
             FTIT_IO *);					

        int (*initICPFunc[2]) 				/** A function pointer pointing to  */									
            (FTIT_configuration* , 		/** the function which actually 	*/
             struct FTIT_execution* ,	/** initializes the iCP. Noticeably	*/ 
             FTIT_topology* ,			/** We need 2 function pointers,	*/ 
             FTIT_checkpoint* , 			/** One for the Level 4 checkpoint  */
             FTIT_dataset*,				/** And one for the remaining cases	*/
             FTIT_IO *);					

        int (*writeVarICPFunc[2]) 		    /** A function pointer pointing to  */
            (int,						/**									*/
             FTIT_configuration* , 		/** the function which actually 	*/
             struct FTIT_execution* ,	/** writes the iCP. Noticeably		*/ 
             FTIT_topology* ,			/** We need 2 function pointers,	*/ 
             FTIT_checkpoint* , 			/** One for the Level 4 checkpoint  */
             FTIT_dataset*,				/** And one for the remaining cases	*/
             FTIT_IO*);					

        int (*finalizeICPFunc[2]) 			/** A function pointer pointing to  */									
            (FTIT_configuration* , 		/** the function which actually 	*/
             struct FTIT_execution* ,	/** finalize the iCP. Noticeably	*/ 
             FTIT_topology* ,			/** We need 2 function pointers,	*/ 
             FTIT_checkpoint* , 			/** One for the Level 4 checkpoint  */
             FTIT_dataset*,				/** And one for the remaining cases	*/
             FTIT_IO *);					
        
        int (*activateHeads) 			/** A function pointer pointing to  */									
            (FTIT_configuration* , 		/** the function which actually 	*/
             struct FTIT_execution* ,	/** finalize the iCP. Noticeably	*/ 
             FTIT_topology* ,			/** We need 2 function pointers,	*/ 
             FTIT_checkpoint* , 			/** One for the Level 4 checkpoint  */
             int ); 			/** One for the Level 4 checkpoint  */


    } FTIT_execution;

#ifdef __cplusplus
}
#endif

#endif // __FTI_TYPES_H__
