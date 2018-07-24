/**
 *  @file   fti.h
 *  @author Leonardo A. Bautista Gomez (leobago@gmail.com)
 *  @date   July, 2013
 *  @brief  Header file for the FTI library.
 */

#ifndef _FTI_H
#define _FTI_H

#include <mpi.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

/*---------------------------------------------------------------------------
  Defines
  ---------------------------------------------------------------------------*/

/** Define RED color for FTI output.                                       */
#define RED   "\x1B[31m"
/** Define ORANGE color for FTI output.                                    */
#define ORG   "\x1B[38;5;202m"
/** Define GREEN color for FTI output.                                     */
#define GRN   "\x1B[32m"
/** Define color RESET for FTI output.                                     */
#define RESET "\x1B[0m"

/** Standard size of buffer and max node size.                             */
#define FTI_BUFS 256
/** Word size used during RS encoding.                                     */
#define FTI_WORD 16
/** Token returned when FTI performs a checkpoint.                         */
#define FTI_DONE 1
/** Token returned if a FTI function succeeds.                             */
#define FTI_SCES 0
/** Token returned if a FTI function fails.                                */
#define FTI_NSCS -1
/** Token returned if recovery fails.                                      */
#define FTI_NREC -2

/** Verbosity level to print only errors.                                  */
#define FTI_EROR 4
/** Verbosity level to print only warning and errors.                      */
#define FTI_WARN 3
/** Verbosity level to print main information.                             */
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
/** indicator for an invalid stage request                                 */

// this is as well the size of the shared memory window exposed by each rank
#define FTI_SI_MAX_NUM (512L*1024L) // 2MB for each rank

/** MD5-hash: unsigned char digest length.                                 */
#define MD5_DIGEST_LENGTH 16
/** MD5-hash: hex converted char digest length.                            */
#define MD5_DIGEST_STRING_LENGTH 33

#ifdef ENABLE_SIONLIB // --> If SIONlib is installed
/** Token for IO mode SIONlib.                                             */
#define FTI_IO_SIONLIB 1004
#endif

/** Token for IO mode HDF5.                                         */
#define FTI_IO_HDF5 1005
#ifdef ENABLE_HDF5 // --> If HDF5 is installed
    #include "hdf5.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

    /*---------------------------------------------------------------------------
      FTI-FF types
      ---------------------------------------------------------------------------*/

    // status field 0xiiiiiiiiiiii0ssa 
    // 's' status indicator (pending, active, success or failed)
    // 'a' 1 if status ID is available 
    // 'i' index of FTIT_StageAppInfo and FTIT_StageHeadInfo arrays
    typedef struct FTIT_StageInfo {
        int nbRequest;
        uint8_t *status;                    /**< status of request              */
        void *request;
        MPI_Win stageWin;
    } FTIT_StageInfo;

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
        long ckptSize;  /**< size of ckpt data                                  */
        long fs;        /**< file size                                          */
        long maxFs;     /**< maximum file size in group                         */
        long ptFs;      /**< partner copy file size                             */
        long timestamp; /**< time when ckpt was created in ns (CLOCK_REALTIME)  */
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
        int id;             /**< id of protected variable                       */
        int idx;            /**< index to corresponding id in pvar array        */
        int containerid;
        bool hascontent;
        long dptr;          /**< data pointer offset				            */
        long fptr;          /**< file pointer offset                            */
        long chunksize;     /**< chunk size stored aof prot. var. in this block */
        long containersize; /**< chunk size stored aof prot. var. in this block */
        unsigned char hash[MD5_DIGEST_LENGTH];  /**< hash of variable chunk     */
    } FTIFF_dbvar;

    /** @typedef    FTIFF_db
     *  @brief      Information about current datablock.
     *
     *  (For FTI-FF only)
     *  Keeps information about the current datablock in file
     *
     */
    typedef struct FTIFF_db {
        int numvars;            /**< number of protected variables in datablock */
        long dbsize;            /**< size of metadata + data for block in bytes */
        FTIFF_dbvar *dbvars;    /**< pointer to related dbvar array             */
        struct FTIFF_db *previous;  /**< link to previous datablock             */
        struct FTIFF_db *next;      /**< link to next datablock                 */
    } FTIFF_db;

    /*---------------------------------------------------------------------------
      New types
      ---------------------------------------------------------------------------*/

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
        int             id;                 /**< ID to search/update dataset.   */
        void            *ptr;               /**< Pointer to the dataset.        */
        long            count;              /**< Number of elements in dataset. */
        FTIT_type*      type;               /**< Data type for the dataset.     */
        int             eleSize;            /**< Element size for the dataset.  */
        long            size;               /**< Total size of the dataset.     */
        int             rank;               /**< Rank of dataset (for HDF5).    */
        int             dimLength[32];      /**< Lenght of each dimention.      */
        char            name[FTI_BUFS];     /**< Name of the dataset.           */
        FTIT_H5Group*   h5group;            /**< Group of this dataset          */
    } FTIT_dataset;

    /** @typedef    FTIT_metadata
     *  @brief      Metadata for restart.
     *
     *  This type stores all the metadata necessary for the restart.
     */
    typedef struct FTIT_metadata {
        int*             exists;             /**< True if metadata exists        */
        long*            maxFs;              /**< Maximum file size.             */
        long*            fs;                 /**< File size.                     */
        long*            pfs;                /**< Partner file size.             */
        char*            ckptFile;           /**< Ckpt file name. [FTI_BUFS]     */
        int*             nbVar;              /**< Number of variables. [FTI_BUFS]*/
        int*             varID;              /**< Variable id for size.[FTI_BUFS]*/
        long*            varSize;            /**< Variable size. [FTI_BUFS]      */
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
        int             metaAlloc;          /**< True if meta allocated.        */
        int             initSCES;           /**< True if FTI initialized.       */
        FTIT_metadata   meta[5];            /**< Metadata for each ckpt level   */
        FTIFF_db         *firstdb;          /**< Pointer to first datablock     */
        FTIFF_db         *lastdb;           /**< Pointer to first datablock     */
        FTIFF_metaInfo  FTIFFMeta;          /**< File meta data for FTI-FF      */
        FTIT_type**     FTI_Type;           /**< Pointer to FTI_Types           */
        FTIT_H5Group**  H5groups;           /**< HDF5 root group.               */
        FTIT_StageInfo* stageInfo;          /**< root of staging requests       */
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
        int             stagingEnabled;
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
        int             ckptTag;            /**< MPI tag for ckpt requests.     */
        int             stageTag;           /**< MPI tag for staging comm.      */
        int             infoTag;            /**< MPI tag for check comm.        */
        int             finalTag;           /**< MPI tag for finalize comm.     */
        int             generalTag;         /**< MPI tag for general comm.      */
        int             test;               /**< TRUE if local test.            */
        int             l3WordSize;         /**< RS encoding word size.         */
        int             ioMode;             /**< IO mode for L4 ckpt.           */
        char            stageDir[FTI_BUFS];
        char            localDir[FTI_BUFS]; /**< Local directory.               */
        char            glbalDir[FTI_BUFS]; /**< Global directory.              */
        char            metadDir[FTI_BUFS]; /**< Metadata directory.            */
        char            lTmpDir[FTI_BUFS];  /**< Local temporary directory.     */
        char            gTmpDir[FTI_BUFS];  /**< Global temporary directory.    */
        char            mTmpDir[FTI_BUFS];  /**< Metadata temporary directory.  */
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
        char            dir[FTI_BUFS];      /**< Checkpoint directory.          */
        char            metaDir[FTI_BUFS];  /**< Metadata directory.            */
        int             isInline;           /**< TRUE if work is inline.        */
        int             ckptIntv;           /**< Checkpoint interval.           */
        int             ckptCnt;            /**< Checkpoint counter.            */

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

    /*---------------------------------------------------------------------------
      Global variables
      ---------------------------------------------------------------------------*/

    /** MPI communicator that splits the global one into app and FTI appart.   */
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

    /*---------------------------------------------------------------------------
      FTI public functions
      ---------------------------------------------------------------------------*/

    int FTI_Init(char *configFile, MPI_Comm globalComm);
    int FTI_Status();
    int FTI_InitType(FTIT_type* type, int size);
    int FTI_InitComplexType(FTIT_type* newType, FTIT_complexType* typeDefinition, int length,
                            size_t size, char* name, FTIT_H5Group* h5group);
    void FTI_AddSimpleField(FTIT_complexType* typeDefinition, FTIT_type* ftiType,
                                size_t offset, int id, char* name);
    void FTI_AddComplexField(FTIT_complexType* typeDefinition, FTIT_type* ftiType,
                                size_t offset, int rank, int* dimLength, int id, char* name);
    int FTI_InitGroup(FTIT_H5Group* h5group, char* name, FTIT_H5Group* parent);
    int FTI_RenameGroup(FTIT_H5Group* h5group, char* name);
    int FTI_Protect(int id, void* ptr, long count, FTIT_type type);
    int FTI_DefineDataset(int id, int rank, int* dimLength, char* name, FTIT_H5Group* h5group);
    long FTI_GetStoredSize(int id);
    void* FTI_Realloc(int id, void* ptr);
    int FTI_BitFlip(int datasetID);
    int FTI_Checkpoint(int id, int level);
    int FTI_GetStageDir( char* stageDir, int maxLen );
    int FTI_GetStageStatus( int ID );
    int FTI_SendFile( char* lpath, char *rpath );
    int FTI_Recover();
    int FTI_Snapshot();
    int FTI_Finalize();
    int FTI_RecoverVar(int id);

#ifdef __cplusplus
}
#endif

#endif /* ----- #ifndef _FTI_H  ----- */
