/**
 *  @file   fti.h
 *  @author Leonardo A. Bautista Gomez (leobago@gmail.com)
 *  @date   July, 2013
 *  @brief  Header file for the FTI library.
 */

#ifndef _FTI_H
#define _FTI_H

#include <mpi.h>

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

/** Hashed string length.                                                */
#define MD5_DIGEST_LENGTH 17

#ifdef ENABLE_SIONLIB // --> If SIONlib is installed
    /** Token for IO mode SIONlib.                                         */
    #define FTI_IO_SIONLIB 1003
#endif

#ifdef __cplusplus
extern "C" {
#endif

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
    float           value;			    /**< Floating point value.          */
    int             intval;             /**< Integer mapped to do bit edits.*/
    char            byte[4];            /**< Byte array for coarser control.*/
} FTIT_float;

/** @typedef    FTIT_dbvar
 *  @brief      Information about protected variable in datablock.
 *
 *  (For FTI File Format only)
 *  Keeps information about the chunk of the protected variable with id 
 *  stored in the current datablock. 
 *  
 */
typedef struct FTIT_dbvar {
    int id;
    int idx;			   /**< index to corresponding id in pvar array */
    long dptr;			   /**< data pointer offset				        */
    long fptr;			   /**< file pointer offset                     */
    long chunksize;
} FTIT_dbvar;

/** @typedef    FTIT_db
 *  @brief      Information about current datablock.
 *
 *  (For FTI File Format only)
 *  Keeps information about the current datablock in file
 */
typedef struct FTIT_db {
    int numvars;	        /**< number of protected variables in datablock */
    long dbsize;            /**< size of metadata + data in bytes		    */
    FTIT_dbvar *dbvars;     /**< pointer to corresponding dbvar array       */
    struct FTIT_db *previous;		 /**< link to previous datablock        */
    struct FTIT_db *next;			 /**< link to next datablock            */
} FTIT_db;

/** @typedef    FTIT_type
 *  @brief      Type recognized by FTI.
 *
 *  This type allows handling data structures.
 */
typedef struct FTIT_type {
    int             id;                 /**< ID of the data type.           */
    int             size;               /**< Size of the data type.         */
} FTIT_type;

/** @typedef    FTIT_dataset
 *  @brief      Dataset metadata.
 *
 *  This type stores the metadata related with a dataset.
 */
typedef struct FTIT_dataset {
    int             id;                 /**< ID to search/update dataset.   */
    void            *ptr;               /**< Pointer to the dataset.        */
    long            count;              /**< Number of elements in dataset. */
    FTIT_type       type;               /**< Data type for the dataset.     */
    int             eleSize;            /**< Element size for the dataset.  */
    long            size;               /**< Total size of the dataset.     */
    /** MD5 Checksum                    */
    char            checksum[MD5_DIGEST_LENGTH];
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
    int             metaAlloc;          /**< True if meta allocated.        */
    int             initSCES;           /**< True if FTI initialized.       */
    FTIT_metadata   meta[5];            /**< Metadata for each ckpt level   */
    FTIT_db         *firstdb;           /**< Pointer to first datablock     */
    FTIT_db         *lastdb;            /**< Pointer to first datablock     */
    MPI_Comm        globalComm;         /**< Global communicator.           */
    MPI_Comm        groupComm;          /**< Group communicator.            */
} FTIT_execution;

/** @typedef    FTIT_configuration
 *  @brief      Configuration metadata.
 *
 *  This type stores the general configuration metadata.
 */
typedef struct FTIT_configuration {
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
    int             tag;                /**< Tag for MPI messages in FTI.   */
    int             test;               /**< TRUE if local test.            */
    int             l3WordSize;         /**< RS encoding word size.         */
    int             ioMode;             /**< IO mode for L4 ckpt.           */
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
int FTI_Protect(int id, void* ptr, long count, FTIT_type type);
long FTI_GetStoredSize(int id);
void* FTI_Realloc(int id, void* ptr);
int FTI_BitFlip(int datasetID);
int FTI_Checkpoint(int id, int level);
int FTI_Recover();
int FTI_Snapshot();
int FTI_Finalize();
int FTI_RecoverVar(int id);

#ifdef __cplusplus
}
#endif

#endif /* ----- #ifndef _FTI_H  ----- */
