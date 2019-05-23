#ifndef KERNEL_TYPES_H
#define KERNEL_TYPES_H

#include <fti-int/defs.h>
#include <fti-int/hdf5-types.h>
#include <fti-int/ftiff-types.h>

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
    struct FTIT_type**     FTI_Type;           /**< Pointer to FTI_Types           */
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

#endif // KERNEL_TYPES_H
