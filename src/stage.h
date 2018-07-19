#define FTI_STAGE_FAIL 0x11
#define FTI_STAGE_SCES 0x00
#define FTI_STAGE_ACTV 0x01
#define FTI_STAGE_PEND 0x10

#define FTI_SI_APTR( ptr ) ((FTIT_StageAppInfo*)ptr)
#define FTI_SI_HPTR( ptr ) ((FTIT_StageHeadInfo*)ptr)

// this is as well the size of the shared memory window exposed by each rank
#define MAX_NUM_REQ (1024L*1024L) // 1MB for each rank

// macro for stage request counter
#define FTI_NEW_REQ_ID request_counter()

// stage request counter returns -1 if too many requests.
static inline int request_counter() {
    static int req_cnt = 0;
    if( req_cnt < MAX_NUM_REQ ) {
        return req_cnt++;
    } else {
        return -1;
    }
}

typedef struct FTIT_StageHeadInfo {
    uint64_t ID;                            /**< Unique request ID              */
    char lpath[FTI_BUFS];                    /**< file path                      */
    char rpath[FTI_BUFS];                      /**< file name                      */
    size_t offset;                          /**< current offset of file pointer */
    size_t size;                            /**< file size                      */
    struct FTIT_StageHeadInfo *previous;    /**< (internal usage ptr previous)  */
    struct FTIT_StageHeadInfo *next;        /**< (internal usage ptr next)      */
} FTIT_StageHeadInfo;

typedef struct FTIT_StageAppInfo {
    uint32_t ID;                            /**< Unique request ID              */
    MPI_Request mpiReq;
    struct FTIT_StageAppInfo *previous;     /**< (internal usage ptr previous)  */
    struct FTIT_StageAppInfo *next;         /**< (internal usage ptr next)      */
} FTIT_StageAppInfo;


int FTI_InitStage( FTIT_execution *FTI_Exec, FTIT_configuration *FTI_Conf, FTIT_topology *FTI_Topo );
int FTI_InitStageRequestApp( FTIT_execution *FTI_Exec, FTIT_topology *FTI_Topo, uint32_t ID );
int FTI_StageRequestHead( char *lpath, char *rpath, FTIT_configuration *FTI_Conf, 
        FTIT_execution *FTI_Exec, FTIT_topology *FTI_Topo, uint32_t ID );
int FTI_InitStageRequestHead( char* lpath, char *rpath, FTIT_execution *FTI_Exec, 
        FTIT_topology *FTI_Topo, int source, uint32_t ID );
int FTI_SyncStage( char* lpath, char *rpath, FTIT_execution *FTI_Exec, FTIT_configuration *FTI_Conf, uint32_t ID ); 
int FTI_HandleStageRequest(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt, int source);
