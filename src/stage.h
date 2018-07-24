#define FTI_SI_FAIL 0x11
#define FTI_SI_SCES 0x00
#define FTI_SI_ACTV 0x01
#define FTI_SI_PEND 0x10

#define FTI_SI_IAVL 0x0
#define FTI_SI_NAVL 0x1

#define FTI_SI_APTR( ptr ) ((FTIT_StageAppInfo*)ptr)
#define FTI_SI_HPTR( ptr ) ((FTIT_StageHeadInfo*)ptr)

#define FTI_SI_MAX_ID (0x7ffff)

typedef enum {
    FTI_SIF_AVL,
    FTI_SIF_VAL,
    FTI_SIF_IDX
} FTIT_StatusField;

typedef struct FTIT_StageHeadInfo {
    char lpath[FTI_BUFS];                    /**< file path                      */
    char rpath[FTI_BUFS];                      /**< file name                      */
    size_t offset;                          /**< current offset of file pointer */
    size_t size;                            /**< file size                      */
} FTIT_StageHeadInfo;

typedef struct FTIT_StageAppInfo {
    MPI_Request mpiReq;
} FTIT_StageAppInfo;


int FTI_GetRequestID( FTIT_execution *FTI_Exec, FTIT_topology *FTI_Topo ); 
int FTI_InitStage( FTIT_execution *FTI_Exec, FTIT_configuration *FTI_Conf, FTIT_topology *FTI_Topo );
int FTI_InitStageRequestApp( FTIT_execution *FTI_Exec, FTIT_topology *FTI_Topo, uint32_t ID );
int FTI_AsyncStage( char *lpath, char *rpath, FTIT_configuration *FTI_Conf, 
        FTIT_execution *FTI_Exec, FTIT_topology *FTI_Topo, uint32_t ID );
int FTI_InitStageRequestHead( char* lpath, char *rpath, FTIT_execution *FTI_Exec, 
        FTIT_topology *FTI_Topo, int source, uint32_t ID );
int FTI_SyncStage( char* lpath, char *rpath, FTIT_execution *FTI_Exec, FTIT_configuration *FTI_Conf, uint32_t ID ); 
int FTI_HandleStageRequest(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt, int source);
int FTI_GetStatusField( FTIT_execution *FTI_Exec, FTIT_topology *FTI_Topo, int ID, FTIT_StatusField val, int source ); 
int FTI_SetStatusField( FTIT_execution *FTI_Exec, FTIT_topology *FTI_Topo, int ID, uint32_t entry, FTIT_StatusField val, int source );
