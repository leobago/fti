#ifndef _STAGE_H_
#define _STAGE_H_

// 1 bit field
#define FTI_SI_NAVL 0x1
#define FTI_SI_IAVL 0x0

#define FTI_SI_IALL 0x1
#define FTI_SI_NALL 0x0

#define FTI_SI_APTR( ptr ) ((FTIT_StageAppInfo*)ptr)
#define FTI_SI_HPTR( ptr ) ((FTIT_StageHeadInfo*)ptr)

#define FTI_SI_MAX_ID (0x7ffff)

#define FTI_DISABLE_STAGING do{*enableStagingPtr = false;} while(0)
#define FTI_SI_ENABLED (*(bool*)enableStagingPtr)

/** @typedef    FTIT_StatusField
 *  @brief      valid fields of 'status' and 'idxRequest'.
 * 
 *  enum that keeps the particular field identifiers for the 'status'
 *  and 'idxRequest' fields.
 */
typedef enum {
    FTI_SIF_AVL = 0,
    FTI_SIF_VAL,
    FTI_SIF_ALL,
    FTI_SIF_IDX
} FTIT_StatusField;

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


int FTI_GetRequestID( FTIT_execution *FTI_Exec, FTIT_topology *FTI_Topo ); 
int FTI_InitStage( FTIT_execution *FTI_Exec, FTIT_configuration *FTI_Conf, FTIT_topology *FTI_Topo );
int FTI_InitStageRequestApp( FTIT_execution *FTI_Exec, FTIT_topology *FTI_Topo, uint32_t ID );
int FTI_AsyncStage( char *lpath, char *rpath, FTIT_configuration *FTI_Conf, 
        FTIT_execution *FTI_Exec, FTIT_topology *FTI_Topo, int ID );
int FTI_InitStageRequestHead( char* lpath, char *rpath, FTIT_execution *FTI_Exec, 
        FTIT_topology *FTI_Topo, int source, uint32_t ID );
int FTI_SyncStage( char* lpath, char *rpath, FTIT_execution *FTI_Exec, 
        FTIT_topology *FTI_Topo, FTIT_configuration *FTI_Conf, uint32_t ID ); 
int FTI_HandleStageRequest(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt, int source);
int FTI_GetStatusField( FTIT_execution *FTI_Exec, FTIT_topology *FTI_Topo, int ID, FTIT_StatusField val, int source ); 
int FTI_SetStatusField( FTIT_execution *FTI_Exec, FTIT_topology *FTI_Topo, int ID, uint8_t entry, FTIT_StatusField val, int source );
int FTI_GetRequestField( int ID, FTIT_StatusField val ); 
int FTI_SetRequestField( int ID, uint32_t entry, FTIT_StatusField val );
int FTI_FreeStageRequest( FTIT_execution *FTI_Exec, FTIT_topology *FTI_Topo, int ID, int source ); 
void FTI_PrintStageStatus( FTIT_execution *FTI_Exec, FTIT_topology *FTI_Topo, int ID, int source ); 
int FTI_GetRequestIdx( int ID );
void FTI_FinalizeStage( FTIT_execution *FTI_Exec, FTIT_topology *FTI_Topo, FTIT_configuration *FTI_Conf ); 

#endif
