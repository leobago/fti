#ifndef STAGE_FUNC_H
#define STAGE_FUNC_H

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
int FTI_GetRequestField( int ID, FTIT_RequestField val ); 
int FTI_SetRequestField( int ID, uint32_t entry, FTIT_RequestField val );
int FTI_FreeStageRequest( FTIT_execution *FTI_Exec, FTIT_topology *FTI_Topo, int ID, int source ); 
void FTI_PrintStageStatus( FTIT_execution *FTI_Exec, FTIT_topology *FTI_Topo, int ID, int source ); 
int FTI_GetRequestIdx( int ID );
void FTI_FinalizeStage( FTIT_execution *FTI_Exec, FTIT_topology *FTI_Topo, FTIT_configuration *FTI_Conf ); 

#endif // STAGE_FUNC_H
