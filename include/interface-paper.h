#ifndef __INTERFACE_PAPER_H__
#define __INTERFACE_PAPER_H__

#include "fti.h"

#define XFTI_CRASH (*(int*)NULL = 0)

#ifdef __cplusplus
extern "C" {
#endif

extern FTIT_topology* topo;
extern FTIT_execution* exec;
extern FTIT_configuration* conf;
extern FTIT_checkpoint* ckpt;
extern FTIT_dataset* data;

void XFTI_LiberateHeads();
int XFTI_GetNbNodes();
int XFTI_GetNbApprocs();
void XFTI_Crash(); 
void XFTI_CrashNodes( int nbNodes ); 
int XFTI_CreateHostfile( int nbNodes, const char* nodeList, const char* fn );
void XFTI_Init( FTIT_topology* FTI_Topo, FTIT_execution* FTI_Exec, FTIT_configuration* FTI_Conf,
        FTIT_dataset* FTI_Data, FTIT_checkpoint* FTI_Ckpt );
int XFTI_updateKeyCfg( const char* tag, const char* key, const char* value );

#ifdef __cplusplus
}
#endif

#endif // __INTERFACE_PAPER_H__

