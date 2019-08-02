#include "interface-paper.h"

FTIT_topology* topo;
FTIT_execution* exec;
FTIT_configuration* conf;
FTIT_checkpoint* ckpt;
FTIT_dataset* data;

void XFTI_Init( FTIT_topology* FTI_Topo, FTIT_execution* FTI_Exec, FTIT_configuration* FTI_Conf,
        FTIT_dataset* FTI_Data, FTIT_checkpoint* FTI_Ckpt ) 
{
    topo = FTI_Topo;
    exec = FTI_Exec;
    conf = FTI_Conf;
    data = FTI_Data;
    ckpt = FTI_Ckpt;
}
