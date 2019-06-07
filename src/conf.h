#ifndef __CONF_H__
#define __CONF_H__

int FTI_UpdateConf(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        int restart);
int FTI_ReadConf(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt,
        FTIT_injection *FTI_Inje);
int FTI_TestConfig(FTIT_configuration* FTI_Conf, FTIT_topology* FTI_Topo,
        FTIT_checkpoint* FTI_Ckpt, FTIT_execution* FTI_Exec);
int FTI_TestDirectories(FTIT_configuration* FTI_Conf, FTIT_topology* FTI_Topo);
int FTI_CreateDirs(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt);
int FTI_LoadConf(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt,
        FTIT_injection *FTI_Inje);

#endif // __CONF_H__
