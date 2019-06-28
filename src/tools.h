#ifndef __TOOLS_H__
#define __TOOLS_H__

#ifdef ENABLE_HDF5
#include <hdf5.h>
int FTI_DebugCheckOpenObjects(hid_t fid, int rank);
#endif

void FTI_Print(char *msg, int priority);
int FTI_Checksum(FTIT_execution* FTI_Exec, FTIT_dataset* FTI_Data,
      FTIT_configuration* FTI_Conf, char* checksum);
int FTI_VerifyChecksum(char* fileName, char* checksumToCmp);
int FTI_Try(int result, char* message);
void FTI_MallocMeta(FTIT_execution* FTI_Exec, FTIT_topology* FTI_Topo);
void FTI_FreeMeta(FTIT_execution* FTI_Exec);
void FTI_FreeTypesAndGroups(FTIT_execution* FTI_Exec);
int FTI_InitGroupsAndTypes(FTIT_execution* FTI_Exec);
int FTI_InitBasicTypes(FTIT_dataset* FTI_Data);
int FTI_InitExecVars(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt,
        FTIT_injection* FTI_Inje);
int FTI_RmDir(char path[FTI_BUFS], int flag);
int FTI_Clean(FTIT_configuration* FTI_Conf, FTIT_topology* FTI_Topo,
        FTIT_checkpoint* FTI_Ckpt, int level);

int FTI_FindVarInMeta(FTIT_execution *FTI_Exec, FTIT_dataset *FTI_Data, 
        int id, int *currentIndex, int *oldIndex);

#endif // __TOOLS_H__
