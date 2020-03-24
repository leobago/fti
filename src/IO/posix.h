#ifndef __POSIX_H__
#define __POSIX_H__
#ifdef __cplusplus
extern "C"
{
#endif
extern FILE* fileposix;

int FTI_ActivateHeadsPosix(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec, FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt, int status);
void FTI_PosixMD5(unsigned char *dest, void *md5);
int FTI_WritePosixData(FTIT_dataset * data, void *fd);
void* FTI_InitPosix(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec, FTIT_topology* FTI_Topo, FTIT_checkpoint *FTI_Ckpt, FTIT_keymap *FTI_Data);
int FTI_PosixSync(void *fileDesc);
int FTI_PosixRead(void *dest, size_t size, void *fileDesc);
size_t FTI_GetPosixFilePos(void *fileDesc);
int FTI_PosixSeek(size_t pos, void *fileDesc);
int FTI_PosixWrite(void *src, size_t size, void *fileDesc);
int FTI_PosixClose(void *fileDesc);
int FTI_PosixOpen(char *fn, void *fileDesc);
int FTI_RecoverVarInitPOSIX(char* fn);
int FTI_RecoverVarPOSIX(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec, 
FTIT_topology* FTI_Topo, FTIT_checkpoint *FTI_Ckpt, FTIT_dataset *FTI_Data, int id, FILE* fileposix);
int FTI_RecoverVarFinalizePOSIX(FILE* fileposix);
#ifdef __cplusplus
}
#endif
#endif // __POSIX_H__
