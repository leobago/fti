#ifndef __IME_H__
#define __IME_H__
#ifdef __cplusplus
extern "C"
{
#endif
void FTI_IMEMD5(unsigned char *dest, void *md5);
int FTI_WriteIMEData(FTIT_dataset * FTI_DataVar, void *fd);
void* FTI_InitIME(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec, FTIT_topology* FTI_Topo, FTIT_checkpoint *FTI_Ckpt, FTIT_dataset *FTI_Data);
int FTI_IMESync(void *fileDesc);
//int FTI_PosixRead(void *dest, size_t size, void *fileDesc);
size_t FTI_GetIMEFilePos(void *fileDesc);
int FTI_IMESeek(size_t pos, void *fileDesc);
int FTI_IMEWrite(void *src, size_t size, void *fileDesc);
int FTI_IMEClose(void *fileDesc);
int FTI_IMEOpen(char *fn, void *fileDesc);

#ifdef __cplusplus
}
#endif
#endif // __IME_H__
