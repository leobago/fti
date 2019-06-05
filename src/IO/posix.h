#ifndef __POSIX_H__
#define __POSIX_H__

int FTI_PosixOpen(char *fn, void *fileDesc);
int FTI_PosixClose(void *fileDesc);
int FTI_PosixWrite(void *src, size_t size, void *fileDesc);
int FTI_PosixSeek(size_t pos, void *fileDesc);
size_t FTI_GetPosixFilePos(void *fileDesc);
int FTI_PosixRead(void *dest, size_t size, void *fileDesc);
int FTI_PosixSync(void *fileDesc);
void* FTI_InitPosix(void);
int FTI_WritePosixData(FTIT_dataset * FTI_DataVar, void *fd);
void FTI_PosixMD5(unsigned char *dest, void *md5);

#endif // __POSIX_H__
