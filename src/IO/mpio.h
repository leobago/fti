#ifndef __MPIIO_H__
#define __MPIIO_H__

int FTI_MPIOOpen(char *fn, void *fileDesc);
int FTI_MPIOClose(void *fileDesc);
int FTI_MPIOWrite(void *src, size_t size, void *fileDesc);
size_t FTI_GetMPIOFilePos(void *fileDesc);
int FTI_MPIORead(void *dest, size_t size, void *fileDesc);
void *FTI_InitMPIO(void);
int FTI_WriteMPIOData(FTIT_dataset * FTI_DataVar, void *fd);

#endif // __MPIIO_H__
