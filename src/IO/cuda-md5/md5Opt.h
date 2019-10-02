#ifndef __MD5GPU__
#define __MD5GPU__

#include <fti.h>
#include "../../tools.h"
#include "../../interface.h"
#ifdef __cplusplus
extern "C"
{
#endif
typedef unsigned int MD5_u32plus;
int FTI_destroyMD5();
int FTI_initMD5(long cSize, long tempSize, FTIT_configuration *FTI_Conf);
int FTI_MD5CPU(FTIT_dataset *FTI_DataVar);
int FTI_MD5GPU(FTIT_dataset *FTI_DataVar);
int FTI_SyncMD5();
int FTI_startMD5();
int FTI_CLOSE_ASYNC(void *f);
int md5Level(unsigned char *data , MD5_u32plus *Dmd5hash, unsigned long size, cudaStream_t stream);
int FTI_destroyMD5();
int FTI_waitVariable( int id); 
int FTI_SleepWorker();
#ifdef __cplusplus
}
#endif
#endif
