/**
 *  Copyright (c) 2017 Leonardo A. Bautista-Gomez
 *  All rights reserved
 *
 *  @file   md5Opt.h
 */

#ifndef FTI_MD5OPT_H_
#define FTI_MD5OPT_H_

#include <fti.h>
#include "../../interface.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef unsigned int MD5_u32plus;
int FTI_destroyMD5();
int FTI_initMD5(long cSize, long tempSize, FTIT_configuration *FTI_Conf);
int FTI_MD5CPU(FTIT_dataset *data);
int FTI_MD5GPU(FTIT_dataset *data);
int FTI_SyncMD5();
int FTI_startMD5();
int FTI_CLOSE_ASYNC(FILE *f);
#ifdef __cplusplus
}
#endif
#endif  // FTI_MD5OPT_H_
