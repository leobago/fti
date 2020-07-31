/**
 *  Copyright (c) 2017 Leonardo A. Bautista-Gomez
 *  All rights reserved
 *
 *  @file   fti-io.h
 */

#ifndef FTI_FTI_IO_H_
#define FTI_FTI_IO_H_


#include "fti.h"
int FTI_InitFunctionPointers(int ckptIO, FTIT_execution * FTI_Exec);
extern FTIT_IO ftiIO[4];

#endif  // FTI_FTI_IO_H_
