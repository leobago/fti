/**
 *  Copyright (c) 2017 Leonardo A. Bautista-Gomez
 *  All rights reserved
 *
 *  @file   fti-io.h
 */

#ifndef FTI_SRC_FTI_IO_H_
#define FTI_SRC_FTI_IO_H_


#include "interface.h"

int FTI_InitFunctionPointers(int ckptIO, FTIT_execution * FTI_Exec);
extern FTIT_IO ftiIO[4];

#endif  // FTI_SRC_FTI_IO_H_
