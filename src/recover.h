/**
 *  Copyright (c) 2017 Leonardo A. Bautista-Gomez
 *  All rights reserved
 *
 *  @file   recover.h
 */

#ifndef FTI_SRC_RECOVER_H_
#define FTI_SRC_RECOVER_H_

#include "interface.h"

int FTI_CheckFile(char *fn, int32_t fs, char* checksum);
int FTI_CheckErasures(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt,
        int *erased);
int FTI_RecoverFiles(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt);

#endif  // FTI_SRC_RECOVER_H_
