/**
 *  Copyright (c) 2017 Leonardo A. Bautista-Gomez
 *  All rights reserved
 *
 *  @file   tools.h
 */

#ifndef SRC_UTIL_TOOLS_H_
#define SRC_UTIL_TOOLS_H_

#include "../interface.h"

#ifdef __cplusplus
extern "C" {
#endif

void FTI_Print(char *msg, int priority);
int FTI_Checksum(FTIT_execution* FTI_Exec, FTIT_keymap* FTI_Data,
      FTIT_configuration* FTI_Conf, char* checksum);
int FTI_VerifyChecksum(char* fileName, char* checksumToCmp);
int FTI_Try(int result, char* message);
void FTI_FreeTypesAndGroups(FTIT_execution* FTI_Exec);
int FTI_InitGroupsAndTypes(FTIT_execution* FTI_Exec);
int FTI_InitBasicTypes(FTIT_execution* FTI_Exec);
int FTI_InitExecVars(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt,
        FTIT_injection* FTI_Inje);
int FTI_RmDir(char path[FTI_BUFS], int flag);
int FTI_Clean(FTIT_configuration* FTI_Conf, FTIT_topology* FTI_Topo,
        FTIT_checkpoint* FTI_Ckpt, int level);

#ifdef __cplusplus
}
#endif

#endif  // SRC_UTIL_TOOLS_H_
