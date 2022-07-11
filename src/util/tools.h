/**
 *  Copyright (c) 2017 Leonardo A. Bautista-Gomez
 *  All rights reserved
 *
 *  @file   tools.h
 */

#ifndef FTI_SRC_UTIL_TOOLS_H_
#define FTI_SRC_UTIL_TOOLS_H_

#include "../interface.h"

#define FTI_MIN(X, Y) (((X) < (Y)) ? (X) : (Y))

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
int FTI_RmDir(const char* path, int flag);
int FTI_Clean(FTIT_configuration* FTI_Conf, FTIT_topology* FTI_Topo,
        FTIT_checkpoint* FTI_Ckpt, int level);

void FTI_CopyStringOrDefault(char* dest, const char* src, char* fmt, ...);
int FTI_IsTypeComposite(FTIT_Datatype *t);
FTIT_Datatype* FTI_GetCompositeType(fti_id_t handle);

// TODO(alex): the following 2 methods are hidden from the public API
fti_id_t FTI_InitType_opaque(size_t size);
FTIT_Datatype* FTI_GetType(fti_id_t id);
int FTI_FileCopy( const char*, const char* );
int FTI_CreateDirectory( FTIT_topology*, const char*, int );

#ifdef __cplusplus
}
#endif

#endif  // FTI_SRC_UTIL_TOOLS_H_
