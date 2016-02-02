/**
 *  @file   interface.h
 *  @author
 *  @date   February, 2016
 *  @brief  Header file for the FTI library private functions.
 */

#ifndef _FTI_INTERFACE_H
#define _FTI_INTERFACE_H

#include "../deps/iniparser/iniparser.h"
#include "../deps/iniparser/dictionary.h"

#include "../deps/jerasure/galois.h"
#include "../deps/jerasure/jerasure.h"

/*---------------------------------------------------------------------------
                                  Defines
---------------------------------------------------------------------------*/

/** Malloc macro.                                                          */
#define talloc(type, num) (type *)malloc(sizeof(type) * (num))

/*---------------------------------------------------------------------------
                            FTI private functions
---------------------------------------------------------------------------*/

void FTI_Abort();
int FTI_FloatBitFlip(float *target, int bit);
int FTI_DoubleBitFlip(double *target, int bit);

int FTI_UpdateIterTime();
int FTI_WriteCkpt(FTIT_dataset* FTI_Data);
int FTI_GroupClean(int level, int group, int pr);
int FTI_PostCkpt(int group, int fo, int pr);
int FTI_Listen();

int FTI_UpdateConf(int restart);
int FTI_ReadConf(FTIT_injection *FTI_Inje);
int FTI_TestConfig();
int FTI_TestDirectories();
int FTI_CreateDirs();
int FTI_LoadConf(FTIT_injection *FTI_Inje);

int FTI_GetMeta(unsigned long *fs, unsigned long *mfs, int group, int level);
int FTI_WriteMetadata(unsigned long *fs, unsigned long mfs, char* fnl);
int FTI_CreateMetadata(int globalTmp);

int FTI_Local(int group);
int FTI_Ptner(int group);
int FTI_RSenc(int group);
int FTI_Flush(int group, int level);

int FTI_Decode(int fs, int maxFs, int *erased);
int FTI_RecoverL1(int group);
int FTI_RecoverL2(int group);
int FTI_RecoverL3(int group);
int FTI_RecoverL4(int group);

int FTI_CheckFile(char *fn, unsigned long fs);
int FTI_CheckErasures(unsigned long *fs, unsigned long *maxFs, int group, int *erased, int level);
int FTI_RecoverFiles();

int FTI_Clean(int level, int group, int rank);
void FTI_Print(char *msg, int priority);
int FTI_Try(int result, char* message);
int FTI_InitBasicTypes(FTIT_dataset FTI_Data[FTI_BUFS]);
int FTI_RmDir(char path[FTI_BUFS], int flag);
int FTI_Clean(int level, int group, int rank);

int FTI_SaveTopo(char *nameList);
int FTI_ReorderNodes(int *nodeList, char *nameList);
int FTI_BuildNodeList(int *nodeList, char *nameList);
int FTI_CreateComms(int *userProcList, int *distProcList, int* nodeList);
int FTI_Topology();

#endif
