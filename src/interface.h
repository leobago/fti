/**
 *  @file   interface.h
 *  @author
 *  @date   February, 2016
 *  @brief  Header file for the FTI library private functions.
 */

#ifndef _FTI_INTERFACE_H
#define _FTI_INTERFACE_H

#include "fti.h"

#include "../deps/iniparser/iniparser.h"
#include "../deps/iniparser/dictionary.h"

#include "../deps/jerasure/galois.h"
#include "../deps/jerasure/jerasure.h"

#ifdef ENABLE_SIONLIB // --> If SIONlib is installed
    #include <sion.h>
#endif

#include <stdint.h>
#include "../deps/md5/md5.h"

#include <sys/stat.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <errno.h>
#include <math.h>
#include <limits.h>

/*---------------------------------------------------------------------------
                                  Defines
---------------------------------------------------------------------------*/

/** Malloc macro.                                                          */
#define talloc(type, num) (type *)malloc(sizeof(type) * (num))

/*---------------------------------------------------------------------------
                            FTI private functions
---------------------------------------------------------------------------*/
void FTI_PrintMeta(FTIT_execution* FTI_Exec, FTIT_topology* FTI_Topo);
void FTI_Abort();
int FTI_FloatBitFlip(float *target, int bit);
int FTI_DoubleBitFlip(double *target, int bit);
void FTI_Print(char *msg, int priority);

int FTI_UpdateIterTime(FTIT_execution* FTI_Exec);
int FTI_WriteCkpt(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
                  FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt,
                  FTIT_dataset* FTI_Data);
#ifdef ENABLE_SIONLIB // --> If SIONlib is installed
int FTI_WriteSionlib(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
                FTIT_topology* FTI_Topo,FTIT_dataset* FTI_Data);
#endif
int FTI_WriteMPI(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
                FTIT_topology* FTI_Topo,FTIT_dataset* FTI_Data);
int FTI_WritePar(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
                FTIT_topology* FTI_Topo,FTIT_dataset* FTI_Data);
int FTI_WritePosix(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
                    FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt,
                    FTIT_dataset* FTI_Data);
int FTI_PostCkpt(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
                 FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt);
int FTI_Listen(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
               FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt);

int FTI_UpdateConf(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
                   int restart);
int FTI_ReadConf(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
                 FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt,
                 FTIT_injection *FTI_Inje);
int FTI_TestConfig(FTIT_configuration* FTI_Conf, FTIT_topology* FTI_Topo,
                   FTIT_checkpoint* FTI_Ckpt);
int FTI_TestDirectories(FTIT_configuration* FTI_Conf, FTIT_topology* FTI_Topo);
int FTI_CreateDirs(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
                   FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt);
int FTI_LoadConf(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
                 FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt,
                 FTIT_injection *FTI_Inje);

int FTI_GetChecksums(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
                     FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt,
                     char* checksum, char* ptnerChecksum, char* rsChecksum);
int FTI_WriteRSedChecksum(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
                             FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt,
                              int rank, char* checksum);
int FTI_LoadTmpMeta(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
              FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt);
int FTI_LoadMeta(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
              FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt);
int FTI_WriteMetadata(FTIT_configuration* FTI_Conf, FTIT_topology* FTI_Topo,
                      unsigned long *fs, unsigned long mfs, char* fnl, char* checksums);
int FTI_CreateMetadata(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
                       FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt);

int FTI_Local(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
              FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt);
int FTI_Ptner(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
              FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt);
int FTI_RSenc(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
              FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt);
int FTI_Flush(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
              FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt, int level);
int FTI_FlushPosix(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
                  FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt, int level);
int FTI_FlushMPI(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
                  FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt, int level);
#ifdef ENABLE_SIONLIB // --> If SIONlib is installed
int FTI_FlushSionlib(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
                    FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt, int level);
#endif
int FTI_Decode(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
               FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt, int *erased);
int FTI_RecoverL1(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
                  FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt);
int FTI_RecoverL2(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
                  FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt);
int FTI_RecoverL3(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
                  FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt);
int FTI_RecoverL4(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
                  FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt);
int FTI_RecoverL4Posix(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
                  FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt);
int FTI_RecoverL4Mpi(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
                  FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt);
#ifdef ENABLE_SIONLIB // --> If SIONlib is installed
int FTI_RecoverL4Sionlib(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
                  FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt);
#endif
int FTI_CheckFile(char *fn, long fs, char* checksum);
int FTI_CheckErasures(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
                      FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt,
                      int *erased);
int FTI_RecoverFiles(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
                     FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt);

int FTI_Checksum(char* fileName, char* checksum);
int FTI_VerifyChecksum(char* fileName, char* checksumToCmp);
int FTI_Try(int result, char* message);
void FTI_MallocMeta(FTIT_execution* FTI_Exec, FTIT_topology* FTI_Topo);
void FTI_FreeMeta(FTIT_execution* FTI_Exec);
int FTI_InitBasicTypes(FTIT_dataset* FTI_Data);
int FTI_RmDir(char path[FTI_BUFS], int flag);
int FTI_Clean(FTIT_configuration* FTI_Conf, FTIT_topology* FTI_Topo,
              FTIT_checkpoint* FTI_Ckpt, int level);

int FTI_SaveTopo(FTIT_configuration* FTI_Conf, FTIT_topology* FTI_Topo, char *nameList);
int FTI_ReorderNodes(FTIT_configuration* FTI_Conf, FTIT_topology* FTI_Topo,
                     int *nodeList, char *nameList);
int FTI_BuildNodeList(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
                      FTIT_topology* FTI_Topo, int *nodeList, char *nameList);
int FTI_CreateComms(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
                    FTIT_topology* FTI_Topo, int *userProcList,
                    int *distProcList, int* nodeList);
int FTI_Topology(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
                 FTIT_topology* FTI_Topo);

#endif
