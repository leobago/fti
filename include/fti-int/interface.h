/**
 *  Copyright (c) 2017 Leonardo A. Bautista-Gomez
 *  All rights reserved
 *
 *  FTI - A multi-level checkpointing library for C/C++/Fortran applications
 *
 *  Revision 1.0 : Fault Tolerance Interface (FTI)
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice, this
 *  list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright notice,
 *  this list of conditions and the following disclaimer in the documentation
 *  and/or other materials provided with the distribution.
 *
 *  3. Neither the name of the copyright holder nor the names of its contributors
 *  may be used to endorse or promote products derived from this software without
 *  specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 *  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 *  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 *  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 *  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 *  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *  @file   interface.h
 *  @date   October, 2017
 *  @brief  Header file for the FTI library private functions.
 */

#ifndef _FTI_INTERFACE_H
#define _FTI_INTERFACE_H

/* include fti.h for public definitions of
 * FTI_BUFS
 * FTI_WORD
 * FTI_DONE
 * FTI_SCES
 * FTI_NSCS 
 * FTI_NREC
 * FTI_HEAD */
#include <fti.h>
  
#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

#define DBG_MSG(MSG,RANK,...) do { \
    int rank; \
    MPI_Comm_rank(MPI_COMM_WORLD,&rank); \
    if ( rank == RANK ) \
        printf( "%s:%d[DEBUG-%d] " MSG "\n", __FILENAME__,__LINE__,rank, ##__VA_ARGS__); \
    if ( RANK == -1 ) \
        printf( "%s:%d[DEBUG-%d] " MSG "\n", __FILENAME__,__LINE__,rank, ##__VA_ARGS__); \
} while (0)

/** Define RED color for FTI output.                                       */
#define FTI_COLOR_RED   "\x1B[31m"
/** Define ORANGE color for FTI output.                                    */
#define FTI_COLOR_ORG   "\x1B[38;5;202m"
/** Define GREEN color for FTI output.                                     */
#define FTI_COLOR_GRN   "\x1B[32m"
/** Define BLUE color for FTI output.                                       */
#define FTI_COLOR_BLU   "\x1B[34m"
/** Define color RESET for FTI output.                                     */
#define FTI_COLOR_RESET "\x1B[0m"

/** Verbosity level to print only errors.                                  */
#define FTI_EROR 4
/** Verbosity level to print only warning and errors.                      */
#define FTI_WARN 3
/** Verbosity level to print main information.                             */
#define FTI_IDCP 5
/** Verbosity level to print debug messages.                               */
#define FTI_INFO 2
/** Verbosity level to print debug messages.                               */
#define FTI_DBUG 1

/** Token for checkpoint Baseline.                                         */
#define FTI_BASE 990
/** Token for checkpoint Level 1.                                          */
#define FTI_CKTW 991
/** Token for checkpoint Level 2.                                          */
#define FTI_XORW 992
/** Token for checkpoint Level 3.                                          */
#define FTI_RSEW 993
/** Token for checkpoint Level 4.                                          */
#define FTI_PFSW 994
/** Token for end of the execution.                                        */
#define FTI_ENDW 995
/** Token to reject checkpoint.                                            */
#define FTI_REJW 996
/** Token for IO mode Posix.                                               */
#define FTI_IO_POSIX 1001
/** Token for IO mode MPI.                                                 */
#define FTI_IO_MPI 1002
/** Token for IO mode FTI-FF.                                              */
#define FTI_IO_FTIFF 1003
#ifdef ENABLE_SIONLIB // --> If SIONlib is installed
/** Token for IO mode SIONlib.                                             */
#define FTI_IO_SIONLIB 1004
#endif
/** Token for IO mode HDF5.                                         */
#define FTI_IO_HDF5 1005

/** MD5-hash: unsigned char digest length.                                 */
#ifndef MD5_DIGEST_LENGTH
#   define MD5_DIGEST_LENGTH 16
#endif
/** MD5-hash: hex converted char digest length.                            */
#define MD5_DIGEST_STRING_LENGTH 33

// need this parameter in one fti api function
#ifndef ENABLE_HDF5
typedef size_t 	hsize_t;
#endif

// internal header
#include <fti-int/types.h> 
#include <fti-int/ftiff.h> 
#include <fti-int/stage.h>
#include <fti-int/icp.h>
#ifdef ENABLE_HDF5
#include <fti-int/hdf5-fti.h>
#endif
#include <fti-int/utility.h>
#include <fti-int/failure-injection.h>

#include "../deps/iniparser/iniparser.h"
#include "../deps/iniparser/dictionary.h"

#include "../deps/jerasure/include/galois.h"
#include "../deps/jerasure/include/jerasure.h"

#ifdef ENABLE_SIONLIB // --> If SIONlib is installed
#   include <sion.h>
#endif

#ifdef ENABLE_HDF5
#include "hdf5.h"
#include "hdf5_hl.h"
#endif

#include "../deps/md5/md5.h"

#define CHUNK_SIZE 131072    /**< MD5 algorithm chunk size.      */

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <errno.h>
#include <math.h>
#include <limits.h>
#include <inttypes.h>
#include <dirent.h>
#include <stdbool.h>
#include <stdint.h>
#include <time.h>
#include <libgen.h>

#ifdef LUSTRE
#   include "lustreapi.h"
#endif

typedef uintptr_t           FTI_ADDRVAL;        /**< for ptr manipulation       */
typedef void*               FTI_ADDRPTR;        /**< void ptr type              */ 

/*---------------------------------------------------------------------------
  Defines
  ---------------------------------------------------------------------------*/

/** Malloc macro.                                                          */
#define talloc(type, num) (type *)malloc(sizeof(type) * (num))

extern int FTI_filemetastructsize;	/**< size of FTIFF_metaInfo in file */
extern int FTI_dbstructsize;		/**< size of FTIFF_db in file       */
extern int FTI_dbvarstructsize;		/**< size of FTIFF_dbvar in file    */

/*---------------------------------------------------------------------------
  FTI private functions
  ---------------------------------------------------------------------------*/
void FTI_PrintMeta(FTIT_execution* FTI_Exec, FTIT_topology* FTI_Topo);
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
int FTI_HandleCkptRequest(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt);
int FTI_HandleStageRequest(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt, int source);

int FTI_UpdateConf(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        int restart);
int FTI_ReadConf(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt,
        FTIT_injection *FTI_Inje);
int FTI_TestConfig(FTIT_configuration* FTI_Conf, FTIT_topology* FTI_Topo,
        FTIT_checkpoint* FTI_Ckpt, FTIT_execution* FTI_Exec);
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
int FTI_WriteMetadata(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, long* fs, long mfs, char* fnl,
        char* checksums, int* allVarIDs, long* allVarSizes);
int FTI_CreateMetadata(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt,
        FTIT_dataset* FTI_Data);
int FTI_WriteCkptMetaData(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt );
int FTI_LoadCkptMetaData(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt );
int FTI_LoadL4CkptMetaData(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt );

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

int FTI_Checksum(FTIT_execution* FTI_Exec, FTIT_dataset* FTI_Data,
      FTIT_configuration* FTI_Conf, char* checksum);
int FTI_VerifyChecksum(char* fileName, char* checksumToCmp);
int FTI_Try(int result, char* message);
void FTI_MallocMeta(FTIT_execution* FTI_Exec, FTIT_topology* FTI_Topo);
void FTI_FreeMeta(FTIT_execution* FTI_Exec);
void FTI_FreeTypesAndGroups(FTIT_execution* FTI_Exec);
int FTI_InitGroupsAndTypes(FTIT_execution* FTI_Exec);
int FTI_InitBasicTypes(FTIT_dataset* FTI_Data);
int FTI_InitExecVars(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt,
        FTIT_injection* FTI_Inje);
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
int FTI_ArchiveL4Ckpt( FTIT_configuration* FTI_Conf, FTIT_execution *FTI_Exec, FTIT_checkpoint *FTI_Ckpt,
        FTIT_topology *FTI_Topo );
void FTI_PrintStatus( FTIT_execution *FTI_Exec, FTIT_topology *FTI_Topo, int ID, int source );

#endif
