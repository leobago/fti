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

#include "fti.h"

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

#include <stdint.h>
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

#ifdef LUSTRE
#   include "lustreapi.h"
#endif

/*---------------------------------------------------------------------------
  Defines
  ---------------------------------------------------------------------------*/

/** Malloc macro.                                                          */
#define talloc(type, num) (type *)malloc(sizeof(type) * (num))

typedef uintptr_t           FTI_ADDRVAL;        /**< for ptr manipulation       */
typedef void*               FTI_ADDRPTR;        /**< void ptr type              */ 

// datablock size in file
extern int FTI_dbstructsize;		    /**< size of FTIT_db struct in file */
//    = sizeof(int)     /* numvars */
//    + sizeof(long);   /* dbsize */


/** @typedef    FTIT_execution
 *  @brief      Execution metadata.
 *
 *  This type stores all the dynamic metadata related to the current execution
 */
typedef struct FTIT_execution {
    char            id[FTI_BUFS];       /**< Execution ID.                  */
    int             ckpt;               /**< Checkpoint flag.               */
    int             reco;               /**< Recovery flag.                 */
    int             ckptLvel;           /**< Checkpoint level.              */
    int             ckptIntv;           /**< Ckpt. interval in minutes.     */
    int             lastCkptLvel;       /**< Last checkpoint level.         */
    int             wasLastOffline;     /**< TRUE if last ckpt. offline.    */
    double          iterTime;           /**< Current wall time.             */
    double          lastIterTime;       /**< Time spent in the last iter.   */
    double          meanIterTime;       /**< Mean iteration time.           */
    double          globMeanIter;       /**< Global mean iteration time.    */
    double          totalIterTime;      /**< Total main loop time spent.    */
    unsigned int    syncIter;           /**< To check mean iter. time.      */
    int             syncIterMax;        /**< Maximal synch. intervall.      */
    unsigned int    minuteCnt;          /**< Checkpoint minute counter.     */
    unsigned int    ckptCnt;            /**< Checkpoint number counter.     */
    unsigned int    ckptIcnt;           /**< Iteration loop counter.        */
    unsigned int    ckptID;             /**< Checkpoint ID.                 */
    unsigned int    ckptNext;           /**< Iteration for next checkpoint. */
    unsigned int    ckptLast;           /**< Iteration for last checkpoint. */
    long            ckptSize;           /**< Checkpoint size.               */
    unsigned int    nbVar;              /**< Number of protected variables. */
    unsigned int    nbVarStored;        /**< Nr. prot. var. stored in file  */
    unsigned int    nbType;             /**< Number of data types.          */
    int             nbGroup;            /**< Number of protected groups.    */
    int             metaAlloc;          /**< True if meta allocated.        */
    int             initSCES;           /**< True if FTI initialized.       */
    FTIT_metadata   meta[5];            /**< Metadata for each ckpt level   */
    FTIFF_db         *firstdb;          /**< Pointer to first datablock     */
    FTIFF_db         *lastdb;           /**< Pointer to first datablock     */
    FTIFF_metaInfo  FTIFFMeta;          /**< File meta data for FTI-FF      */
    FTIT_type**     FTI_Type;           /**< Pointer to FTI_Types           */
    FTIT_H5Group**  H5groups;           /**< HDF5 root group.               */
    MPI_Comm        globalComm;         /**< Global communicator.           */
    MPI_Comm        groupComm;          /**< Group communicator.            */
    cudaStream_t    cStream;            /**< CUDA stream.                   */
    cudaEvent_t     cEvents[2];         /**< CUDA event.                    */
    void*           cHostBufs[2];       /**< CUDA host buffer.              */
    MD5_CTX         mdContext;          /**< MD5 Checksum.                  */
    unsigned int    nbKernels;          /**< Number of protected kernels    */
    FTIT_gpuInfo    gpuInfo[FTI_BUFS];  /**< Info about GPU interruptions   */
  } FTIT_execution;


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
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt)
        __attribute__((noreturn));

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

#ifdef ENABLE_HDF5
    int FTI_WriteHDF5(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
                      FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt,
                      FTIT_dataset* FTI_Data);
    int FTI_RecoverHDF5(FTIT_execution* FTI_Exec, FTIT_checkpoint* FTI_Ckpt,
                        FTIT_dataset* FTI_Data);
    int FTI_RecoverVarHDF5(FTIT_execution* FTI_Exec, FTIT_checkpoint* FTI_Ckpt,
                            FTIT_dataset* FTI_Data, int id);
#endif

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
#ifdef ENABLE_HDF5
    void FTI_CreateComplexType(FTIT_type* ftiType, FTIT_type** FTI_Type);
    void FTI_CloseComplexType(FTIT_type* ftiType, FTIT_type** FTI_Type);
    void FTI_CreateGroup(FTIT_H5Group* ftiGroup, hid_t parentGroup, FTIT_H5Group** FTI_Group);
    void FTI_OpenGroup(FTIT_H5Group* ftiGroup, hid_t parentGroup, FTIT_H5Group** FTI_Group);
    void FTI_CloseGroup(FTIT_H5Group* ftiGroup, FTIT_H5Group** FTI_Group);
#endif
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

#endif
