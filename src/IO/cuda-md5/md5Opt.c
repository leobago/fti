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
 *  @author Konstantinos Parasyris (konstantinos.parasyris@bsc.es)
 *  @file   diff-checkpoint.c
 *  @date   February, 2018
 *  @brief  Routines to compute the MD5 checksum  
 */


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include <fti.h>
#include "md5Opt.h"
#include "../../interface.h"
#define CPU 1
#define GPU 2
#define CFILE 3

int MD5GPU(FTIT_dataset *);
int MD5CPU(FTIT_dataset *);
int usesAsync = 0;

pthread_t thread;
pthread_mutex_t worker;
pthread_mutex_t application;
int32_t totalWork = 0;
int32_t worker_exit = 0;
int deviceId;
unsigned char* (*cpuHash)(const unsigned char *data, uint64_t nBytes,
 unsigned char *hash);
int32_t tempBufferSize;
int32_t md5ChunkSize;


/*-------------------------------------------------------------------------*/
/**
  @brief     This function initializes the MD5 checksum functions  for DCP
  @param     cSize Size of the chunk  
  @param     tempSize Size of intermediate buffers (Not used in this file) 
  @param     FTI_Conf Pointer to the configuration options 
  @return     integer         FTI_SCES if successfu.

  This function initializes parameters for the computation of DCP MD5 checksums
 **/
/*-------------------------------------------------------------------------*/
int FTI_initMD5(int32_t cSize, int32_t tempSize, FTIT_configuration *FTI_Conf) {
    if (FTI_Conf->dcpInfoPosix.cachedCkpt)
        usesAsync = 1;
    else
        usesAsync = 0;

    cpuHash = FTI_Conf->dcpInfoPosix.hashFunc;
    tempBufferSize = tempSize;
    md5ChunkSize = cSize;
    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief     This function computes the checksums of an Protected Variable 
  @param     data Variable We need to compute the checksums
  @return     integer         FTI_SCES if successfu.

  This function computes the checksums of a specific variable 
 **/
/*-------------------------------------------------------------------------*/
int MD5CPU(FTIT_dataset *data) {
    int64_t dataSize = data->size;
    unsigned char block[md5ChunkSize];
    size_t i;
    unsigned char *ptr = (unsigned char *) data->ptr;
    for (i = 0 ; i < data->size; i+=md5ChunkSize) {
        unsigned int blockId = i/md5ChunkSize;
        unsigned int hashIdx = blockId*16;
        unsigned int chunkSize = ((dataSize-i) < md5ChunkSize) ?
         dataSize-i: md5ChunkSize;
        if (chunkSize < md5ChunkSize) {
            memset(block, 0x0, md5ChunkSize);
            memcpy(block, &ptr[i], chunkSize);
            cpuHash(block, md5ChunkSize,
             &data->dcpInfoPosix.currentHashArray[hashIdx]);
        } else {
            cpuHash(&ptr[i], md5ChunkSize,
             &data->dcpInfoPosix.currentHashArray[hashIdx]);
        }
    }
    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief     This function computes the checksums of a Protected Variable 
  @param     data Variable We need to compute the checksums
  @return     integer         FTI_SCES if successfu.

  This function performs the computation of DCP MD5 checksums
  it is actually an interface between the FTI and the async methods
 **/
/*-------------------------------------------------------------------------*/
int FTI_MD5CPU(FTIT_dataset *data) {
    return MD5CPU(data);
}


/*-------------------------------------------------------------------------*/
/**
  @brief     This function computes the checksums of a Protected Variable (GPU) 
  @param     data Variable We need to compute the checksums
  @return     integer         FTI_SCES if successfu.

  This function initializes parameters for the computation of DCP MD5 checksums
  it is actually an interface between the FTI and the async methods
 **/
/*-------------------------------------------------------------------------*/
int FTI_MD5GPU(FTIT_dataset *data) {
    return 1;
}

/*-------------------------------------------------------------------------*/
/**
  @brief     This function synchronizes the file writes with the stable storages 
  @param     f pointer to the file to be synchronized 
  @return     integer         FTI_SCES if successfull.

    This function is an interface, atm not implemented for NON-GPU optimized 
    checkpoints.
 **/
/*-------------------------------------------------------------------------*/
int FTI_CLOSE_ASYNC(FILE *f) {
    return 1;
}


/*-------------------------------------------------------------------------*/
/**
  @brief     This function synchronizes the computation of the checksum with 
             the current thread
  @return     integer         FTI_SCES if successfu.


  This function should not be called when compiling without GPU optimizations
   **/
/*-------------------------------------------------------------------------*/
int FTI_SyncMD5() {
    return FTI_SCES;
}


/*-------------------------------------------------------------------------*/
/**
  @brief     This function fires the async thread to start computing work 
  @return     integer         FTI_SCES if successfull.

  This function should not be called when compiling without GPU optimizations
 **/
/*-------------------------------------------------------------------------*/
int FTI_startMD5() {
    return FTI_SCES;
}


/*-------------------------------------------------------------------------*/
/**
  @brief     This function destroys the internal MD5 data structures 
  @return     integer         FTI_SCES if successfull.

  This function should not be called when compiling without GPU optimizations
 **/
/*-------------------------------------------------------------------------*/
int FTI_destroyMD5() {
    return FTI_SCES;
}
