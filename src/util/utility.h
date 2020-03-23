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
 *  @file   utility.h
 *  @date   October, 2017
 *  @brief  Utility functions for the FTI library.
 */

#ifndef __UTILITY__
#define __UTILITY__

#include <fti.h>
#include "../deps/md5/md5.h"

typedef struct{
    FTIT_configuration* FTI_Conf;   // Configuration of the FTI
    FTIT_topology *FTI_Topo;        // Topology of the nodes
    MPI_Offset offset;              // Offset of the Rank in the file
    size_t loffset;                 // Offset in the local file
    int err;                        // Errors
    MPI_Info info;                  // MPI info of the file
    MPI_File pfh;                   // File descriptor
    char flag;                      // Flags used to open the file
    MD5_CTX integrity;              // integrity of the file
} WriteMPIInfo_t;

typedef struct{
    FILE *f;                        // Posix file descriptor
    size_t offset;                  // offset in the file
    char flag;                      // flags to open the file
    MD5_CTX integrity;              // integrity of the file
}WritePosixInfo_t;

#ifdef ENABLE_IME_NATIVE
typedef struct{
    int f;                          // IME native file descriptor
    size_t offset;                  // offset in the file
    int flag;                       // flags to open the file
    mode_t mode;                    // mode the file has been opened
    MD5_CTX integrity;              // integrity of the file
}WriteIMEInfo_t;
#endif

typedef struct{
    WritePosixInfo_t write_info;    // Posix Write info descriptor 
    FTIT_configuration *FTI_Conf;   // FTI Configuration
    FTIT_checkpoint *FTI_Ckpt;      // FTI Checkpoint options
    FTIT_execution *FTI_Exec;       // FTI execution options
    FTIT_topology *FTI_Topo;        // FTI node topology
    size_t layerSize;               // size of the dcp layer
}WriteDCPPosixInfo_t;

typedef struct{
    FILE *f;                        // Posix file descriptor
    size_t offset;                  // offset in the file
    char flag;                      // flags to open the file
    MD5_CTX integrity;              // integrity of the file
    FTIT_configuration *FTI_Conf;   // FTI Configuration
    FTIT_checkpoint *FTI_Ckpt;      // FTI Checkpoint options
    FTIT_execution *FTI_Exec;       // FTI execution options
    FTIT_topology *FTI_Topo;        // FTI node topology
    FTIT_keymap *FTI_Data;
}WriteFTIFFInfo_t;

#ifdef ENABLE_HDF5
typedef struct{
    FTIT_execution *FTI_Exec;       // Execution environment
    FTIT_keymap *FTI_Data;         // FTI Data
    FTIT_topology *FTI_Topo;         // FTI Data
    FTIT_configuration *FTI_Conf;         // FTI Data
    hid_t file_id;                  // File Id
}WriteHDF5Info_t;

int FTI_HDF5Open(char *fn, void *fileDesc);
int FTI_HDF5Close(void *fileDesc);
void *FTI_InitHDF5(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec, FTIT_topology* FTI_Topo, FTIT_checkpoint *FTI_Ckpt, FTIT_keymap *FTI_Data);
int FTI_WriteHDF5Data(FTIT_dataset * data, void *write_info);
int FTI_WriteHDF5(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt,
        FTIT_keymap* FTI_Data);
size_t FTI_GetHDF5FilePos(void *);
#endif

#ifdef ENABLE_SIONLIB
#include <sion.h>
typedef struct{
    int sid;
    int *file_map;
    int *ranks;
    int *rank_map;
    size_t loffset;
    sion_int64* chunkSizes;
}WriteSionInfo_t;

int FTI_WriteSionData(FTIT_dataset * data, void *fd);
void* FTI_InitSion(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec, FTIT_topology* FTI_Topo, FTIT_checkpoint *FTI_Ckpt, FTIT_keymap *FTI_Data);
int FTI_SionClose(void *fileDesc);
size_t FTI_GetSionFilePos(void *fileDesc);
#endif

// Wrappers around MPIO
int FTI_MPIOOpen(char *fn, void *fileDesc);
int FTI_MPIOClose(void *fileDesc);
int FTI_MPIOWrite(void *src, size_t size, void *fileDesc);
int FTI_MPIORead(void *src, size_t size, void *fileDesc);
size_t FTI_GetMPIOFilePos(void *fileDesc);

void *FTI_InitMPIO(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec, FTIT_topology* FTI_Topo, FTIT_checkpoint *FTI_Ckpt, FTIT_keymap *FTI_Data);
int FTI_WriteMPIOData(FTIT_dataset * data, void *write_info);

//Wrappers around dcp POSIX

size_t FTI_GetDCPPosixFilePos(void *fileDesc);
void *FTI_InitDCPPosix(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec, FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt, FTIT_keymap* FTI_Data);
int FTI_WritePosixDCPData(FTIT_dataset *data, void *fd);
int FTI_PosixDCPClose(void *fileDesc);

int copyDataFromDevive(FTIT_execution* FTI_Exec, FTIT_keymap* FTI_Data);



#ifdef ENABLE_SIONLIB 
int write_sion(void *src, size_t size, void *opaque);
#endif
#endif
