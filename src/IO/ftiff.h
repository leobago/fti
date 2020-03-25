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
 *  @author Kai Keller (kellekai@gmx.de)
 *  @file   ftiff.h
 *  @date   October, 2017
 *  @brief  Header file for the FTI File Format (FTI-FF).
 */

#ifndef _FTIFF_H
#define _FTIFF_H

#include "fti.h"
#ifndef FTI_NOZLIB
#   include "zlib.h"
#endif
#include <assert.h>
#include <string.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#define MBR_CNT(TYPE) int TYPE ## _mbrCnt
#define MBR_BLK_LEN(TYPE) int TYPE ## _mbrBlkLen[]
#define MBR_TYPES(TYPE) MPI_Datatype TYPE ## _mbrTypes[]
#define MBR_DISP(TYPE) MPI_Aint TYPE ## _mbrDisp[]

#define CKPT_FN_FORMAT(level, backup) ( ( backup ) ? ( ( level == 2 ) ? "Ckpt%d-Pcof%d.fti" : "Ckpt%d-RSed%d.fti" ) : "Ckpt%d-Rank%d.fti" )

extern int FTI_filemetastructsize;	/**< size of FTIFF_metaInfo in file */
extern int FTI_dbstructsize;		/**< size of FTIFF_db in file       */
extern int FTI_dbvarstructsize;		/**< size of FTIFF_dbvar in file    */

extern char *filemmap; 
extern struct stat filestats;

/**

  +-------------------------------------------------------------------------+
  |   FTI-FF TYPES                                                          |
  +-------------------------------------------------------------------------+

 **/

/** @typedef    dcpBLK_t
 *  @brief      unsigned short (0 - 65535).
 *  
 *  Type that keeps the block sizes inside the hash meta data. 
 *  unsigned short is a trade off between memory occupation and block 
 *  size range.
 */
typedef unsigned short dcpBLK_t;

/** @typedef    FTIFF_headInfo
 *  @brief      Runtime meta info for the heads.
 *
 *  keeps all necessary meta data information for the heads, in order to
 *  perform the checkpoint.
 *
 */
typedef struct FTIFF_headInfo {
    int exists;
    int nbVar;
    char ckptFile[FTI_BUFS];
    long maxFs;
    long fs;
    long pfs;
    int isDcp;
} FTIFF_headInfo;

/** @typedef    FTIFF_RecoveryInfo
 *  @brief      Meta data for L2 recovery.
 *
 *  keeps meta data information that needs to be exchanged between the ranks.
 *
 */
typedef struct FTIFF_RecoveryInfo {
    int FileExists;
    int BackupExists;
    int ckptId;
    int rightIdx;
    long maxFs;
    long fs;
    long bfs;
} FTIFF_RecoveryInfo;

/**

  +-------------------------------------------------------------------------+
  |   MPI DERIVED DATA TYPES                                                |
  +-------------------------------------------------------------------------+

 **/

// ID MPI types
enum {
    FTIFF_HEAD_INFO,
    FTIFF_RECO_INFO,
    FTIFF_NUM_MPI_TYPES
};

// declare MPI datatypes
extern MPI_Datatype FTIFF_MpiTypes[FTIFF_NUM_MPI_TYPES];

typedef struct FTIFF_MPITypeInfo {
    MPI_Datatype        raw;
    int                 mbrCnt;
    int*                mbrBlkLen;
    MPI_Datatype*       mbrTypes;
    MPI_Aint*           mbrDisp;
} FTIFF_MPITypeInfo;

/**

  +-------------------------------------------------------------------------+
  |   FUNCTION DECLARATIONS                                                 |
  +-------------------------------------------------------------------------+

 **/
int FTI_FinalizeFtiff( void *fd );
size_t FTI_DummyFilePos(void *ignore);
int FTI_WriteFtiffData( FTIT_dataset* data, void *fd );
void* FTI_InitFtiff( FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt,
        FTIT_keymap* FTI_Data );
void FTIFF_InitMpiTypes();
int FTIFF_DeserializeFileMeta( FTIFF_metaInfo* meta, char* buffer_ser );
int FTIFF_DeserializeDbMeta( FTIFF_db* db, char* buffer_ser );
int FTIFF_DeserializeDbVarMeta( FTIFF_dbvar* dbvar, char* buffer_ser );
int FTIFF_SerializeFileMeta( FTIFF_metaInfo* meta, char* buffer_ser );
int FTIFF_SerializeDbMeta( FTIFF_db* db, char* buffer_ser );
int FTIFF_SerializeDbVarMeta( FTIFF_dbvar* dbvar, char* buffer_ser );
void FTIFF_FreeDbFTIFF(FTIFF_db* last);
int FTIFF_Recover( FTIT_execution *FTI_Exec, FTIT_keymap *FTI_Data, FTIT_checkpoint *FTI_Ckpt );
int FTIFF_RecoverVar(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec, FTIT_topology* FTI_Topo, 
    FTIT_checkpoint *FTI_Ckpt, FTIT_keymap *FTI_Data, int id);
int FTIFF_UpdateDatastructVarFTIFF( FTIT_execution* FTI_Exec, 
        FTIT_dataset* data, FTIT_configuration* FTI_Conf );
int FTIFF_ReadDbFTIFF( FTIT_configuration *FTI_Conf, FTIT_execution *FTI_Exec, FTIT_checkpoint* FTI_Ckpt, FTIT_keymap* FTI_Data );
int FTIFF_LoadMetaPostprocessing( FTIT_execution* FTI_Exec, FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt, FTIT_configuration* FTI_Conf, int proc );
int FTIFF_GetFileChecksum( FTIFF_metaInfo *FTIFF_Meta, int fd, char *checksum );
int FTIFF_createHashesDbVarFTIFF( FTIT_execution* FTI_Exec );
int FTIFF_finalizeDatastructFTIFF( FTIT_execution* FTI_Exec );
int FTIFF_writeMetaDataFTIFF( FTIT_execution* FTI_Exec, WriteFTIFFInfo_t *fd );
int FTIFF_CreateMetadata( FTIT_execution* FTI_Exec, FTIT_topology* FTI_Topo,
        FTIT_configuration* FTI_Conf );
int FTIFF_RequestFileName( char* dir, int rank, int level, int dcp, int backup, char* fn );
int FTIFF_LoadFileMeta(int fd, FTIFF_metaInfo* fm );
int FTIFF_OpenCkptFile(char* fn, int oflag);
int FTIFF_GetEncodedFileChecksum( FTIFF_metaInfo *FTIFFMeta, int fd, char *checksum ); 
int FTIFF_RequestRecoveryInfo( FTIFF_RecoveryInfo* info, char* dir, int rank, int level, bool dcp, bool backup );
int FTIFF_CheckL1RecoverInit( FTIT_execution* FTI_Exec, FTIT_topology* FTI_Topo,
        FTIT_checkpoint* FTI_Ckpt, FTIT_configuration* FTI_Conf );
int FTIFF_CheckL2RecoverInit( FTIT_execution* FTI_Exec, FTIT_topology* FTI_Topo,
        FTIT_checkpoint* FTI_Ckpt, FTIT_configuration* FTI_Conf, int *exists);
int FTIFF_CheckL3RecoverInit( FTIT_execution* FTI_Exec, FTIT_topology* FTI_Topo,
        FTIT_checkpoint* FTI_Ckpt, int* erased);
int FTIFF_CheckL4RecoverInit( FTIT_execution* FTI_Exec, FTIT_topology* FTI_Topo,
        FTIT_checkpoint* FTI_Ckpt);
void FTIFF_GetHashMetaInfo( unsigned char *hash, FTIFF_metaInfo *FTIFFMeta );
void FTIFF_GetHashdb( unsigned char *hash, FTIFF_db *db );
void FTIFF_GetHashdbvar( unsigned char *hash, FTIFF_dbvar *dbvar );
void FTIFF_SetHashChunk( FTIFF_dbvar *dbvar, FTIT_keymap* FTI_Data ); 
void FTIFF_PrintDataStructure( int rank, FTIT_execution* FTI_Exec );
int FTI_ProcessDBVar(FTIT_execution *FTI_Exec, FTIT_configuration *FTI_Conf, FTIFF_dbvar *currentdbvar, 
        FTIT_dataset *data, unsigned char *hashchk, WriteFTIFFInfo_t *fd, long *dcpSize, unsigned char **dptr);
int FTI_RecoverVarInitFTIFF(char* fn);
int FTI_RecoverVarFinalizeFTIFF();
#endif
