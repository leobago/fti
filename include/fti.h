/**
 *  Copyright (c) 2017 Leonardo A. Bautista-Gomez
 *  All rights reserved
 *
 *  @file   fti.h
 *  @author Leonardo A. Bautista Gomez (leobago@gmail.com)
 *  @date   July, 2013
 *  @brief  Header file for the FTI library.
 */

#ifndef FTI_INCLUDE_FTI_H_
#define FTI_INCLUDE_FTI_H_

/** Standard size of buffer and max node size.                             */
#define FTI_BUFS 256
/** Word size used during RS encoding.                                     */
#define FTI_WORD 16
/** Token returned when FTI performs a checkpoint.                         */
#define FTI_DONE 1
/** Token returned if a FTI function succeeds.                             */
#define FTI_SCES 0
/** Token returned if a FTI function fails.                                */
#define FTI_NSCS -1
/** Token returned if recovery fails.                                      */
#define FTI_NREC -2
/** Token that indicates a head process in user space                      */
#define FTI_HEAD 2

/** status 'failed' for stage requests                                     */
#define FTI_SI_FAIL 0x4
/** status 'succeed' for stage requests                                    */
#define FTI_SI_SCES 0x3
/** status 'active' for stage requests                                     */
#define FTI_SI_ACTV 0x2
/** status 'pending' for stage requests                                    */
#define FTI_SI_PEND 0x1
/** status 'not initialized' for stage requests                            */
#define FTI_SI_NINI 0x0

/** Identifier abstraction for FTI internal objects                        */
typedef int fti_id_t;
/** FTI v1.4 and backwards data type handling compatibility                */
#define FTIT_type fti_id_t

#include "fti-intern.h"

#ifdef __cplusplus
extern "C" {
#endif

  /*---------------------------------------------------------------------------
    Global variables
    ---------------------------------------------------------------------------*/

  /** MPI communicator that splits the global one into app and FTI appart.   */
  extern MPI_Comm FTI_COMM_WORLD;

  /** FTI data type for chars.                                               */
  extern fti_id_t FTI_CHAR;
  /** FTI data type for short integers.                                      */
  extern fti_id_t FTI_SHRT;
  /** FTI data type for integers.                                            */
  extern fti_id_t FTI_INTG;
  /** FTI data type for long integers.                                       */
  extern fti_id_t FTI_LONG;
  /** FTI data type for unsigned chars.                                      */
  extern fti_id_t FTI_UCHR;
  /** FTI data type for unsigned short integers.                             */
  extern fti_id_t FTI_USHT;
  /** FTI data type for unsigned integers.                                   */
  extern fti_id_t FTI_UINT;
  /** FTI data type for unsigned long integers.                              */
  extern fti_id_t FTI_ULNG;
  /** FTI data type for single floating point.                               */
  extern fti_id_t FTI_SFLT;
  /** FTI data type for double floating point.                               */
  extern fti_id_t FTI_DBLE;
  /** FTI data type for long doble floating point.                           */
  extern fti_id_t FTI_LDBE;

  /*---------------------------------------------------------------------------
    FTI public functions
    ---------------------------------------------------------------------------*/

  int FTI_Init(const char *configFile, MPI_Comm globalComm);
  int FTI_Status();
  int FTI_InitGroup(FTIT_H5Group* h5group, const char* name,
   FTIT_H5Group* parent);
  int FTI_RenameGroup(FTIT_H5Group* h5group, const char* name);
  int FTI_Protect(int id, void* ptr, int32_t count, fti_id_t tid);
  int FTI_SetAttribute(int id, FTIT_attribute attribute,
          FTIT_attributeFlag flag);
  int FTI_DefineDataset(int id, int rank, int* dimLength, const char* name,
   FTIT_H5Group* h5group);
  int FTI_DefineGlobalDataset(int id, int rank, FTIT_hsize_t* dimLength,
   const char* name, FTIT_H5Group* h5group, fti_id_t tid);
  int FTI_AddSubset(int id, int rank, FTIT_hsize_t* offset,
   FTIT_hsize_t* count, int did);
  int FTI_RecoverDatasetDimension(int did);
  FTIT_hsize_t* FTI_GetDatasetSpan(int did, int rank);
  int FTI_GetDatasetRank(int did);
  int FTI_UpdateGlobalDataset(int id, int rank, FTIT_hsize_t* dimLength);
  int FTI_UpdateSubset(int id, int rank, FTIT_hsize_t* offset,
   FTIT_hsize_t* count, int did);
  int32_t FTI_GetStoredSize(int id);
  void* FTI_Realloc(int id, void* ptr);
  int FTI_BitFlip(int datasetID);
  int FTI_Checkpoint(int id, int level);
  int FTI_GetStageDir(char* stageDir, int maxLen);
  int FTI_GetStageStatus(int ID);
  int FTI_SendFile(const char* lpath, const char *rpath);
  int FTI_Recover();
  int FTI_Snapshot();
  int FTI_Finalize();
  int FTI_RecoverVar(int id);
  int FTI_RecoverVarInit();
  int FTI_RecoverVarFinalize();
  int FTI_InitICP(int id, int level, bool activate);
  int FTI_AddVarICP(int varID);
  int FTI_FinalizeICP();
  int FTI_setIDFromString(const char *name);
  int FTI_getIDFromString(const char *name);
  FTIT_allConfiguration FTI_GetConfig(const char* configFile,
   MPI_Comm globalComm);
  int FTI_RecoverVarInit();
  int FTI_RecoverVarFinalize();

  // FTI data type handling functions
  int FTI_InitType(fti_id_t* type, int size);
  fti_id_t FTI_InitCompositeType(const char* name, size_t size,
   FTIT_H5Group* h5g);
  int FTI_AddScalarField(fti_id_t id, const char* name, fti_id_t fid,
   size_t offset);
  int FTI_AddVectorField(fti_id_t id, const char* name, fti_id_t tid,
   size_t offset, int ndims, int* dim_sizes);

#ifdef __cplusplus
}
#endif

#endif  // FTI_INCLUDE_FTI_H_
