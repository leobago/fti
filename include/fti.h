/**
 *  @file   fti.h
 *  @author Leonardo A. Bautista Gomez (leobago@gmail.com)
 *  @date   July, 2013
 *  @brief  Header file for the FTI library.
 */

#ifndef _FTI_H
#define _FTI_H

#include <mpi.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef GPUSUPPORT
#include <cuda_runtime_api.h>
#endif

#define _FTI_PUBLIC

/*---------------------------------------------------------------------------
  Defines
  ---------------------------------------------------------------------------*/

///** Standard size of buffer and max node size.                             */
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

#include <fti-int/interface.h>

#ifdef ENABLE_HDF5 // --> If HDF5 is installed
#include "hdf5.h"
#endif
// need this parameter in one fti api function
#ifndef ENABLE_HDF5
typedef size_t 	hsize_t;
#endif

#ifdef __cplusplus
extern "C" {
#endif

  /*---------------------------------------------------------------------------
    Global variables
    ---------------------------------------------------------------------------*/

  /** MPI communicator that splits the global one into app and FTI appart.   */
  extern MPI_Comm FTI_COMM_WORLD;

  /** FTI data type for chars.                                               */
  extern FTIT_type FTI_CHAR;
  /** FTI data type for short integers.                                      */
  extern FTIT_type FTI_SHRT;
  /** FTI data type for integers.                                            */
  extern FTIT_type FTI_INTG;
  /** FTI data type for long integers.                                       */
  extern FTIT_type FTI_LONG;
  /** FTI data type for unsigned chars.                                      */
  extern FTIT_type FTI_UCHR;
  /** FTI data type for unsigned short integers.                             */
  extern FTIT_type FTI_USHT;
  /** FTI data type for unsigned integers.                                   */
  extern FTIT_type FTI_UINT;
  /** FTI data type for unsigned long integers.                              */
  extern FTIT_type FTI_ULNG;
  /** FTI data type for single floating point.                               */
  extern FTIT_type FTI_SFLT;
  /** FTI data type for double floating point.                               */
  extern FTIT_type FTI_DBLE;
  /** FTI data type for long doble floating point.                           */
  extern FTIT_type FTI_LDBE;

  /*---------------------------------------------------------------------------
    FTI public functions
    ---------------------------------------------------------------------------*/

  int FTI_Init(char *configFile, MPI_Comm globalComm);
  int FTI_Status();
  int FTI_InitType(FTIT_type* type, int size);
  int FTI_InitComplexType(FTIT_type* newType, FTIT_complexType* typeDefinition, int length,
      size_t size, char* name, FTIT_H5Group* h5group);
  void FTI_AddSimpleField(FTIT_complexType* typeDefinition, FTIT_type* ftiType,
      size_t offset, int id, char* name);
  void FTI_AddComplexField(FTIT_complexType* typeDefinition, FTIT_type* ftiType,
      size_t offset, int rank, int* dimLength, int id, char* name);
  int FTI_InitGroup(FTIT_H5Group* h5group, char* name, FTIT_H5Group* parent);
  int FTI_RenameGroup(FTIT_H5Group* h5group, char* name);
  int FTI_Protect(int id, void* ptr, long count, FTIT_type type);
  int FTI_DefineDataset(int id, int rank, int* dimLength, char* name, FTIT_H5Group* h5group);
  int FTI_DefineGlobalDataset(int id, int rank, hsize_t* dimLength, char* name, FTIT_H5Group* h5group, FTIT_type type);
  int FTI_AddSubset( int id, int rank, hsize_t* offset, hsize_t* count, int did );
  long FTI_GetStoredSize(int id);
  void* FTI_Realloc(int id, void* ptr);
  int FTI_BitFlip(int datasetID);
  int FTI_Checkpoint(int id, int level);
  int FTI_GetStageDir( char* stageDir, int maxLen );
  int FTI_GetStageStatus( int ID );
  int FTI_SendFile( char* lpath, char *rpath );
  int FTI_Recover();
  int FTI_Snapshot();
  int FTI_Finalize();
  int FTI_RecoverVar(int id);
  int FTI_InitICP(int id, int level, bool activate);
  int FTI_AddVarICP( int varID ); 
  int FTI_FinalizeICP(); 

#ifdef __cplusplus
}
#endif

#endif /* ----- #ifndef _FTI_H  ----- */
