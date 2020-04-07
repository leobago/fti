/**
 *  @file   fti.h
 *  @author Leonardo A. Bautista Gomez (leobago@gmail.com)
 *  @date   July, 2013
 *  @brief  Header file for the FTI library.
 */

#ifndef __FTI_H__
#define __FTI_H__

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


#ifdef __cplusplus
extern "C" {
#endif

#include "fti-intern.h"
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

    int FTI_Init(const char *configFile, MPI_Comm globalComm);
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
    int FTI_DefineGlobalDataset(int id, int rank, FTIT_hsize_t* dimLength, const char* name, FTIT_H5Group* h5group, FTIT_type type);
    int FTI_AddSubset( int id, int rank, FTIT_hsize_t* offset, FTIT_hsize_t* count, int did );
    int FTI_RecoverDatasetDimension( int did ); 
    FTIT_hsize_t* FTI_GetDatasetSpan( int did, int rank );
    int FTI_GetDatasetRank( int did );
    int FTI_UpdateGlobalDataset(int id, int rank, FTIT_hsize_t* dimLength );
    int FTI_UpdateSubset( int id, int rank, FTIT_hsize_t* offset, FTIT_hsize_t* count, int did );
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
    int FTI_setIDFromString( char *name );
    int FTI_getIDFromString( char *name );
    int FTI_Finalize_ReInit();
    int FTI_GetNodeID();
    int FTI_GetGroupSize();
    int FTI_GetNodeSize();
    int FTI_isSimulatedExecution();
    char *FTI_GetLocalDirectory();
    int FTI_InitOpt(const char *config, MPI_Comm globalComm, int *failedNOdes, int numFailedNodes );
#ifdef __cplusplus
}
#endif

#endif // __FTI_H__
