#ifndef HDF5_FUNC_H
#define HDF5_FUNC_H

#include <fti-int/types.h>

int FTI_WriteHDF5(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
                  FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt,
                  FTIT_dataset* FTI_Data);
int FTI_RecoverHDF5(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec, FTIT_checkpoint* FTI_Ckpt,
                    FTIT_dataset* FTI_Data);
int FTI_RecoverVarHDF5(FTIT_execution* FTI_Exec, FTIT_checkpoint* FTI_Ckpt,
                        FTIT_dataset* FTI_Data, int id);
int FTI_WriteHDF5Var(FTIT_dataset* FTI_DataVar);
int FTI_CheckHDF5File(char* fn, long fs, char* checksum);
int FTI_OpenGlobalDatasets( FTIT_execution* FTI_Exec );
herr_t FTI_ReadSharedFileData( FTIT_dataset FTI_Data );
int FTI_H5CheckSingleFile( FTIT_configuration* FTI_Conf, int * ckptID );
int FTI_ScanGroup( hid_t gid, char* fn );
int FTI_CheckDimensions( FTIT_dataset * FTI_Data, FTIT_execution * FTI_Exec );
void FTI_FreeVPRMem( FTIT_execution* FTI_Exec, FTIT_dataset* FTI_Data ); 

herr_t FTI_WriteSharedFileData( FTIT_dataset FTI_Data );
void FTI_CreateComplexType(FTIT_type* ftiType, FTIT_type** FTI_Type);
void FTI_CloseComplexType(FTIT_type* ftiType, FTIT_type** FTI_Type);
void FTI_CreateGroup(FTIT_H5Group* ftiGroup, hid_t parentGroup, FTIT_H5Group** FTI_Group);
void FTI_OpenGroup(FTIT_H5Group* ftiGroup, hid_t parentGroup, FTIT_H5Group** FTI_Group);
void FTI_CloseGroup(FTIT_H5Group* ftiGroup, FTIT_H5Group** FTI_Group);
int FTI_CreateGlobalDatasets( FTIT_execution* FTI_Exec );
int FTI_CloseGlobalDatasets( FTIT_execution* FTI_Exec );

#endif // HDF5_FUNC_H
