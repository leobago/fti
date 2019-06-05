#ifndef __HDF5_FTI_H__
#define __HDF5_FTI_H__

#define FTI_HDF5_MAX_DIM 32

#ifdef ENABLE_HDF5
int FTI_RecoverHDF5(void);
int FTI_RecoverVarHDF5(int id);
int FTI_GetDatasetRankReco( hid_t did );
int FTI_GetDatasetSpanReco( hid_t did, hsize_t * span );
int FTI_WriteHDF5Var(FTIT_dataset* FTI_DataVar);
int FTI_CheckHDF5File(char* fn, long fs, char* checksum);
int FTI_OpenGlobalDatasets(void);
herr_t FTI_ReadSharedFileData( FTIT_dataset data );
int FTI_H5CheckSingleFile( int * ckptID );
int FTI_ScanGroup( hid_t gid, char* fn );
int FTI_CheckDimensions(void);
void FTI_FreeVPRMem(void); 
int FTI_CommitDataType(FTIT_dataset *FTI_DataVar);
herr_t FTI_WriteSharedFileData( FTIT_dataset );
void FTI_CreateComplexType(FTIT_type* ftiType, FTIT_type** FTI_Type);
void FTI_CloseComplexType(FTIT_type* ftiType, FTIT_type** FTI_Type);
void FTI_CreateGroup(FTIT_H5Group* ftiGroup, hid_t parentGroup, FTIT_H5Group** FTI_Group);
void FTI_OpenGroup(FTIT_H5Group* ftiGroup, hid_t parentGroup, FTIT_H5Group** FTI_Group);
void FTI_CloseGroup(FTIT_H5Group* ftiGroup, FTIT_H5Group** FTI_Group);
int FTI_CreateGlobalDatasets(void);
int FTI_CloseGlobalDatasets(void);
#endif

#endif // __HDF5_FTI_H__
