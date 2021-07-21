/**
 *  Copyright (c) 2017 Leonardo A. Bautista-Gomez
 *  All rights reserved
 *
 *  @file   hdf5-fti.h
 */

#ifndef FTI_SRC_IO_HDF5_FTI_H_
#define FTI_SRC_IO_HDF5_FTI_H_

#define FTI_HDF5_MAX_DIM 32

#ifdef ENABLE_HDF5

int FTI_ActivateHeadsHDF5(FTIT_configuration* FTI_Conf,
    FTIT_execution* FTI_Exec, FTIT_topology* FTI_Topo,
    FTIT_checkpoint* FTI_Ckpt, int status);
void *FTI_InitHDF5(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
 FTIT_topology* FTI_Topo, FTIT_checkpoint *FTI_Ckpt, FTIT_keymap *FTI_Data);
int FTI_RecoverHDF5(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
 FTIT_checkpoint* FTI_Ckpt, FTIT_keymap* FTI_Data);
int FTI_RecoverVarHDF5(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
 FTIT_checkpoint* FTI_Ckpt, FTIT_keymap* FTI_Data, int id);
int FTI_RecoverVarInitHDF5(FTIT_configuration* FTI_Conf,
 FTIT_execution* FTI_Exec, FTIT_checkpoint* FTI_Ckpt);
int FTI_RecoverVarFinalizeHDF5(FTIT_configuration* FTI_Conf,
 FTIT_execution* FTI_Exec, FTIT_checkpoint* FTI_Ckpt, FTIT_keymap* FTI_Data);
int FTI_ReadHDF5Var(FTIT_dataset *data);
int FTI_GetDatasetRankReco(hid_t did);
int FTI_GetDatasetSpanReco(hid_t did, hsize_t * span);
int FTI_WriteHDF5Var(FTIT_dataset *data, FTIT_execution* FTI_Exec);
int FTI_CheckHDF5File(char* fn, uint64_t fs, char* checksum);
int FTI_OpenGlobalDatasets(FTIT_execution* FTI_Exec);
herr_t FTI_ReadSharedFileData(FTIT_dataset FTI_Data);
int FTI_H5CheckSingleFile(FTIT_configuration* FTI_Conf, int * ckptID);
int FTI_ScanGroup(hid_t gid, char* fn);
int FTI_CheckDimensions(FTIT_keymap * FTI_Data, FTIT_execution * FTI_Exec);
void FTI_FreeVPRMem(FTIT_execution* FTI_Exec, FTIT_keymap* FTI_Data);
herr_t FTI_WriteSharedFileData(FTIT_dataset FTI_Data);
void FTI_CreateComplexType(FTIT_Datatype* ftiType);
void FTI_CloseComplexType(FTIT_Datatype* ftiType);
void FTI_CreateGroup(FTIT_H5Group* ftiGroup, hid_t parentGroup,
 FTIT_H5Group** FTI_Group);
void FTI_OpenGroup(FTIT_H5Group* ftiGroup, hid_t parentGroup,
 FTIT_H5Group** FTI_Group);
void FTI_CloseGroup(FTIT_H5Group* ftiGroup, FTIT_H5Group** FTI_Group);
int FTI_CreateGlobalDatasets(FTIT_execution* FTI_Exec);
int FTI_CreateGlobalDatasetsAsGroups(FTIT_execution* FTI_Exec);
int FTI_CloseGlobalDatasets(FTIT_execution* FTI_Exec);
int FTI_CloseGlobalDatasetsAsGroups(FTIT_execution* FTI_Exec);
int FTI_FlushH5SingleFile(FTIT_execution* FTI_Exec,
 FTIT_configuration* FTI_Conf, FTIT_topology* FTI_Topo);
int FTI_FinalizeH5SingleFile(FTIT_execution* FTI_Exec,
 FTIT_configuration* FTI_Conf, FTIT_topology* FTI_Topo,
 FTIT_checkpoint* FTI_Ckpt, double t);
#endif

#endif  // FTI_SRC_IO_HDF5_FTI_H_
