/**
 *  Copyright (c) 2017 Leonardo A. Bautista-Gomez
 *  All rights reserved
 *
 *  @file   meta.h
 */

#ifndef FTI_META_H_
#define FTI_META_H_

int FTI_GetChecksums(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt,
        char* checksum, char* ptnerChecksum, char* rsChecksum);
int FTI_WriteRSedChecksum(FTIT_configuration* FTI_Conf,
        FTIT_execution* FTI_Exec, FTIT_topology* FTI_Topo,
        FTIT_checkpoint* FTI_Ckpt, int rank, char* checksum);
int FTI_LoadMetaPostprocessing(FTIT_configuration* FTI_Conf,
        FTIT_execution* FTI_Exec, FTIT_topology* FTI_Topo,
        FTIT_checkpoint* FTI_Ckpt, int proc);
int FTI_LoadMetaRecovery(FTIT_configuration* FTI_Conf,
        FTIT_execution* FTI_Exec, FTIT_topology* FTI_Topo,
        FTIT_checkpoint* FTI_Ckpt);
int FTI_LoadMetaDcp(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt);
int FTI_LoadMetaDataset(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt,
        FTIT_keymap* FTI_Data);
int FTI_WriteMetadata(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt, int32_t* fs,
        int32_t mfs, char* fnl, char* checksums, int* allVarIDs,
        int* allVarTypeIDs, int* allVarTypeSizes,
        int32_t* allVarSizes, uint32_t* allLayerSizes, char* allLayerHashes,
        int32_t *allVarPositions, char *allCharIds);
int FTI_CreateMetadata(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt,
        FTIT_keymap* FTI_Data);
int FTI_WriteCkptMetaData(FTIT_configuration* FTI_Conf,
        FTIT_execution* FTI_Exec, FTIT_topology* FTI_Topo,
        FTIT_checkpoint* FTI_Ckpt);
int FTI_LoadCkptMetaData(FTIT_configuration* FTI_Conf,
        FTIT_execution* FTI_Exec, FTIT_topology* FTI_Topo,
        FTIT_checkpoint* FTI_Ckpt);
int FTI_LoadL4CkptMetaData(FTIT_configuration* FTI_Conf,
        FTIT_execution* FTI_Exec, FTIT_topology* FTI_Topo,
        FTIT_checkpoint* FTI_Ckpt);

#endif  // FTI_META_H_
