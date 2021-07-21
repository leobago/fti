/**
 *  Copyright (c) 2017 Leonardo A. Bautista-Gomez
 *  All rights reserved
 *
 *  @file   meta.h
 */

#ifndef FTI_SRC_META_H_
#define FTI_SRC_META_H_

#include "interface.h"

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
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt, uint64_t* fs,
        uint64_t mfs, char* fnl, char* checksums, int* allVarIDs,
        int* allRanks, uint64_t* allCounts,
        int* allVarTypeIDs, int* allVarTypeSizes,
        uint64_t* allVarSizes, uint64_t* allLayerSizes, char* allLayerHashes,
        uint64_t *allVarPositions, char *allNames, char *allCharIds);
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

#endif  // FTI_SRC_META_H_
