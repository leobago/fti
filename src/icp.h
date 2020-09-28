/**
 *  Copyright (c) 2017 Leonardo A. Bautista-Gomez
 *  All rights reserved
 *
 *  @file   icp.h
 */

#ifndef FTI_SRC_ICP_H_
#define FTI_SRC_ICP_H_

#include <mpi.h>
#include <stdio.h>
#ifdef ENABLE_HDF5
#include <hdf5.h>
#include <hdf5_hl.h>
#endif

#include "interface.h"

#define FTI_ICP_NINI 0
#define FTI_ICP_ACTV 1
#define FTI_ICP_FAIL 2

int FTI_startICP(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt,
        FTIT_keymap* FTI_Data, FTIT_IO *io);
int FTI_WriteVar(int varID, FTIT_configuration* FTI_Conf,
        FTIT_execution* FTI_Exec, FTIT_topology* FTI_Topo,
        FTIT_checkpoint* FTI_Ckpt, FTIT_keymap* FTI_Data, FTIT_IO *io);
int FTI_FinishICP(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt,
        FTIT_keymap* FTI_Data, FTIT_IO *io);

#if 0
int FTI_WriteSionlibVar(int varID, FTIT_configuration* FTI_Conf,
        FTIT_execution* FTI_Exec, FTIT_topology* FTI_Topo,
        FTIT_checkpoint* FTI_Ckpt, FTIT_keymap* FTI_Data);

int FTI_InitSionlibICP(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt,
        FTIT_keymap* FTI_Data);

int FTI_FinalizeSionlibICP(FTIT_configuration* FTI_Conf,
        FTIT_execution* FTI_Exec, FTIT_topology* FTI_Topo,
        FTIT_checkpoint* FTI_Ckpt, FTIT_keymap* FTI_Data);
#endif

#endif  // FTI_SRC_ICP_H_
