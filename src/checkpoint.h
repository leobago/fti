/**
 *  Copyright (c) 2017 Leonardo A. Bautista-Gomez
 *  All rights reserved
 *
 *  @file   checkpoint.h
 */

#ifndef FTI_SRC_CHECKPOINT_H_
#define FTI_SRC_CHECKPOINT_H_

#include "interface.h"

int FTI_UpdateIterTime(FTIT_execution* FTI_Exec);
int FTI_WriteCkpt(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt,
        FTIT_keymap* FTI_Data);
int FTI_PostCkpt(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt, FTIT_keymap* FTI_Data);
int FTI_Listen(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt, FTIT_keymap* FTI_Data);
int FTI_HandleCkptRequest(FTIT_configuration* FTI_Conf,
 FTIT_execution* FTI_Exec, FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt, FTIT_keymap* FTI_Data);
int FTI_Write(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt,
        FTIT_keymap* FTI_Data, FTIT_IO *FTI_IO);

#endif  // FTI_SRC_CHECKPOINT_H_
