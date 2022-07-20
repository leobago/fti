/**
 *  Copyright (c) 2017 Leonardo A. Bautista-Gomez
 *  All rights reserved
 *
 *  @file   fti-ext.h
 *  @author Kai Keller (kai.rasmus.keller@gmail.com)
 *  @date   July, 2022
 *  @brief  Header file for the FTI library extensions.
 */

#ifndef FTI_SRC_FTI_KERNEL_H_
#define FTI_SRC_FTI_KERNEL_H_

#include "interface.h"

/** General configuration information used by FTI.                         */
extern FTIT_configuration FTI_Conf;

/** Checkpoint information for all levels of checkpoint.                   */
extern FTIT_checkpoint FTI_Ckpt[5];

/** Dynamic information for this execution.                                */
extern FTIT_execution FTI_Exec;

/** Topology of the system.                                                */
extern FTIT_topology FTI_Topo;

/** id map that holds metadata for protected datasets                      */
extern FTIT_keymap* FTI_Data;

/** SDC injection model and all the required information.                  */
extern FTIT_injection FTI_Inje;

#endif  // FTI_SRC_FTI_KERNEL_H_

