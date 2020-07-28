/**
 *  Copyright (c) 2017 Leonardo A. Bautista-Gomez
 *  All rights reserved
 *
 *  @file   postckpt.h
 */

#ifndef FTI_POSTCKPT_H_
#define FTI_POSTCKPT_H_

int FTI_Local(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt);
int FTI_Ptner(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt);
int FTI_RSenc(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt);
int FTI_Flush(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt, int level);
int FTI_FlushPosix(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt, int level);
int FTI_FlushMPI(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt, int level);
#ifdef ENABLE_SIONLIB  // --> If SIONlib is installed
int FTI_FlushSionlib(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt, int level);
#endif
int FTI_ArchiveL4Ckpt(FTIT_configuration* FTI_Conf, FTIT_execution *FTI_Exec,
        FTIT_checkpoint *FTI_Ckpt, FTIT_topology *FTI_Topo);

#endif  // FTI_POSTCKPT_H_
