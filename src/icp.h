#ifndef __ICP_H__
#define __ICP_H__

#include <mpi.h>
#ifdef ENABLE_HDF5
#include "hdf5.h"
#include "hdf5_hl.h"
#endif
#include <stdio.h>

#define FTI_ICP_NINI 0
#define FTI_ICP_ACTV 1
#define FTI_ICP_FAIL 2

int FTI_startICP(FTIT_IO *io);
int FTI_WriteVar(int varID, FTIT_IO *io);
int FTI_FinishICP(FTIT_IO *io);

int FTI_WriteFtiffVar(int varID, FTIT_IO *io);
int FTI_InitFtiffICP(FTIT_IO *io);
int FTI_FinalizeFtiffICP(FTIT_IO *io);
#if 0
int FTI_WriteSionlibVar(int varID);
int FTI_InitSionlibICP();
int FTI_FinalizeSionlibICP();
#endif
#endif // __ICP_H__
