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

#define FTI_GT(NUM1, NUM2) ((NUM1) > (NUM2)) ? NUM1 : NUM2
#define FTI_PO_FH FILE*
#define FTI_FF_FH int
#define FTI_MI_FH MPI_File
#ifdef ENABLE_SIONLIB // --> If SIONlib is installed
#   define FTI_SL_FH int
#endif
#ifdef ENABLE_HDF5 // --> If HDF5 is desired
#   define FTI_H5_FH hid_t
#endif

#if !defined (ENABLE_SIONLIB) && !defined (ENABLE_HDF5) 
#   define FTI_ICP_FH_SIZE FTI_GT( sizeof(FTI_PO_FH), FTI_GT( sizeof(FTI_FF_FH), sizeof(FTI_MI_FH) ) )
#elif !defined (ENABLE_SIONLIB) && defined (ENABLE_HDF5)
#   define FTI_ICP_FH_SIZE FTI_GT( \
        FTI_GT( sizeof(FTI_PO_FH), sizeof(FTI_H5_FH) ), \
        FTI_GT( sizeof(FTI_FF_FH), sizeof(FTI_MI_FH) ) )
#elif defined (ENABLE_SIONLIB) && !defined (ENABLE_HDF5)
#   define FTI_ICP_FH_SIZE FTI_GT( \
        FTI_GT( sizeof(FTI_PO_FH), sizeof(FTI_SL_FH) ), \
        FTI_GT( sizeof(FTI_FF_FH), sizeof(FTI_MI_FH) ) )
#elif defined (ENABLE_SIONLIB) && defined (ENABLE_HDF5)
#   define FTI_ICP_FH_SIZE FTI_GT( \
        FTI_GT( sizeof(FTI_PO_FH), sizeof(FTI_H5_FH) ), \
        FTI_GT( sizeof(FTI_FF_FH),\
        FTI_GT( sizeof(FTI_SL_FH), sizeof(FTI_MI_FH) ) ) )
#endif

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
