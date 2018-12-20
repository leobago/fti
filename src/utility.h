#ifndef __UTILITY__
#include <string.h>

#include "interface.h"
#include "ftiff.h"

#include "api_cuda.h"


typedef struct
{
  FTIT_configuration* FTI_Conf;
  MPI_File pfh;
  MPI_Offset offset;
  int err;
} WriteMPIInfo_t;


int write_posix(void *src, size_t size, void *opaque);
int write_mpi(void *src, size_t size, void *opaque);
int copyDataFromDevive(FTIT_execution* FTI_Exec, FTIT_dataset* FTI_Data);

#ifdef ENABLE_SIONLIB 
int write_sion(void *src, size_t size, void *opaque);
#endif
#endif
