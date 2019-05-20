#ifndef __UTILITY__
#include <string.h>

#include "interface.h"
#include "IO/ftiff.h"

#include "api_cuda.h"


typedef struct
{
	FTIT_configuration* FTI_Conf;
	FTIT_topology *FTI_Topo;
	MPI_Offset offset;
	int err;
	MPI_Info info;
	MPI_File pfh;
	char flag;
} WriteMPIInfo_t;


int FTI_MPIOOpen(char *fn, void *fileDesc);
int FTI_MPIOClose(void *fileDesc);
int FTI_MPIOWrite(void *src, size_t size, void *fileDesc);
int FTI_MPIORead(void *src, size_t size, void *fileDesc);

int write_posix(void *src, size_t size, void *opaque);
int copyDataFromDevive(FTIT_execution* FTI_Exec, FTIT_dataset* FTI_Data);

#ifdef ENABLE_SIONLIB 
int write_sion(void *src, size_t size, void *opaque);
#endif
#endif
