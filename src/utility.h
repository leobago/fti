#ifndef __UTILITY__
#define __UTILITY__
#include <string.h>

#include "IO/ftiff.h"



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

typedef struct{
	FILE *f;
	size_t offset;
	char flag;
}WritePosixInfo_t;

#ifdef ENABLE_HDF5
typedef struct{
	FTIT_execution *FTI_Exec;
	FTIT_dataset *FTI_Data;
	hid_t file_id;
}WriteHDF5_t;

int FTI_HDF5Open(char *fn, void *fileDesc);
int FTI_HDF5Close(void *fileDesc);
int FTI_HDF5Write(void *src, size_t size, void *fileDesc);
int FTI_HDF5Read(void *src, size_t size, void *fileDesc);
#endif

// Wrappers around MPIO
int FTI_MPIOOpen(char *fn, void *fileDesc);
int FTI_MPIOClose(void *fileDesc);
int FTI_MPIOWrite(void *src, size_t size, void *fileDesc);
int FTI_MPIORead(void *src, size_t size, void *fileDesc);

//Wrappers around POSIX IO
int FTI_PosixOpen(char *fn, void *fileDesc);
int FTI_PosixClose(void *fileDesc);
int FTI_PosixWrite(void *src, size_t size, void *fileDesc);
int FTI_PosixRead(void *src, size_t size, void *fileDesc);
int FTI_PosixSync(void *fileDesc);
int FTI_PosixSeek(size_t pos, void *fileDesc);


int copyDataFromDevive(FTIT_execution* FTI_Exec, FTIT_dataset* FTI_Data);

#ifdef ENABLE_SIONLIB 
int write_sion(void *src, size_t size, void *opaque);
#endif
#endif
