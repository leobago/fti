#ifndef __UTILITY__
#define __UTILITY__
#include <string.h>

#include "IO/ftiff.h"
#include "../deps/md5/md5.h"



typedef struct{
	FTIT_configuration* FTI_Conf;
	FTIT_topology *FTI_Topo;
	MPI_Offset offset;
	int err;
	MPI_Info info;
	MPI_File pfh;
	char flag;
	MD5_CTX integrity;
} WriteMPIInfo_t;

typedef struct{
	FILE *f;
	size_t offset;
	char flag;
	MD5_CTX integrity;
}WritePosixInfo_t;

#ifdef ENABLE_HDF5
typedef struct{
	FTIT_execution *FTI_Exec;
	FTIT_dataset *FTI_Data;
	hid_t file_id;
}WriteHDF5Info_t;

int FTI_HDF5Open(char *fn, void *fileDesc);
int FTI_HDF5Close(void *fileDesc);
WriteHDF5Info_t *FTI_InitHDF5(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec, FTIT_topology* FTI_Topo, FTIT_checkpoint *FTI_Ckpt, FTIT_dataset *FTI_Data);
int FTI_WriteHDF5Data(FTIT_dataset * FTI_DataVar, void *write_info);
#endif

// Wrappers around MPIO
int FTI_MPIOOpen(char *fn, void *fileDesc);
int FTI_MPIOClose(void *fileDesc);
int FTI_MPIOWrite(void *src, size_t size, void *fileDesc);
int FTI_MPIORead(void *src, size_t size, void *fileDesc);
size_t FTI_GetMPIOFilePos(void *fileDesc);

WriteMPIInfo_t *FTI_InitMpi(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec, FTIT_topology* FTI_Topo, FTIT_checkpoint *FTI_Ckpt, FTIT_dataset *FTI_Data);
int FTI_WriteMPIOData(FTIT_dataset * FTI_DataVar, WriteMPIInfo_t *write_info);

//Wrappers around POSIX IO
int FTI_PosixOpen(char *fn, void *fileDesc);
int FTI_PosixClose(void *fileDesc);
int FTI_PosixWrite(void *src, size_t size, void *fileDesc);
int FTI_PosixRead(void *src, size_t size, void *fileDesc);
int FTI_PosixSync(void *fileDesc);
int FTI_PosixSeek(size_t pos, void *fileDesc);
WritePosixInfo_t *FTI_InitPosix(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec, FTIT_topology* FTI_Topo, FTIT_checkpoint *FTI_Ckpt, FTIT_dataset *FTI_Data);
int FTI_WritePosixData(FTIT_dataset * FTI_DataVar, WritePosixInfo_t *write_info);
size_t FTI_GetPosixFilePos(void *fileDesc);



int copyDataFromDevive(FTIT_execution* FTI_Exec, FTIT_dataset* FTI_Data);

#ifdef ENABLE_SIONLIB 
int write_sion(void *src, size_t size, void *opaque);
#endif
#endif
