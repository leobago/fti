#ifndef __UTILITY__
#define __UTILITY__

#include <mpi.h>
#include <stdio.h>
#include "../deps/md5/md5.h"

typedef struct{
    MPI_Offset offset;              // Offset of the Rank in the file
    int err;                        // Errors
    MPI_Info info;                  // MPI info of the file
    MPI_File pfh;                   // File descriptor
    char flag;                      // Flags used to open the file
    MD5_CTX integrity;              // integrity of the file
} WriteMPIInfo_t;

typedef struct{
    FILE *f;                        // Posix file descriptor
    size_t offset;                  // offset in the file
    char flag;                      // flags to open the file
    MD5_CTX integrity;              // integrity of the file
}WritePosixInfo_t;

typedef struct{
    WritePosixInfo_t write_info;    // Posix Write info descriptor 
    size_t layerSize;               // size of the dcp layer
}WriteDCPPosixInfo_t;

#ifdef ENABLE_HDF5
typedef struct{
    hid_t file_id;                  // File Id
}WriteHDF5Info_t;

int FTI_HDF5Open(char *fn, void *fileDesc);
int FTI_HDF5Close(void *fileDesc);
void *FTI_InitHDF5();
int FTI_WriteHDF5Data(FTIT_dataset * FTI_DataVar, void *write_info);
int FTI_WriteHDF5();
size_t FTI_GetHDF5FilePos(void *);
#endif

// Wrappers around MPIO
int FTI_MPIOOpen(char *fn, void *fileDesc);
int FTI_MPIOClose(void *fileDesc);
int FTI_MPIOWrite(void *src, size_t size, void *fileDesc);
int FTI_MPIORead(void *src, size_t size, void *fileDesc);
size_t FTI_GetMPIOFilePos(void *fileDesc);

void *FTI_InitMPIO();
int FTI_WriteMPIOData(FTIT_dataset * FTI_DataVar, void *write_info);

//Wrappers around POSIX IO
int FTI_PosixOpen(char *fn, void *fileDesc);
int FTI_PosixClose(void *fileDesc);
int FTI_PosixWrite(void *src, size_t size, void *fileDesc);
int FTI_PosixRead(void *src, size_t size, void *fileDesc);
int FTI_PosixSync(void *fileDesc);
int FTI_PosixSeek(size_t pos, void *fileDesc);
void *FTI_InitPosix();
int FTI_WritePosixData(FTIT_dataset * FTI_DataVar, void* write_info);
size_t FTI_GetPosixFilePos(void *fileDesc);
void FTI_PosixMD5(unsigned char *, void *);


//Wrappers around dcp POSIX

size_t FTI_GetDCPPosixFilePos(void *fileDesc);
void *FTI_InitDCPPosix();
int FTI_WritePosixDCPData(FTIT_dataset *FTI_DataVar, void *fd);
int FTI_PosixDCPClose(void *fileDesc);

int copyDataFromDevive();



#ifdef ENABLE_SIONLIB 
int write_sion(void *src, size_t size, void *opaque);
#endif
#endif
