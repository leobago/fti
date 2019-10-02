#ifndef __UTILITY__
#define __UTILITY__

#include <fti.h>
#include "../deps/md5/md5.h"

typedef struct{
    FTIT_configuration* FTI_Conf;   // Configuration of the FTI
    FTIT_topology *FTI_Topo;        // Topology of the nodes
    MPI_Offset offset;              // Offset of the Rank in the file
    size_t loffset;                 // Offset in the local file
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
    FTIT_configuration *FTI_Conf;   // FTI Configuration
    FTIT_checkpoint *FTI_Ckpt;      // FTI Checkpoint options
    FTIT_execution *FTI_Exec;       // FTI execution options
    FTIT_topology *FTI_Topo;        // FTI node topology
    FTIT_dataset *FTI_Data;
    size_t layerSize;               // size of the dcp layer
}WriteDCPPosixInfo_t;

typedef struct{
    FILE *f;                        // Posix file descriptor
    size_t offset;                  // offset in the file
    char flag;                      // flags to open the file
    MD5_CTX integrity;              // integrity of the file
    FTIT_configuration *FTI_Conf;   // FTI Configuration
    FTIT_checkpoint *FTI_Ckpt;      // FTI Checkpoint options
    FTIT_execution *FTI_Exec;       // FTI execution options
    FTIT_topology *FTI_Topo;        // FTI node topology
    FTIT_dataset *FTI_Data;
}WriteFTIFFInfo_t;

#ifdef ENABLE_HDF5
typedef struct{
    FTIT_execution *FTI_Exec;       // Execution environment
    FTIT_dataset *FTI_Data;         // FTI Data
    FTIT_topology *FTI_Topo;         // FTI Data
    FTIT_configuration *FTI_Conf;         // FTI Data
    hid_t file_id;                  // File Id
}WriteHDF5Info_t;

int FTI_HDF5Open(char *fn, void *fileDesc);
int FTI_HDF5Close(void *fileDesc);
void *FTI_InitHDF5(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec, FTIT_topology* FTI_Topo, FTIT_checkpoint *FTI_Ckpt, FTIT_dataset *FTI_Data);
int FTI_WriteHDF5Data(FTIT_dataset * FTI_DataVar, void *write_info);
int FTI_WriteHDF5(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt,
        FTIT_dataset* FTI_Data);
size_t FTI_GetHDF5FilePos(void *);
#endif

#ifdef ENABLE_SIONLIB
#include <sion.h>
typedef struct{
    int sid;
    int *file_map;
    int *ranks;
    int *rank_map;
    size_t loffset;
    sion_int64* chunkSizes;
}WriteSionInfo_t;

int FTI_WriteSionData(FTIT_dataset * FTI_DataVar, void *fd);
void* FTI_InitSion(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec, FTIT_topology* FTI_Topo, FTIT_checkpoint *FTI_Ckpt, FTIT_dataset *FTI_Data);
int FTI_SionClose(void *fileDesc);
size_t FTI_GetSionFilePos(void *fileDesc);
#endif

// Wrappers around MPIO
int FTI_MPIOOpen(char *fn, void *fileDesc);
int FTI_MPIOClose(void *fileDesc);
int FTI_MPIOWrite(void *src, size_t size, void *fileDesc);
int FTI_MPIORead(void *src, size_t size, void *fileDesc);
size_t FTI_GetMPIOFilePos(void *fileDesc);

void *FTI_InitMPIO(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec, FTIT_topology* FTI_Topo, FTIT_checkpoint *FTI_Ckpt, FTIT_dataset *FTI_Data);
int FTI_WriteMPIOData(FTIT_dataset * FTI_DataVar, void *write_info);

//Wrappers around dcp POSIX

size_t FTI_GetDCPPosixFilePos(void *fileDesc);
void *FTI_InitDCPPosix(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec, FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt, FTIT_dataset* FTI_Data);
int FTI_WritePosixDCPData(FTIT_dataset *FTI_DataVar, void *fd);
int FTI_PosixDCPClose(void *fileDesc);

int copyDataFromDevive(FTIT_execution* FTI_Exec, FTIT_dataset* FTI_Data);
int FTI_AsyncDCPClose(void *fileDesc);



#ifdef ENABLE_SIONLIB 
int write_sion(void *src, size_t size, void *opaque);
#endif
#endif
