#ifndef FTIFF_FUNC_H
#define FTIFF_FUNC_H

#ifndef FTI_NOZLIB
#   include "zlib.h"
#else
extern const uint32_t crc32_tab[];

static inline uint32_t crc32_raw(const void *buf, size_t size, uint32_t crc)
{
    const uint8_t *p = (const uint8_t *)buf;

    while (size--)
        crc = crc32_tab[(crc ^ *p++) & 0xFF] ^ (crc >> 8);
    return (crc);
}

static inline uint32_t crc32(const void *buf, size_t size)
{
    uint32_t crc;

    crc = crc32_raw(buf, size, ~0U);
    return (crc ^ ~0U);
}
#endif

#include <assert.h>
#include <string.h>


void FTIFF_InitMpiTypes();
int FTIFF_DeserializeFileMeta( FTIFF_metaInfo* meta, char* buffer_ser );
int FTIFF_DeserializeDbMeta( FTIFF_db* db, char* buffer_ser );
int FTIFF_DeserializeDbVarMeta( FTIFF_dbvar* dbvar, char* buffer_ser );
int FTIFF_SerializeFileMeta( FTIFF_metaInfo* meta, char* buffer_ser );
int FTIFF_SerializeDbMeta( FTIFF_db* db, char* buffer_ser );
int FTIFF_SerializeDbVarMeta( FTIFF_dbvar* dbvar, char* buffer_ser );
void FTIFF_FreeDbFTIFF(FTIFF_db* last);
int FTIFF_Recover( FTIT_execution *FTI_Exec, FTIT_dataset *FTI_Data, FTIT_checkpoint *FTI_Ckpt );
int FTIFF_RecoverVar( int id, FTIT_execution *FTI_Exec, FTIT_dataset *FTI_Data, FTIT_checkpoint *FTI_Ckpt );
int FTIFF_UpdateDatastructVarFTIFF( FTIT_execution* FTI_Exec, 
        FTIT_dataset* FTI_Data, FTIT_configuration* FTI_Conf, 
        int pvar_idx );
int FTIFF_ReadDbFTIFF( FTIT_configuration *FTI_Conf, FTIT_execution *FTI_Exec, FTIT_checkpoint* FTI_Ckpt );
int FTIFF_GetFileChecksum( FTIFF_metaInfo *FTIFF_Meta, FTIT_checkpoint* FTI_Ckpt, int fd, char *checksum );
int FTIFF_WriteFTIFF(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt,
        FTIT_dataset* FTI_Data);
int FTIFF_createHashesDbVarFTIFF( FTIT_execution* FTI_Exec, FTIT_dataset* FTI_Data );
int FTIFF_finalizeDatastructFTIFF( FTIT_execution* FTI_Exec, FTIT_dataset* FTI_Data );
int FTIFF_writeMetaDataFTIFF( FTIT_execution* FTI_Exec, int fd );
int FTIFF_CreateMetadata( FTIT_execution* FTI_Exec, FTIT_topology* FTI_Topo,
        FTIT_dataset* FTI_Data, FTIT_configuration* FTI_Conf );
int FTIFF_CheckL1RecoverInit( FTIT_execution* FTI_Exec, FTIT_topology* FTI_Topo,
        FTIT_checkpoint* FTI_Ckpt, FTIT_configuration* FTI_Conf );
int FTIFF_CheckL2RecoverInit( FTIT_execution* FTI_Exec, FTIT_topology* FTI_Topo,
        FTIT_checkpoint* FTI_Ckpt, FTIT_configuration* FTI_Conf, int *exists);
int FTIFF_CheckL3RecoverInit( FTIT_execution* FTI_Exec, FTIT_topology* FTI_Topo,
        FTIT_checkpoint* FTI_Ckpt, int* erased);
int FTIFF_CheckL4RecoverInit( FTIT_execution* FTI_Exec, FTIT_topology* FTI_Topo,
        FTIT_checkpoint* FTI_Ckpt);
void FTIFF_GetHashMetaInfo( unsigned char *hash, FTIFF_metaInfo *FTIFFMeta );
void FTIFF_GetHashdb( unsigned char *hash, FTIFF_db *db );
void FTIFF_GetHashdbvar( unsigned char *hash, FTIFF_dbvar *dbvar );
void FTIFF_SetHashChunk( FTIFF_dbvar *dbvar, FTIT_dataset* FTI_Data ); 
void FTIFF_PrintDataStructure( int rank, FTIT_execution* FTI_Exec, FTIT_dataset* FTI_Data );

// dcp

int FTI_ProcessDBVar(FTIT_execution *FTI_Exec, FTIT_configuration *FTI_Conf, FTIFF_dbvar *currentdbvar, 
                     FTIT_dataset *FTI_Data, unsigned char *hashchk, int fd, char *fn, long *dcpSize, unsigned char **dptr);

int FTI_InitDcp(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec, FTIT_dataset* FTI_Data);
int FTI_FinalizeDcp( FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec ); 
int FTI_InitNextHashData(FTIT_DataDiffHash *hashes);
int FTI_FreeDataDiff( FTIT_DataDiffHash *dhash);
dcpBLK_t FTI_GetDiffBlockSize(); 
int FTI_GetDcpMode(); 
int FTI_ReallocateDataDiff( FTIT_DataDiffHash *dhash, long nbHashes);
int FTI_InitBlockHashArray( FTIFF_dbvar* dbvar ); 
int FTI_CollapseBlockHashArray( FTIT_DataDiffHash* hashes, long chunkSize); 
int FTI_ExpandBlockHashArray( FTIT_DataDiffHash* dataHash, long chunkSize ); 
long FTI_CalcNumHashes( long chunkSize ); 
int FTI_HashCmp( long hashIdx, FTIFF_dbvar* dbvar, unsigned char *ptr );
int FTI_UpdateDcpChanges(FTIT_dataset* FTI_Data, FTIT_execution* FTI_Exec); 
int FTI_ReceiveDataChunk(unsigned char** buffer_addr, size_t* buffer_size, FTIFF_dbvar* dbvar,  FTIT_dataset* FTI_Data, unsigned char *startAddr, size_t *totalBytes ); 

#endif // FTIFF_FUNC_H
