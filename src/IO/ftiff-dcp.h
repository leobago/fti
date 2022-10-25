/**
 *  Copyright (c) 2017 Leonardo A. Bautista-Gomez
 *  All rights reserved
 *
 *  @file   ftiff-dcp.h
 */

#ifndef FTI_SRC_IO_FTIFF_DCP_H_
#define FTI_SRC_IO_FTIFF_DCP_H_

#define CHUNK_SIZE 131072    /**< MD5 algorithm chunk size.      */

int FTI_InitDcp(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec);
int FTI_FinalizeDcp(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec);
int FTI_InitNextHashData(FTIT_DataDiffHash *hashes);
int FTI_FreeDataDiff(FTIT_DataDiffHash *dhash);
dcpBLK_t FTI_GetDiffBlockSize();
int FTI_GetDcpMode();
int FTI_ReallocateDataDiff(FTIT_DataDiffHash *dhash, int32_t nbHashes);
int FTI_InitBlockHashArray(FTIFF_dbvar* dbvar);
int FTI_CollapseBlockHashArray(FTIT_DataDiffHash* hashes, int64_t chunkSize);
int FTI_ExpandBlockHashArray(FTIT_DataDiffHash* dataHash, int64_t chunkSize);
int32_t FTI_CalcNumHashes(int64_t chunkSize);
int FTI_HashCmp(int32_t hashIdx, FTIFF_dbvar* dbvar, unsigned char *ptr);
int FTI_UpdateDcpChanges(FTIT_execution* FTI_Exec);
int FTI_ReceiveDataChunk(unsigned char** buffer_addr, size_t* buffer_size,
 FTIFF_dbvar* dbvar,  unsigned char *startAddr, size_t *totalBytes);

#endif  // FTI_SRC_IO_FTIFF_DCP_H_
