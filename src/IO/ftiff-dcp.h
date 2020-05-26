/**
 *  Copyright (c) 2017 Leonardo A. Bautista-Gomez
 *  All rights reserved
 *
 *  @file   ftiff-dcp.h
 */

#ifndef FTI_FTIFF_DCP_H_
#define FTI_FTIFF_DCP_H_

#define CHUNK_SIZE 131072    /**< MD5 algorithm chunk size.      */

int FTI_InitDcp(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec);
int FTI_FinalizeDcp(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec);
int FTI_InitNextHashData(FTIT_DataDiffHash *hashes);
int FTI_FreeDataDiff(FTIT_DataDiffHash *dhash);
dcpBLK_t FTI_GetDiffBlockSize();
int FTI_GetDcpMode();
int FTI_ReallocateDataDiff(FTIT_DataDiffHash *dhash, long nbHashes);
int FTI_InitBlockHashArray(FTIFF_dbvar* dbvar);
int FTI_CollapseBlockHashArray(FTIT_DataDiffHash* hashes, long chunkSize);
int FTI_ExpandBlockHashArray(FTIT_DataDiffHash* dataHash, long chunkSize);
long FTI_CalcNumHashes(long chunkSize);
int FTI_HashCmp(long hashIdx, FTIFF_dbvar* dbvar, unsigned char *ptr);
int FTI_UpdateDcpChanges(FTIT_execution* FTI_Exec);
int FTI_ReceiveDataChunk(unsigned char** buffer_addr, size_t* buffer_size,
 FTIFF_dbvar* dbvar,  unsigned char *startAddr, size_t *totalBytes);

#endif  // FTI_FTIFF_DCP_H_
