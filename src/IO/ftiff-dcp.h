#ifndef __FTIFF_DCP_H__
#define __FTIFF_DCP_H__

#define CHUNK_SIZE 131072    /**< MD5 algorithm chunk size.      */

int FTI_ProcessDBVar( FTIFF_dbvar *currentdbvar, unsigned char *hashchk, WritePosixInfo_t *fd, char *fn, long *dcpSize, unsigned char **dptr);

int FTI_InitDcp(void);
int FTI_FinalizeDcp(void); 
int FTI_InitNextHashData(FTIT_DataDiffHash *hashes);
int FTI_FreeDataDiff( FTIT_DataDiffHash *dhash);
dcpBLK_t FTI_GetDiffBlockSize(void); 
int FTI_GetDcpMode(void); 
int FTI_ReallocateDataDiff( FTIT_DataDiffHash *dhash, long nbHashes);
int FTI_InitBlockHashArray( FTIFF_dbvar* dbvar ); 
int FTI_CollapseBlockHashArray( FTIT_DataDiffHash* hashes, long chunkSize); 
int FTI_ExpandBlockHashArray( FTIT_DataDiffHash* dataHash, long chunkSize ); 
long FTI_CalcNumHashes( long chunkSize ); 
int FTI_HashCmp( long hashIdx, FTIFF_dbvar* dbvar, unsigned char *ptr );
int FTI_UpdateDcpChanges(void); 
int FTI_ReceiveDataChunk(unsigned char** buffer_addr, size_t* buffer_size, FTIFF_dbvar* dbvar, unsigned char *startAddr, size_t *totalBytes ); 

#endif // __FTIFF_DCP_H__
