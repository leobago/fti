#ifndef __FTIFF_DCP_H__
#define __FTIFF_DCP_H__

int FTI_ProcessDBVar(FTIT_execution *FTI_Exec, FTIT_configuration *FTI_Conf, FTIFF_dbvar *currentdbvar, 
        FTIT_dataset *FTI_Data, unsigned char *hashchk, WritePosixInfo_t *fd, char *fn, long *dcpSize, unsigned char **dptr);

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

#endif // __FTIFF_DCP_H__
