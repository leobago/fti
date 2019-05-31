#ifndef __POSIX_DCP_H__
#define __POSIX_DCP_H__

int FTI_WritePosixDcp(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt,
        FTIT_dataset* FTI_Data);
int FTI_CheckFileDcpPosix(char* fn, long fs, char* checksum);
int FTI_VerifyChecksumDcpPosix(char* fileName);
void* FTI_DcpPosixRecoverRuntimeInfo( int tag, void* exec_, void* conf_ );
int FTI_RecoverDcpPosix( FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec, FTIT_checkpoint* FTI_Ckpt, FTIT_dataset* FTI_Data );
int FTI_RecoverVarDcpPosix( FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec, FTIT_checkpoint* FTI_Ckpt, FTIT_dataset* FTI_Data, int id );
int FTI_DataGetIdx( int varId, FTIT_execution* FTI_Exec, FTIT_dataset* FTI_Data );
char* FTI_GetHashHexStr( const unsigned char* hash, int digestWidth, char* hashHexStr );

#endif // __POSIX_DCP_H__
