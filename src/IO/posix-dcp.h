/**
 *  Copyright (c) 2017 Leonardo A. Bautista-Gomez
 *  All rights reserved
 *
 *  @file   posix-dcp.h
 */

#ifndef FTI_SRC_IO_POSIX_DCP_H_
#define FTI_SRC_IO_POSIX_DCP_H_

#ifndef MD5_DIGEST_LENGTH
#   define MD5_DIGEST_LENGTH 16  // 128 bits
#endif
#ifndef CRC32_DIGEST_LENGTH
#   define CRC32_DIGEST_LENGTH 4  // 32 bits
#endif
#define CMP_DIGEST_LENGTH 1  // boolean

#define MAX_BLOCK_IDX 0x3fffffff
#define MAX_VAR_ID 0x3ffff

#define DCP_POSIX_EXEC_TAG 0
#define DCP_POSIX_CONF_TAG 1
#define DCP_POSIX_INIT_TAG -1

int FTI_CheckFileDcpPosix(char* fn, int64_t fs, char* checksum);
int FTI_VerifyChecksumDcpPosix(char* fileName);
void* FTI_DcpPosixRecoverRuntimeInfo(int tag, void* exec_, void* conf_);
int FTI_RecoverDcpPosix(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
 FTIT_checkpoint* FTI_Ckpt, FTIT_keymap* FTI_Data);
int FTI_RecoverVarDcpPosix(FTIT_configuration* FTI_Conf,
 FTIT_execution* FTI_Exec, FTIT_checkpoint* FTI_Ckpt,
 FTIT_keymap* FTI_Data, int id);
char* FTI_GetHashHexStr(unsigned char* hash, int digestWidth,
 char* hashHexStr);
// wrapper for CRC32 hash algorithm
unsigned char* CRC32(const unsigned char *n, unsigned long nBytes,
 unsigned char *hash);
unsigned char* _CRC32_wrap(const unsigned char *n, const unsigned char *o, unsigned long nBytes,
 unsigned char *hash);
unsigned char* _MD5_wrap(const unsigned char *n, const unsigned char *o, unsigned long nBytes,
 unsigned char *hash);
unsigned char* _COMPARE(const unsigned char *n, const unsigned char *o, unsigned long nBytes,
 unsigned char *hash); 

bool _MD5_cmp( FTIT_dataset* data, int hashIdx  );
bool _CRC32_cmp( FTIT_dataset* data, int hashIdx  );
bool _CMP_cmp( FTIT_dataset* data, int hashIdx  );

int FTI_BlockHashDcp (FTIT_configuration* FTI_Conf,FTIT_execution* FTI_Exec, 
  FTIT_dataset* FTI_Data, void *block_new, void *block_old ,uint64_t nBytes,unsigned char *hash) ;
int FTI_RecoverVarDcpPosixInit();
int FTI_RecoverVarDcpPosixFinalize();
#endif  // FTI_SRC_IO_POSIX_DCP_H_
