#ifndef __POSTRECO_H__
#define __POSTRECO_H__

int FTI_SendCkptFileL2( int destination, int ptner);
int FTI_RecvCkptFileL2( int source, int ptner);
int FTI_Decode(int *erased);
int FTI_RecoverL1(void);
int FTI_RecoverL2(void);
int FTI_RecoverL3(void);
int FTI_RecoverL4(void);
int FTI_RecoverL4Posix(void);
int FTI_RecoverL4Mpi(void);
#ifdef ENABLE_SIONLIB // --> If SIONlib is installed
int FTI_RecoverL4Sionlib(void);
#endif

#endif // __POSTRECO_H__
