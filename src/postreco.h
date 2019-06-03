#ifndef __POSTRECO_H__
#define __POSTRECO_H__

int FTI_SendCkptFileL2( int destination, int ptner);
int FTI_RecvCkptFileL2( int source, int ptner);
int FTI_Decode(int *erased);
int FTI_RecoverL1();
int FTI_RecoverL2();
int FTI_RecoverL3();
int FTI_RecoverL4();
int FTI_RecoverL4Posix();
int FTI_RecoverL4Mpi();
#ifdef ENABLE_SIONLIB // --> If SIONlib is installed
int FTI_RecoverL4Sionlib();
#endif

#endif // __POSTRECO_H__
