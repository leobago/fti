#ifndef __POSTCKPT_H__
#define __POSTCKPT_H__

int FTI_Local(void);
int FTI_Ptner(void);
int FTI_SendCkpt( int destination, int postFlag );
int FTI_RecvPtner(int source, int postFlag);
int FTI_RSenc(void);
int FTI_Flush(int level);
int FTI_FlushPosix(int level);
int FTI_FlushMPI(int level);
#ifdef ENABLE_SIONLIB // --> If SIONlib is installed
int FTI_FlushSionlib(nt level);
#endif
int FTI_ArchiveL4Ckpt(void);

#endif // __POSTCKPT_H__
