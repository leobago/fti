#ifndef __CHECKPOINT_H__
#define __CHECKPOINT_H__

int FTI_UpdateIterTime(void);
int FTI_WriteCkpt(void);
#ifdef ENABLE_SIONLIB // --> If SIONlib is installed
int FTI_WriteSionlib(void);
#endif
int FTI_PostCkpt(void);
int FTI_Listen(void);
int FTI_HandleCkptRequest(void);
int FTI_Write(FTIT_IO *FTI_IO);

#endif // __CHECKPOINT_H__
