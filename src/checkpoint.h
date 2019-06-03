#ifndef __CHECKPOINT_H__
#define __CHECKPOINT_H__

int FTI_UpdateIterTime();
int FTI_WriteCkpt();
#ifdef ENABLE_SIONLIB // --> If SIONlib is installed
int FTI_WriteSionlib();
#endif
int FTI_PostCkpt();
int FTI_Listen();
int FTI_HandleCkptRequest();
int FTI_Write(FTIT_IO *FTI_IO);

#endif // __CHECKPOINT_H__
