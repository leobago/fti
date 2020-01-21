#ifndef __FTI_UTILAPI__
#define __FTI_UTILAPI__

#define SUCCESS 1
#define ERROR -1


int FTI_InitUtil(char *configFile);
int FTI_GetNumberOfCkptIds(int *numCkpts);
int FTI_GetCkptIds(int *ckptIds);
int FTI_FinalizeUtil();

#endif
