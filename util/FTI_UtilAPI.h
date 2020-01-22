#ifndef __FTI_UTILAPI__
#define __FTI_UTILAPI__

#define SUCCESS 1
#define ERROR -1


int FTI_InitUtil(char *configFile);
int FTI_GetNumberOfCkptIds(int *numCkpts);
int FTI_GetCkptIds(int *ckptIds);
int FTI_FinalizeUtil();
int FTI_GetUserRanks(int *numRanks);
int FTI_VerifyCkpt(int collection, int ckpt);
int FTI_GetNumVars(int ckptId, int rank);
int FTI_readVariable(int varId, int ckptId, int rank, char **varName, unsigned char **buf, size_t *size);

#endif
