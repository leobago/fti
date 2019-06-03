#ifndef __RECOVER_H__
#define __RECOVER_H__

int FTI_CheckFile(char *fn, long fs, char* checksum);
int FTI_CheckErasures(int *erased);
int FTI_RecoverFiles();

#endif // __RECOVER_H__
