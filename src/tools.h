#ifndef __TOOLS_H__
#define __TOOLS_H__

void FTI_Print( char *msg, int priority );
int FTI_Checksum( char* checksum );
int FTI_VerifyChecksum( char* fileName, char* checksumToCmp );
int FTI_Try( int result, char* message );
void FTI_MallocMeta(void);
void FTI_FreeMeta(void);
void FTI_FreeTypesAndGroups(void);
int FTI_InitGroupsAndTypes(void);
int FTI_InitBasicTypes(void);
int FTI_InitExecVars(void);
int FTI_RmDir( char path[FTI_BUFS], int flag );
int FTI_Clean( int level );


#endif // __TOOLS_H__
