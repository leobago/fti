#ifndef __TOOLS_H__
#define __TOOLS_H__

void FTI_Print( char *msg, int priority );
int FTI_Checksum( char* checksum );
int FTI_VerifyChecksum( char* fileName, char* checksumToCmp );
int FTI_Try( int result, char* message );
void FTI_MallocMeta();
void FTI_FreeMeta();
void FTI_FreeTypesAndGroups();
int FTI_InitGroupsAndTypes();
int FTI_InitBasicTypes();
int FTI_InitExecVars();
int FTI_RmDir( char path[FTI_BUFS], int flag );
int FTI_Clean( int level );


#endif // __TOOLS_H__
