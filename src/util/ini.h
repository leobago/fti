#ifndef FTI_INI_H
#define FTI_INI_H

#include "../../deps/iniparser/iniparser.h"
#include "../../deps/iniparser/dictionary.h"

typedef enum FTIT_inimode {
    FTI_INI_OPEN,
    FTI_INI_CREATE,
    FTI_INI_APPEND,
} FTIT_inimode;

typedef struct FTIT_iniparser {
    dictionary*                 dict;
    char                        file[FTI_BUFS];
    char*       (*getString)    ( struct FTIT_iniparser*, const char* );
    int         (*getInt)       ( struct FTIT_iniparser*, const char* );
    int         (*getLong)      ( struct FTIT_iniparser*, const char* );
    int         (*set)          ( struct FTIT_iniparser*, const char*, const char* );
    int         (*dump)        ( struct FTIT_iniparser* );
    int         (*clear)        ( struct FTIT_iniparser* );
} FTIT_iniparser;

int FTI_Iniparser( FTIT_iniparser*, const char*, FTIT_inimode );
char* FTI_IniparserGetString( FTIT_iniparser*, const char* );
int FTI_IniparserGetInt( FTIT_iniparser*, const char* key );
int FTI_IniparserGetLong( FTIT_iniparser*, const char* key );
int FTI_IniparserSet( FTIT_iniparser*, const char*, const char* );
int FTI_IniparserDump( FTIT_iniparser* );
int FTI_IniparserClear( FTIT_iniparser* );

#endif // FTI_INI_H
  
