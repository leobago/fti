#ifndef FTI_INI_H
#define FTI_INI_H

#include "../deps/iniparser/iniparser.h"
#include "../deps/iniparser/dictionary.h"

typedef struct FTIT_iniparser {
    dictionary* dict;
    char*       (*getString)    ( struct FTIT_iniparser*, const char* );
    int         (*getInt)       ( struct FTIT_iniparser*, const char* );
    int         (*getLong)      ( struct FTIT_iniparser*, const char* );
    int         (*clear)      ( struct FTIT_iniparser* );
} FTIT_iniparser;

int FTI_Iniparser( FTIT_iniparser*, const char* );
char* FTI_IniparserGetString( FTIT_iniparser*, const char* );
int FTI_IniparserGetInt( FTIT_iniparser*, const char* key );
int FTI_IniparserGetLong( FTIT_iniparser*, const char* key );
int FTI_IniparserClear( FTIT_iniparser* );

#endif // FTI_INI_H
  
