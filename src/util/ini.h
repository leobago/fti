#ifndef FTI_INI_H
#define FTI_INI_H

#include "../deps/iniparser/iniparser.h"
#include "../deps/iniparser/dictionary.h"

typedef struct FTIT_iniparser {
    dictionary* dict;
    char*       (*getString)( const char* );
    int         (*getInt)( const char* );
    int         (*getLong)( const char* );
    int         (*destroy)( void );
} FTIT_iniparser;

int FTI_IniparserCreate( FTIT_iniparser*, const char* );
char* FTI_IniparserGetString( const char* );
int FTI_IniparserGetInt( const char* key );
int FTI_IniparserGetLong( const char* key );
int FTI_IniparserDestroy( void );

#endif // FTI_INI_H
  
