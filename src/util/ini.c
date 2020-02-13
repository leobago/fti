#include "../interface.h"

int FTI_Iniparser( FTIT_iniparser* self, const char* inifile )
{

    char err[FTI_BUFS];

    if( self == NULL )
    {
        FTI_Print("iniparser context is NULL.", FTI_EROR);
        return FTI_NSCS;
    }

    if( inifile == NULL )
    {
        FTI_Print("iniparser file is NULL.", FTI_EROR);
        return FTI_NSCS;
    }

    if (access(inifile, R_OK) != 0) {
        return FTI_NSCS;
    }
 
    self->dict = iniparser_load( inifile );
    
    if (self->dict == NULL) {
        snprintf( err, FTI_BUFS, "Iniparser failed to parse the file ('%s').", inifile );
        FTI_Print( err, FTI_WARN );
        return FTI_NSCS;
    }
    
    self->getString = FTI_IniparserGetString;
    self->getInt = FTI_IniparserGetInt;
    self->getLong = FTI_IniparserGetLong;
    self->clear = FTI_IniparserClear;

    return FTI_SCES;

}

char* FTI_IniparserGetString( FTIT_iniparser* self, const char* key )
{   
    static char nullstr = '\0';
    
    char* string = iniparser_getstring( self->dict, key, NULL );

    if( string == NULL ) return &nullstr;

    return string;
    
}

int FTI_IniparserGetInt( FTIT_iniparser* self, const char* key )
{   

    return iniparser_getint( self->dict, key, -1 );
    
}

int FTI_IniparserGetLong( FTIT_iniparser* self, const char* key )
{   

    return iniparser_getlint( self->dict, key, -1 );
    
}

int FTI_IniparserClear( FTIT_iniparser* self )
{
    if( self == NULL ) return FTI_SCES;
    iniparser_freedict( self->dict );
    return FTI_SCES;
}
