#include "../interface.h"

static FTIT_iniparser* self;

int FTI_IniparserCreate( FTIT_iniparser* ini, const char* inifile )
{

    char err[FTI_BUFS];

    if( ini == NULL )
    {
        FTI_Print("ini is NULL.", FTI_EROR);
        return FTI_NSCS;
    }

    if( inifile == NULL )
    {
        FTI_Print("inifile is NULL.", FTI_EROR);
        return FTI_NSCS;
    }

    if (access(inifile, R_OK) != 0) {
        return FTI_NSCS;
    }
 
    ini->dict = iniparser_load( inifile );
    
    if (ini == NULL) {
        snprintf( err, FTI_BUFS, "Iniparser failed to parse the file ('%s').", inifile );
        FTI_Print( err, FTI_WARN );
        return FTI_NSCS;
    }
    
    self = ini;
    self->getString = FTI_IniparserGetString;
    self->getInt = FTI_IniparserGetInt;
    self->getLong = FTI_IniparserGetLong;
    self->destroy = FTI_IniparserDestroy;

    return FTI_SCES;

}

char* FTI_IniparserGetString( const char* key )
{   
    static char nullstr = '\0';
    
    char* string = iniparser_getstring( self->dict, key, NULL );

    if( string == NULL ) return &nullstr;

    return string;
    
}

int FTI_IniparserGetInt( const char* key )
{   

    return iniparser_getint( self->dict, key, -1 );
    
}

int FTI_IniparserGetLong( const char* key )
{   

    return iniparser_getlint( self->dict, key, -1 );
    
}

int FTI_IniparserDestroy()
{
    if( self == NULL ) return FTI_SCES;
    iniparser_freedict( self->dict );
    return FTI_SCES;
}
