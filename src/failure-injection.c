#include <fti-int/defs.h> 
#include <fti-int/types.h> 
#include "failure-injection.h"
#include <stdlib.h>
#include <fti.h>
#include <string.h>
#include <stdio.h>

static float _PROBABILITY ;
static char _FUNCTION[FTI_BUFS];

float PROBABILITY() {
    return _PROBABILITY;
}

unsigned int FUNCTION( const char *testFunction ) {
    int len = ( strlen(testFunction) > FTI_BUFS ) ? FTI_BUFS : strlen(testFunction);
    int res = strncmp( testFunction, _FUNCTION, len );
    return !res;
}

void FTI_InitFIIO() {

    char *env;
    if ( (env = getenv("FTI_FI_PROBABILITY")) != NULL ) {
        _PROBABILITY = atof(env);
    } else {
        _PROBABILITY = 0.01;
    }
    if ( (env = getenv("FTI_FI_FUNCTION")) != NULL ) {
        strncpy( _FUNCTION, env, FTI_BUFS-1 );
        _FUNCTION[FTI_BUFS-1] = '\0';
    } else {
        _FUNCTION[0] = '\0';
    }

}
