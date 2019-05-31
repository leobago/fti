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

/*-------------------------------------------------------------------------*/
/**
  @brief      It corrupts a bit of the given float.
  @param      target          Pointer to the float to corrupt.
  @param      bit             Position of the bit to corrupt.
  @return     integer         FTI_SCES if successful.

  This function filps the bit of the target float.

 **/
/*-------------------------------------------------------------------------*/
int FTI_FloatBitFlip(float* target, int bit)
{
    if (bit >= 32 || bit < 0) {
        return FTI_NSCS;
    }
    int* corIntPtr = (int*)target;
    int corInt = *corIntPtr;
    corInt = corInt ^ (1 << bit);
    corIntPtr = &corInt;
    float* fp = (float*)corIntPtr;
    *target = *fp;
    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      It corrupts a bit of the given float.
  @param      target          Pointer to the float to corrupt.
  @param      bit             Position of the bit to corrupt.
  @return     integer         FTI_SCES if successful.

  This function filps the bit of the target float.

 **/
/*-------------------------------------------------------------------------*/
int FTI_DoubleBitFlip(double* target, int bit)
{
    if (bit >= 64 || bit < 0) {
        return FTI_NSCS;
    }
    FTIT_double myDouble;
    myDouble.value = *target;
    int bitf = (bit >= 32) ? bit - 32 : bit;
    int half = (bit >= 32) ? 1 : 0;
    FTI_FloatBitFlip(&(myDouble.floatval[half]), bitf);
    *target = myDouble.value;
    return FTI_SCES;
}

