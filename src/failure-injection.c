#include "failure-injection.h"
#include <stdlib.h>

static float __PROBABILITY__ ;

float PROBABILITY() {
    return __PROBABILITY__;
}

void init_fi() {

    char *env;
    if ( (env = getenv("FTI_FI_PROBABILITY")) != NULL ) {
        __PROBABILITY__ = atof(env);
    } else {
        __PROBABILITY__ = 0.01;
    }

}
