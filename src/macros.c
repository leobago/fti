#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdarg.h>
#include "interface.h"

__attribute__ ((sentinel))
    void cleanup(char *pattern, ...) {
        va_list args;
        va_start(args, pattern);
        while (*pattern!= '\0') {
            switch (*pattern++) {
                case 'p':
                    free(va_arg(args, void*));
                    break;
                case 'f':
                    fclose(va_arg(args, void*));
                    break;
                default:
                    FTI_Print("Unknown pattern in error Clean UP",FTI_WARN);
            }
        }

        va_end(args);
    }

