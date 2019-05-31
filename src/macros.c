/**
 *  Copyright (c) 2017 Leonardo A. Bautista-Gomez
 *  All rights reserved
 *
 *  FTI - A multi-level checkpointing library for C/C++/Fortran applications
 *
 *  Revision 1.0 : Fault Tolerance Interface (FTI)
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice, this
 *  list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright notice,
 *  this list of conditions and the following disclaimer in the documentation
 *  and/or other materials provided with the distribution.
 *
 *  3. Neither the name of the copyright holder nor the names of its contributors
 *  may be used to endorse or promote products derived from this software without
 *  specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 *  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 *  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 *  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 *  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 *  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *  @file   macros.c
 *  @date   May, 2019
 *  @brief  API functions for the FTI library.
 */



#include "interface.h"


/*-------------------------------------------------------------------------*/
/**
  @brief      Varidic function that cleans the execution when an error occurs.
  @param      patttern.       a series of characters, f denotes that the respective argument is a file to be closed p corresponds to a pointer to be freed 
  @param      ....            Pointers corresponding to files or pointers.
  @return     integer         FTI_SCES if successful.

    This functions cleans up the local environment after an error occurs.
 **/
/*-------------------------------------------------------------------------*/
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

