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
 *  @file   macros.h
 *  @date   May, 2019
 *  @brief  API functions for the FTI library.
 */



#ifndef FTI_MACROS_H_
#define FTI_MACROS_H_

void cleanup(char* pattern, ...);

#define MKDIR(a, b)                                                      \
    do {                                                                 \
        if (mkdir(a, b) == -1) {                                         \
            if (errno != EEXIST) {                                       \
                char ErrorString[400];                                   \
                snprintf(ErrorString, sizeof(ErrorString), "FILE %s FUNC %s:%d Cannot create directory: %s", __FILE__, __FUNCTION__, __LINE__, a); \
                FTI_Print(ErrorString, FTI_EROR);                        \
                return FTI_NSCS;                                         \
            }                                                            \
        }                                                                \
    } while (0)

#define RENAME(a, b)                                                     \
    do {                                                                 \
        errno = 0;                                                       \
        if (rename(a, b) != 0) {                                         \
            char ErrorString[1024];                                      \
            snprintf(ErrorString, sizeof(ErrorString), "FILE %s FUNC %s:%d Cannot rename : %s to %s", __FILE__, __FUNCTION__, __LINE__, a, b); \
            FTI_Print(ErrorString, FTI_EROR);                            \
            errno = 0;                                                   \
            return FTI_NSCS;                                             \
        }                                                                \
    } while (0)


#define FREAD(errorCode, bytes, buff, size, number, fd, format, ...)     \
    do {                                                                 \
        bytes = fread(buff, size, number, fd);                           \
        if (ferror(fd)) {                                                \
            char ErrorString[400];                                       \
            snprintf(ErrorString, sizeof(ErrorString), "FILE %s FUNC %s:%d Error Reading File Bytes Read : %d", __FILE__, __FUNCTION__, __LINE__, (int32_t)bytes);  \
            FTI_Print(ErrorString, FTI_EROR);                            \
            cleanup(format, __VA_ARGS__, NULL);                          \
            fclose(fd);                                                  \
            return errorCode;                                            \
        }                                                                \
    } while (0)

#define FWRITE(errorCode, bytes, buff, size, number, fd, format, ...)    \
    do {                                                                 \
        bytes = fwrite(buff, size, number, fd);                          \
        if (ferror(fd)) {                                                \
            char ErrorString[400];                                       \
            snprintf(ErrorString, sizeof(ErrorString), "FILE %s FUNC %s:%d Error Writing File Bytes Written : %d", __FILE__, __FUNCTION__, __LINE__, (int32_t)bytes); \
            FTI_Print(ErrorString, FTI_EROR);                 \
            cleanup(format, __VA_ARGS__, NULL);               \
            fclose(fd);                                       \
            return errorCode;                                 \
        }                                                     \
    } while (0)

#define TRY_ALLOC(dest, dtype, count) dest = (dtype*) calloc(sizeof(dtype), count);\
                                     if (dest == NULL)

#endif  // FTI_MACROS_H_
