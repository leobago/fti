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
 *  @brief  Defines wrapper for POSIX write functions to enable failure injection.
 *
 *  In order ro enable the Failure Injection for I/O (FIIO) mechanism,
 *  we need to pass -DENABLE_FI_IO to the cmake command. We can inject
 *  failures for the write (or fwrite) functions in the following
 *  functions:
 *  
 *  - FTI_WritePosix
 *  - FTIFF_WriteFTIFF
 *  - FTI_RecvPtner
 *  - FTI_RSenc
 *  - FTI_FlushPosix
 *
 *  In order to select the function where we want to inject the failure,
 *  we need to set the environment variable FTI_FI_FUNCTION. For
 *  instance:
 *
 *  FTI_FI_FUNCTION=FTI_WritePosix mpirun -n 8 ./application
 *
 *  We can set the probability for the failure to hapen by setting the
 *  environment variable FTI_FI_PROBABILITY. For instance to inject a
 *  failure in function FTI_WritePosix with probability of 0.5:
 *
 *  FTI_FI_FUNCTION=FTI_WritePosix FTI_FI_PROBABILITY=0.5 mpirun -n 8 ./application
 *
 *  The default value for the probability is 0.1.
 *
 *  @author Kai Keller (kellekai@gmx.de)
 *  @file   failure-injection.h
 *  @date   December, 2018 
 */
#ifndef FTI_FAILURE_INJECTION_H_
#define FTI_FAILURE_INJECTION_H_

#include <fti.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdint.h>
#include <limits.h>

static inline int64_t get_ruint() {
    int64_t buffer;
    int fd = open("/dev/urandom", O_RDWR);
    read(fd, &buffer, 8);
    close(fd);
    return buffer%INT_MAX;
}

void FTI_InitFIIO();
float PROBABILITY();
unsigned int FUNCTION(const char *testFunction);

#ifdef ENABLE_FTI_FI_IO
#define FTI_FI_WRITE(ERR, FD, BUF, COUNT, FN) \
    do { \
        if (FUNCTION(__FUNCTION__)) { \
            if (get_ruint() < ((int64_t)((double)PROBABILITY()*INT_MAX))) { \
                close(FD); \
                FD = open(FN, O_RDONLY); \
            }  \
        } \
        ERR = write(FD, BUF, COUNT); \
        (void)(ERR); \
    } while (0)
#define FTI_FI_FWRITE(ERR, BUF, SIZE, COUNT, FSTREAM, FN) \
    do { \
        if (FUNCTION(__FUNCTION__)) { \
            if (get_ruint() < ((int64_t)((double)PROBABILITY()*INT_MAX))) { \
                fclose(FSTREAM); \
                FSTREAM = fopen(FN, "rb"); \
            } \
        } \
        ERR = fwrite(BUF, SIZE, COUNT, FSTREAM); \
        (void)(ERR); \
    } while (0)
#else
#define FTI_FI_WRITE(ERR, FD, BUF, COUNT, FN) (ERR = write(FD, BUF, COUNT))
#define FTI_FI_FWRITE(ERR, BUF, SIZE, COUNT, FSTREAM, FN) (ERR = fwrite(BUF, SIZE, COUNT, FSTREAM))
#endif

int FTI_FloatBitFlip(float *target, int bit);
int FTI_DoubleBitFlip(double *target, int bit);

#endif  // FTI_FAILURE_INJECTION_H_
