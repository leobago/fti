
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
 *  @file   diff-checkpoint.h
 *  @date   February, 2018
 *  @brief  Differential checkpointing header file.
 */

#include "interface.h"
#include "checksum.h"
#include <time.h>

/**                                                                             */
/** Global Constans                                                             */

unsigned char *MD5(const unsigned char *d, unsigned long n,
                         unsigned char *md);

uint32_t
fletcher32(const uint16_t *data, size_t len)
{
        uint32_t c0, c1;
        unsigned int i;

        for (c0 = c1 = 0; len >= 360; len -= 360) {
                for (i = 0; i < 360; ++i) {
                        c0 = c0 + *data++;
                        c1 = c1 + c0;
                }
                c0 = c0 % 65535;
                c1 = c1 % 65535;
        }
        for (i = 0; i < len; ++i) {
                c0 = c0 + *data++;
                c1 = c1 + c0;
        }
        c0 = c0 % 65535;
        c1 = c1 % 65535;
        return (c1 << 16 | c0);
}

const uint32_t MOD_ADLER = 65521;

uint32_t adler32(unsigned char *data, size_t len)
/*
    where data is the location of the data in physical memory and
    len is the length of the data in bytes
*/
{
    uint32_t a = 1, b = 0;
    size_t index;

    // Process each byte of the data in order
    for (index = 0; index < len; ++index)
    {
        a = (a + data[index]) % MOD_ADLER;
        b = (b + a) % MOD_ADLER;
    }

    return (b << 16) | a;
}

typedef struct              FTIT_HashBlock
{
    unsigned char*          md5hash;
    uint32_t                bit32hash;
    bool                    dirty;
    bool                    isValid;

}FTIT_HashBlock;

typedef struct              FTIT_DataDiffHash
{
    FTIT_HashBlock*         hashBlocks;
    FTI_ADDRVAL             basePtr;
    long                    nbBlocks;
    long                    totalSize;
    int                     id;

}FTIT_DataDiffHash;

typedef struct              FTIT_DataDiffInfoHash
{
    FTIT_DataDiffHash*      dataDiff;
    int                     nbProtVar;

}FTIT_DataDiffInfoHash;

