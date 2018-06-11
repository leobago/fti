
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

/**                                                                             */
/** Global Constans                                                             */

#define DIFF_BLOCK_SIZE 4096

typedef struct              FTIT_DataRange
{
    FTI_ADDRVAL             offset;
    FTI_ADDRVAL             size;

}FTIT_DataRange;

typedef struct              FTIT_DataDiffSignal
{
    FTIT_DataRange*         ranges;
    FTI_ADDRVAL             basePtr;
    long                    totalSize;
    long                    rangeCnt;
    int                     id;

}FTIT_DataDiffSignal;

typedef struct              FTIT_DataDiffInfoSignal
{
    FTIT_DataDiffSignal*    dataDiff;
    int                     nbProtVar;

}FTIT_DataDiffInfoSignal;

typedef struct              FTIT_HashBlock
{
    unsigned char*          hash;
    bool                    dirty;

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

//typedef struct              FTIT_PageRange 
//{
//    FTI_ADDRVAL             basePtr;
//    long                  size;
//
//}FTIT_PageRange;
//
//typedef struct              FTIT_PageInfo       /**< bag for dirty pages        */
//{
//    FTI_ADDRVAL*            dirtyPages;         /**< dirty pages array          */
//    FTIT_PageRange*         protPageRanges;     /**< dirty pages array          */
//    long                  dirtyPagesCount;    /**< # of dirty pages           */
//    long                  protPagesCount;     /**< # of dirty pages           */
//
//}FTIT_PageInfo;

/** Function Declarations                                                       */



