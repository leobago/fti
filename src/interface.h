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
 *  @file   interface.h
 *  @date   October, 2017
 *  @brief  Header file for the FTI library private functions.
 */

#ifndef _FTI_INTERFACE_H
#define _FTI_INTERFACE_H



#include "fti.h"

#include "meta.h"
#include "interface.h"
#include "api_cuda.h"
#include "postreco.h"
#include "tools.h"
#include "conf.h"
#include "checkpoint.h"
#include "stage.h"
#include "FTI_IO.h"
#include "topo.h"
#include "IO/posix-dcp.h"
#include "IO/hdf5-fti.h"
#include "IO/ftiff.h"
#include "IO/ftiff-dcp.h"
#include "failure-injection.h"
#include "postckpt.h"
#include "recover.h"
#include "fortran/ftif.h"
#include "incremental-checkpoint.h"
#include "macros.h"
#include "utility.h"


#include "../deps/iniparser/iniparser.h"
#include "../deps/iniparser/dictionary.h"

#include "../deps/jerasure/include/galois.h"
#include "../deps/jerasure/include/jerasure.h"

#ifdef ENABLE_SIONLIB // --> If SIONlib is installed
#   include <sion.h>
#endif

#ifdef ENABLE_HDF5
#include "hdf5.h"
#include "hdf5_hl.h"
#endif


#include <stdint.h>

#define CHUNK_SIZE 131072    /**< MD5 algorithm chunk size.      */

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <errno.h>
#include <math.h>
#include <limits.h>
#include <inttypes.h>
#include <dirent.h>
#include <stdbool.h>
#include <stdint.h>
#include <time.h>
#include <libgen.h>

#ifdef LUSTRE
#   include "lustreapi.h"
#endif

/*---------------------------------------------------------------------------
  Defines
  ---------------------------------------------------------------------------*/

/** Malloc macro.                                                          */
#define talloc(type, num) (type *)malloc(sizeof(type) * (num))

extern int FTI_filemetastructsize;	/**< size of FTIFF_metaInfo in file */
extern int FTI_dbstructsize;		/**< size of FTIFF_db in file       */
extern int FTI_dbvarstructsize;		/**< size of FTIFF_dbvar in file    */

typedef uintptr_t           FTI_ADDRVAL;        /**< for ptr manipulation       */
typedef void*               FTI_ADDRPTR;        /**< void ptr type              */ 

#ifdef FTI_NOZLIB
extern const uint32_t crc32_tab[];

static inline uint32_t crc32_raw(const void *buf, size_t size, uint32_t crc)
{
    const uint8_t *p = (const uint8_t *)buf;

    while (size--)
        crc = crc32_tab[(crc ^ *p++) & 0xFF] ^ (crc >> 8);
    return (crc);
}

static inline uint32_t crc32(const void *buf, size_t size)
{
    uint32_t crc;

    crc = crc32_raw(buf, size, ~0U);
    return (crc ^ ~0U);
}
#endif
/*---------------------------------------------------------------------------
  FTI private functions
  ---------------------------------------------------------------------------*/

//void FTI_PrintMeta(FTIT_execution* FTI_Exec, FTIT_topology* FTI_Topo);

//int FTI_WritePar(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
//        FTIT_topology* FTI_Topo,FTIT_dataset* FTI_Data);
//int FTI_HandleStageRequest(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
//        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt, int source);









#endif

// DIFFERENTIAL CHECKPOINTING


// DIFFERENTIAL CHECKPOINTING POSIX


//INCREMENTAL CHECKPOINTING FOR FTIFF


// INCREMENTAL CHECKPOINTING



