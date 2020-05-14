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

#ifndef FTI_INTERFACE_H_
#define FTI_INTERFACE_H_
#ifdef __cplusplus
extern "C" {
#endif
#include "fti.h"

#include "fortran/ftif.h"

#include "util/ini.h"
#include "util/dataset.h"
#include "util/keymap.h"
#include "util/metaqueue.h"
#include "util/macros.h"
#include "util/utility.h"
#include "util/failure-injection.h"

#include "IO/posix.h"
#include "IO/posix-dcp.h"
#include "IO/hdf5-fti.h"
#include "IO/ftiff.h"
#include "IO/ftiff-dcp.h"
#include "IO/ime.h"

#include "./meta.h"
#include "./api-cuda.h"
#include "./postreco.h"
#include "util/tools.h"
#include "./dcp.h"
#include "./conf.h"
#include "./checkpoint.h"
#include "./stage.h"
#include "./fti-io.h"
#include "./topo.h"
#include "./postckpt.h"
#include "./recover.h"
#include "./icp.h"

#include "deps/md5/md5.h"
#include "deps/iniparser/iniparser.h"
#include "deps/iniparser/dictionary.h"

#include "deps/jerasure/include/galois.h"
#include "deps/jerasure/include/jerasure.h"

#ifdef ENABLE_SIONLIB  // --> If SIONlib is installed
#   include <sion.h>
#endif

#ifdef ENABLE_IME_NATIVE  // --> If IME native API is installed
#   include <ime_native.h>
#endif

#ifdef LUSTRE
#   include "lustreapi.h"
#endif
#ifdef __cplusplus
}
#endif
#endif  // FTI_INTERFACE_H_
