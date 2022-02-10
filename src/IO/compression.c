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
 *  @file   compression.c
 *  @date   May, 2019
 *  @brief  Funtions to support posix checkpointing.
 */


#include "../interface.h"

int FTI_InitCompression( FTIT_dataset* data )
{
  data->compression.ptr = malloc(data->size);
  data->compression.size = FTI_DoubleToFloat( data );
}

int FTI_FiniCompression( FTIT_dataset* data )
{
  free(data->compression.ptr);
}

int FTI_InitDecompression( FTIT_dataset* data )
{
  data->compression.ptr = malloc(data->sizeStored);
}

int FTI_FiniDecompression( FTIT_dataset* data )
{
  FTI_FloatToDouble( data );
  free(data->compression.ptr);
}

int64_t FTI_DoubleToFloat( FTIT_dataset* data )
{
  double* values_d = (double*) data->ptr; 
  float* values_f = (float*) data->compression.ptr;
  for(int64_t i=0; i<data->count; i++) {
    values_f[i] = (float) values_d[i];
  }
  return data->count*sizeof(float);
}

int64_t FTI_FloatToDouble( FTIT_dataset* data )
{
  double* values_d = (double*) data->ptr; 
  float* values_f = (float*) data->compression.ptr;
  for(int64_t i=0; i<data->count; i++) {
    values_d[i] = (double) values_f[i];
  }
}
