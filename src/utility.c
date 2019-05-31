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
 *  @file   utility.c
 *  @date   October, 2017
 *  @brief  API functions for the FTI library.
 */

#include "interface.h"

#ifdef ENABLE_SIONLIB 
/*-------------------------------------------------------------------------*/
/**
  @brief     Writes data to a file using the SION library
  @param     src    The location of the data to be written 
  @param     size   The number of bytes that I need to write 
  @param     opaque A pointer to the file descriptor  
  @return    integer FTI_SCES if successful.

  Writes the data to a file using the SION library. 

 **/
/*-------------------------------------------------------------------------*/
int write_sion(void *src, size_t size, void *opaque)
{
    int *sid= (int *)opaque;
    int res = sion_fwrite(src, size, 1, *sid);
    if (res < 0 ){
        return FTI_NSCS;
    }
    return FTI_SCES;
}
#endif

/*-------------------------------------------------------------------------*/
/**
  @brief     copies all data of GPU variables to a CPU memory location 
  @param     FTI_Exec Execution Meta data. 
  @param     FTI_Data        Dataset metadata.
  @return    integer FTI_SCES if successful.

  Copis data from the GPU side to the CPU memory  

 **/
/*-------------------------------------------------------------------------*/

int copyDataFromDevive(FTIT_execution* FTI_Exec, FTIT_dataset* FTI_Data){
#ifdef GPUSUPPORT
    int i;
    for (i = 0; i < FTI_Exec->nbVar; i++) {
        if ( FTI_Data[i].isDevicePtr ){
            FTI_copy_from_device( FTI_Data[i].ptr, FTI_Data[i].devicePtr,FTI_Data[i].size,FTI_Exec);
        }
    }
#endif
    return FTI_SCES;
}

