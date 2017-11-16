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
 *  @file   ftif.c
 *  @author Julien Bigot <julien.bigot@cea.fr>
 *  @date   February, 2016
 *  @brief Interface to call FTI from Fortran
 */

#include "fti.h"
#include "interface.h"
#include "ftif.h"

/** @brief Fortran wrapper for FTI_Init, Initializes FTI.
 *
 * @return the error status of FTI
 * @param configFile (IN) the name of the configuration file as a
 *        \0 terminated string
 * @param globalComm (INOUT) the "world" communicator, FTI will replace it
 *        with a communicator where its own processes have been removed.
 */
int FTI_Init_fort_wrapper(char* configFile, int* globalComm)
{
    int ierr = FTI_Init(configFile, MPI_Comm_f2c(*globalComm));
    *globalComm = MPI_Comm_c2f(FTI_COMM_WORLD);
    return ierr;
}

/**
 *   @brief      Initializes a data type.
 *   @param      type            The data type to be intialized.
 *   @param      size            The size of the data type to be intialized.
 *   @return     integer         FTI_SCES if successful.
 *
 *   This function initalizes a data type. the only information needed is the
 *   size of the data type, the rest is black box for FTI.
 *
 **/
int FTI_InitType_wrapper(FTIT_type** type, int size)
{
    *type = talloc(FTIT_type, 1);
    return FTI_InitType(*type, size);
}

/**
 @brief      Stores or updates a pointer to a variable that needs to be protected.
 @param      id              ID for searches and update.
 @param      ptr             Pointer to the data structure.
 @param      count           Number of elements in the data structure.
 @param      type            Type of elements in the data structure.
 @return     integer         FTI_SCES if successful.

 This function stores a pointer to a data structure, its size, its ID,
 its number of elements and the type of the elements. This list of
 structures is the data that will be stored during a checkpoint and
 loaded during a recovery.

 **/
/*-------------------------------------------------------------------------*/
int FTI_Protect_wrapper(int id, void* ptr, long count, FTIT_type* type)
{
    return FTI_Protect(id, ptr, count, *type);
}
