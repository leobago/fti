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

#include <ctype.h>
#include <stdio.h>

#include "fti.h"
#include "../interface.h"
#include "ftif.h"

#define TYPECODE_NONE 0
#define TYPECODE_INT 1
#define TYPECODE_FLOAT 2

/** @brief Fortran wrapper for FTI_Init, Initializes FTI.
 *
 * @return the error status of FTI
 * @param configFile (IN) the name of the configuration file as a
 *        \0 terminated string
 * @param globalComm (INOUT) the "world" communicator, FTI will replace it
 *        with a communicator where its own processes have been removed.
 */
int FTI_Init_fort_wrapper(char* configFile, int* globalComm) {
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
int FTI_InitType_wrapper(FTIT_type** type, int size) {
    *type = talloc(FTIT_type, 1);
    return FTI_InitType(*type, size);
}

/**
 *   @brief      Initialize a FTIT_Type structure for a Fortran primitive type
 *   @param      type            The data type to be intialized
 *   @param      name            Fortran typename string
 *   @param      size            Type size in bytes
 *   @return     integer         FTI_SCES if successful
 *
 *   This method first try to associate the Fortran primitive type to a C type.
 *   It does so by parsing the name and then the size of the data type.
 *   If there is no direct correlation between C an Fortran, create a new type.
 *
 *   WARNING: We assume that C and Fortran types share the same binary format.
 *   For instance, the C float is usually defined by the IEEE 754 format.
 *   We would assume that Fortran real(4) types are also encoded as IEEE 754.
 *   This is usually the case but might be an error source on some compilers.
 **/
int FTI_InitPrimitiveType_C(FTIT_type** type, const char *name, int size) {
    int typecode = TYPECODE_NONE;
    char *dest;
    int i = 0;
    // Copy the type name to lower case ignoring non-letters
    dest  = talloc(char, strlen(name));
    for (const char *p=name; *p; ++p) {
      if ((*p >= 'a' && *p <= 'z') || (*p >= 'A' && *p <= 'Z'))
        dest[i++] = tolower(*p);
    }
    dest[i] = '\0';
    // Discover the fundamental format behind the type name
    if (strcmp(dest, "integer") == 0 ||
      strcmp(dest, "logical") == 0 ||
      strcmp(dest, "character") == 0)
      typecode = TYPECODE_INT;
    else if (strcmp(dest, "real") == 0)
      typecode = TYPECODE_FLOAT;
    free(dest);
    // Find the static FTIT_Type object mapped to the primitive
    switch (typecode) {
    case TYPECODE_INT:
      switch (size) {
      case sizeof(char):
        *type = &FTI_CHAR;
        break;
      case sizeof(short):
        *type = &FTI_SHRT;
        break;
      case sizeof(int):
        *type = &FTI_INTG;
        break;
      case sizeof(long):
        *type = &FTI_LONG;
        break;
      default:
        return FTI_InitType_wrapper(type, size);
      }
      break;
    case TYPECODE_FLOAT:
      switch (size) {
      case sizeof(float):
        *type = &FTI_SFLT;
        break;
      case sizeof(double):
        *type = &FTI_DBLE;
        break;
      case sizeof(long double):
        *type = &FTI_LDBE;
        break;
      default:
        return FTI_InitType_wrapper(type, size);
      }
      break;
      default:
        return FTI_InitType_wrapper(type, size);
    }
    return FTI_SCES;
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
int FTI_Protect_wrapper(int id, void* ptr, int32_t count, FTIT_type* type) {
    return FTI_Protect(id, ptr, count, *type);
}

int FTI_SetAttribute_string_wrapper(int id, char* attribute, int flag) {
    if ( (flag & FTI_ATTRIBUTE_NAME) == FTI_ATTRIBUTE_NAME ) {
        FTIT_attribute att;
        strncpy(att.name, attribute, FTI_BUFS);
        return FTI_SetAttribute(id, att, flag);
    }
    return FTI_SCES;
}

int FTI_SetAttribute_long_array_wrapper(int id, int ndims,
        int64_t* attribute, int flag) {
    if ( (flag & FTI_ATTRIBUTE_DIM) == FTI_ATTRIBUTE_DIM ) {
        FTIT_attribute att;
        att.dim.ndims = ndims;
        int i = 0; for (; i < ndims; i++) {
            if (attribute[i] < 0) return FTI_NSCS;
            att.dim.count[i] = attribute[i];
        }
        return FTI_SetAttribute(id, att, flag);
    }
    return FTI_SCES;
}

/**
 *   @brief      Initializes a complex hdf5 data type.
 *   @param      newType         The data type to be intialized.
 *   @param      typeDefinition  The definition of the data type to be intialized.
 *   @param      length          Number of fields in structure.
 *   @param      size            Size of the structure.
 *   @param      name            Name of the structure.
 *   @return     integer         FTI_SCES if successful.
 *
 *   This function initalizes a complex data type. the information needed is passed
 *   in typeDefinition, the rest is black box for FTI.
 *
 **/
int FTI_InitComplexType_wrapper(FTIT_type** newType,
 FTIT_complexType* typeDefinition, int length, size_t size, char* name,
 FTIT_H5Group* h5group) {
    *newType = talloc(FTIT_type, 1);
    return FTI_InitComplexType(*newType, typeDefinition,
            length, size, name, h5group);
}
