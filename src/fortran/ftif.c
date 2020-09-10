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

#include "ftif.h"

#define TYPECODE_NONE 0
#define TYPECODE_INT 1
#define TYPECODE_FLOAT 2
#define TYPECODE_CHAR 3

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
 *   @brief      Registers a Fortran primitive in FTI type system
 *   @param      name            Fortran data type mnemonic string
 *   @param      size            The data type size in bytes
 *   @return     fti_id_t        A handle for the new data type
 *
 *   This method first try to associate the Fortran primitive type to a C type.
 *   It does so by parsing the type mnemonic and then its size.
 *   Integers and logicals are mapped to integers of size 1, 2, 4 and 8.
 *   Reals are mapped to float, real and long real.
 *   Character is mapped to char regardless of size (byte array).
 *   If there is no direct correlation between C an Fortran, create a new type.
 *
 *   WARNING: We assume that C and Fortran types share the same binary format.
 *   For instance, the C float is defined by the IEEE 754 format.
 *   We would assume that Fortran real(4) types are also encoded as IEEE 754.
 *   This is usually the case but might be an error source on some compilers.
 **/
fti_id_t FTI_InitPrimitiveType_C(const char *name, int size) {
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
      strcmp(dest, "logical") == 0)
      typecode = TYPECODE_INT;
    else if (strcmp(dest, "real") == 0)
      typecode = TYPECODE_FLOAT;
    else if (strcmp(dest, "character") == 0)
      typecode = TYPECODE_CHAR;
    free(dest);
    // Find the static FTIT_Type object mapped to the primitive
    switch (typecode) {
    case TYPECODE_INT:
      switch (size) {
      case sizeof(char):
        return FTI_CHAR;
      case sizeof(short):
        return FTI_SHRT;
      case sizeof(int):
        return FTI_INTG;
      case sizeof(long):
        return FTI_LONG;
      default:
        return FTI_InitType(size);
      }
      break;
    case TYPECODE_FLOAT:
      switch (size) {
      case sizeof(float):
        return FTI_SFLT;
      case sizeof(double):
        return FTI_DBLE;
      case sizeof(long double):
        return FTI_LDBE;
      default:
        return FTI_InitType(size);
      }
      break;
      case TYPECODE_CHAR:
        return FTI_CHAR;
      default:
        return FTI_InitType(size);
    }
    return FTI_SCES;
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
 *   @brief      Initializes an hdf5-like empty complex data type.
 *   @param      size            Size of the structure.
 *   @param      name            Name of the structure.
 *   @return     integer         FTI_SCES if successful, FTI_NSCS otherwise.
 *
 *   The components are added with FTI_AddSimpleField and FTI_AddComplexField.
 *
 **/
fti_id_t FTI_InitComplexType_wrapper(char* name, int size) {
  return FTI_InitComplexType(name, size, NULL);
}
