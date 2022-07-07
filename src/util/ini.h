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
 *  @file   ini.c
 *  @date   March, 2020
 *  @author Kai Keller (kellekai@gmx.de)
 *  @brief  methods that simplify the usage of the iniparser library.
 */

#ifndef FTI_INI_H_
#define FTI_INI_H_

#include "../deps/iniparser/iniparser.h"
#include "../deps/iniparser/dictionary.h"

/**--------------------------------------------------------------------------
  
  
  @brief        File access mode for FTI_Iniparser function.

  FTI_INI_OPEN: The filename passed to \ref FTI_Iniparser has to
  be accessible, otherwise the initialization will fail.  


  FTI_INI_CREATE: A new file is created. If file exists already,
  it will be deleted.  
  
  
  FTI_INI_APPEND: If file exists, only the changes will be applied.
  If the file does not exists, a new file is created.


--------------------------------------------------------------------------**/
typedef enum FTIT_inimode {
    /**< pass to FTI_Iniparser -> open existing file       */
    FTI_INI_OPEN,
    /**< pass to FTI_Iniparser -> create new file          */
    FTI_INI_CREATE,
    /**< pass to FTI_Iniparser -> append to existing file or create new one. */
    FTI_INI_APPEND,
} FTIT_inimode;

/**--------------------------------------------------------------------------
  
  
  @brief        FTI_Iniparser handle.

  This structure serves as the opaque iniparser handle. It contains the 
  pointer to the dictionary and the belonging filepath.

  The structure mimics C++ class behaviour to simplify the handling. The 
  FTI functions that operate on the dictionary are members of this 
  structure.


--------------------------------------------------------------------------**/
typedef struct FTIT_iniparser {
    dictionary* dict;           /**< Pointer to iniparser dictionary       */
    char        filetmp[FTI_BUFS]; /**< Path to corresponding file            */
    char        file[FTI_BUFS]; /**< Path to corresponding file            */
    int         (*getSections)(struct FTIT_iniparser*, char**, int);
    bool        (*isSection)(struct FTIT_iniparser*, const char*);
    char*       (*getString)(struct FTIT_iniparser*, const char*);
    int         (*getInt)(struct FTIT_iniparser*, const char*);
    bool        (*getBool)(struct FTIT_iniparser*, const char*);
    long         (*getLong)(struct FTIT_iniparser*, const char*);
    int         (*set)(struct FTIT_iniparser*, const char*,
                                 const char*);
    int         (*dump)(struct FTIT_iniparser*);
    int         (*clear)(struct FTIT_iniparser*);
} FTIT_iniparser;

/**--------------------------------------------------------------------------
  
  
  @brief        Initializes instance of FTIT_iniparser.

  This function initializes the dictionary. If mode = \ref FTI_INI_OPEN, 
  the dictionary will be created by parsing the file passed to the function.
  The same applies for mode = \ref FTI_INI_APPEND, if the file exists. If not,
  the bahavior is the same as for mode = \ref FTI_INI_CREATE. 
  If mode = FTI_INI_CREATE, the file passed to the function is created. If 
  the file already exists, it will be removed.

  @param        self[out]    <b> FTIT_iniparser* </b> Handle that needs to
  be passed to the other FTI_Iniparser functions. Cannot be NULL.
  @param        inifile[in]  <b> const char* </b> Path to ini file.
  @param        mode[in]     <b> FTIT_inimode </b> File access mode.
  
  @return                       \ref FTI_SCES on success.  
                                \ref FTI_NSCS upon failure.
 

--------------------------------------------------------------------------**/
int FTI_Iniparser(FTIT_iniparser*, const char*, FTIT_inimode);

/**--------------------------------------------------------------------------
  
  
  @brief        Requests string value for key.

  This function returns the string value that is set for key. If key was not
  found, the null string '\0' is returned.

  @param        self[in]    <b> FTIT_iniparser* </b> FTI_Iniparser handle.
  @param        n[out]      <b> int* </b> number of sections.
  
  @return                       String array of section names.
                                NULL on error.
 

--------------------------------------------------------------------------**/
int FTI_IniparserGetSections(FTIT_iniparser*, char**, int);

bool FTI_IniparserIsSection(FTIT_iniparser*, const char*);

/**--------------------------------------------------------------------------
  
  
  @brief        Requests string value for key.

  This function returns the string value that is set for key. If key was not
  found, the null string '\0' is returned.

  @param        self[in]    <b> FTIT_iniparser* </b> FTI_Iniparser handle.
  @param        key[in]     <b> const char* </b> dictionary key.
  
  @return                       Dictionary value on success.  
                                Null string ('\0') if key not found.
 

--------------------------------------------------------------------------**/
char* FTI_IniparserGetString(FTIT_iniparser*, const char*);

/**--------------------------------------------------------------------------
  
  
  @brief        Requests integer value for key.

  This function returns the integer value that is set for key. If key was not
  found, -1 is returned.

  @param        self[in]    <b> FTIT_iniparser* </b> FTI_Iniparser handle.
  @param        key[in]     <b> const char* </b> dictionary key.
  
  @return                       Dictionary value on success.  
                                <b> int </b> -1 if key not found.
 

--------------------------------------------------------------------------**/
int FTI_IniparserGetInt(FTIT_iniparser*, const char* key);
  
// FIXME error handling! we have to change the getBool function
// currently it does not allow error handling.
bool FTI_IniparserGetBool(FTIT_iniparser*, const char* key);

/**--------------------------------------------------------------------------
  
  
  @brief        Requests long integer value for key.

  This function returns the long integer value that is set for key. If key was not
  found, -1 is returned.

  @param        self[in]    <b> FTIT_iniparser* </b> FTI_Iniparser handle.
  @param        key[in]     <b> const char* </b> dictionary key.
  
  @return                       Dictionary value on success.  
                                <b> long </b> -1 if key not found.
 

--------------------------------------------------------------------------**/
long FTI_IniparserGetLong(FTIT_iniparser*, const char* key);

/**--------------------------------------------------------------------------
  
  
  @brief        Sets key to val.

  This function assigns val to key. If for val NULL is passed, the key
  will be the name of a section in the iniparser file.

  @param        self[in/out] <b> FTIT_iniparser* </b> FTI_Iniparser handle.
  @param        key[in]      <b> const char* </b> dictionary key.
  @param        val[in]      <b> const char* </b> key value.
  
  @return                       \ref FTI_SCES on success.  
                                \ref FTI_NSCS on failure.

 
--------------------------------------------------------------------------**/
int FTI_IniparserSet(FTIT_iniparser*, const char*, const char*);

/**--------------------------------------------------------------------------
  
  
  @brief        Writes dictionary to file.

  This function writes the dictionary to the iniparser file. 

  @param        self[in] <b> FTIT_iniparser* </b> FTI_Iniparser handle.
  
  @return                       \ref FTI_SCES on success.  
                                \ref FTI_NSCS upon failure.


--------------------------------------------------------------------------**/
int FTI_IniparserDump(FTIT_iniparser*);

/**--------------------------------------------------------------------------
  
  
  @brief        Destroys FTI_Iniparser instance.

  This function frees all the memory allocated for the dictionary by calling
  the corresponding iniparser library function.

  @param        self[in] <b> FTIT_iniparser* </b> FTI_Iniparser handle.
  
  @return                       \ref FTI_SCES on success.  
                                \ref FTI_NSCS upon failure.


--------------------------------------------------------------------------**/
int FTI_IniparserClear(FTIT_iniparser*);

#endif  // FTI_INI_H_

