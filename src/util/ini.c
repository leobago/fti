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

#include "../interface.h"

int FTI_Iniparser(FTIT_iniparser* self, const char* inifile,
 FTIT_inimode mode) {
    char err[FTI_BUFS];

    if (self == NULL) {
        FTI_Print("iniparser context is NULL.", FTI_EROR);
        return FTI_NSCS;
    }

    if (inifile == NULL) {
        FTI_Print("iniparser file is NULL.", FTI_EROR);
        return FTI_NSCS;
    }

    bool create = (mode == FTI_INI_CREATE) ? true : false;

    if (access(inifile, R_OK) != 0) {
        if (mode == FTI_INI_OPEN) {
            snprintf(err, FTI_BUFS,
             "Iniparser failed to open meta file ('%s').", inifile);
            FTI_Print(err, FTI_DBUG);
            return FTI_NSCS;
        }  else if (mode == FTI_INI_APPEND) {
            create = true;
        }
    }

    self->dict = NULL;

    if (create) {
        self->dict = dictionary_new(0);
    } else {
        self->dict = iniparser_load(inifile);
    }

    if (self->dict == NULL) {
        if (mode == FTI_INI_OPEN) {
            snprintf(err, FTI_BUFS,
             "Iniparser failed to parse the file ('%s').", inifile);
        } else if (mode == FTI_INI_APPEND) {
            snprintf(err, FTI_BUFS,
             "Iniparser failed to parse the file ('%s').", inifile);
        } else {
            snprintf(err, FTI_BUFS, "Unknown iniparser mode.");
        }
        FTI_Print(err, FTI_WARN);
        return FTI_NSCS;
    }

    self->getString = FTI_IniparserGetString;
    self->getInt = FTI_IniparserGetInt;
    self->getLong = FTI_IniparserGetLong;
    self->getInt64 = FTI_IniparserGetInt64;
    self->set = FTI_IniparserSet;
    self->dump = FTI_IniparserDump;
    self->clear = FTI_IniparserClear;
    strncpy(self->file, inifile, FTI_BUFS);

    return FTI_SCES;
}

int FTI_IniparserSet(FTIT_iniparser* self, const char* key, const char* val) {
    if (self == NULL) {
        FTI_Print("iniparser context is NULL.", FTI_EROR);
        return FTI_NSCS;
    }

    return (iniparser_set(self->dict, key, val) == 0) ? FTI_SCES : FTI_NSCS;
}

char* FTI_IniparserGetString(FTIT_iniparser* self, const char* key) {
    static char nullstr = '\0';

    if (self == NULL) {
        FTI_Print("iniparser context is NULL.", FTI_EROR);
        return &nullstr;
    }

    char* string = iniparser_getstring(self->dict, key, NULL);

    if (string == NULL) return &nullstr;

    return string;
}

int FTI_IniparserGetInt(FTIT_iniparser* self, const char* key) {
    if (self == NULL) {
        FTI_Print("iniparser context is NULL.", FTI_EROR);
        return FTI_NSCS;
    }

    return iniparser_getint(self->dict, key, -1);
}

int FTI_IniparserGetLong(FTIT_iniparser* self, const char* key) {
    if (self == NULL) {
        FTI_Print("iniparser context is NULL.", FTI_EROR);
        return FTI_NSCS;
    }

    return iniparser_getlint(self->dict, key, -1);
}

int64_t FTI_IniparserGetInt64(FTIT_iniparser* self, const char* key) {
    if (self == NULL) {
        FTI_Print("iniparser context is NULL.", FTI_EROR);
	return FTI_NSCS;
    }

    char *string = iniparser_getstring(self->dict, key, NULL);
    if (string == NULL) return -1;

    return strtoll(string, NULL, 0);
}

int FTI_IniparserDump(FTIT_iniparser* self) {
    if (self == NULL) {
        FTI_Print("iniparser context is NULL.", FTI_EROR);
        return FTI_NSCS;
    }

    char err[FTI_BUFS];

    FILE* fd = fopen(self->file, "w");
    if (fd == NULL) {
        snprintf(err, FTI_BUFS, "Iniparser failed to open file '%s'",
         self->file);
        FTI_Print(err, FTI_WARN);

        return FTI_NSCS;
    }

    // Write metadata
    iniparser_dump_ini(self->dict , fd);

    if (fclose(fd) != 0) {
        snprintf(err, FTI_BUFS, "Iniparser failed to close file '%s'",
         self->file);
        FTI_Print(err, FTI_WARN);

        return FTI_NSCS;
    }

    return FTI_SCES;
}

int FTI_IniparserClear(FTIT_iniparser* self) {
    if (self == NULL) {
        FTI_Print("iniparser context is NULL.", FTI_EROR);
        return FTI_NSCS;
    }

    iniparser_freedict(self->dict);
    return FTI_SCES;
}
