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
 *  @file   tools.c
 *  @date   October, 2017
 *  @brief  Utility functions for the FTI library.
 */

#include "interface.h"

// This class is meant to be a singleton
static FTIT_keymap self = {0};

int FTI_KeyMap( FTIT_keymap** instance, size_t type_size, size_t max_key )
{
    
    if( type_size == 0 ) {
        FTI_Print("Call to FTI_KeyMap with typesize '0' is invalid",FTI_EROR);
        return FTI_NSCS;
    }

    assert( self.initialized == false && "Only one instance of FTIT_keymap is allowed!" );
    *instance = &self;

    self._type_size = type_size;
    self._max_id = max_key;
    self._key = talloc( int, max_key+1 );

    int i=0; for(; i<max_key+1; i++)
        self._key[i] = -1;

    self.push_back = FTI_KeyMapPushBack;
    self.get = FTI_KeyMapGet;
    self.clear = FTI_KeyMapClear;
    self.check = FTI_KeyMapCheckError;
    self.check_range = FTI_KeyMapCheckRange;
    self.initialized = true;
}

int FTI_KeyMapPushBack( void* new_item, int id )
{
    if( !self.initialized ) {
        FTI_Print("keymap not initialized",FTI_EROR);
        return FTI_NSCS;
    }
    
    if( new_item == NULL ) {
        FTI_Print("Call to FTI_PushBack with 'NULL' data",FTI_EROR);
        return FTI_NSCS;
    }

    if( id < 0 ) {
        FTI_Print("id for FTIT_keymap has to be positive",FTI_EROR);
        return FTI_NSCS;
    }

    if( id > self._max_id ) {
        char str[FTI_BUFS];
        snprintf( str, FTI_BUFS, "id is larger than 'max_id = %d' for keymap", self._max_id );
        FTI_Print( str, FTI_EROR);
        return FTI_NSCS;
    }
    
    if( self._key[id] != -1 ) {
        char str[FTI_BUFS];
        snprintf( str, FTI_BUFS, "Requested ID='%d' is already in use", id );
        FTI_Print( str, FTI_EROR);
        return FTI_NSCS;
    }
        
    size_t new_size = self._size;
    size_t new_used = self._used + 1;

    if( new_used > self._size ) {
        size_t new_block = (self._size > 0) ? self._size * self._type_size : self._type_size;
        new_size = (new_block > FTI_MAX_REALLOC) ? self._size + FTI_MAX_REALLOC/self._type_size : (self._size > 0) ? self._size*2 : 2;
        void* alloc = realloc( self.data, new_size * self._type_size );
        if(!alloc) {
            FTI_Print("Failed to extent keymap size", FTI_EROR);
            return FTI_NSCS;
        }
        self.data = alloc;
    }
    
    memcpy(self.data + self._used*self._type_size, new_item, self._type_size);
    self._key[id] = self._used;
    self._used = new_used;
    self._size = new_size;

    return FTI_SCES;
}

void* FTI_KeyMapGet( int id )
{
    
    if( !self.initialized ) {
        FTI_Print("keymap not initialized",FTI_EROR);
        return NULL;
    }

    self._error=false;

    if( id < 0 ) {
        FTI_Print("id has to be positive",FTI_EROR);
        self._error=true;
        return NULL;
    }

    if( id > self._max_id ) { 
        FTI_Print("id is larger than 'max_id' for keymap",FTI_EROR);
        self._error=true;
        return NULL;
    }
    
    size_t check_pos = self._key[id];
    
    if( check_pos == -1 ) {
        // id not in use
        return NULL;
    }
   
    if( check_pos > (self._used - 1) ) {
        FTI_Print("data location out of bounds", FTI_EROR );
        self._error=true;
        return NULL;
    }
    
    return self.data + check_pos * self._type_size;

}

bool FTI_KeyMapCheckError()
{
    if( !self.initialized ) {
        FTI_Print("keymap not initialized",FTI_EROR);
        return true;
    }

    bool val = self._error;
    self._error = false;
    return val;
}

bool FTI_KeyMapCheckRange( int max )
{
    if( !self.initialized ) {
        FTI_Print("keymap not initialized",FTI_EROR);
        return false;
    }

    return (self._used <= max);
}

int FTI_KeyMapClear() 
{
    if( !self.initialized ) {
        FTI_Print("keymap not initialized",FTI_EROR);
        return FTI_NSCS;
    }

    self._size = 0;
    self._used = 0;
    free(self.data);
    self.data = NULL;
    free(self._key);
    self._key = NULL;
    return FTI_SCES;
}
