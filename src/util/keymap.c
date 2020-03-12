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
 *  @file   keymap.c
 *  @date   March, 2020
 *  @author Kai Keller (kellekai@gmx.de)
 *  @brief  methods for key-value container.
 */

#include "../interface.h"

/*-------------------------------------------------------------------------*/
/** 
    @var static FTIT_keymap self  
    Singleton instance of the key value container.
**/
/*-------------------------------------------------------------------------*/
static FTIT_keymap self;

int FTI_KeyMap( FTIT_keymap** instance, long type_size, long max_key )
{

    if( type_size == 0 ) {
        FTI_Print("Call to FTI_KeyMap with typesize '0' is invalid",FTI_EROR);
        return FTI_NSCS;
    }

    assert( self.initialized == false && "Only one instance of FTIT_keymap is allowed!" );
    *instance = &self;

    self._type_size = type_size;
    self._max_key = max_key;
    self._key = talloc( int, max_key+1 );

    int i=0; for(; i<max_key+1; i++)
        self._key[i] = -1;

    self.push_back = FTI_KeyMapPushBack;
    self.data = FTI_KeyMapData;
    self.get = FTI_KeyMapGet;
    self.clear = FTI_KeyMapClear;
    self.initialized = true;

    return FTI_SCES;

}

int FTI_KeyMapPushBack( void* new_item, int key )
{

    char str[FTI_BUFS];
    
    if( !self.initialized ) {
        FTI_Print("keymap not initialized",FTI_EROR);
        return FTI_NSCS;
    }

    if( new_item == NULL ) {
        FTI_Print("Call to FTI_PushBack with 'NULL' data",FTI_EROR);
        return FTI_NSCS;
    }

    if( key < 0 ) {
        FTI_Print("key for FTIT_keymap has to be positive",FTI_EROR);
        return FTI_NSCS;
    }

    if( key > self._max_key ) {
        snprintf( str, FTI_BUFS, "key is larger than 'max_key = %d' for keymap", self._max_key );
        FTI_Print( str, FTI_EROR);
        return FTI_NSCS;
    }

    if( self._key[key] != -1 ) {
        snprintf( str, FTI_BUFS, "Requested key='%d' is already in use", key );
        FTI_Print( str, FTI_EROR);
        return FTI_NSCS;
    }

    long new_size = self._size;
    long new_used = self._used + 1;

    if( new_used > self._size ) {
        
        // double container size each time limit is reached except 
        // new extra chunk would be larger than FTI_MAX_REALLOC * self._type_size

        if( self._size == 0 ) {

            new_size = FTI_MIN_REALLOC;
        
        } else {

            new_size = ( self._size > FTI_MAX_REALLOC ) ? self._size + FTI_MAX_REALLOC : self._size * 2;

        }

        void* alloc = realloc( self._data, new_size * self._type_size );
        
        if(!alloc) {
            FTI_Print("Failed to extent keymap size", FTI_EROR);
            return FTI_NSCS;
        }

        self._data = alloc;
    
    }

    memcpy(self._data + self._used*self._type_size, new_item, self._type_size);
    
    self._key[key] = self._used;
    self._used = new_used;
    self._size = new_size;

    return FTI_SCES;

}

int FTI_KeyMapData( FTIT_dataset** data, int n )
{

    if( !self.initialized ) {
        FTI_Print("keymap not initialized",FTI_EROR);
        return FTI_NSCS;
    }

    if( n > self._used ) {
        FTI_Print("keymap out of bounds",FTI_EROR);
        return FTI_NSCS;
    }

    *data = self._data;

    return FTI_SCES;

}

int FTI_KeyMapGet( FTIT_dataset** data, int key )
{

    if( !self.initialized ) {
        FTI_Print("keymap not initialized",FTI_EROR);
        return FTI_NSCS;
    }

    if( key < 0 ) {
        FTI_Print("key has to be positive",FTI_EROR);
        return FTI_NSCS;
    }

    if( key > self._max_key ) { 
        FTI_Print("key is larger than 'max_key' for keymap",FTI_EROR);
        return FTI_NSCS;
    }

    long check_pos = self._key[key];

    if( check_pos > (self._used - 1) ) {
        FTI_Print("data location out of bounds", FTI_EROR );
        return FTI_NSCS;
    }

    if( check_pos == -1 ) {
        // key not in use
        *data = NULL;
        return FTI_SCES;
    }

    *data = self._data + check_pos * self._type_size;

    return FTI_SCES;

}

int FTI_KeyMapClear() 
{

    if( !self.initialized ) {
        FTI_Print("keymap not initialized",FTI_EROR);
        return FTI_NSCS;
    }

    free(self._data);
    free(self._key);
    
    FTIT_keymap reset = {0};
    
    self = reset;
    
    return FTI_SCES;

}
