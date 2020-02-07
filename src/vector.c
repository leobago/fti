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

int FTI_VectorKey( FTIT_vectorkey* self, size_t type_size, FTIT_configuration FTI_Conf )
{
    if( self == NULL ) {
        FTI_Print("Call to FTI_VectorKey with 'NULL' vector",FTI_EROR);
        return FTI_NSCS;
    }
    
    if( type_size == 0 ) {
        FTI_Print("Call to FTI_VectorKey with typesize '0' is invalid",FTI_EROR);
        return FTI_NSCS;
    }

    self->_type_size = type_size;
    self->_size = 0;
    self->_used = 0;
    self->_data = NULL;
    self->_max_id = FTI_Conf.maxVarId;
    self->_key = talloc( int, self->_max_id+1 );

    int i=0; for(; i<FTI_Conf.maxVarId+1; i++)
        self->_key[i] = -1;

    self->push_back = FTI_VectorKeyPushBack;
    self->data = FTI_VectorKeyGet;
    self->clear = FTI_VectorKeyClear;
}

int FTI_VectorKeyPushBack( FTIT_vectorkey* self, void* new_item, int id )
{
    if( self == NULL ) {
        FTI_Print("Call to FTI_PushBack with 'NULL' vector",FTI_EROR);
        return FTI_NSCS;
    }
    
    if( new_item == NULL ) {
        FTI_Print("Call to FTI_PushBack with 'NULL' data",FTI_EROR);
        return FTI_NSCS;
    }

    if( id < 0 ) {
        FTI_Print("ids have to be positive",FTI_EROR);
        return FTI_NSCS;
    }

    if( id > self->_max_id ) { 
        FTI_Print("id is larger than 'max_id' for vector",FTI_EROR);
        return FTI_NSCS;
    }

    size_t new_size = self->_size;
    size_t new_used = self->_used + 1;

    if( new_used > self->_size ) {
        size_t new_block = (self->_size > 0) ? self->_size * self->_type_size : self->_type_size;
        new_size = (new_block > FTI_MAX_REALLOC) ? self->_size + FTI_MAX_REALLOC/self->_type_size : (self->_size > 0) ? self->_size*2 : 2;
        void* alloc = realloc( self->_data, new_size * self->_type_size );
        if(!alloc) {
            FTI_Print("Failed to extent selftor size", FTI_EROR);
            return FTI_NSCS;
        }
        self->_data = alloc;
    }
    
    memcpy(self->_data + self->_used*self->_type_size, new_item, self->_type_size);
    self->_key[id] = self->_used;
    self->_used = new_used;
    self->_size = new_size;
}

void* FTI_VectorKeyGet( FTIT_vectorkey* self, int id )
{
    if( self == NULL ) {
        FTI_Print("Call to FTI_VectorKeyGet with 'NULL' vector",FTI_EROR);
        return NULL;
    }

    if( id < 0 ) {
        FTI_Print("ids have to be positive",FTI_EROR);
        return NULL;
    }

    if( id > self->_max_id ) { 
        FTI_Print("id is larger than 'max_id' for vector",FTI_EROR);
        return NULL;
    }
    
    size_t check_pos = self->_key[id];
    
    if( check_pos == -1 ) {
        FTI_Print("id is invalid", FTI_EROR );
        return NULL;
    }

    if( check_pos > (self->_size - 1) ) {
        FTI_Print("data location out of bounds", FTI_EROR );
        return NULL;
    }
    
    return self->_data + check_pos * self->_type_size;
}

void* FTI_VectorKeyClear( FTIT_vectorkey* self ) 
{
        self->_size = 0;
        self->_used = 0;
        free(self->_data);
        free(self->_key);
}
