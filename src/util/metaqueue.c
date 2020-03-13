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
 *  @file   metaqueue.c
 *  @date   March, 2020
 *  @author Kai Keller (kellekai@gmx.de)
 *  @brief  methods for FTIT_mqueue, a queue for the FTI checkpoint meta data (FTIT_metadata).
 */

#include "../interface.h"

int FTI_MetadataQueue( FTIT_mqueue* q )
{

    if( q == NULL ) {
        FTI_Print("metadata queue context is NULL", FTI_WARN);
        return FTI_NSCS;
    }

    FTIT_mqueue init = {0};
    *q = init;

    q->_front = NULL;

    q->push = FTI_MetadataQueuePush;
    q->pop = FTI_MetadataQueuePop;
    q->empty = FTI_MetadataQueueEmpty;
    q->clear = FTI_MetadataQueueClear;
    
    q->_initialized = true;

    return FTI_SCES;

}

int FTI_MetadataQueuePush( FTIT_mqueue* mqueue, FTIT_metadata data )
{

    if( mqueue == NULL ) {
        FTI_Print("metadata queue context is NULL", FTI_WARN);
        return FTI_NSCS;
    }
    
    if( !mqueue->_initialized ) {
        FTI_Print("metadata queue is not initialized", FTI_WARN);
        return FTI_NSCS;
    }

    FTIT_mnode* old = mqueue->_front;
    FTIT_mnode* new = malloc( sizeof(FTIT_mnode) );

    new->_data = malloc( sizeof(FTIT_metadata) );
    *new->_data = data;
    new->_next = NULL; 

    mqueue->_front = new;
    mqueue->_front->_next = old;

    return FTI_SCES;

}

int FTI_MetadataQueuePop( FTIT_mqueue* mqueue, FTIT_metadata* data )
{

    if( mqueue == NULL ) {
        FTI_Print("metadata queue context is NULL", FTI_WARN);
        return FTI_NSCS;
    }
    
    if( !mqueue->_initialized ) {
        FTI_Print("metadata queue is not initialized", FTI_WARN);
        return FTI_NSCS;
    }
    
    if( !mqueue->_front ) return FTI_NSCS;
    if( !mqueue->_front->_data ) return FTI_NSCS;

    if( data )
        memcpy( data, mqueue->_front->_data, sizeof(FTIT_metadata) );

    FTIT_mnode* pop = mqueue->_front;

    mqueue->_front = mqueue->_front->_next;

    free(pop->_data);
    free(pop);

    return FTI_SCES;

}

bool FTI_MetadataQueueEmpty( FTIT_mqueue* mqueue )
{
    
    if( mqueue == NULL ) {
        FTI_Print("metadata queue context is NULL", FTI_WARN);
        return true;
    }
    
    if( !mqueue->_initialized ) {
        FTI_Print("metadata queue is not initialized", FTI_WARN);
        return true;
    }

    return (mqueue->_front == NULL);

}

int FTI_MetadataQueueClear( FTIT_mqueue* mqueue )
{
    
    if( mqueue == NULL ) {
        FTI_Print("metadata queue context is NULL", FTI_WARN);
        return FTI_NSCS;
    }

    if( !mqueue->_initialized ) {
        FTI_Print("metadata queue is not initialized", FTI_WARN);
        return FTI_NSCS;
    }
    
    while( !mqueue->empty( mqueue ) )
        mqueue->pop( mqueue, NULL );
        
    return FTI_SCES;

}


