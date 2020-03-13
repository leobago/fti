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


