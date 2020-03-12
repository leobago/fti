#include "../interface.h"

static FTIT_mqueue* mqueue;

int FTI_MetadataQueue( FTIT_mqueue* q )
{
    if( q == NULL ) {
        FTI_Print("metadata queue context is NULL", FTI_WARN);
        return FTI_NSCS;
    }

    q->_front = NULL;

    q->push = FTI_MetadataQueuePush;
    q->pop = FTI_MetadataQueuePop;
    q->empty = FTI_MetadataQueueEmpty;
    q->clear = FTI_MetadataQueueClear;

    mqueue = q;

    return FTI_SCES;
}

int FTI_MetadataQueuePush( FTIT_metadata data )
{

    if( mqueue == NULL ) {
        FTI_Print("metadata queue context is NULL", FTI_WARN);
        return FTI_NSCS;
    }

    FTIT_mnode* old = mqueue->_front;
    FTIT_mnode* new = malloc( sizeof(FTIT_mnode) );

    new->_data = malloc( sizeof(FTIT_metadata) );
    memcpy( new->_data, &data, sizeof(FTIT_metadata) );
    new->_next = NULL; 

    mqueue->_front = new;
    mqueue->_front->_next = old;

    return FTI_SCES;

}

int FTI_MetadataQueuePop( FTIT_metadata* data )
{

    if( !mqueue ) return FTI_NSCS;
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

bool FTI_MetadataQueueEmpty()
{
    if( !mqueue ) return true;
    return (mqueue->_front == NULL);
}

void FTI_MetadataQueueClear()
{
    while( !mqueue->empty() )
        mqueue->pop( NULL );
}


