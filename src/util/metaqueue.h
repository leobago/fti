#ifndef __METAQUEUE_H__
#define __METAQUEUE_H__

int FTI_MetadataQueue( FTIT_mqueue* );
bool FTI_MetadataQueueEmpty( void );
int FTI_MetadataQueuePush( FTIT_metadata );
int FTI_MetadataQueuePop( FTIT_metadata* );
void FTI_MetadataQueueClear( void );

#endif // __METAQUEUE_H__

