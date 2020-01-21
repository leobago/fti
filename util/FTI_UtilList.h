#ifndef __FTI_LIST__
#define __FTI_LIST__

typedef struct node_t{
    void *data;
    struct node_t *next;
    struct node_t *prev;
    int id;
}node;

typedef struct FTI_pool{
    node *head, *tail;
    int (*destroyer)( void * );
}FTIpool;

int createPool(FTIpool **data, int (*func)( void * ));
int destroyPool(FTIpool **data);
int addNode(FTIpool **data, void *ptr, int id);
void execOnAllNodes(FTIpool **pool, void (*func)(void *data));

#endif
