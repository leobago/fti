#include <stdlib.h>
#include <stdio.h>

#include "FTI_UtilList.h"
#include "FTI_UtilMacros.h"
#include "FTI_UtilAPI.h"

int createPool(FTIpool **data, int (*func)( void * )){
    FTIpool* root = NULL;
    *data = NULL;
    MALLOC(root, 1, FTIpool);
    root->tail = root->head = NULL;
    root->destroyer = func;
    *data = root;
    return SUCCESS;
}

int destroyPool(FTIpool **data){
    node *ptr = (*data)->head;
    while ( ptr ){
        node *next = ptr->next;
        if ( (*data)->destroyer(ptr->data) != SUCCESS){
            fprintf(stderr,"Could not delete node");
        }
        ptr->next = NULL;
        ptr->prev = NULL;
        FREE(ptr->data);
        ptr->data = NULL;
        FREE(ptr);            
        ptr = next;
    }
    FREE(*data);
    *data = NULL;
    ptr = NULL;
    return SUCCESS;
}

int addNode(FTIpool **pool, void *data, int id){
    node *tmp = (*pool)->head; 
    node *newNode = NULL;
    MALLOC(newNode,1, node);
    newNode->data = data;
    newNode->id = id;

    if ((*pool)->head == NULL){
        newNode->next = NULL;
        newNode->prev = NULL;
        (*pool)->head = newNode;
        (*pool)->tail = newNode;
        return SUCCESS;
    }
    while ( tmp!= NULL && tmp->id < id ){
        tmp = tmp->next;
    }

    // I am going to add on the tail
    if ( tmp == NULL ){
        (*pool)->tail->next = newNode;
        newNode->next = NULL;
        newNode->prev = (*pool)->tail;
        (*pool)->tail = newNode;
    }else {
        newNode->next = tmp;
        newNode->prev = tmp->prev;

        if ( tmp->prev != NULL){
            tmp->prev->next = newNode;
        }
        else{
            (*pool)->head = newNode;
        }

        tmp->prev = newNode;

    }
    return SUCCESS;
}

void execOnAllNodes(FTIpool **pool, void (*func)(void *data)){
    node *ptr = (*pool)->head;
    while (ptr){
        func(ptr->data);
        ptr = ptr->next;
    }

}

int dummy(void *ptr){
    return SUCCESS;
}
