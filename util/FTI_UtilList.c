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
 *  @file   FTI_UtilList.c
 *  @author konstantinos Parasyris (koparasy)
 *  @date   23 January, 2020
 *  @brief  FTI API for lists.
 */

#include <stdlib.h>
#include <stdio.h>

#include "FTI_UtilList.h"
#include "FTI_UtilMacros.h"
#include "FTI_UtilAPI.h"

/*-------------------------------------------------------------------------*/
/**
  @brief      Creates a data-pool. 
  @param      pool pointer which will point to the data pool.
  @param      func function which will be called upon deletion of the pool.
  @return     integer         SUCCESS if successful.

  This function allocates and initializes the data structure for a pool
 **/
/*-------------------------------------------------------------------------*/
int createPool(FTIPool **data, int (*func)( void * )){
    FTIPool* root = NULL;
    *data = NULL;
    MALLOC(root, 1, FTIPool);
    root->tail = root->head = NULL;
    root->destroyer = func;
    *data = root;
    return SUCCESS;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      destroys a data-pool. 
  @param      pool pointer which will point to the data pool.
  @return     integer         SUCCESS if successful.

  This function de-allocates and destroys the data structure for a pool
 **/
/*-------------------------------------------------------------------------*/
int destroyPool(FTIPool **data){
    FTINode *ptr = (*data)->head;
    while ( ptr ){
        FTINode *next = ptr->next;
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

/*-------------------------------------------------------------------------*/
/**
  @brief      Adds a data item in the pool. 
  @param      pool pointer which points to the data pool.
  @param      data ptr to data to be added in the pool.
  @param      id value which is used to sort the data.
  @return     integer         SUCCESS if successful.

    This function adds the data value in the pool, the id is 
    used to keep data sorted.
 **/
/*-------------------------------------------------------------------------*/
int addNode(FTIPool **pool, void *data, int id){
    FTINode *tmp = (*pool)->head; 
    FTINode *newNode = NULL;
    MALLOC(newNode,1, FTINode);
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


/*-------------------------------------------------------------------------*/
/**
  @brief      Execute specific functionality on each data in the pool. 
  @param      pool pointer which points to the data pool.
  @param      func Function to be execute on each data value. 
  @return     void 

    This function executes the function on each node.
 **/
/*-------------------------------------------------------------------------*/
void execOnAllNodes(FTIPool **pool, void (*func)(void *data)){
    FTINode *ptr = (*pool)->head;
    while (ptr){
        func(ptr->data);
        ptr = ptr->next;
    }

}

/*-------------------------------------------------------------------------*/
/**
  @brief      Search in the poll for an item with id. 
  @param      pool pointer which points to the data pool.
  @param      id to search for
  @return     void *pointer to the data element. 

    This function executes the function on each node.
 **/
/*-------------------------------------------------------------------------*/
void *search(FTIPool *pool, int id){
    FTINode *ptr = pool->head;
    while(ptr){
        if (ptr->id == id ){
            return ptr->data;
        }
        ptr=ptr->next;
    }
    return NULL;
}

