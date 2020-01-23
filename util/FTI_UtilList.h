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
 *  @file   FTI_UtilList.h
 *  @author konstantinos Parasyris (koparasy)
 *  @date   23 January, 2020
 *  @brief  FTI API for lists.
 */



#ifndef __FTI_LIST__
#define __FTI_LIST__

    /** @typedef   FTINode 
     *  @brief      internal data structure of the list.
     *  
     *  The node is a generic container to store any information in a list.
     *  identified by a specific id.
     */

typedef struct node_t{
    void *data;                     /**< Pointer pointing to the data stored **/
    struct node_t *next;            /**< Pointer pointing to Next node       **/
    struct node_t *prev;            /**< Pointer pointing to Prev node       **/
    int id;                         /**< identifier                          **/
}FTINode;

    /** @typedef   FTIpool 
     *  @brief      internal data structure to store all data 
     *  
     */

typedef struct FTI_pool{
    FTINode *head, *tail;              /**< start and end of the pool **/
    int (*destroyer)( void * );        /**< Function pointer to destroy node data**/
}FTIPool;

int createPool(FTIPool **data, int (*func)( void * ));
int destroyPool(FTIPool **data);
int addNode(FTIPool **data, void *ptr, int id);
void execOnAllNodes(FTIPool **pool, void (*func)(void *data));
void *search(FTIPool *pool, int id);

#endif
