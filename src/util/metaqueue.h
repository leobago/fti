#ifndef __METAQUEUE_H__
#define __METAQUEUE_H__

/**--------------------------------------------------------------------------
  
  
  @brief Initializes passed instance of FTIT_mqueue.       

  This function initializes the metadata queue instance passed. The structure
  FTIT_mqueue mimics C++ class behavior to simplify the usage. All functions
  belonging to this class can bee called with the corresponding struct member
  of \ref FTIT_mqueue.
  
  @param        mqueue[out] <b> void* </b>  Pointer to FTIT_mqueue instance.
  @return                       \ref FTI_SCES if successful.  
                                \ref FTI_NSCS on failure.


--------------------------------------------------------------------------**/
int FTI_MetadataQueue( FTIT_mqueue* );

/**--------------------------------------------------------------------------
  
  
  @brief Checks if meta data queue has elements.       

  This function checks for elements in the meta data queue. If no elements
  are found, the function returns true.
  
  @param        mqueue[in] <b> void* </b>  Pointer to FTIT_mqueue instance.
  
  @return                       \ref <b> true </b> if empty.  
                                \ref <b> false </b> if not empty.


--------------------------------------------------------------------------**/
bool FTI_MetadataQueueEmpty( FTIT_mqueue* );

/**--------------------------------------------------------------------------
  
  
  @brief Pushes new element at the front of the meta data queue.       

  This function inserts an element at the front of the meta data queue. The
  element can be requested and removed by the call to 
  \ref FTI_MetaDataQueuePop.
  
  @param        mqueue[in/out] <b> void* </b>  Pointer to FTIT_mqueue instance.
  @param        mqueue[in] <b> FTIT_metadata </b>  meta data element to push 
  on the queue.
  
  @return                       \ref FTI_SCES if successful.  
                                \ref FTI_NSCS on failure.


--------------------------------------------------------------------------**/
int FTI_MetadataQueuePush( FTIT_mqueue*, FTIT_metadata );

/**--------------------------------------------------------------------------
  
  
  @brief Returns the element at the front of the meta data queue.       

  This function returns the element at the front of the meta data queue. After
  the call to this function, the element is removed and the next element 
  placed at the front of the meta data queue. If the queue is empty, the call
  will be considered unsuccsessful.
  
  @param        mqueue[in/out] <b> void* </b>  Pointer to FTIT_mqueue instance.
  @param        mqueue[out] <b> FTIT_metadata* </b>  Instance to store the meta
  data element from the queue to.
  
  @return                       \ref FTI_SCES if successful.  
                                \ref FTI_NSCS on failure or when called on
                                an empty queue.


--------------------------------------------------------------------------**/
int FTI_MetadataQueuePop( FTIT_mqueue*, FTIT_metadata* );

/**--------------------------------------------------------------------------
  
  
  @brief frees all elements in the meta data queue.       

  This function removes all the elements of the queue and frees the allocated 
  memory. 
  
  @param        mqueue[in/out] <b> void* </b>  Pointer to FTIT_mqueue instance.
  
  @return                       \ref FTI_SCES if successful.  
                                \ref FTI_NSCS on failure.


--------------------------------------------------------------------------**/
int FTI_MetadataQueueClear( FTIT_mqueue* );

#endif // __METAQUEUE_H__

