#ifndef __FTI__KEYMAP

#ifdef __cplusplus
extern "C" {
#endif

    /** Minimum size for dynamic reallocation (in number of elements) */
    const static size_t FTI_MIN_REALLOC = 32;
    
    /** Maximum size for dynamic reallocation (in number of elements) ~ 10 Mb for FTIT_dataset */
    const static size_t FTI_MAX_REALLOC = 10*1024;  

    /**--------------------------------------------------------------------------
      
      
      @brief        Keymap structure.
    
      This structure holds all nessecary information for the key value container
      'keymap'. To mimic C++ class functionality for simpler handling, the structure
      contains function pointer members for:  
     

      FTI_KeyMapPushBack -> push_back   
      FTI_KeyMapData -> data   
      FTI_KeyMapGet -> get and  
      FTI_KeyMapClear -> clear.

    
    --------------------------------------------------------------------------**/
    typedef struct FTIT_keymap {

        bool    initialized;    /**< True iff instance exists                  */
        long    _type_size;     /**< Size of keymap elements                   */
        long    _size;          /**< Capacity of keymap                        */
        long    _used;          /**< Number of elements in keymap              */
        int     _max_key;       /**< Maximum value for key                     */       
        void*   _data;          /**< Pointer to first element in keymap        */
        int*    _key;           /**< Lookup table for key -> location in keymap*/
        int     (*push_back)    ( void*, int );
        int     (*data)         ( FTIT_dataset**, int );
        int     (*get)          ( FTIT_dataset**, int );
        int     (*clear)        ( void );
    
    } FTIT_keymap;

    /**--------------------------------------------------------------------------
      
      
      @brief        Initialize keymap singleton.
    
      This function initialized the keymap. Only a single instance of keymap is 
      allowed at a time. The function expects a pointer to an FTIT_keymap pointer, 
      the size of one element of the keymap and the maximum value for the key. 
      after the successful call, <code>*instance</code> points to the static
      variable \ref self. The call to the function if the keymap was already 
      initialized is erroneous and is protected by assert. 
      
      @param        instance[out]     <b> FTIT_keymap** </b>  Pointer to be set to
      the static instance of the key value container.
      @param        type_size[in]     <b> long          </b>  Element size of container.
      @param        max_key[in]       <b> long          </b>  Maximum value for Key.
      @return                       \ref FTI_SCES if successful.  
                                    \ref FTI_NSCS on failure.

    
    --------------------------------------------------------------------------**/
    int     FTI_KeyMap              ( FTIT_keymap**, long, long );
    
    /**--------------------------------------------------------------------------
      
      
      @brief        Insert new element to the keymap.
    
      This function inserts a new element to the keymap. The allocation happens
      dynamically and is inspired by the C++ vector push_back method. each time
      the addition of an element leads to the necessity of a reallocation, the 
      new allocation size will be set to twice the current size. However, the 
      maximum reallocation size can be controlled by the variable \ref FTI_MAX_REALLOC. 
      The first allocation size (first insertion) is controled by variable 
      \ref FTI_MIN_REALLOC. The new item will be copied so that we are not in danger 
      for inconsistent pointer values when the passed pointer goes out of scope.
      
      @param        new_item[in]    <b> void*   </b>  Pointer to new element.
      @param        key[in]         <b> int     </b>  Key of new element.
      @return                       \ref FTI_SCES if successful  
                                    \ref FTI_NSCS on failure
     
    
    --------------------------------------------------------------------------**/
    int     FTI_KeyMapPushBack      ( void*, int );
    
    /**--------------------------------------------------------------------------
      
      
      @brief        Request pointer to first element in the keymap.
    
      This function requests a pointer to the first element of the keymap. The
      function checks if it contains at least n elements.
      
      @param        data[out]   <b> FTIT_dataset** </b>  pointer to be set to the
      first element of keymap.
      @param        n[in]       <b> int            </b>  number of elements that
      should at least contained in the keymap.
      @return                       \ref FTI_SCES if successful  
                                    \ref FTI_NSCS on failure
     
    
    --------------------------------------------------------------------------**/
    int     FTI_KeyMapData          ( FTIT_dataset**, int );
    
    /**--------------------------------------------------------------------------
      
      
      @brief        Request pointer to element with a certain key in the keymap.
    
      This function requests a pointer to the element of the keymap with a certain key. 
      The function. If no element with key was found, the passed pointer is set to NULL.
      This case is considered to be a successful call.
      
      @param        data[out]   <b> FTIT_dataset** </b>  pointer to be set to the
      element with key in keymap.
      @param        key[in]     <b> int            </b>  key of requested element
      @return                       \ref FTI_SCES if successful  
                                    \ref FTI_NSCS on failure
     
    
    --------------------------------------------------------------------------**/
    int     FTI_KeyMapGet           ( FTIT_dataset**, int );
    
    /**--------------------------------------------------------------------------
      
      
      @brief        Resets the singleton instance of the keymap and frees buffers.
    
      This function frees all allocated memory in the keymap and resets all members
      to the initial state. After the call to this function \ref FTI_KeyMap can be
      called again safely to create a new singleton keamyp instance.
      
      @return                       \ref FTI_SCES if successful  
                                    \ref FTI_NSCS on failure
     
    
    --------------------------------------------------------------------------**/
    int     FTI_KeyMapClear         ( void ); 

#ifdef __cplusplus
}
#endif

#endif // __FTI__KEYMAP
