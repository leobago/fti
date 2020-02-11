#ifndef __FTI__KEYMAP


#ifdef __cplusplus
extern "C" {
#endif
    
    typedef struct FTIT_keymap {
        size_t  _type_size;
        size_t  _size;
        size_t  _used;
        int     _max_id;
        void*   _data;
        int*    _key;
        int     (*push_back)( struct FTIT_keymap* , void*, int );
        int     (*get)( struct FTIT_keymap* , int, void* );
        int   (*clear)( struct FTIT_keymap* );
        int   (*data)( struct FTIT_keymap*, void** );
    } FTIT_keymap;

    int FTI_KeyMap( FTIT_keymap*, size_t, FTIT_configuration );
    int FTI_KeyMapPushBack( FTIT_keymap* , void*, int );
    int FTI_KeyMapGet( FTIT_keymap* , int, void* );
    int FTI_KeyMapData( FTIT_keymap*, void** ); 
    int FTI_KeyMapClear( FTIT_keymap* ); 

#ifdef __cplusplus
}
#endif

#endif // __FTI__KEYMAP
