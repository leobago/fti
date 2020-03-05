#ifndef __FTI__KEYMAP

#ifdef __cplusplus
extern "C" {
#endif

    typedef struct FTIT_keymap {
        size_t  _type_size;
        size_t  _size;
        size_t  _used;
        int     _max_id;
        bool    _error;
        void*   data;
        int*    _key;
        bool    (*check)();
        bool    (*check_range)( int );
        int     (*push_back)( void*, int );
        void*   (*get)( int );
        int     (*clear)();
    } FTIT_keymap;

    int FTI_KeyMap( FTIT_keymap*, size_t, FTIT_configuration );
    int FTI_KeyMapPushBack( void*, int );
    void* FTI_KeyMapGet( int );
    int FTI_KeyMapClear( void ); 
    bool FTI_KeyMapCheckError( void ); 
    bool FTI_KeyMapCheckRange( int ); 

#ifdef __cplusplus
}
#endif

#endif // __FTI__KEYMAP
