#ifndef __FTI__KEYMAP

#ifdef __cplusplus
extern "C" {
#endif

    const static int FTI_MIN_REALLOC = 32;

    typedef struct FTIT_keymap {
        bool    initialized;
        long  _type_size;
        long  _size;
        long  _used;
        int     _max_id;
        void*   _data;
        int*    _key;
        int     (*push_back)    ( void*, int );
        int     (*data)         ( FTIT_dataset**, int );
        int     (*get)          ( FTIT_dataset**, int );
        int     (*clear)        ( void );
    } FTIT_keymap;

    int     FTI_KeyMap              ( FTIT_keymap**, long, long );
    int     FTI_KeyMapPushBack      ( void*, int );
    int     FTI_KeyMapData          ( FTIT_dataset**, int );
    int     FTI_KeyMapGet           ( FTIT_dataset**, int );
    int     FTI_KeyMapClear         ( void ); 

#ifdef __cplusplus
}
#endif

#endif // __FTI__KEYMAP
