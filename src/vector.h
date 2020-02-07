#ifndef __FTI__VECTOR


#ifdef __cplusplus
extern "C" {
#endif
    
    typedef struct FTIT_vectorkey {
        size_t  _type_size;
        size_t  _size;
        size_t  _used;
        int     _max_id;
        void*   _data;
        int*    _key;
        int     (*push_back)(struct FTIT_vectorkey* , void*, int);
        void*   (*data)(struct FTIT_vectorkey* , int);
        void*   (*clear)(struct FTIT_vectorkey* );
    } FTIT_vectorkey;

    int FTI_VectorKey( FTIT_vectorkey*, size_t, FTIT_configuration );
    int FTI_VectorKeyPushBack( FTIT_vectorkey* , void*, int );
    void* FTI_VectorKeyGet( FTIT_vectorkey* , int );
    void* FTI_VectorKeyClear( FTIT_vectorkey* ); 

#ifdef __cplusplus
}
#endif

#endif // __FTI__VECTOR
