#ifndef __UTIL_MACROS__
#define __UTIL_MACROS__

#include "FTI_List.h"

#define ERROR -1
#define SUCCESS 1


#define OPEN(out,name,type) \
    do{\
    out = fopen(name,type); \
    if (out == NULL ){\
        fprintf(stderr, "%s:%s:%d Could Not Open File: %s\n", __FILE__, __func__, __LINE__, name);\
        return ERROR;\
        }\
    }while(0)

#define CLOSE(desc)\
    do{\
        if( fclose(desc) != 0 ){\
            fprintf(stderr, "%s:%s:%d Could Not Close File\n", __FILE__, __func__, __LINE__);\
            return ERROR;\
        }\
    }while(0)

#define MALLOC(ptr, elements, type)\
    do {\
        ptr = (type*) malloc (sizeof(type)*(elements));\
        if ( ptr == NULL){\
            fprintf(stderr, "%s:%s:%d Could Not allocate memory\n", __FILE__, __func__, __LINE__);\
            return ERROR;\
        }\
    }while(0)

#define CHDIR(path)\
    do {\
        int tmp = chdir(path);\
        if (tmp != 0){\
            int errnum = errno;\
            fprintf(stderr, "%s : %s :%d\t Could Not Change Directory to %s\n", __FILE__, __func__, __LINE__,path);\
            return ERROR;\
        }\
    }while(0)

#define FREE(ptr)\
    do{\
        if (ptr){\
            free(ptr);\
            ptr=NULL;\
        }\
    }while(0)

#define OPENDIR(a,path)\
    do{\
    a = opendir(path); \
    if (a == NULL){ \
        fprintf(stderr, "%s : %s :%d\t Could Not Open Directory %s\n", __FILE__, __func__, __LINE__,path);\
        return ERROR;\
    }\
    }while(0)

#define CLOSEDIR(a)\
    do{\
        if (closedir(dr) != 0 ){\
            fprintf(stderr, "%s : %s :%d\t Could Not Close Directory\n", __FILE__, __func__, __LINE__);\
            return ERROR;\
        }\
    }while(0)


#endif

