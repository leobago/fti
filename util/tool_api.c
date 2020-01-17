#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <dirent.h>
#include <unistd.h>
#include <assert.h>


#include "../deps/iniparser/iniparser.h"
#include "tool_api.h"


FTI_Info *info = NULL; 

#define OPEN(out,name,type) \
    do{\
    out = fopen(name,type); \
    if (out == NULL ){\
        fprintf(stderr, "%s:%s:%d Could Not Open File: %s\n", __FILE__, __func__, __LINE__, name);\
        return ERROR;\
    }while(0)

#define MALLOC(ptr, elements, type)\
    do {\
        ptr = (type*) malloc (sizeof(type)*(elements));\
        printf("%d Allocating %ld data\n", __LINE__, sizeof(type)*elements);\
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

int exists(char *pathToDir){
    DIR* dir = opendir(pathToDir);
    if (dir) {
        closedir(dir);
        return 1;
    } 
    return 0;
}



int getDirs(char *pathToConfigFile, char **name, char **path ){
    char *fileName = strrchr(pathToConfigFile,'/') + 1;
    int pathSize = strlen(fileName) ;
    MALLOC(*name, pathSize +1, char);

    int length = fileName - pathToConfigFile;
    MALLOC(*path, length + 1, char);

    strncpy(*path,pathToConfigFile, length); 
    strncpy(*name, fileName, pathSize);
    (*path)[length] = '\0';
    (*name)[pathSize] = '\0';

    return SUCCESS;
}

int initEnvironment(char *pathToConfigFile){
    assert(pathToConfigFile);
    dictionary *ini = NULL;

    MALLOC(info,1,FTI_Info);

    getDirs(pathToConfigFile, &info->configName,&info->configFileDir);
    info->execDir = getcwd(NULL, 0);

    CHDIR(info->configFileDir);
    ini = iniparser_load(info->configName);

    if (ini == NULL) {
        fprintf(stderr,"%s:%s:%d Iniparser failed to parse the conf. file.",__FILE__, __func__, __LINE__);
        return ERROR;
    }

    char *par = iniparser_getstring(ini, "Basic:ckpt_dir", NULL);
    int elements = strlen(par) + 1;
    MALLOC(info->localDir,elements, char); 
    strcpy(info->localDir,  par);

    par = iniparser_getstring(ini, "Basic:glbl_dir", NULL);
    elements = strlen(par) + 1;
    MALLOC(info->globalDir,elements, char); 
    strcpy(info->globalDir,  par);

    par = iniparser_getstring(ini, "Basic:meta_dir", NULL);
    elements = strlen(par) + 1;
    MALLOC(info->metaDir,elements, char); 
    strcpy(info->metaDir, par);

    par = iniparser_getstring(ini, "restart:exec_id", NULL);
    elements = strlen(par) + 1;
    MALLOC(info->execId, elements, char);
    strcpy(info->execId, par );

    info->localTest = (int)iniparser_getint(ini, "Advanced:local_test", -1);
    info->groupSize =  (int)iniparser_getint(ini, "Basic:group_size", -1); 
    info->head = (int)iniparser_getint(ini, "Basic:head", 0);
    info->nodeSize = (int)iniparser_getint(ini, "Basic:node_size", -1);
    info->userRanks= info->nodeSize - info->head;

    iniparser_freedict(ini);
    return SUCCESS;
}

int printInfo(FTI_Info *ptr){
   printf("==========================FTI CONFIG INFO=========================\n"); 
   printf("meta dir:\t %s\n", ptr->metaDir);
   printf("glbl dir:\t %s\n", ptr->globalDir);
   printf("locl dir:\t %s\n", ptr->localDir);
   printf("exec id :\t %s\n", ptr->execId);
   printf("Group Size : %d\n", ptr->groupSize);
   printf("Node Size  : %d\n", ptr->nodeSize);
   printf("userRanks  : %d\n", ptr->userRanks);
   printf("head       : %d\n", ptr->head);
   printf("==================================================================\n"); 
   return SUCCESS;
}

int destroEnvironment(FTI_Info *info){
    FREE(info->execDir);
    FREE(info->configFileDir);
    FREE(info->configName);
    FREE(info->metaDir);
    FREE(info->globalDir);
    FREE(info->localDir);
    FREE(info->execId);
    FREE(info); 
}

int main(int argc, char *argv[]){
   initEnvironment(argv[1]); 
   printInfo(info);
   destroEnvironment(info);
   return 0;
}




