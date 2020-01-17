#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <dirent.h>
#include <unistd.h>
#include <assert.h>


#include "../deps/iniparser/iniparser.h"
#include "tool_api.h"
#include "util_macros.h"

FTI_Info *info = NULL; 

static int cmpFunc( const void *a, const void *b){
    FTI_CkptFile *A = (FTI_CkptFile *) a;
    FTI_CkptFile *B = (FTI_CkptFile *) b;
    return A->globalRank- B->globalRank;
}

static int getDirs(char *pathToConfigFile, char **name, char **path ){
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

int isMetaFile(char *name){
    int sec, group;
    int ret = sscanf(name,"sector%d-group%d.fti",&sec, &group);
    return ret;
}

int readMetaFile(FTI_CkptFile *ckptFile, char *directory, char *fileName, int groupSize){
    dictionary *ini = NULL;
    char buff[BUFF_SIZE];
    size_t fullNameSize = strlen(directory) + strlen(fileName) + 5;
    char *fullName = NULL;
    char *par;
    int i;

    MALLOC(fullName, fullNameSize, char);
    int size = sprintf(fullName, "%s/%s", directory, fileName);
    printf("FullName is %s\n", fullName);

    ini = iniparser_load(fullName);

    if (ini == NULL) {
        fprintf(stderr,"%s:%s:%d Iniparser failed to parse the conf. file.",__FILE__, __func__, __LINE__);
        return ERROR;
    }

    for ( i = 0; i < groupSize; i++){
        sprintf(buff, "%d:ckpt_file_name",i);
        par = iniparser_getstring(ini,buff, NULL);
        int size= strlen(par);
        int ckptId, rank;
        MALLOC(ckptFile[i].name, size+1, char);
        strcpy(ckptFile[i].name, par);
        sscanf(par, "Ckpt%d-Rank%d.fti", &ckptId, &rank);
        ckptFile[i].globalRank= rank;

        int numVars = -1;
        int id;
        do { 
            numVars++;
            sprintf(buff, "%d:Var%d_id", i, numVars);
            id = iniparser_getint(ini, buff, -1);
        }while(id != -1);

        ckptFile[i].numVars = numVars;

        MALLOC(ckptFile[i].variables, numVars, FTI_DataVar);
        sprintf(buff,"%d:ckpt_checksum",i);
        par = iniparser_getstring(ini,buff,NULL);
        size = strlen(par);
        MALLOC(ckptFile[i].md5hash, size+1, char);
        strcpy(ckptFile[i].md5hash, par);

        FTI_DataVar *vars = ckptFile[i].variables;

        for (int k = 0; k < numVars; k++){
            sprintf(buff, "%d:Var%d_id", i, k);
            id = iniparser_getint(ini, buff, -1);
            vars[k].id = id;
            sprintf(buff, "%d:Var%d_pos", i, k);
            vars[k].pos = iniparser_getlint(ini, buff, -1);
            sprintf(buff, "%d:Var%d_size", i, k);
            vars[k].size = iniparser_getlint(ini, buff, -1);
            vars[k].buf = NULL;
            sprintf(buff, "%d:Var%d_name", i, k);
            par = iniparser_getstring(ini,buff,NULL);

            size = strlen(par);
            MALLOC(vars[k].name, size+1, char);
            strcpy(vars[k].name, par);
        }
    }
    iniparser_freedict(ini);
    FREE(fullName);
    return SUCCESS;
}

int readMetaDataFiles(FTI_Info *info, FTI_CkptFile **files){
    // I need to check how many meta data files I have. 
    char metaPath[BUFF_SIZE]; 
    sprintf(metaPath,"%s/%s/l4/",info->metaDir, info->execId);
    struct dirent *de; 
    DIR *dr = NULL; 
    size_t totalRanks = 0;
    int rank = 0;
    OPENDIR( dr,metaPath );   

    while ((de = readdir(dr)) != NULL){
        if (isMetaFile(de->d_name))
            totalRanks++;        
    }

    CLOSEDIR(dr); 
    totalRanks = totalRanks*info->groupSize;
    info->userRanks= totalRanks;
    MALLOC(*files, totalRanks, FTI_CkptFile);
    
    
    OPENDIR( dr,metaPath );   
    while ((de = readdir(dr)) != NULL){
        if (isMetaFile(de->d_name)){
            readMetaFile(&(*files)[rank], metaPath, de->d_name, info->groupSize);
            rank += info->groupSize;
        }
    }

    qsort(*files, totalRanks, sizeof(FTI_CkptFile), cmpFunc);
    for ( int i = 0; i < totalRanks; i++){
        (*files)[i].applicationRank = i;
        printf("%d -- %d -- %d\n", i, (*files)[i].globalRank, (*files)[i].applicationRank);
    }



    CLOSEDIR(dr); 
}

int destroyEnvironment(FTI_Info *info, FTI_CkptFile *files){
    int numRanks = info->userRanks;
    FREE(info->execDir);
    FREE(info->configFileDir);
    FREE(info->configName);
    FREE(info->metaDir);
    FREE(info->globalDir);
    FREE(info->localDir);
    FREE(info->execId);
    FREE(info); 

    for ( int i = 0; i < numRanks; i++){
        FREE(files[i].name);           
        FREE(files[i].md5hash);           
        for ( int j = 0; j < files[i].numVars; j++){
            FREE(files[i].variables[j].name);
            FREE(files[i].variables[j].buf);
        }
        FREE(files[i].variables);
    }
    FREE(files);
}

void printCkptInfo(FTI_CkptFile *files, FTI_Info *info){
    int i, j;
    for ( i = 0 ; i < info->userRanks; i++){
        printf("===================================%s=============================================\n", files[i].name);
        printf("\tNum Vars %d\n", files[i].numVars);
        printf("\tHash %s\n", files[i].md5hash);
        printf("\tGlobal Rank %d\n", files[i].globalRank);
        printf("\tApplication Rank %d\n", files[i].applicationRank);
        for ( j = 0; j < files[i].numVars; j++){
            printf("____________________________________________________________\n");
            printf("\t\t Name:%s id:%d\n",files[i].variables[j].name, files[i].variables[j].id);
            printf("\t\t size:%ld\n",files[i].variables[j].size);
            printf("\t\t position:%ld\n",files[i].variables[j].pos);
            printf("____________________________________________________________\n");
        }
    }
    
}

int main(int argc, char *argv[]){
    FTI_CkptFile *files;
    initEnvironment(argv[1]); 
    readMetaDataFiles(info, &files);
    printInfo(info);
    destroyEnvironment(info, files);
    return 0;
}




