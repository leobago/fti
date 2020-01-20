#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <dirent.h>
#include <unistd.h>
#include <assert.h>


#include "../deps/iniparser/iniparser.h"
#include "../deps/md5/md5.h"
#include "tool_api.h"
#include "util_macros.h"
#include "FTI_List.h"

FTI_Info *info = NULL; 
FTI_Collection collection;

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
    int ckptId;

    MALLOC(fullName, fullNameSize, char);
    int size = sprintf(fullName, "%s/%s", directory, fileName);
    printf("FullName is %s\n", fullName);

    ini = iniparser_load(fullName);

    if (ini == NULL) {
        fprintf(stderr,"%s:%s:%d Iniparser failed to parse the conf. file.",__FILE__, __func__, __LINE__);
        return ERROR;
    }

    sprintf(buff, "ckpt_info:ckpt_id");
    ckptId = iniparser_getint(ini,buff, -1);

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
    return ckptId;
}


int readMetaDataFiles(FTI_Info *info, FTI_CkptFile **files, int *ckptId, char *path){
    char metaPath[BUFF_SIZE]; 
    sprintf(metaPath,"%s/%s/%s/",info->metaDir, info->execId, path);
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
            *ckptId = readMetaFile(&(*files)[rank], metaPath, de->d_name, info->groupSize);
            rank += info->groupSize;
        }
    }

    qsort(*files, totalRanks, sizeof(FTI_CkptFile), cmpFunc);

    for ( int i = 0; i < totalRanks; i++){
        (*files)[i].applicationRank = i;
        (*files)[i].verified = 0;
        printf("%d -- %d -- %d\n", i, (*files)[i].globalRank, (*files)[i].applicationRank);
    }

    CLOSEDIR(dr); 

    return totalRanks;

}


int readAllMetaDataFiles(FTI_Info *info, FTIpool  *allCkpts){
    FTI_Collection *newCollection;
    MALLOC(newCollection,1, FTI_Collection);
    int ckpts = -1;
    ckpts = readMetaDataFiles(info, &(newCollection->files),&(newCollection->ckptId), "l4");
    if (ckpt < 0){
        fprintf(stderr, "Some kind of an error occured on readMetaFiles\n");
        return ERROR;
    }

    newCollection->ckptId = ckpts;
    return SUCCESS;

}

int destroyCollection( void* ptr){
    FTI_Collection *collection = (FTI_Collection *) ptr;
    int numRanks = collection->numCkpts;
    FTI_CkptFile *files = collection->files;
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
    FREE(collection->pathTockpts);
    FREE(collection->pathToMeta);
}

int destroyInfo(FTI_Info *info){
    FREE(info->execDir);
    FREE(info->configFileDir);
    FREE(info->configName);
    FREE(info->metaDir);
    FREE(info->localDir);
    FREE(info->globalDir);
    FREE(info->execId);
    FREE(info);
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

int verifyCkpt(FTI_Info *info, FTI_CkptFile *file){
    char tmp[BUFF_SIZE];
    unsigned char hash[MD5_DIGEST_LENGTH];
    char checksum[MD5_DIGEST_STRING_LENGTH];   //calculated checksum
    int ii = 0;
    int bytes;
    unsigned char *data;
    int i;
    FILE *ckpt;

    if ( file->verified )
        return SUCCESS;

    sprintf(tmp,"%s/%s/%s",info->globalDir, info->execId,  file->name);
    OPEN(ckpt,tmp,"rb");

    MD5_CTX mdContext;
    MD5_Init (&mdContext);
    MALLOC(data, MAX_BUFF, char);

    while ((bytes = fread (data, 1, MAX_BUFF, ckpt)) != 0) {
        MD5_Update (&mdContext, data, bytes);
    }

    MD5_Final (hash, &mdContext);

    for(i = 0; i < MD5_DIGEST_LENGTH; i++) {
        sprintf(&checksum[ii], "%02x", hash[i]);
        ii += 2;
    }

    if (strcmp(checksum, file->md5hash) != 0) {
        fprintf(stderr, "Checksums do not match:\nFile Hash : %s \nMeta Hash : %s\n",checksum, file->md5hash);
        CLOSE(ckpt);
        return ERROR;
    }
    
    CLOSE(ckpt);
    FREE(data);
    file->verified = 1;
    return SUCCESS;
}

int readVariable( FTI_DataVar *var, FTI_CkptFile *ckpt, unsigned char **data, size_t *size){
    FILE *fd = NULL;
    unsigned char *tmpdata = NULL;
    char tmp[BUFF_SIZE];
    size_t index = 0;
    int bytes;

    *size = var->size;
    printf("Size is %ld\n", *size);
    if ( var->buf ){
        (*data) = var->buf;
        return SUCCESS;
    }

    MALLOC(tmpdata, (*size) , unsigned char);
    sprintf(tmp,"%s/%s/l4/%s",info->globalDir, info->execId, ckpt->name);
    OPEN(fd,tmp,"rb");
    fseek(fd, var->pos, SEEK_SET); 
    size_t bytesToRead = *size;
    do{
        int MIN = bytesToRead > MAX_BUFF ? MAX_BUFF: bytesToRead;
        bytes = fread (&tmpdata[index], 1, MIN, fd);
        bytesToRead -= bytes;
        index+=bytes;
    }while(bytes > 0 );

    var->buf = tmpdata;
    (*data) = tmpdata;
    CLOSE(fd);

    return SUCCESS;
}

int readVarById(int id, unsigned char **ptr, int rank, size_t *size ){
  int i;
  if ( rank >= info->userRanks){
      fprintf(stderr, "You are requesting ckpt data from a rank that does not exist\n");
      return ERROR;
  }

  FTI_CkptFile *ckpt = &collection.files[rank];      
  if ( ckpt->verified == 0){
      fprintf(stderr, "You are requesting to read a ckpt which you have not verified\n");
  }

  FTI_DataVar *variable = NULL;

  for ( i = 0; i < ckpt->numVars; i++){
      if ( id == ckpt->variables[i].id){
          variable = &ckpt->variables[i];
          break;
      }
  }

  if (!variable ){
      fprintf(stderr, "Could not find requested variable\n");
      return ERROR;
  }

  return readVariable(variable,ckpt,  ptr, size);

}

int readVarByName(char *name, unsigned char **ptr, int rank, size_t *size ){
  int i;
  if ( rank >= info->userRanks){
      fprintf(stderr, "You are requesting ckpt data from a rank that does not exist\n");
      return ERROR;
  }

  FTI_CkptFile *ckpt = &collection.files[rank];      
  if ( ckpt->verified == 0){
      fprintf(stderr, "You are requesting to read a ckpt which you have not verified\n");
  }

  FTI_DataVar *variable = NULL;

  for ( i = 0; i < ckpt->numVars; i++){
      if ( strcmp(name,ckpt->variables[i].name) == 0){
          variable = &ckpt->variables[i];
          break;
      }
  }

  if (!variable ){
      fprintf(stderr, "Could not find requested variable\n");
      return ERROR;
  }

  return readVariable(variable,ckpt,  ptr, size);

}


int main(int argc, char *argv[]){
    unsigned char *ptr = NULL;
    size_t size;
    initEnvironment(argv[1]); 

    FTIpool *allCkpts;

    if ( createPool(&allCkpts,destroyCollection) != SUCCESS){
        fprintf(stderr,"Could not create pools\n");
        return 0;
    }

    readAllMetaDataFiles(info, allCkpts);
    for ( int i = 0; i < info->userRanks; i++){
        verifyCkpt(info, &collection.files[i]);
    }
    printCkptInfo(collection.files, info);
    readVarById(0, &ptr, 0, &size);
    readVarByName("PRESS", &ptr, 3, &size);
    destroyInfo(info);
    info = NULL;
    return 0;
}




