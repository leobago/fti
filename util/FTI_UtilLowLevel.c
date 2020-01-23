/**
 *  Copyright (c) 2017 Leonardo A. Bautista-Gomez
 *  All rights reserved
 *
 *  FTI - A multi-level checkpointing library for C/C++/Fortran applications
 *
 *  Revision 1.0 : Fault Tolerance Interface (FTI)
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice, this
 *  list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright notice,
 *  this list of conditions and the following disclaimer in the documentation
 *  and/or other materials provided with the distribution.
 *
 *  3. Neither the name of the copyright holder nor the names of its contributors
 *  may be used to endorse or promote products derived from this software without
 *  specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 *  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 *  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 *  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 *  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 *  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *  @file   FTI_UtiLowLevel.c
 *  @author konstantinos Parasyris (koparasy)
 *  @date   23 January, 2020
 *  @brief  FTI API for lists.
 */



#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <dirent.h>
#include <unistd.h>
#include <assert.h>


#include "../deps/iniparser/iniparser.h"
#include "../deps/md5/md5.h"

#include "FTI_UtilLowLevel.h"
#include "FTI_UtilMacros.h"
#include "FTI_UtilList.h"
#include "FTI_UtilAPI.h"

FTIInfo *info = NULL; 
FTICollection collection;
FTIPool *allCkpts = NULL;
FTICollection *latest = NULL;


/*-------------------------------------------------------------------------*/
/**
  @brief      comparison function of to ckpt file based on global rank.
  @param      a pointer to first ckpt 
  @param b pointer to second ckpt 
  @return    integer 

    This function is used to order ckpt files.

 **/
/*-------------------------------------------------------------------------*/

static int cmpFunc( const void *a, const void *b){
    FTICkptFile *A = (FTICkptFile *) a;
    FTICkptFile *B = (FTICkptFile *) b;
    return A->globalRank- B->globalRank;
}


/*-------------------------------------------------------------------------*/
/**
  @brief      Function that gets a path and seperates path and file name 
  @param      pathToConfigFile path to file
  @param      name it will contain the name of the file
  @param      path it will contain the path of the file
  @return    integer SUCCESS or ERROR 

    The function splits a file path to a path and a file name

 **/
/*-------------------------------------------------------------------------*/

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

/*-------------------------------------------------------------------------*/
/**
  @brief      Fixes paths of config files 
  @param      Global Dir or Meta Dir 
  @param      configFileDir path to config file 
  @param      fullpath  will contain the absolute path of the "par" file 
  @return    integer SUCCESS or ERROR 

    
    Fix abosolute directoy

 **/
/*-------------------------------------------------------------------------*/
int processPath(char *par, char **configFileDir, char **fullPath){
    int elements;
    if ( par == NULL ){
        fprintf(stderr, " Directory is not mentioned in configuration file exiting....\n");
        return ERROR;
    }

    if ( par[0] == '/' ){
        elements = strlen(par) + 1;
        MALLOC(*fullPath,elements, char); 
        strcpy(*fullPath,  par);
    }
    else if ( par[0] == '.' && par[1] == '/' ){
        fprintf(stderr,"WARNING PATHS in configuration are relative (%s)\n", par); 
        fprintf(stderr, "I will try to open them relatively to directory of config file\n");
        elements = strlen(*configFileDir) + strlen(&par[2]);

        MALLOC(*fullPath,elements+2, char); 
        sprintf(*fullPath,"%s/%s", *configFileDir, &par[2]);
        (*fullPath)[elements+1] = '\0';
    }
    else{
        fprintf(stderr,"WARNING PATHS in configuration are relative (%s)\n", par); 
        fprintf(stderr, "I will try to open them relatively to directory of config file\n");
        elements = strlen(*configFileDir) + strlen(par);
        MALLOC(*fullPath,elements+2, char); 
        sprintf(*fullPath,"%s/%s", *configFileDir, par);
        (*fullPath)[elements+0] = '\0';
    }
    return SUCCESS;
}


/*-------------------------------------------------------------------------*/
/**
  @brief      Read the config file and initializes all necessary info 
  @param      pathToConfigFile path of configuration 
  @return    integer SUCCESS or ERROR 

    It reads the configuration file and allocates/initializes all necessary
    information of the FTIInfo structure.

 **/
/*-------------------------------------------------------------------------*/
int initEnvironment(char *pathToConfigFile){
    assert(pathToConfigFile);
    dictionary *ini = NULL;
    char *path, *path1;
    char *par;
    int elements;
    int ret;
    int length = 0;
    MALLOC(info,1,FTIInfo);

    getDirs(pathToConfigFile, &path1 ,&path);
    info->execDir = getcwd(NULL, 0);

    CHDIR(path);
    info->configFileDir= getcwd(NULL, 0);
    length = strlen(info->configFileDir) + strlen(path1);
    MALLOC(info->configName, length + 2, char);
    sprintf(info->configName,"%s/%s", info->configFileDir, path1);

    FREE(path); 
    FREE(path1); 
    path = NULL;
    path1 = NULL;

    CHDIR(info->execDir);

    ini = iniparser_load(info->configName);

    if (ini == NULL) {
        fprintf(stderr,"%s:%s:%d Iniparser failed to parse the conf. file.",__FILE__, __func__, __LINE__);
        return ERROR;
    }

    par = iniparser_getstring(ini, "Basic:glbl_dir", NULL);
    ret = processPath(par, &info->configFileDir, &info->globalDir);
    if (ret != SUCCESS)
        return ERROR;

    par = iniparser_getstring(ini, "Basic:meta_dir", NULL);
    ret = processPath(par, &info->configFileDir, &info->metaDir);
    if (ret != SUCCESS)
        return ERROR;

    par = iniparser_getstring(ini, "restart:exec_id", NULL);
    elements = strlen(par) + 1;
    MALLOC(info->execId, elements, char);
    strcpy(info->execId, par );

    info->localTest = (int)iniparser_getint(ini, "Advanced:local_test", -1);
    info->groupSize =  (int)iniparser_getint(ini, "Basic:group_size", -1); 
    info->head = (int)iniparser_getint(ini, "Basic:head", 0);
    info->nodeSize = (int)iniparser_getint(ini, "Basic:node_size", -1);
    info->numCheckpoints = 0;

    iniparser_freedict(ini);
    return SUCCESS;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Prints information read from the configuration file 
  @return    integer SUCCESS or ERROR 

    Prints data to stdout
 **/
/*-------------------------------------------------------------------------*/

int printInfo(){
    printf("==========================FTI CONFIG INFO=========================\n"); 
    printf("meta dir:\t %s\n", info->metaDir);
    printf("glbl dir:\t %s\n", info->globalDir);
    printf("exec id :\t %s\n", info->execId);
    printf("Group Size : %d\n", info->groupSize);
    printf("Node Size  : %d\n", info->nodeSize);
    printf("userRanks  : %d\n", info->userRanks);
    printf("head       : %d\n", info->head);
    printf("==================================================================\n"); 
    return SUCCESS;
}


/*-------------------------------------------------------------------------*/
/**
  @brief     Checks if file is metadata file 
  @param      name of possible meta data file 
  @return    integer > 0 in sucessfull  

  Checks whether the file fits the metadata file name structure
 **/
/*-------------------------------------------------------------------------*/
int isMetaFile(char *name){
    int sec, group;
    int ret = sscanf(name,"sector%d-group%d.fti",&sec, &group);
    return ret;
}


/*-------------------------------------------------------------------------*/
/**
  @brief    Checks whether this is is a ckpt directory 
  @param      name of possible checkpoint directory 
  @return    integer > 0 in sucessfull  

  Checks whether the file fits the checkpoint directory name 
 **/
/*-------------------------------------------------------------------------*/

int isCkptDir(char *name){
    int ckptId;
    int ret = sscanf(name,"Ckpt_%d",&ckptId);
    return ret;
}


/*-------------------------------------------------------------------------*/
/**
  @brief     Reads Meta data file of a checkpoint 
  @param     ckptFile structure pointing to the checkpoint file 
  @param     directory path to files 
  @param     fileName name of meta data file 
  @param     Number of data values in metadata file 
  @return    integer > 0 in sucessfull  

    Reads a metadata file which contains information of a specific group of checkpoints.
 **/
/*-------------------------------------------------------------------------*/
int readMetaFile(FTICkptFile *ckptFile, char *directory, char *fileName, int groupSize){
    dictionary *ini = NULL;
    char buff[BUFF_SIZE];
    size_t fullNameSize = strlen(directory) + strlen(fileName) + 5;
    char *fullName = NULL;
    char *par;
    int i;
    int ckptId;

    MALLOC(fullName, fullNameSize, char);
    int size = sprintf(fullName, "%s/%s", directory, fileName);

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
        ckptFile[i].pathToFile = NULL;
        ckptFile[i].fd = NULL;

        int numVars = -1;
        int id;
        do { 
            numVars++;
            sprintf(buff, "%d:Var%d_id", i, numVars);
            id = iniparser_getint(ini, buff, -1);
        }while(id != -1);

        ckptFile[i].numVars = numVars;

        MALLOC(ckptFile[i].variables, numVars, FTIDataVar);
        sprintf(buff,"%d:ckpt_checksum",i);
        par = iniparser_getstring(ini,buff,NULL);
        size = strlen(par);
        MALLOC(ckptFile[i].md5hash, size+1, char);
        strcpy(ckptFile[i].md5hash, par);

        FTIDataVar *vars = ckptFile[i].variables;

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

/*-------------------------------------------------------------------------*/
/**
  @brief     Reads all Meta data file of a a specific checkpoint
  @param     info structure pointing to the information
  @param     **files all checkpoint files
  @param     ckptId Stores the ckpt id of this checkpoint. 
  @param     metaPath path to meta data file 
  @param     globalDir path to global data
  @return    integer > 0 in sucessfull  

    Reads all metadata files which contains information of a specific ckpt id
   
 **/
/*-------------------------------------------------------------------------*/

int readMetaDataFiles(FTIInfo *info, FTICkptFile **files, int *ckptId, char *metaPath, char *glblPath){
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
    MALLOC(*files, totalRanks, FTICkptFile);


    OPENDIR( dr,metaPath );   
    while ((de = readdir(dr)) != NULL){
        if (isMetaFile(de->d_name)){
            *ckptId = readMetaFile(&(*files)[rank], metaPath, de->d_name, info->groupSize);
            rank += info->groupSize;
        }
    }

    qsort(*files, totalRanks, sizeof(FTICkptFile), cmpFunc);

    for ( int i = 0; i < totalRanks; i++){
        char tmp[BUFF_SIZE];
        (*files)[i].applicationRank = i;
        (*files)[i].verified = 0;
        sprintf(tmp,"%s/%s/%s/%s",info->globalDir, info->execId,glblPath, (*files)[i].name);
        int length = strlen(tmp);
        MALLOC((*files)[i].pathToFile,length+1, char);
        strcpy((*files)[i].pathToFile, tmp);
        (*files)[i].pathToFile[length] = '\0';
    }

    CLOSEDIR(dr); 

    return totalRanks;

}

/*-------------------------------------------------------------------------*/
/**
  @brief     Reads all Meta data file of a the entire application execution
  @param     info information of this application execution 
  @param     allCkpts structure containing all ckpts  
  @return    integer > 0 in sucessfull  

    Reads all metadata files which contains information of an entire application
    execution
   
 **/
/*-------------------------------------------------------------------------*/
int readAllMetaDataFiles(FTIInfo *info, FTIPool  *allCkpts){
    FTICollection *newCollection;
    struct dirent *de; 
    DIR *dr = NULL; 
    MALLOC(newCollection,1, FTICollection);
    char metaPath[BUFF_SIZE]; 
    char archive[BUFF_SIZE];
    int ckpts = -1;
    // Here I read all meta files of last ckpt
    sprintf(metaPath,"%s/%s/l4/",info->metaDir, info->execId);
    ckpts = readMetaDataFiles(info, &(newCollection->files),&(newCollection->ckptId), metaPath, "l4");
    if (ckpts < 0){
        fprintf(stderr, "Some kind of an error occured on readMetaFiles\n");
        return ERROR;
    }

    newCollection->numCkpts= ckpts;
    info->numCheckpoints++;
    addNode(&allCkpts, newCollection, newCollection->ckptId);

    sprintf(metaPath,"%s/%s/l4_archive/",info->metaDir, info->execId);
    dr = opendir(metaPath);

    if ( dr == NULL ){
        return SUCCESS;
    }

    OPENDIR( dr,metaPath );   
    while ((de = readdir(dr)) != NULL){
        if ( isCkptDir(de->d_name) ){
            MALLOC(newCollection,1, FTICollection);
            sprintf(archive,"%s/%s/",metaPath,de->d_name);
            ckpts = readMetaDataFiles(info, &(newCollection->files),&(newCollection->ckptId), archive, "l4_archive");
            if (ckpts < 0){
                fprintf(stderr, "Some kind of an error occured on readMetaFiles\n");
                return ERROR;
            }
            newCollection->numCkpts= ckpts;
            addNode(&allCkpts, newCollection, newCollection->ckptId);
            info->numCheckpoints++;
        }
    }

    CLOSEDIR(dr);


    return SUCCESS;

}

/*-------------------------------------------------------------------------*/
/**
  @brief     De allocates a collection 
  @param     ptr pointer pointing to the collection to be freed. 
  @return    integer > 0 in sucessfull  

    De allocates all information of collection.
   
 **/
/*-------------------------------------------------------------------------*/

int destroyCollection( void* ptr){
    FTICollection *collection = (FTICollection *) ptr;
    int numRanks = collection->numCkpts;
    FTICkptFile *files = collection->files;
    for ( int i = 0; i < numRanks; i++){
        FREE(files[i].name);           
        FREE(files[i].md5hash);           
        for ( int j = 0; j < files[i].numVars; j++){
            FREE(files[i].variables[j].name);
            FREE(files[i].variables[j].buf);
        }
        FREE(files[i].variables);
        FREE(files[i].pathToFile);
        if (files[i].fd)
            CLOSE(files[i].fd); 
    }
    FREE(files);
    latest = NULL;
    return SUCCESS;
}

/*-------------------------------------------------------------------------*/
/**
  @brief     De allocates the information 
  @param     @return    integer > 0 in sucessfull  

    De allocates all information of collection.
   
 **/
/*-------------------------------------------------------------------------*/
int destroyInfo(){
    FREE(info->execDir);
    FREE(info->configFileDir);
    FREE(info->configName);
    FREE(info->metaDir);
    FREE(info->globalDir);
    FREE(info->execId);
    FREE(info);
}

/*-------------------------------------------------------------------------*/
/**
  @brief     Prints all values of a checkpoint collection 
  @param     Pointer to data to be printed.
  @return    void 

    Prints data to stdout
 **/
/*-------------------------------------------------------------------------*/
void printCollection(void *ptr){
    int i,j;
    FTICollection *collection = (FTICollection *) ptr;
    FTICkptFile *files = collection->files;
    for ( i = 0; i < collection-> numCkpts; i++){
        printf("===================================%s=============================================\n", files[i].name);
        printf("\tNum Vars %d\n", files[i].numVars);
        printf("\tHash %s\n", files[i].md5hash);
        printf("\tGlobal Rank %d\n", files[i].globalRank);
        printf("\tApplication Rank %d\n", files[i].applicationRank);
        printf("\t Path:%s\n", files[i].pathToFile);
        for ( j = 0; j < files[i].numVars; j++){
            printf("____________________________________________________________\n");
            printf("\t\t Name:%s id:%d\n",files[i].variables[j].name, files[i].variables[j].id);
            printf("\t\t size:%ld\n",files[i].variables[j].size);
            printf("\t\t position:%ld\n",files[i].variables[j].pos);
            printf("____________________________________________________________\n");
        }
    }
}


/*-------------------------------------------------------------------------*/
/**
  @brief     Checks whether a ckpt file is corrupted or not 
  @param     info pointer pointing to the information of this execution.
  @param     file Checkpoint file to verify.
  @return    int SUCCESS or ERROR 


    Verifies a checkpoint
 **/
/*-------------------------------------------------------------------------*/
int verifyCkpt(FTIInfo *info, FTICkptFile *file){
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

    OPEN(ckpt, file->pathToFile,"rb");

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

    //    CLOSE(ckpt);
    FREE(data);
    file->verified = 1;
    file->fd = ckpt;
    return SUCCESS;
}


/*-------------------------------------------------------------------------*/
/**
  @brief     Reads a variable from a specific pointer
  @param     var Variable to read
  @param     ckpt Checkpoint file to read from.
  @param     data Data read from ckpt file.
  @param     size Size of the data it read.
  @return    int SUCCESS or ERROR 

   Reads Data from a specific checkpoint file 
 **/
/*-------------------------------------------------------------------------*/
int readVariable( FTIDataVar *var, FTICkptFile *ckpt, unsigned char **data, size_t *size){
    FILE *fd = NULL;
    unsigned char *tmpdata = NULL;
    char tmp[BUFF_SIZE];
    size_t index = 0;
    int bytes;

    *size = var->size;

    if ( var->buf ){
        (*data) = var->buf;
        return SUCCESS;
    }

    MALLOC(tmpdata, (*size) , unsigned char);
    OPEN(fd,ckpt->pathToFile,"rb");
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

/*-------------------------------------------------------------------------*/
/**
  @brief     Reads a variable by referencing with an index 
  @param     index Index of the variable we want to read
  @param     ptr Will point to the Data values we read
  @param     ckpt Checkpoint file from which we will read from.
  @param     varName Will contain the Name of the protected variable.
  @param     varId will contain the id of the variable we read.
  @return    int SUCCESS or ERROR 

   Reads Data from a specific checkpoint file 
 **/
/*-------------------------------------------------------------------------*/
int readVarByIndex(int index, unsigned char **ptr, FTICkptFile *ckpt, size_t *size, char **varName, int *varId ){

    if ( ckpt->verified == 0){
        fprintf(stderr, "WARN: You are requesting to read a ckpt which you have not verified\n");
    }

    if ( index >= ckpt->numVars ){
        fprintf(stderr, "You are requesting to read a variable by index which exceeds the total number of stored variables\n");
        return ERROR; 
    }

    FTIDataVar *variable = NULL;

    variable = &ckpt->variables[index];
    (*varName) = ckpt->variables[index].name;
    *varId = variable->id;

    if (!variable ){
        fprintf(stderr, "Could not find requested variable\n");
        return ERROR;
    }

    return readVariable(variable,ckpt,  ptr, size);

}


/*-------------------------------------------------------------------------*/
/**
    @brief Initializes data structures to store the all the  collection of checkpoints
    @return int SUCCESS or ERROR

 **/
/*-------------------------------------------------------------------------*/
int FTI_CreatePool(){
    if ( createPool(&allCkpts,destroyCollection) != SUCCESS){
        fprintf(stderr,"Could not create pools\n");
        return ERROR;
    }
    return SUCCESS;
}

/*-------------------------------------------------------------------------*/
/**
    @brief Initializes the utilities environment 
    @param configFile configuration file from which we initialize
    @return int SUCCESS or ERROR

    Initializes all the data structures of the utility.

 **/
/*-------------------------------------------------------------------------*/
int FTI_LLInitEnvironment(char *configFile){
    int ret = 0;
    ret = initEnvironment(configFile); 

    if ( ret != SUCCESS )
        return ERROR;

    ret = FTI_CreatePool();
    if (ret != SUCCESS )
        return ERROR;

    ret = readAllMetaDataFiles(info, allCkpts);
    if ( ret != SUCCESS )
        return ERROR;

    return SUCCESS;
}

/*-------------------------------------------------------------------------*/
/**
    @brief Returns the number of checkpoints 
    @return int  if >= 0 the number of collective checkpoints 
    
    Returns the number of collective checkpoint

 **/
/*-------------------------------------------------------------------------*/
int FTI_LLGetNumCheckpoints(){
    if (info != NULL )
        return info->numCheckpoints;
    return ERROR;        
}


/*-------------------------------------------------------------------------*/
/**
    @brief Returns the checkpoint IDs for all collections
    @param ckptIds on return it contains the checkpoint ids
    @return  int SUCCESS 
    
    Returns the number of collective checkpoint

 **/
/*-------------------------------------------------------------------------*/
int FTI_LLGetCkptID(int *ckptIds){
    FTINode *iter = allCkpts->head;
    int cnt= 0;
    while ( iter != NULL){
        FTICollection *data = (FTICollection *) iter->data;
        ckptIds[cnt++] = data->ckptId;
        iter = iter->next;
    }
    return SUCCESS;
}

/*-------------------------------------------------------------------------*/
/**
    @brief Finalizes the utility 
    @return  int SUCCESS 
    
    Deallocates and closes all files
 **/
/*-------------------------------------------------------------------------*/
int FTI_LLFinalizeUtil(){
    int ret = destroyInfo();
    if (ret == ERROR )
        return ret;
    ret = destroyPool(&allCkpts);
    if (ret == ERROR )
        return ret;
    return ret;
}

/*-------------------------------------------------------------------------*/
/**
    @brief Returns the number of application ranks 
    @return  int if > 0 it contains the number of application ranks. 
    
 **/
/*-------------------------------------------------------------------------*/
int FTI_LLGetNumUserRanks(){
    if (info != NULL){
        return info->userRanks;
    }
    return ERROR;
}

/*-------------------------------------------------------------------------*/
/**
    @brief Returns the number of variables for a specific rank/ckpt 
    @param ckptId Checkpoint id we are concerned about.
    @param rank rank.
    @return  int if > 0 it contains the number of application ranks. 
 **/
/*-------------------------------------------------------------------------*/
int FTI_LLGetNumVars(int ckptId, int rank){

    FTICollection *coll= NULL;
    if ( latest && latest-> ckptId == ckptId) {
        coll = latest;
    }else{
        coll = (FTICollection*) search(allCkpts, ckptId);
        latest = coll;
    }

    if (!coll){
        fprintf(stderr, "%s:%s:%d Could Not Find requestd Ckpt\n", __FILE__, __func__, __LINE__);
        return ERROR;
    }

    if ( rank > coll->numCkpts ){
        fprintf(stderr, "%s:%s:%d We do not have that many ckpts.\n", __FILE__, __func__, __LINE__);
        return ERROR;
    }

    return coll->files[rank].numVars;
}


/*-------------------------------------------------------------------------*/
/**
    @brief Verifies a specifc ckpt  
    @param ckptId Checkpoint ID we will verify.
    @param rank Verify data of the speficied rank
    @return  int if > 0 it contains the number of application ranks. 
 **/
/*-------------------------------------------------------------------------*/
int FTI_LLverifyCkpt( int ckptId, int rank){
    int ret;
    FTICollection *coll= NULL;
    if ( latest && latest-> ckptId == ckptId) {
        coll = latest;
    }else{
        coll = (FTICollection*) search(allCkpts, ckptId);
        latest = coll;
    }

    if (!coll){
        fprintf(stderr, "%s:%s:%d Could Not Find requestd Ckpt\n", __FILE__, __func__, __LINE__);
        return ERROR;
    }

    if ( rank > coll->numCkpts ){
        fprintf(stderr, "%s:%s:%d We do not have that many ckpts.\n", __FILE__, __func__, __LINE__);
        return ERROR;
    }

    ret = verifyCkpt(info, &(coll->files[rank]));
    return ret;
}


/*-------------------------------------------------------------------------*/
/**
    @brief Reads a specific variable referenced by index. 
    @param index of the variable.
    @param ckptId Checkpoint id we will use to export data
    @param rank Data used to be property of the speficied rank
    @param **varName stores the name of the protected variable (if defined) 
    @param *varId stores the variable id  
    @param **buf will contain data after read.
    @param *size it will contain the size of the data we read
    @return SUCCESS or ERROR

    reads a specific variable identified by the index 
 **/
/*-------------------------------------------------------------------------*/
int FTI_LLreadVariable(int varIndex, int ckptId, int rank, char **varName, int *varId, unsigned char **buf, size_t *size){
    int ret;
    FTICollection *coll= NULL;
    if ( latest && latest-> ckptId == ckptId) {
        coll = latest;
    }else{
        coll = (FTICollection*) search(allCkpts, ckptId);
        latest = coll;
    }

    if (!coll){
        fprintf(stderr, "%s:%s:%d Could Not Find requestd Ckpt %d\n", __FILE__, __func__, __LINE__,ckptId);
        return ERROR;
    }

    if ( rank > coll->numCkpts ){
        fprintf(stderr, "%s:%s:%d We do not have that many ckpts.\n", __FILE__, __func__, __LINE__);
        return ERROR;
    }

    ret = readVarByIndex(varIndex, buf, &(coll->files[rank]) , size , varName, varId);
}

