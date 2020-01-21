#include <stdio.h>
#include <stdlib.h>
#include <assert.h>


#include "FTI_UtilLowLevel.h"
#include "FTI_UtilAPI.h"

#define PRINTERROR()\
        fprintf(stderr, "Error in :%s\n",__func__)

int FTI_InitUtil(char *configFile){
    printf("I am being called with %s\n",configFile);
    int ret = FTI_LLInitEnvironment(configFile); 
    if (ret != SUCCESS ){
        PRINTERROR();
    }
    return ret; 
}

int FTI_GetNumberOfCkptIds(int *numCkpts){
   int ret = FTI_LLGetNumCheckpoints(); 
   if ( ret == ERROR){
        PRINTERROR();
        return ERROR; 
    }
   *numCkpts = ret;
   return SUCCESS;
}

int FTI_GetCkptIds(int *ckptIds){
   assert(ckptIds);
   int ret = FTI_LLGetCkptID(ckptIds); 
   if ( ret != SUCCESS ){
       PRINTERROR();
       return ERROR;
   }
   return SUCCESS;
}


int FTI_FinalizeUtil(){
    int ret = FTI_LLFinalizeUtil();
    if (ret != SUCCESS){
        PRINTERROR();
        return ERROR;
    }
    return SUCCESS;
}


