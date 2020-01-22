#include <stdio.h>
#include <stdlib.h>
#include <assert.h>


#include "FTI_UtilLowLevel.h"
#include "FTI_UtilAPI.h"

#define PRINTERROR()\
        fprintf(stderr, "Error in :%s\n",__func__)

int FTI_InitUtil(char *configFile){
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

int FTI_GetUserRanks(int *numRanks){
    assert(numRanks);
    *numRanks = FTI_LLGetNumUserRanks();
    if (*numRanks == ERROR ){
        PRINTERROR();
        return ERROR;
    }
    return SUCCESS;

}

int FTI_VerifyCkpt(int collection, int ckpt){
    int ret =  FTI_LLverifyCkpt(collection, ckpt);
    if (ret != SUCCESS){
        PRINTERROR();
        return ERROR;
    }
    return SUCCESS;

}

int FTI_GetNumVars(int collection, int ckpt){
    int ret = FTI_LLGetNumVars(collection, ckpt);
    if (ret == ERROR){
        PRINTERROR();
        return ERROR;
    }
    return ret ;

}

int FTI_readVariable(int varId, int ckptId, int rank, char **varName, unsigned char **buf, size_t *size){
    int ret = FTI_LLreadVariable(varId, ckptId, rank, varName, buf, size);
    if (ret == ERROR){
        PRINTERROR();
        return ERROR;
    }
    return ret ;

}

