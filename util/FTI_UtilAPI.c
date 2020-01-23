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
 *  @file   FTI_UtilAPI.c
 *  @author konstantinos Parasyris (koparasy)
 *  @date   23 January, 2020
 *  @brief  API functions for the FTI library.
 */
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>


#include "FTI_UtilLowLevel.h"
#include "FTI_UtilAPI.h"

#define PRINTERROR()\
    fprintf(stderr, "Error in :%s\n",__func__)

/*-------------------------------------------------------------------------*/
/**
  @brief      Initializes FTI Utility.
  @param      configFile      FTI configuration file.
  @return     integer         SUCCESS if successful.

 This function initializes all internal data structures to represent an entire 
 checkpoint execution. 

 **/
/*-------------------------------------------------------------------------*/

int FTI_InitUtil(char *configFile){
    int ret = FTI_LLInitEnvironment(configFile); 
    if (ret != SUCCESS ){
        PRINTERROR();
    }
    return ret; 
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Returns the number of total checkpoints performed.
  @param      *numCkpts pointer in which it stores the number of checkpoints 
  @return     integer         SUCCESS if successful.

  This function sets *numCkpts to the total number of Checkpoints performed 
  during an application execution.
 **/
/*-------------------------------------------------------------------------*/

int FTI_GetNumberOfCkptIds(int *numCkpts){
    int ret = FTI_LLGetNumCheckpoints(); 
    if ( ret == ERROR){
        PRINTERROR();
        return ERROR; 
    }
    *numCkpts = ret;
    return SUCCESS;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Set the checkpoint id of each checkpoint.
  @param      *ckptIds pointer in which it stores the ids of checkpoints 
  @return     integer         SUCCESS if successful.

  This function sets each index of ckptIds to the respective checkpoint id.
  The function assumes that ckptIds has enough space to store all the ids. 
 **/
/*-------------------------------------------------------------------------*/
int FTI_GetCkptIds(int *ckptIds){
    assert(ckptIds);
    int ret = FTI_LLGetCkptID(ckptIds); 
    if ( ret != SUCCESS ){
        PRINTERROR();
        return ERROR;
    }
    return SUCCESS;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Finalize the utility.
  @return     integer         SUCCESS if successful.

  This function de-allocates and closes any file used by the utility
 **/
/*-------------------------------------------------------------------------*/

int FTI_FinalizeUtil(){
    int ret = FTI_LLFinalizeUtil();
    if (ret != SUCCESS){
        PRINTERROR();
        return ERROR;
    }
    return SUCCESS;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Sets numRanks to the number of ranks participated in the 
              application execution.
  @param      *numRanks pointer in which it stores the number of ranks 
  @return     integer         SUCCESS if successful.

              Sets numRanks to the number of ranks participated in the 
              application execution.

 **/
/*-------------------------------------------------------------------------*/
int FTI_GetUserRanks(int *numRanks){
    assert(numRanks);
    *numRanks = FTI_LLGetNumUserRanks();
    if (*numRanks == ERROR ){
        PRINTERROR();
        return ERROR;
    }
    return SUCCESS;

}

/*-------------------------------------------------------------------------*/
/**
  @brief      Verifies the checkpoint with id ckptId which was created by
              the specified rank 
  @param      ckptId checkpoint id 
  @param      rank specific application rank 
  @return     integer         SUCCESS if successful.

  Reads the checkpoint file and checks whether the is some kind of 
  in respect to the data stored during application execution corruption.
 **/
/*-------------------------------------------------------------------------*/
int FTI_VerifyCkpt(int ckptId, int rank){
    int ret =  FTI_LLverifyCkpt(ckptId, rank);
    if (ret != SUCCESS){
        PRINTERROR();
        return ERROR;
    }
    return SUCCESS;

}

/*-------------------------------------------------------------------------*/
/**
  @brief      retuns the number of variables stored in this checkpoint 
  @param      ckptId checkpoint id 
  @param      rank specific application rank 
  @return     integer         SUCCESS if successful.

  Reads the checkpoint file and checks whether the is some kind of 
  in respect to the data stored during application execution corruption.
 **/
/*-------------------------------------------------------------------------*/
int FTI_GetNumVars(int ckptId , int rank){
    int ret = FTI_LLGetNumVars(ckptId, rank);
    if (ret == ERROR){
        PRINTERROR();
        return ERROR;
    }
    return ret ;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Reads the specified variable from a checkpoint file 
  @param      ckptId checkpoint id 
  @param      rank specific application rank 
  @param      varName on return it contains the variable string name
  @param      varId on return it contains the variable id.
  @param      buff on return points to the read data
  @param      size on return it contains the size of the read data. 
  @return     integer         SUCCESS if successful.

    This function exports the specified variable from the ckpt. The buf ptr
    is allocated by the utility and will be de-allocated by the utility. 
    It should be used for read only operations. In case of trying to read the
    same variable the same ptr will be returned.
 **/
/*-------------------------------------------------------------------------*/
int FTI_readVariableByIndex(int varIndex, int ckptId, int rank, char **varName, int *varId, unsigned char **buf, size_t *size){
    int ret = FTI_LLreadVariable(varIndex, ckptId, rank, varName, varId,  buf, size);
    if (ret == ERROR){
        PRINTERROR();
        return ERROR;
    }
return ret ;

}

