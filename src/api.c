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
 *  @file   api.c
 *  @date   October, 2017
 *  @brief  API functions for the FTI library.
 */


#include "interface.h"


#include "ftiff.h"

#ifdef GPUSUPPORT
#include <cuda_runtime_api.h>
#endif 

#include "api_cuda.h"
#include "utility.h"


/** General configuration information used by FTI.                         */
static FTIT_configuration FTI_Conf;

/** Checkpoint information for all levels of checkpoint.                   */
static FTIT_checkpoint FTI_Ckpt[5];

/** Dynamic information for this execution.                                */
static FTIT_execution FTI_Exec;

/** Topology of the system.                                                */
static FTIT_topology FTI_Topo;

/** Array of datasets and all their internal information.                  */
static FTIT_dataset FTI_Data[FTI_BUFS];

/** SDC injection model and all the required information.                  */
static FTIT_injection FTI_Inje;

/** MPI communicator that splits the global one into app and FTI appart.   */
MPI_Comm FTI_COMM_WORLD;

/** FTI data type for chars.                                               */
FTIT_type FTI_CHAR;
/** FTI data type for short integers.                                      */
FTIT_type FTI_SHRT;
/** FTI data type for integers.                                            */
FTIT_type FTI_INTG;
/** FTI data type for long integers.                                       */
FTIT_type FTI_LONG;
/** FTI data type for unsigned chars.                                      */
FTIT_type FTI_UCHR;
/** FTI data type for unsigned short integers.                             */
FTIT_type FTI_USHT;
/** FTI data type for unsigned integers.                                   */
FTIT_type FTI_UINT;
/** FTI data type for unsigned long integers.                              */
FTIT_type FTI_ULNG;
/** FTI data type for single floating point.                               */
FTIT_type FTI_SFLT;
/** FTI data type for double floating point.                               */
FTIT_type FTI_DBLE;
/** FTI data type for long doble floating point.                           */
FTIT_type FTI_LDBE;


/*-------------------------------------------------------------------------*/
/**
  @brief      Initializes FTI.
  @param      configFile      FTI configuration file.
  @param      globalComm      Main MPI communicator of the application.
  @return     integer         FTI_SCES if successful.

  This function initializes the FTI context and prepares the heads to wait
  for checkpoints. FTI processes should never get out of this function. In
  case of a restart, checkpoint files should be recovered and in place at the
  end of this function.

 **/
/*-------------------------------------------------------------------------*/
int FTI_Init(char* configFile, MPI_Comm globalComm)
{
#ifdef ENABLE_FTI_FI_IO
    FTI_InitFIIO();
#endif
    FTI_InitExecVars(&FTI_Conf, &FTI_Exec, &FTI_Topo, FTI_Ckpt, &FTI_Inje);
    FTI_Exec.globalComm = globalComm;
    MPI_Comm_rank(FTI_Exec.globalComm, &FTI_Topo.myRank);
    MPI_Comm_size(FTI_Exec.globalComm, &FTI_Topo.nbProc);
    snprintf(FTI_Conf.cfgFile, FTI_BUFS, "%s", configFile);
    FTI_Conf.verbosity = 1; //Temporary needed for output in FTI_LoadConf.
    FTI_Exec.initSCES = 0;
    FTI_Inje.timer = MPI_Wtime();
    FTI_COMM_WORLD = globalComm; // Temporary before building topology. Needed in FTI_LoadConf and FTI_Topology to communicate.
    FTI_Topo.splitRank = FTI_Topo.myRank; // Temporary before building topology. Needed in FTI_Print.
    int res = FTI_Try(FTI_LoadConf(&FTI_Conf, &FTI_Exec, &FTI_Topo, FTI_Ckpt, &FTI_Inje), "load configuration.");
    if (res == FTI_NSCS) {
        return FTI_NSCS;
    }
    res = FTI_Try(FTI_Topology(&FTI_Conf, &FTI_Exec, &FTI_Topo), "build topology.");
    if (res == FTI_NSCS) {
        return FTI_NSCS;
    }
    FTI_Try(FTI_InitGroupsAndTypes(&FTI_Exec), "malloc arrays for groups and types.");
    FTI_Try(FTI_InitBasicTypes(FTI_Data), "create the basic data types.");
    if (FTI_Topo.myRank == 0) {
        FTI_Try(FTI_UpdateConf(&FTI_Conf, &FTI_Exec, FTI_Exec.reco), "update configuration file.");
    }
    MPI_Barrier(FTI_Exec.globalComm); //wait for myRank == 0 process to save config file
    FTI_MallocMeta(&FTI_Exec, &FTI_Topo);
    res = FTI_Try(FTI_LoadMeta(&FTI_Conf, &FTI_Exec, &FTI_Topo, FTI_Ckpt), "load metadata");
    if (res == FTI_NSCS) {
        FTI_FreeMeta(&FTI_Exec);
        return FTI_NSCS;
    }
    if( FTI_Conf.ioMode == FTI_IO_FTIFF ) {
        FTIFF_InitMpiTypes();
    }
    if( FTI_Conf.stagingEnabled ) {
        FTI_InitStage( &FTI_Exec, &FTI_Conf, &FTI_Topo );
    }
    FTI_Exec.initSCES = 1;
    if (FTI_Topo.amIaHead) { // If I am a FTI dedicated process
        if (FTI_Exec.reco) {
            res = FTI_Try(FTI_RecoverFiles(&FTI_Conf, &FTI_Exec, &FTI_Topo, FTI_Ckpt), "recover the checkpoint files.");
            if (res != FTI_SCES) {
                FTI_Exec.reco = 0;
                FTI_Exec.initSCES = 2; //Could not recover all ckpt files
            }
        }
        FTI_Listen(&FTI_Conf, &FTI_Exec, &FTI_Topo, FTI_Ckpt); //infinite loop inside, can stop only by callling FTI_Finalize
        // FTI_Listen only returns if FTI_Conf.keepHeadsAlive is TRUE
        return FTI_HEAD;
    }
    else { // If I am an application process
#ifdef GPUSUPPORT   
    CUDA_ERROR_CHECK(cudaStreamCreate(&FTI_Exec.cStream));
    CUDA_ERROR_CHECK(cudaEventCreateWithFlags(&FTI_Exec.cEvents[0], cudaEventBlockingSync | cudaEventDisableTiming));
    CUDA_ERROR_CHECK(cudaEventCreateWithFlags(&FTI_Exec.cEvents[1], cudaEventBlockingSync | cudaEventDisableTiming));
    CUDA_ERROR_CHECK(cudaHostAlloc(&FTI_Exec.cHostBufs[0], FTI_Conf.cHostBufSize, cudaHostAllocDefault));
    CUDA_ERROR_CHECK(cudaHostAlloc(&FTI_Exec.cHostBufs[1], FTI_Conf.cHostBufSize, cudaHostAllocDefault));
#endif
        // call in any case. treatment for diffCkpt disabled inside initializer.
        if( FTI_Conf.dcpEnabled ) {
            FTI_InitDcp( &FTI_Conf, &FTI_Exec, FTI_Data );
        }
        if (FTI_Exec.reco) {
            res = FTI_Try(FTI_RecoverFiles(&FTI_Conf, &FTI_Exec, &FTI_Topo, FTI_Ckpt), "recover the checkpoint files.");
            if (FTI_Conf.ioMode == FTI_IO_FTIFF && res == FTI_SCES) {
                res += FTI_Try( FTIFF_ReadDbFTIFF( &FTI_Conf, &FTI_Exec, FTI_Ckpt ), "Read FTIFF meta information" );
            }
            FTI_Exec.ckptCnt = FTI_Exec.ckptID;
            FTI_Exec.ckptCnt++;
            if (res != FTI_SCES) {
                FTI_Exec.reco = 0;
                FTI_Exec.initSCES = 2; //Could not recover all ckpt files (or failed reading meta; FTI-FF)
                FTI_Print("FTI has been initialized.", FTI_INFO);
                return FTI_NREC;
            }
        }
        FTI_Print("FTI has been initialized.", FTI_INFO);
        return FTI_SCES;
    }
}

/*-------------------------------------------------------------------------*/
/**
  @brief      It returns the current status of the recovery flag.
  @return     integer         FTI_Exec.reco.

  This function returns the current status of the recovery flag.

 **/
/*-------------------------------------------------------------------------*/
int FTI_Status()
{
    return FTI_Exec.reco;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      It initializes a data type.
  @param      type            The data type to be intialized.
  @param      size            The size of the data type to be intialized.
  @return     integer         FTI_SCES if successful.

  This function initalizes a data type. The only information needed is the
  size of the data type, the rest is black box for FTI. Types saved as byte array
  in case of HDF5 format.

 **/
/*-------------------------------------------------------------------------*/
int FTI_InitType(FTIT_type* type, int size)
{  
    type->id = FTI_Exec.nbType;
    type->size = size;
    type->structure = NULL;

#ifdef ENABLE_HDF5
    type->h5group = FTI_Exec.H5groups[0];

    // Maps FTI types to HDF5 types
    switch (FTI_Exec.nbType) {
        case 0:
            type->h5datatype = H5T_NATIVE_CHAR; break;
        case 1:
            type->h5datatype = H5T_NATIVE_SHORT; break;
        case 2:
            type->h5datatype = H5T_NATIVE_INT; break;
        case 3:
            type->h5datatype = H5T_NATIVE_LONG; break;
        case 4:
            type->h5datatype = H5T_NATIVE_UCHAR; break;
        case 5:
            type->h5datatype = H5T_NATIVE_USHORT; break;
        case 6:
            type->h5datatype = H5T_NATIVE_UINT; break;
        case 7:
            type->h5datatype = H5T_NATIVE_ULONG; break;
        case 8:
            type->h5datatype = H5T_NATIVE_FLOAT; break;
        case 9:
            type->h5datatype = H5T_NATIVE_DOUBLE; break;
        case 10:
            type->h5datatype = H5T_NATIVE_LDOUBLE; break;
        default:
            type->h5datatype = -1; break; //to mark as closed
    }
#endif

    //make a clone of the type in case the user won't store pointer
    FTI_Exec.FTI_Type[FTI_Exec.nbType] = malloc(sizeof(FTIT_type));
    *FTI_Exec.FTI_Type[FTI_Exec.nbType] = *type;

    FTI_Exec.nbType = FTI_Exec.nbType + 1;

    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
    @brief      It initializes a complex data type.
    @param      newType         The data type to be intialized.
    @param      typeDefinition  Structure definition of the new type.
    @param      length          Number of fields in structure
    @param      size            Size of the structure.
    @param      name            Name of the structure.
    @param      h5group         Group of the type.
    @return     integer         FTI_SCES if successful.

    This function initalizes a simple data type. New type can only consists
    fields of flat FTI types (no arrays). Type definition must include:
        - length                => number of fields in the new type
        - field[].type          => types of the field in the new type
        - field[].name          => name of the field in the new type
        - field[].rank          => number of dimentions of the field
        - field[].dimLength[]   => length of each dimention of the field

 **/
/*-------------------------------------------------------------------------*/
int FTI_InitComplexType(FTIT_type* newType, FTIT_complexType* typeDefinition, int length, size_t size, char* name, FTIT_H5Group* h5group)
{
    if (h5group == NULL) {
        h5group = FTI_Exec.H5groups[0];
    }
    if (length < 1) {
        FTI_Print("Type can't conain less than 1 type.", FTI_WARN);
        return FTI_NSCS;
    }
    if (length > 255) {
        FTI_Print("Type can't conain more than 255 types.", FTI_WARN);
        return FTI_NSCS;
    }
    int i;
    for (i = 0; i < length; i++) {
        if (typeDefinition->field[i].rank < 1) {
            FTI_Print("Type rank must be greater than 0.", FTI_WARN);
            return FTI_NSCS;
        }
        if (typeDefinition->field[i].rank > 32) {
            FTI_Print("Maximum rank is 32.", FTI_WARN);
            return FTI_NSCS;
        }
        int j;
        for (j = 0; j < typeDefinition->field[i].rank; j++) {
            if (typeDefinition->field[i].dimLength[j] < 1) {
                char str[FTI_BUFS];
                snprintf(str, FTI_BUFS, "(%s, index: %d) Type dimention length must be greater than 0.", typeDefinition->field[i].name, i);
                FTI_Print(str, FTI_WARN);
                return FTI_NSCS;
            }
        }
    }

    newType->id = FTI_Exec.nbType;
    newType->size = size;
    //assign type definition to type structure (types, names, ranks, dimLengths)
    typeDefinition->length = length;
    if (name == NULL || !strlen(name)) {
        sprintf(typeDefinition->name, "Type%d", newType->id);
    } else {
        strncpy(typeDefinition->name, name, FTI_BUFS);
    }

    #ifdef ENABLE_HDF5
        newType->h5datatype = -1; //to mark as closed
        newType->h5group = FTI_Exec.H5groups[h5group->id];
    #endif

    //make a clone of the type definition in case the user won't store pointer
    newType->structure = malloc(sizeof(FTIT_complexType));
    *newType->structure = *typeDefinition;

    //append a space for new type
    FTI_Exec.FTI_Type = realloc(FTI_Exec.FTI_Type, sizeof(FTIT_type*) * (FTI_Exec.nbType + 1));

    //make a clone of the type in case the user won't store pointer
    FTI_Exec.FTI_Type[FTI_Exec.nbType] = malloc(sizeof(FTIT_type));
    *FTI_Exec.FTI_Type[FTI_Exec.nbType] = *newType;

    FTI_Exec.nbType = FTI_Exec.nbType + 1;

    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
    @brief      It adds a simple field in complex data type.
    @param      typeDefinition  Structure definition of the complex data type.
    @param      ftiType         Type of the field
    @param      offset          Offset of the field (use offsetof)
    @param      id              Id of the field (start with 0)
    @param      name            Name of the field (put NULL if want default)
    @return     integer         FTI_SCES if successful.

    This function adds a field to the complex datatype. Use offsetof macro to
    set offset. First ID must be 0, next one must be +1. If name is NULL FTI
    will set "T${id}" name. Sets rank and dimLength to 1.

 **/
/*-------------------------------------------------------------------------*/
void FTI_AddSimpleField(FTIT_complexType* typeDefinition, FTIT_type* ftiType, size_t offset, int id, char* name)
{
    typeDefinition->field[id].typeID = ftiType->id;
    typeDefinition->field[id].offset = offset;
    if (name == NULL || !strlen(name)) {
        sprintf(typeDefinition->field[id].name, "T%d", id);
    } else {
        strncpy(typeDefinition->field[id].name, name, FTI_BUFS);
    }
    typeDefinition->field[id].rank = 1;
    typeDefinition->field[id].dimLength[0] = 1;
}

/*-------------------------------------------------------------------------*/
/**
    @brief      It adds a simple field in complex data type.
    @param      typeDefinition  Structure definition of the complex data type.
    @param      ftiType         Type of the field
    @param      offset          Offset of the field (use offsetof)
    @param      rank            Rank of the array
    @param      dimLength       Dimention length for each rank
    @param      id              Id of the field (start with 0)
    @param      name            Name of the field (put NULL if want default)
    @return     integer         FTI_SCES if successful.

    This function adds a field to the complex datatype. Use offsetof macro to
    set offset. First ID must be 0, next one must be +1. If name is NULL FTI
    will set "T${id}" name.

 **/
/*-------------------------------------------------------------------------*/
void FTI_AddComplexField(FTIT_complexType* typeDefinition, FTIT_type* ftiType, size_t offset, int rank, int* dimLength, int id, char* name)
{
    typeDefinition->field[id].typeID = ftiType->id;
    typeDefinition->field[id].offset = offset;
    typeDefinition->field[id].rank = rank;
    int i;
    for (i = 0; i < rank; i++) {
        typeDefinition->field[id].dimLength[i] = dimLength[i];
    }
    if (name == NULL || !strlen(name)) {
        sprintf(typeDefinition->field[id].name, "T%d", id);
    } else {
        strncpy(typeDefinition->field[id].name, name, FTI_BUFS);
    }
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Places the FTI staging directory path into 'stageDir'.
  @param      stageDir        pointer to allocated memory region.
  @param      maxLen          size of allocated memory region in bytes.
  @return     integer         FTI_SCES if successful, FTI_NSCS else.

  This function places the FTI staging directory path in 'stageDir'. If
  allocation size is not sufficiant, no action is perfoprmed and
  FTI_NSCS is returned.
 **/
/*-------------------------------------------------------------------------*/
int FTI_GetStageDir( char* stageDir, int maxLen) 
{
    
    if ( !FTI_Conf.stagingEnabled ) {
        FTI_Print( "'FTI_GetStageDir' -> Staging disabled, no action performed.", FTI_WARN );
        return FTI_NSCS;
    }

    if( stageDir == NULL ) {
        FTI_Print( "invalid value for stageDir ('nil')!", FTI_WARN );
        return FTI_NSCS;
    }

    if( maxLen < 1 ) { 
        char errstr[FTI_BUFS];
        snprintf( errstr, FTI_BUFS, "invalid value for maxLen ('%d')!", maxLen );
        FTI_Print( errstr, FTI_WARN );
        return FTI_NSCS;
    }

    int len = strlen(FTI_Conf.stageDir);
    if( maxLen < len+1 ) {
        FTI_Print( "insufficient buffer size (maxLen too small)!", FTI_WARN );
        return FTI_NSCS;
    }

    strncpy( stageDir, FTI_Conf.stageDir, FTI_BUFS );

    return FTI_SCES;

}


/*-------------------------------------------------------------------------*/
/**
  @brief      Returns status of staging request.
  @param      ID            ID of staging request.
  @return     integer       Status of staging request on success, 
                            FTI_NSCS else.

  This function returns the status of the staging request corresponding
  to ID. The ID is returned by the function 'FTI_SendFile'. The status
  may be one of the five possible statuses:
  
  @par
  FTI_SI_FAIL - Stage request failed
  FTI_SI_SCES - Stage request succeed
  FTI_SI_ACTV - Stage request is currently processed
  FTI_SI_PEND - Stage request is pending
  FTI_SI_NINI - There is no stage request with this ID
  
  @note If the status is FTI_SI_NINI, the ID is either invalid or the
  request was finished (succeeded or failed). In the latter case,
  'FTI_GetStageStatus' returns FTI_SI_FAIL or FTI_SI_SCES and frees the
  stage request ressources. In the consecutive call it will then return
  FTI_SI_NINI.
 **/
/*-------------------------------------------------------------------------*/
int FTI_GetStageStatus( int ID )
{

    if ( !FTI_Conf.stagingEnabled ) {
        FTI_Print( "'FTI_GetStageStatus' -> Staging disabled, no action performed.", FTI_WARN );
        return FTI_NSCS;
    }

    // indicator if we still need the request structure allocated
    // (i.e. send buffer not released by MPI)
    bool free_req = true;

    // get status of request
    int status;
    status = FTI_GetStatusField( &FTI_Exec, &FTI_Topo, ID, FTI_SIF_VAL, FTI_Topo.nodeRank );  

    // check if pending
    if ( status == FTI_SI_PEND ) {
        int flag = 1, idx;
        // if pending check if we can free the send buffer
        if ( (idx = FTI_GetRequestIdx(ID)) >= 0 ) { 
            MPI_Test( &(FTI_SI_APTR(FTI_Exec.stageInfo->request)[idx].mpiReq), &flag, MPI_STATUS_IGNORE );
        }
        if ( flag == 0 ) {
            free_req = false;
        }
    }

    if ( free_req ) {
        FTI_FreeStageRequest( &FTI_Exec, &FTI_Topo, ID, FTI_Topo.nodeRank );
    }
   
    if ( (status==FTI_SI_FAIL) || (status==FTI_SI_SCES) ) {
        FTI_SetStatusField( &FTI_Exec, &FTI_Topo, ID, FTI_SI_NINI, FTI_SIF_VAL, FTI_Topo.nodeRank );
        FTI_SetStatusField( &FTI_Exec, &FTI_Topo, ID, FTI_SI_IAVL, FTI_SIF_AVL, FTI_Topo.nodeRank );
    }

    return status;
        
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Copies file asynchronously from 'lpath' to 'rpath'.
  @param      lpath           absolute path local file.
  @param      rpath           absolute path remote file.
  @return     integer         Request handle (ID) on success, FTI_NSCS else.

  This function may be used to copy a file local on the nodes via the
  FTI head process asynchronously to the PFS. The file will not be
  removed after successful transfer, however, if stored in the directory
  returned by 'FTI_GetStageDir' it will be removed during
  'FTI_Finalize'.
  
  @par
  If staging is enabled but no head process, the staging will be
  performed synchronously (i.e. by the calling rank).
 **/
/*-------------------------------------------------------------------------*/
int FTI_SendFile( char* lpath, char *rpath )
{ 

    if ( !FTI_Conf.stagingEnabled ) {
        FTI_Print( "'FTI_SendFile' -> Staging disabled, no action performed.", FTI_WARN );
        return FTI_NSCS;
    }

    int ID = FTI_NSCS;
    
    // discard if path is NULL
    if ( lpath == NULL ){
        FTI_Print( "local path field is NULL!", FTI_WARN );
        return FTI_NSCS;
    }

    if ( rpath == NULL ){
        FTI_Print( "remote path field is NULL!", FTI_WARN );
        return FTI_NSCS;
    }

    // asign new request ID
    // note: if ID found, FTI_Exec->stageInfo->status[ID] is set to not available
    int reqID = FTI_GetRequestID( &FTI_Exec, &FTI_Topo );
    if (reqID < 0) {
        FTI_Print("Too many stage requests!", FTI_WARN);
        return FTI_NSCS;
    }
    ID = reqID;

    FTI_InitStageRequestApp( &FTI_Exec, &FTI_Topo, ID );
    
    if ( FTI_Topo.nbHeads == 0 ) {

        if ( FTI_SyncStage( lpath, rpath, &FTI_Exec, &FTI_Topo, &FTI_Conf, ID ) != FTI_SCES ) {
            FTI_Print("synchronous staging failed!", FTI_WARN);
            return FTI_NSCS;
        }

    }
    
    if ( FTI_Topo.nbHeads > 0 ) {
        
        if ( FTI_AsyncStage( lpath, rpath, &FTI_Conf, &FTI_Exec, &FTI_Topo, ID ) != FTI_SCES ) {
            FTI_Print("asynchronous staging failed!", FTI_WARN);
            return FTI_NSCS;
        }
    
    }
    
    return ID;

}

/*-------------------------------------------------------------------------*/
/**
  @brief      It initialize a HDF5 group
  @param      h5group         H5 group that we want to initialize
  @param      name            Name of the H5 group
  @param      parent          Parent H5 group
  @return     integer         FTI_SCES if successful.

    Initialize group defined by user. If parent is NULL this mean parent will
    be set to root group.

 **/
/*-------------------------------------------------------------------------*/
int FTI_InitGroup(FTIT_H5Group* h5group, char* name, FTIT_H5Group* parent)
{
    if (parent == NULL) {
        //child of root
        parent = FTI_Exec.H5groups[0];
    }
    FTIT_H5Group* parentInArray = FTI_Exec.H5groups[parent->id];
    //check if this parent has that child
    int i;
    for (i = 0; i < parentInArray->childrenNo; i++) {
        if (strcmp(FTI_Exec.H5groups[parentInArray->childrenID[i]]->name, name) == 0) {
            char str[FTI_BUFS];
            snprintf(str, FTI_BUFS, "Group %s already has the %s child.", parentInArray->name, name);
            return FTI_NSCS;
        }
    }
    h5group->id = FTI_Exec.nbGroup;
    h5group->childrenNo = 0;
    strncpy(h5group->name, name, FTI_BUFS);
#ifdef ENABLE_HDF5
    h5group->h5groupID = -1; //to mark as closed
#endif

    //make a clone of the group in case the user won't store pointer
    FTI_Exec.H5groups[FTI_Exec.nbGroup] = malloc(sizeof(FTIT_H5Group));
    *FTI_Exec.H5groups[FTI_Exec.nbGroup] = *h5group;

    //assign a child and increment the childrenNo
    parentInArray->childrenID[parentInArray->childrenNo] = FTI_Exec.nbGroup;
    parentInArray->childrenNo++;

    FTI_Exec.nbGroup = FTI_Exec.nbGroup + 1;

    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Renames a HDF5 group
  @param      h5group         H5 group that we want to rename
  @param      name            New name of the H5 group
  @return     integer         FTI_SCES if successful.

    This function renames HDF5 group defined by user.

 **/
/*-------------------------------------------------------------------------*/
int FTI_RenameGroup(FTIT_H5Group* h5group, char* name) {
    strncpy(FTI_Exec.H5groups[h5group->id]->name, name, FTI_BUFS);
    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      It sets/resets the pointer and type to a protected variable.
  @param      id              ID for searches and update.
  @param      ptr             Pointer to the data structure.
  @param      count           Number of elements in the data structure.
  @param      type            Type of elements in the data structure.
  @return     integer         FTI_SCES if successful.

  This function stores a pointer to a data structure, its size, its ID,
  its number of elements and the type of the elements. This list of
  structures is the data that will be stored during a checkpoint and
  loaded during a recovery. It resets the pointer to a data structure,
  its size, its number of elements and the type of the elements if the
  dataset was already previously registered.

 **/
/*-------------------------------------------------------------------------*/
int FTI_Protect(int id, void* ptr, long count, FTIT_type type)
{
  if (FTI_Exec.initSCES == 0) {
    FTI_Print("FTI is not initialized.", FTI_WARN);
    return FTI_NSCS;
  }

  char str[FTI_BUFS]; //For console output
#ifdef GPUSUPPORT 
  FTIT_ptrinfo ptrInfo;
  int res;
  if (( res = FTI_Try( FTI_get_pointer_info((const void*) ptr, &ptrInfo),"FTI_Protect: determine pointer type")) != FTI_SCES )  
    return res;
#endif

  int i;
  for (i = 0; i < FTI_BUFS; i++) {
    if (id == FTI_Data[i].id) { //Search for dataset with given id
      long prevSize = FTI_Data[i].size;
#ifdef GPUSUPPORT
      if ( ptrInfo.type == FTIT_PTRTYPE_CPU) {
        FTI_Data[i].isDevicePtr = false;
        FTI_Data[i].devicePtr= NULL;
        FTI_Data[i].ptr = ptr;
      }
      else if( ptrInfo.type == FTIT_PTRTYPE_GPU ){
        FTI_Data[i].isDevicePtr = true;
        FTI_Data[i].devicePtr= ptr;
        FTI_Data[i].ptr = NULL; //(void *) malloc (type.size *count);
        if (FTI_Conf.ioMode == FTI_IO_FTIFF || FTI_Conf.ioMode == FTI_IO_HDF5){
          FTI_Data[i].ptr = (void *) malloc (type.size *count);
          if (FTI_Data[i].ptr == NULL){
            FTI_Print("Could Not Allocate Extra Buffer for GPU data\n",FTI_EROR);
            return FTI_NSCS;
          }
        } 
      }
      else{
          FTI_Print("ptr Should be either a device location or a cpu location\n",FTI_EROR);
          FTI_Data[i].ptr = NULL; //(void *) malloc (type.size *count);
          return FTI_NSCS;
      }
#else            
      FTI_Data[i].isDevicePtr = false;
      FTI_Data[i].devicePtr= NULL;
      FTI_Data[i].ptr = ptr;
#endif  
      FTI_Data[i].count = count;
      FTI_Data[i].type = FTI_Exec.FTI_Type[type.id];
      FTI_Data[i].eleSize = type.size;
      FTI_Data[i].size = type.size * count;
      FTI_Data[i].dimLength[0] = count;
      FTI_Exec.ckptSize = FTI_Exec.ckptSize + ((type.size * count) - prevSize);
      sprintf(str, "Variable ID %d reseted. Current ckpt. size per rank is %.2fMB.", id, (float) FTI_Exec.ckptSize / (1024.0 * 1024.0));
      FTI_Print(str, FTI_DBUG);
      return FTI_SCES;
    }
  }
  //Id could not be found in datasets

  //If too many variables exit FTI.
  if (FTI_Exec.nbVar >= FTI_BUFS) {
    FTI_Print("Unable to register variable. Too many variables already registered.", FTI_WARN);
    return FTI_NSCS;
  }

  //Adding new variable to protect
  FTI_Data[FTI_Exec.nbVar].id = id;
#ifdef GPUSUPPORT
  if ( ptrInfo.type == FTIT_PTRTYPE_CPU) {
    FTI_Data[FTI_Exec.nbVar].isDevicePtr = false;
    FTI_Data[FTI_Exec.nbVar].devicePtr= NULL;
    FTI_Data[FTI_Exec.nbVar].ptr = ptr;
    sprintf(str, "Variable ID %d to protect.  isDevice %d %p %p", id,  FTI_Data[FTI_Exec.nbVar].isDevicePtr, FTI_Data[FTI_Exec.nbVar].ptr,FTI_Data[FTI_Exec.nbVar].devicePtr);
    FTI_Print(str, FTI_INFO);
  }
  else if( ptrInfo.type == FTIT_PTRTYPE_GPU ){
    FTI_Data[FTI_Exec.nbVar].isDevicePtr = true;
    FTI_Data[FTI_Exec.nbVar].devicePtr= ptr;
    FTI_Data[FTI_Exec.nbVar].ptr = NULL; //(void *) malloc (type.size *count);
    if (FTI_Conf.ioMode == FTI_IO_FTIFF || FTI_Conf.ioMode == FTI_IO_HDF5){
      FTI_Data[FTI_Exec.nbVar].ptr =  (void *) malloc (type.size *count);
      if (FTI_Data[FTI_Exec.nbVar].ptr == NULL){
        FTI_Print("Could Not Allocate Extra Buffer for GPU data\n",FTI_EROR);
        return FTI_NSCS;
      }
      sprintf(str, "Variable ID %d to protect.  isDevice %d %p %p", id,  FTI_Data[FTI_Exec.nbVar].isDevicePtr, FTI_Data[FTI_Exec.nbVar].ptr,FTI_Data[FTI_Exec.nbVar].devicePtr);
      FTI_Print(str, FTI_INFO);
    } 
  }
   else{
      FTI_Print("ptr Should be either a device location or a cpu location\n",FTI_EROR);
      FTI_Data[FTI_Exec.nbVar].ptr = NULL; //(void *) malloc (type.size *count);
      return FTI_NSCS;
    }
#else            
  FTI_Data[FTI_Exec.nbVar].isDevicePtr = false;
  FTI_Data[FTI_Exec.nbVar].devicePtr= NULL;
  FTI_Data[FTI_Exec.nbVar].ptr = ptr;
#endif  
  FTI_Data[FTI_Exec.nbVar].count = count;
  FTI_Data[FTI_Exec.nbVar].type = FTI_Exec.FTI_Type[type.id];
  FTI_Data[FTI_Exec.nbVar].eleSize = type.size;
  FTI_Data[FTI_Exec.nbVar].size = type.size * count;
  FTI_Data[FTI_Exec.nbVar].rank = 1;
  FTI_Data[FTI_Exec.nbVar].dimLength[0] = FTI_Data[FTI_Exec.nbVar].count;
  FTI_Data[FTI_Exec.nbVar].h5group = FTI_Exec.H5groups[0];
  sprintf(FTI_Data[FTI_Exec.nbVar].name, "Dataset_%d", id);
  FTI_Exec.ckptSize = FTI_Exec.ckptSize + (type.size * count);
  sprintf(str, "Variable ID %d to protect. Current ckpt. size per rank is %.2fMB. isDevice %d Device: %p CPU: %p", id, (float) FTI_Exec.ckptSize / (1024.0 * 1024.0), FTI_Data[FTI_Exec.nbVar].isDevicePtr, FTI_Data[FTI_Exec.nbVar].devicePtr,FTI_Data[FTI_Exec.nbVar].ptr);
  FTI_Exec.nbVar = FTI_Exec.nbVar + 1;
  FTI_Print(str, FTI_INFO);
  return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
    @brief      Defines the dataset
    @param      id              ID for searches and update.
    @param      rank            Rank of the array
    @param      dimLength       Dimention length for each rank
    @param      name            Name of the dataset in HDF5 file.
    @param      h5group         Group of the dataset. If Null then "/"
    @return     integer         FTI_SCES if successful.

    This function gives FTI all information needed by HDF5 to correctly save
    the dataset in the checkpoint file.

 **/
/*-------------------------------------------------------------------------*/
int FTI_DefineDataset(int id, int rank, int* dimLength, char* name, FTIT_H5Group* h5group)
{
    if (FTI_Exec.initSCES == 0) {
        FTI_Print("FTI is not initialized.", FTI_WARN);
        return FTI_NSCS;
    }
    
    if (rank > 0 && dimLength == NULL) {
        FTI_Print("If rank > 0, the dimLength cannot be NULL.", FTI_WARN);
        return FTI_NSCS;
    }
    if (rank > 32) {
        FTI_Print("Maximum rank is 32.", FTI_WARN);
        return FTI_NSCS;
    }

    char str[FTI_BUFS]; //For console output

    int i;
    for (i = 0; i < FTI_BUFS; i++) {
        if (id == FTI_Data[i].id) { //Search for dataset with given id
            //check if size is correct
            int expectedSize = 1;
            int j;
            for (j = 0; j < rank; j++) {
                expectedSize *= dimLength[j]; //compute the number of elements
            }

            if (rank > 0) {
                if (expectedSize != FTI_Data[i].count) {
                    sprintf(str, "Trying to define datasize: number of elements %d, but the dataset count is %ld.", expectedSize, FTI_Data[i].count);
                    FTI_Print(str, FTI_WARN);
                    return FTI_NSCS;
                }
                FTI_Data[i].rank = rank;
                for (j = 0; j < rank; j++) {
                    FTI_Data[i].dimLength[j] = dimLength[j];
                }
            }

            if (h5group != NULL) {
                FTI_Data[i].h5group = FTI_Exec.H5groups[h5group->id];
            }

            if (name != NULL) {
                strncpy(FTI_Data[i].name, name, FTI_BUFS);
            }
            
            return FTI_SCES;
        }
    }

    sprintf(str, "The dataset #%d not initialized. Use FTI_Protect first.", id);
    FTI_Print(str, FTI_WARN);
    return FTI_NSCS;
}


/*-------------------------------------------------------------------------*/
/**
  @brief      Returns size saved in metadata of variable
  @param      id              Variable ID.
  @return     long            Returns size of variable or 0 if size not saved.

  This function returns size of variable of given ID that is saved in metadata.
  This may be different from size of variable that is in the program. If this
  function it's called when recovery it returns size from metadata file, if it's
  called after checkpoint it returns size saved in temporary metadata. If there
  is no size saved in metadata it returns 0.
 **/
/*-------------------------------------------------------------------------*/
long FTI_GetStoredSize(int id)
{
    if (FTI_Exec.initSCES == 0) {
        FTI_Print("FTI is not initialized.", FTI_WARN);
        return 0;
    }

    int i;
    //Search first in temporary metadata (always the newest)
    for (i = 0; i < FTI_BUFS; i++) {
        if (FTI_Exec.meta[0].varID[i] == id) {
            if (FTI_Exec.meta[0].varSize[i] != 0) {
                return FTI_Exec.meta[0].varSize[i];
            }
            break;
        }
    }
    //If couldn't find in temporary metadata, search in last level checkpoint
    //(this means no checkpoint was taken in current execution)
    for (i = 0; i < FTI_BUFS; i++) {
        if (FTI_Exec.meta[FTI_Exec.ckptLvel].varID[i] == id) {
            return FTI_Exec.meta[FTI_Exec.ckptLvel].varSize[i];
        }
    }
    return 0;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Reallocates dataset to last checkpoint size.
  @param      id              Variable ID.
  @param      ptr             Pointer to the variable.
  @return     ptr             Pointer if successful, NULL otherwise
  This function loads the checkpoint data size from the metadata
  file, reallacates memory and updates data size information.
 **/
/*-------------------------------------------------------------------------*/
void* FTI_Realloc(int id, void* ptr)
{
    if (FTI_Exec.initSCES == 0) {
        FTI_Print("FTI is not initialized.", FTI_WARN);
        return ptr;
    }

    FTI_Print("Trying to reallocate dataset.", FTI_DBUG);
    if (FTI_Exec.reco) {
        char str[FTI_BUFS];
        int i;
        for (i = 0; i < FTI_BUFS; i++) {
            if (id == FTI_Data[i].id) {
                long oldSize = FTI_Data[i].size;
                FTI_Data[i].size = FTI_Exec.meta[FTI_Exec.ckptLvel].varSize[i];
                sprintf(str, "Reallocated size: %ld", FTI_Data[i].size);
                FTI_Print(str, FTI_DBUG);
                if (FTI_Data[i].size == 0) {
                    sprintf(str, "Cannot allocate 0 size.");
                    FTI_Print(str, FTI_DBUG);
                    return ptr;
                }
                ptr = realloc (ptr, FTI_Data[i].size);
                FTI_Data[i].ptr = ptr;
                FTI_Data[i].count = FTI_Data[i].size / FTI_Data[i].eleSize;
                FTI_Exec.ckptSize += FTI_Data[i].size - oldSize;
                sprintf(str, "Dataset #%d reallocated.", FTI_Data[i].id);
                FTI_Print(str, FTI_INFO);
                break;
            }
        }
    }
    else {
        FTI_Print("This is not a recovery. Couldn't reallocate memory.", FTI_WARN);
    }
    return ptr;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      It corrupts a bit of the given float.
  @param      target          Pointer to the float to corrupt.
  @param      bit             Position of the bit to corrupt.
  @return     integer         FTI_SCES if successful.

  This function filps the bit of the target float.

 **/
/*-------------------------------------------------------------------------*/
int FTI_FloatBitFlip(float* target, int bit)
{
    if (bit >= 32 || bit < 0) {
        return FTI_NSCS;
    }
    int* corIntPtr = (int*)target;
    int corInt = *corIntPtr;
    corInt = corInt ^ (1 << bit);
    corIntPtr = &corInt;
    float* fp = (float*)corIntPtr;
    *target = *fp;
    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      It corrupts a bit of the given float.
  @param      target          Pointer to the float to corrupt.
  @param      bit             Position of the bit to corrupt.
  @return     integer         FTI_SCES if successful.

  This function filps the bit of the target float.

 **/
/*-------------------------------------------------------------------------*/
int FTI_DoubleBitFlip(double* target, int bit)
{
    if (bit >= 64 || bit < 0) {
        return FTI_NSCS;
    }
    FTIT_double myDouble;
    myDouble.value = *target;
    int bitf = (bit >= 32) ? bit - 32 : bit;
    int half = (bit >= 32) ? 1 : 0;
    FTI_FloatBitFlip(&(myDouble.floatval[half]), bitf);
    *target = myDouble.value;
    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Bit-flip injection following the injection instructions.
  @param      datasetID       ID of the dataset where to inject.
  @return     integer         FTI_SCES if successful.

  This function injects the given number of bit-flips, at the given
  frequency and in the given location (rank, dataset, bit position).

 **/
/*-------------------------------------------------------------------------*/
int FTI_BitFlip(int datasetID)
{
    if (FTI_Exec.initSCES == 0) {
        FTI_Print("FTI is not initialized.", FTI_WARN);
        return FTI_NSCS;
    }

    if (FTI_Inje.rank == FTI_Topo.splitRank) {
        if (datasetID >= FTI_Exec.nbVar) {
            return FTI_NSCS;
        }
        if (FTI_Inje.counter < FTI_Inje.number) {
            if ((MPI_Wtime() - FTI_Inje.timer) > FTI_Inje.frequency) {
                if (FTI_Inje.index < FTI_Data[datasetID].count) {
                    char str[FTI_BUFS];
                    if (FTI_Data[datasetID].type->id == 9) { // If it is a double
                        double* target = FTI_Data[datasetID].ptr + FTI_Inje.index;
                        double ori = *target;
                        int res = FTI_DoubleBitFlip(target, FTI_Inje.position);
                        FTI_Inje.counter = (res == FTI_SCES) ? FTI_Inje.counter + 1 : FTI_Inje.counter;
                        FTI_Inje.timer = (res == FTI_SCES) ? MPI_Wtime() : FTI_Inje.timer;
                        sprintf(str, "Injecting bit-flip in dataset %d, index %d, bit %d : %f => %f",
                                datasetID, FTI_Inje.index, FTI_Inje.position, ori, *target);
                        FTI_Print(str, FTI_WARN);
                        return res;
                    }
                    if (FTI_Data[datasetID].type->id == 8) { // If it is a float
                        float* target = FTI_Data[datasetID].ptr + FTI_Inje.index;
                        float ori = *target;
                        int res = FTI_FloatBitFlip(target, FTI_Inje.position);
                        FTI_Inje.counter = (res == FTI_SCES) ? FTI_Inje.counter + 1 : FTI_Inje.counter;
                        FTI_Inje.timer = (res == FTI_SCES) ? MPI_Wtime() : FTI_Inje.timer;
                        sprintf(str, "Injecting bit-flip in dataset %d, index %d, bit %d : %f => %f",
                                datasetID, FTI_Inje.index, FTI_Inje.position, ori, *target);
                        FTI_Print(str, FTI_WARN);
                        return res;
                    }
                }
            }
        }
    }
    return FTI_NSCS;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      It takes the checkpoint and triggers the post-ckpt. work.
  @param      id              Checkpoint ID.
  @param      level           Checkpoint level.
  @return     integer         FTI_SCES if successful.

  This function starts by blocking on a receive if the previous ckpt. was
  offline. Then, it updates the ckpt. information. It writes down the ckpt.
  data, creates the metadata and the post-processing work. This function
  is complementary with the FTI_Listen function in terms of communications.

 **/
/*-------------------------------------------------------------------------*/
int FTI_Checkpoint(int id, int level)
{
     
    char str[FTI_BUFS]; //For console output
    
    if (FTI_Exec.initSCES == 0) {
        FTI_Print("FTI is not initialized.", FTI_WARN);
        return FTI_NSCS;
    }

    if ((level < FTI_MIN_LEVEL_ID) || (level > FTI_MAX_LEVEL_ID)) {
        FTI_Print("Invalid level id! Aborting checkpoint creation...", FTI_WARN);
        return FTI_NSCS;
    }
    if ((level > FTI_L4) && (level < FTI_L4_DCP)) {
        snprintf( str, FTI_BUFS, "dCP only implemented for level 4! setting to level %d...", level - 4 );
        FTI_Print(str, FTI_WARN);
        level -= 4; 
    }

    int ckptFirst = !FTI_Exec.ckptID; //ckptID = 0 if first checkpoint
    FTI_Exec.ckptID = id;

    // reset dcp requests.
    FTI_Ckpt[4].isDcp = false;
    if ( level == FTI_L4_DCP ) {
        if ( FTI_Conf.ioMode == FTI_IO_FTIFF ) {
            if ( FTI_Conf.dcpEnabled ) {
                FTI_Ckpt[4].isDcp = true;
            } else {
                FTI_Print("L4 dCP requested, but dCP is disabled!", FTI_WARN);
            }
        } else {
            FTI_Print("L4 dCP requested, but dCP needs FTI-FF!", FTI_WARN);
        }
        level = 4;
    }

    double t0 = MPI_Wtime(); //Start time
    if (FTI_Exec.wasLastOffline == 1) { // Block until previous checkpoint is done (Async. work)
        int lastLevel;
        MPI_Recv(&lastLevel, 1, MPI_INT, FTI_Topo.headRank, FTI_Conf.generalTag, FTI_Exec.globalComm, MPI_STATUS_IGNORE);
        if (lastLevel != FTI_NSCS) { //Head sends level of checkpoint if post-processing succeed, FTI_NSCS Otherwise
            FTI_Exec.lastCkptLvel = lastLevel; //Store last successful post-processing checkpoint level
            sprintf(str, "LastCkptLvel received from head: %d", lastLevel);
            FTI_Print(str, FTI_DBUG);
        } else {
            FTI_Print("Head failed to do post-processing after previous checkpoint.", FTI_WARN);
        }
    }
    
    double t1 = MPI_Wtime(); //Time after waiting for head to done previous post-processing
    int lastCkptLvel = FTI_Exec.ckptLvel; //Store last successful writing checkpoint level in case of failure
    FTI_Exec.ckptLvel = level; //For FTI_WriteCkpt
    int res = FTI_Try(FTI_WriteCkpt(&FTI_Conf, &FTI_Exec, &FTI_Topo, FTI_Ckpt, FTI_Data), "write the checkpoint.");
    double t2 = MPI_Wtime(); //Time after writing checkpoint
   

    // set hasCkpt flags true
    if ( FTI_Conf.dcpEnabled && FTI_Ckpt[4].isDcp ) {
        FTIFF_db* currentDB = FTI_Exec.firstdb;
        currentDB->update = false;
        do {    
            int varIdx;
            for(varIdx=0; varIdx<currentDB->numvars; ++varIdx) {
                FTIFF_dbvar* currentdbVar = &(currentDB->dbvars[varIdx]);
                currentdbVar->hasCkpt = true;
                currentdbVar->update = false;
            }
        }
        while ( (currentDB = currentDB->next) != NULL );    
        
    
        FTI_UpdateDcpChanges(FTI_Data, &FTI_Exec);
        FTI_Ckpt[4].hasDcp = true;
    }

    // FTIFF: send meta info to the heads
    FTIFF_headInfo *headInfo;
    if (!FTI_Ckpt[FTI_Exec.ckptLvel].isInline) { // If postCkpt. work is Async. then send message
        FTI_Exec.wasLastOffline = 1;
        // Head needs ckpt. ID to determine ckpt file name.
        int value = FTI_BASE + FTI_Exec.ckptLvel; //Token to send to head
        if (res != FTI_SCES) { //If Writing checkpoint failed
            FTI_Exec.ckptLvel = lastCkptLvel; //Set previous ckptLvel
            value = FTI_REJW; //Send reject checkpoint token to head
        }
        MPI_Send(&value, 1, MPI_INT, FTI_Topo.headRank, FTI_Conf.ckptTag, FTI_Exec.globalComm);
        // FTIFF: send meta info to the heads
        if( FTI_Conf.ioMode == FTI_IO_FTIFF && value != FTI_REJW ) {
            headInfo = malloc(sizeof(FTIFF_headInfo));
            headInfo->exists = FTI_Exec.meta[0].exists[0];
            headInfo->nbVar = FTI_Exec.meta[0].nbVar[0];
            headInfo->maxFs = FTI_Exec.meta[0].maxFs[0];
            headInfo->fs = FTI_Exec.meta[0].fs[0];
            headInfo->pfs = FTI_Exec.meta[0].pfs[0];
            headInfo->isDcp = (FTI_Ckpt[4].isDcp) ? 1 : 0;
            if( FTI_Conf.dcpEnabled && FTI_Ckpt[4].isDcp ) {
                strncpy(headInfo->ckptFile, FTI_Ckpt[4].dcpName, FTI_BUFS);
            } else {
                strncpy(headInfo->ckptFile, FTI_Exec.meta[0].ckptFile, FTI_BUFS);
            }            
	    MPI_Send(headInfo, 1, FTIFF_MpiTypes[FTIFF_HEAD_INFO], FTI_Topo.headRank, FTI_Conf.generalTag, FTI_Exec.globalComm);
            MPI_Send(FTI_Exec.meta[0].varID, headInfo->nbVar, MPI_INT, FTI_Topo.headRank, FTI_Conf.generalTag, FTI_Exec.globalComm);
            MPI_Send(FTI_Exec.meta[0].varSize, headInfo->nbVar, MPI_LONG, FTI_Topo.headRank, FTI_Conf.generalTag, FTI_Exec.globalComm);
            free(headInfo);
        }

    }
    else { //If post-processing is inline
        FTI_Exec.wasLastOffline = 0;
        if (res != FTI_SCES) { //If Writing checkpoint failed
            FTI_Exec.ckptLvel = FTI_REJW - FTI_BASE; //The same as head call FTI_PostCkpt with reject ckptLvel if not success
        }
        res = FTI_Try(FTI_PostCkpt(&FTI_Conf, &FTI_Exec, &FTI_Topo, FTI_Ckpt), "postprocess the checkpoint.");
        if (res == FTI_SCES) { //If post-processing succeed
            FTI_Exec.lastCkptLvel = FTI_Exec.ckptLvel; //Store last successful post-processing checkpoint level
        }
    }
    double t3 = MPI_Wtime(); //Time after post-processing
    if (res != FTI_SCES) {
        sprintf(str, "Checkpoint with ID %d at Level %d failed.", FTI_Exec.ckptID, FTI_Exec.ckptLvel);
        FTI_Print(str, FTI_WARN);
        return FTI_NSCS;
    }

    sprintf(str, "Ckpt. ID %d (L%d) (%.2f MB/proc) taken in %.2f sec. (Wt:%.2fs, Wr:%.2fs, Ps:%.2fs)",
            FTI_Exec.ckptID, FTI_Exec.ckptLvel, FTI_Exec.ckptSize / (1024.0 * 1024.0), t3 - t0, t1 - t0, t2 - t1, t3 - t2);
    FTI_Print(str, FTI_INFO);
    
    if ( FTI_Conf.dcpEnabled && FTI_Ckpt[4].isDcp ) {
        
        long norder_data, norder_dcp;
        char corder_data[3], corder_dcp[3];
        long DCP_TB = (1024L*1024L*1024L*1024L);
        long DCP_GB = (1024L*1024L*1024L);
        long DCP_MB = (1024L*1024L);
        if ( FTI_Exec.FTIFFMeta.dataSize > DCP_TB ) {
            norder_data = DCP_TB;
            snprintf( corder_data, 3, "TB" );
        } else if ( FTI_Exec.FTIFFMeta.dataSize > DCP_GB ) {
            norder_data = DCP_GB;
            snprintf( corder_data, 3, "GB" );
        } else {
            norder_data = DCP_MB;
            snprintf( corder_data, 3, "MB" );
        }
        if ( FTI_Exec.FTIFFMeta.dcpSize > DCP_TB ) {
            norder_dcp = DCP_TB;
            snprintf( corder_dcp, 3, "TB" );
        } else if ( FTI_Exec.FTIFFMeta.dcpSize > DCP_GB ) {
            norder_dcp = DCP_GB;
            snprintf( corder_dcp, 3, "GB" );
        } else {
            norder_dcp = DCP_MB;
            snprintf( corder_dcp, 3, "MB" );
        }

        if ( FTI_Topo.splitRank != 0 ) {
            snprintf( str, FTI_BUFS, "Local CP data: %.2lf %s, Local dCP update: %.2lf %s, dCP share: %.2lf%%",
                    (double)FTI_Exec.FTIFFMeta.dataSize/norder_data, corder_data,
                    (double)FTI_Exec.FTIFFMeta.dcpSize/norder_dcp, corder_dcp,
                    ((double)FTI_Exec.FTIFFMeta.dcpSize/FTI_Exec.FTIFFMeta.dataSize)*100 );
            FTI_Print( str, FTI_DBUG );
        } else {
            snprintf( str, FTI_BUFS, "Total CP data: %.2lf %s, Total dCP update: %.2lf %s, dCP share: %.2lf%%",
                    (double)FTI_Exec.FTIFFMeta.dataSize/norder_data, corder_data,
                    (double)FTI_Exec.FTIFFMeta.dcpSize/norder_dcp, corder_dcp,
                    ((double)FTI_Exec.FTIFFMeta.dcpSize/FTI_Exec.FTIFFMeta.dataSize)*100 );
        }
        
        FTI_Print(str, FTI_IDCP);
    }
    if (ckptFirst && FTI_Topo.splitRank == 0) {
        //Setting recover flag to 1 (to recover from current ckpt level)
        FTI_Try(FTI_UpdateConf(&FTI_Conf, &FTI_Exec, 1), "update configuration file.");
        FTI_Exec.initSCES = 1; //in case FTI couldn't recover all ckpt files in FTI_Init
    }
    
    if ( FTI_Conf.dcpEnabled && FTI_Ckpt[4].isDcp ) {
        FTI_Ckpt[4].isDcp = false;
    }
 
    return FTI_DONE;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Initialize an incremental checkpoint.
  @param      id              Checkpoint ID.
  @param      level           Checkpoint level.
  @param      activate        Boolean expression.
  @return     integer         FTI_SCES if successful.

  This function defines the environment for the incremental checkpointing
  mechanism. The iCP mechanism consists of three functions: FTI_InitICP,
  FTI_AddVarICP and FTI_FinalizeICP. The two functions FTI_InitICP and
  FTI_FinalizeICP define the iCP region within the user may write the
  protected variables in any order. The iCP region is active, when the
  expression passed through 'activate' evaluates to TRUE.

  @note This function is not blocking for POSIX, FTI-FF and HDF5, but,
  blocking for MPI-IO. This is due to the collective open call in MPI_IO
 **/
/*-------------------------------------------------------------------------*/
int FTI_InitICP(int id, int level, bool activate)
{
    if (FTI_Exec.initSCES == 0) {
        FTI_Print("FTI is not initialized.", FTI_WARN);
        return FTI_NSCS;
    }
     
    // only step in if activate TRUE.
    if ( !activate ) {
        return FTI_SCES;
    }
   
    // reset iCP meta info (i.e. set counter to zero etc.)
    memset( &(FTI_Exec.iCPInfo), 0x0, sizeof(FTIT_iCPInfo) );

    // init iCP status with failure
    FTI_Exec.iCPInfo.status = FTI_ICP_FAIL;
    FTI_Exec.iCPInfo.result = FTI_NSCS;

    int res = FTI_NSCS;
    
    char str[FTI_BUFS]; //For console output

    if (FTI_Exec.initSCES == 0) {
        FTI_Print("FTI is not initialized.", FTI_WARN);
        return FTI_NSCS;
    }

    if ((level < FTI_MIN_LEVEL_ID) || (level > FTI_MAX_LEVEL_ID)) {
        FTI_Print("Invalid level id! Aborting checkpoint creation...", FTI_WARN);
        return FTI_NSCS;
    }
    if ((level > FTI_L4) && (level < FTI_L4_DCP)) {
        snprintf( str, FTI_BUFS, "dCP only implemented for level 4! setting to level %d...", level - 4 );
        FTI_Print(str, FTI_WARN);
        level -= 4; 
    }

    FTI_Exec.iCPInfo.lastCkptID = FTI_Exec.ckptID;
    FTI_Exec.iCPInfo.isFirstCp = !FTI_Exec.ckptID; //ckptID = 0 if first checkpoint
    FTI_Exec.ckptID = id;

    // reset dcp requests.
    FTI_Ckpt[4].isDcp = false;
    if ( level == FTI_L4_DCP ) {
        if ( FTI_Conf.ioMode == FTI_IO_FTIFF ) {
            if ( FTI_Conf.dcpEnabled ) {
                FTI_Ckpt[4].isDcp = true;
            } else {
                FTI_Print("L4 dCP requested, but dCP is disabled!", FTI_WARN);
            }
        } else {
            FTI_Print("L4 dCP requested, but dCP needs FTI-FF!", FTI_WARN);
        }
        level = 4;
    }

    FTI_Exec.iCPInfo.t0 = MPI_Wtime(); //Start time
    if (FTI_Exec.wasLastOffline == 1) { // Block until previous checkpoint is done (Async. work)
        int lastLevel;
        MPI_Recv(&lastLevel, 1, MPI_INT, FTI_Topo.headRank, FTI_Conf.generalTag, FTI_Exec.globalComm, MPI_STATUS_IGNORE);
        if (lastLevel != FTI_NSCS) { //Head sends level of checkpoint if post-processing succeed, FTI_NSCS Otherwise
            FTI_Exec.lastCkptLvel = lastLevel; //Store last successful post-processing checkpoint level
            sprintf(str, "LastCkptLvel received from head: %d", lastLevel);
            FTI_Print(str, FTI_DBUG);
        } else {
            FTI_Print("Head failed to do post-processing after previous checkpoint.", FTI_WARN);
        }
    }

    FTI_Exec.iCPInfo.t1 = MPI_Wtime(); //Time after waiting for head to done previous post-processing
    FTI_Exec.iCPInfo.lastCkptLvel = FTI_Exec.ckptLvel; //Store last successful writing checkpoint level in case of failure
    FTI_Exec.ckptLvel = level; //For FTI_WriteCkpt

    //If checkpoint is inlin and level 4 save directly to PFS
    if (FTI_Ckpt[4].isInline && FTI_Exec.ckptLvel == 4) {

        if ( !(FTI_Conf.dcpEnabled && FTI_Ckpt[4].isDcp) ) {
            FTI_Print("Saving to temporary global directory", FTI_DBUG);

            //Create global temp directory
            if (mkdir(FTI_Conf.gTmpDir, 0777) == -1) {
                if (errno != EEXIST) {
                    FTI_Print("Cannot create global directory", FTI_EROR);
                    return FTI_NSCS;
                }
            }
        } else {
            if ( !FTI_Ckpt[4].hasDcp ) {
                if (mkdir(FTI_Ckpt[4].dcpDir, 0777) == -1) {
                    if (errno != EEXIST) {
                        FTI_Print("Cannot create global dCP directory", FTI_EROR);
                        return FTI_NSCS;
                    }
                }
            }
        }

        switch (FTI_Conf.ioMode) {
#ifdef ENABLE_SIONLIB 
            case FTI_IO_SIONLIB:
                FTI_Print("iCP is currently not supported using SIONlib I/O, falling back to POSIX", FTI_WARN);
#if 0
                res = FTI_Try(FTI_InitSionlibICP(&FTI_Conf, &FTI_Exec, &FTI_Topo, FTI_Ckpt, FTI_Data), "Initialize iCP (Sionlib).");
                break;
#endif // 0
#endif // If SIONlib is installed
            case FTI_IO_POSIX:
                res = FTI_Try(FTI_InitPosixICP(&FTI_Conf, &FTI_Exec, &FTI_Topo, FTI_Ckpt, FTI_Data), "Initialize iCP (POSIX I/O).");
                break;
            case FTI_IO_MPI:
                res = FTI_Try(FTI_InitMpiICP(&FTI_Conf, &FTI_Exec, &FTI_Topo, FTI_Ckpt, FTI_Data), "Initialize iCP (MPI-IO).");
                break;
            case FTI_IO_FTIFF:
		copyDataFromDevive( &FTI_Exec, FTI_Data );
                res = FTI_Try(FTI_InitFtiffICP(&FTI_Conf, &FTI_Exec, &FTI_Topo, FTI_Ckpt, FTI_Data), "Initialize iCP (FTI-FF).");
                break;
#ifdef ENABLE_HDF5 //If HDF5 is installed
            case FTI_IO_HDF5:
		copyDataFromDevive( &FTI_Exec, FTI_Data );
                res = FTI_Try(FTI_InitHdf5ICP(&FTI_Conf, &FTI_Exec, &FTI_Topo, FTI_Ckpt, FTI_Data), "Initialize iCP (HDF5).");
                break;
#endif
        }
    }
    else {
        if ( !(FTI_Conf.dcpEnabled && FTI_Ckpt[4].isDcp) ) {
            FTI_Print("Saving to temporary local directory", FTI_DBUG);
            //Create local temp directory
            if (mkdir(FTI_Conf.lTmpDir, 0777) == -1) {
                if (errno != EEXIST) {
                    FTI_Print("Cannot create local directory", FTI_EROR);
                }
            }
        } else {
            if ( !FTI_Ckpt[4].hasDcp ) {
                if (mkdir(FTI_Ckpt[1].dcpDir, 0777) == -1) {
                    if (errno != EEXIST) {
                        FTI_Print("Cannot create global dCP directory", FTI_EROR);
                        return FTI_NSCS;
                    }
                }
            }
        }
        switch (FTI_Conf.ioMode) {
            case FTI_IO_FTIFF:
		copyDataFromDevive( &FTI_Exec, FTI_Data );
                res = FTI_Try(FTI_InitFtiffICP(&FTI_Conf, &FTI_Exec, &FTI_Topo, FTI_Ckpt, FTI_Data), "Initialize iCP (FTI-FF).");
                break;
#ifdef ENABLE_HDF5 //If HDF5 is installed
      case FTI_IO_HDF5:
        copyDataFromDevive( &FTI_Exec, FTI_Data );
        res = FTI_Try(FTI_InitHdf5ICP(&FTI_Conf, &FTI_Exec, &FTI_Topo, FTI_Ckpt, FTI_Data), "Initialize iCP (HDF5).");
        break;
#endif
            default:
                res = FTI_Try(FTI_InitPosixICP(&FTI_Conf, &FTI_Exec, &FTI_Topo, FTI_Ckpt, FTI_Data), "Initialize iCP (POSIX I/O).");
                break;
        }

    }

    if ( res == FTI_SCES ) {
        FTI_Exec.iCPInfo.status = FTI_ICP_ACTV;
    }

    return res;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Write variable into the CP file.
  @param      id              Protected variable ID.
  @return     integer         FTI_SCES if successful.
    
  With this function, the user may write the protected datasets in any
  order into the checkpoint file. However, before the call to
  FTI_FinalizeICP, all protected variables must have been written into
  the file.
 **/
/*-------------------------------------------------------------------------*/
int FTI_AddVarICP( int varID ) 
{
    if (FTI_Exec.initSCES == 0) {
        FTI_Print("FTI is not initialized.", FTI_WARN);
        return FTI_NSCS;
    }

    // only step in if iCP was successfully initialized
    if ( FTI_Exec.iCPInfo.status == FTI_ICP_NINI ) {
        return FTI_SCES;
    }
    if ( FTI_Exec.iCPInfo.status == FTI_ICP_FAIL ) {
        return FTI_NSCS;
    }

    char str[FTI_BUFS];

    bool validID = false;
    
    int i;
    // check if dataset with 'varID' exists.
    for(i=0; i<FTI_Exec.nbVar; ++i) {
        validID |= (FTI_Data[i].id == varID);
    }
    if( !validID ) {
        snprintf( str, FTI_BUFS, "FTI_AddVarICP: dataset ID: %d is invalid!", varID );
        FTI_Print(str, FTI_WARN);
        return FTI_NSCS;
    }
    
    // check if dataset was not already written.
    for(i=0; i<FTI_Exec.iCPInfo.countVar; ++i) {
        validID &= !(FTI_Exec.iCPInfo.isWritten[i] == varID);
    }
    if( !validID ) {
        snprintf( str, FTI_BUFS, "Dataset with ID: %d was already successfully written!", varID );
        FTI_Print(str, FTI_WARN);
        return FTI_NSCS;
    }
    
    int res;

    switch (FTI_Conf.ioMode) {
#ifdef ENABLE_SIONLIB //If SIONlib is installed
        case FTI_IO_SIONLIB:
#endif
        case FTI_IO_POSIX:
            res = FTI_Try(FTI_WritePosixVar(varID, &FTI_Conf, &FTI_Exec, &FTI_Topo, FTI_Ckpt, FTI_Data), "write dataset to ckpt file.");
            break;
        case FTI_IO_MPI:
            if (FTI_Ckpt[4].isInline && FTI_Exec.ckptLvel == 4) {
                res = FTI_Try(FTI_WriteMpiVar(varID, &FTI_Conf, &FTI_Exec, &FTI_Topo, FTI_Ckpt, FTI_Data), "Write dataset (iCP) (MPI-IO).");
            } else {
                res = FTI_Try(FTI_WritePosixVar(varID, &FTI_Conf, &FTI_Exec, &FTI_Topo, FTI_Ckpt, FTI_Data), "write dataset to ckpt file.");
            }
            break;
#if 0
#ifdef ENABLE_SIONLIB //If SIONlib is installed
        case FTI_IO_SIONLIB:
            res = FTI_WriteSionlibVar(varID, &FTI_Conf, &FTI_Exec, &FTI_Topo, FTI_Ckpt, FTI_Data);
            break;
#endif
#endif
        case FTI_IO_FTIFF:
            res = FTI_Try(FTI_WriteFtiffVar(varID, &FTI_Conf, &FTI_Exec, &FTI_Topo, FTI_Ckpt, FTI_Data), "Initialize iCP (FTI-FF).");
            break;
#ifdef ENABLE_HDF5 //If HDF5 is installed
        case FTI_IO_HDF5:
            res = FTI_Try(FTI_WriteHdf5Var(varID, &FTI_Conf, &FTI_Exec, &FTI_Topo, FTI_Ckpt, FTI_Data), "Initialize iCP (HDF5).");
            break;
#endif
    }

    if ( res == FTI_SCES ) {
        FTI_Exec.iCPInfo.isWritten[FTI_Exec.iCPInfo.countVar++] = varID;
    }

    return res;

}

/*-------------------------------------------------------------------------*/
/**
  @brief      Finalize an incremental checkpoint.
  @return     integer         FTI_SCES if successful.
    
  This function finalizes an incremental checkpoint. In contrast to
  InitICP, this function is collective on the communicator
  FTI_COMM_WORLD and blocking.
 **/
/*-------------------------------------------------------------------------*/
int FTI_FinalizeICP() 
{
    if (FTI_Exec.initSCES == 0) {
        FTI_Print("FTI is not initialized.", FTI_WARN);
        return FTI_NSCS;
    }

    // if iCP uninitialized, don't step in.
    if ( FTI_Exec.iCPInfo.status == FTI_ICP_NINI ) {
        return FTI_SCES;
    }

    int allRes[2];
    int locRes[2] = { (int)(FTI_Exec.iCPInfo.result==FTI_SCES), (int)(FTI_Exec.iCPInfo.countVar==FTI_Exec.nbVar) };
    //Check if all processes have written all the datasets failure free.
    MPI_Allreduce(locRes, allRes, 2, MPI_INT, MPI_SUM, FTI_COMM_WORLD);
    if (allRes[0] != FTI_Topo.nbNodes*FTI_Topo.nbApprocs) {
        FTI_Exec.iCPInfo.status = FTI_ICP_FAIL;
        FTI_Print("Not all variables were successfully written!.", FTI_EROR);
    }
    if (allRes[1] != FTI_Topo.nbNodes*FTI_Topo.nbApprocs) {
        FTI_Exec.iCPInfo.status = FTI_ICP_FAIL;
        FTI_Print("Not all datasets were added to the CP file!.", FTI_EROR);
    }

    char str[FTI_BUFS];
    int resCP;
    int resPP;
    
    // Close files for each I/O
    switch (FTI_Conf.ioMode) {
#ifdef ENABLE_SIONLIB //If SIONlib is installed
        case FTI_IO_SIONLIB:
#endif
        case FTI_IO_POSIX:
            resCP = FTI_FinalizePosixICP(&FTI_Conf, &FTI_Exec, &FTI_Topo, FTI_Ckpt, FTI_Data);
            break;
        case FTI_IO_MPI:
            if (FTI_Ckpt[4].isInline && FTI_Exec.ckptLvel == 4) {
                resCP = FTI_Try(FTI_FinalizeMpiICP(&FTI_Conf, &FTI_Exec, &FTI_Topo, FTI_Ckpt, FTI_Data), "Finalize iCP (MPI-IO).");
            } else {
                resCP = FTI_FinalizePosixICP(&FTI_Conf, &FTI_Exec, &FTI_Topo, FTI_Ckpt, FTI_Data);
            }
            break;
#if 0
#ifdef ENABLE_SIONLIB //If SIONlib is installed
        case FTI_IO_SIONLIB:
            resCP = FTI_Try(FTI_FinalizeSionlibICP(&FTI_Conf, &FTI_Exec, &FTI_Topo, FTI_Ckpt, FTI_Data), "Finalize iCP (Sionlib).");
            break;
#endif
#endif
        case FTI_IO_FTIFF:
            resCP = FTI_Try(FTI_FinalizeFtiffICP(&FTI_Conf, &FTI_Exec, &FTI_Topo, FTI_Ckpt, FTI_Data), "Initialize iCP (FTI-FF).");
            break;
#ifdef ENABLE_HDF5 //If HDF5 is installed
        case FTI_IO_HDF5:
            resCP = FTI_Try(FTI_FinalizeHdf5ICP(&FTI_Conf, &FTI_Exec, &FTI_Topo, FTI_Ckpt, FTI_Data), "Initialize iCP (HDF5).");
            break;
#endif
    }
    
    if( resCP == FTI_SCES ) {
        resCP = FTI_Try(FTI_CreateMetadata(&FTI_Conf, &FTI_Exec, &FTI_Topo, FTI_Ckpt, FTI_Data), "create metadata.");
    }

    if ( resCP != FTI_SCES ) {
        FTI_Exec.iCPInfo.status = FTI_ICP_FAIL;
        sprintf(str, "Checkpoint with ID %d at Level %d failed.", FTI_Exec.ckptID, FTI_Exec.ckptLvel);
        FTI_Print(str, FTI_WARN);
    }
    
    double t2 = MPI_Wtime(); //Time after writing checkpoint

    if ( FTI_Conf.dcpEnabled && FTI_Ckpt[4].isDcp ) {
        // After dCP update store total data and dCP sizes in application rank 0
        long dcpStats[2]; // 0:totalDcpSize, 1:totalDataSize
        long sendBuf[] = { FTI_Exec.FTIFFMeta.dcpSize, FTI_Exec.FTIFFMeta.dataSize };
        MPI_Reduce( sendBuf, dcpStats, 2, MPI_LONG, MPI_SUM, 0, FTI_COMM_WORLD );
        if ( FTI_Topo.splitRank ==  0 ) {
            FTI_Exec.FTIFFMeta.dcpSize = dcpStats[0]; 
            FTI_Exec.FTIFFMeta.dataSize = dcpStats[1];
        }
    }

    // TODO this has to come inside postckpt on success! 
    if ( (FTI_Conf.dcpEnabled || FTI_Conf.keepL4Ckpt) && (FTI_Topo.splitRank == 0) ) {
        FTI_WriteCkptMetaData( &FTI_Conf, &FTI_Exec, &FTI_Topo, FTI_Ckpt );
    }

    // TODO this also has to come inside postckpt upon successful call!
    // set hasCkpt flags true
    if ( FTI_Conf.dcpEnabled && FTI_Ckpt[4].isDcp ) {
        FTIFF_db* currentDB = FTI_Exec.firstdb;
        currentDB->update = false;
        do {    
            int varIdx;
            for(varIdx=0; varIdx<currentDB->numvars; ++varIdx) {
                FTIFF_dbvar* currentdbVar = &(currentDB->dbvars[varIdx]);
                currentdbVar->hasCkpt = true;
                currentdbVar->update = false;
            }
        }
        while ( (currentDB = currentDB->next) != NULL );    


        FTI_UpdateDcpChanges(FTI_Data, &FTI_Exec);
        FTI_Ckpt[4].hasDcp = true;
    }

    // FTIFF: send meta info to the heads
    FTIFF_headInfo *headInfo;
    if (!FTI_Ckpt[FTI_Exec.ckptLvel].isInline) { // If postCkpt. work is Async. then send message
        FTI_Exec.wasLastOffline = 1;
        // Head needs ckpt. ID to determine ckpt file name.
        int value = FTI_BASE + FTI_Exec.ckptLvel; //Token to send to head
        if (FTI_Exec.iCPInfo.status == FTI_ICP_FAIL) { //If Writing checkpoint failed
            FTI_Exec.ckptLvel = FTI_Exec.iCPInfo.lastCkptLvel; //Set previous ckptLvel
            value = FTI_REJW; //Send reject checkpoint token to head
        }
        MPI_Send(&value, 1, MPI_INT, FTI_Topo.headRank, FTI_Conf.ckptTag, FTI_Exec.globalComm);
        // FTIFF: send meta info to the heads
        if( FTI_Conf.ioMode == FTI_IO_FTIFF && value != FTI_REJW ) {
            headInfo = malloc(sizeof(FTIFF_headInfo));
            headInfo->exists = FTI_Exec.meta[0].exists[0];
            headInfo->nbVar = FTI_Exec.meta[0].nbVar[0];
            headInfo->maxFs = FTI_Exec.meta[0].maxFs[0];
            headInfo->fs = FTI_Exec.meta[0].fs[0];
            headInfo->pfs = FTI_Exec.meta[0].pfs[0];
            headInfo->isDcp = (FTI_Ckpt[4].isDcp) ? 1 : 0;
            if( FTI_Conf.dcpEnabled && FTI_Ckpt[4].isDcp ) {
                strncpy(headInfo->ckptFile, FTI_Ckpt[4].dcpName, FTI_BUFS);
            } else {
                strncpy(headInfo->ckptFile, FTI_Exec.meta[0].ckptFile, FTI_BUFS);
            }            
            MPI_Send(headInfo, 1, FTIFF_MpiTypes[FTIFF_HEAD_INFO], FTI_Topo.headRank, FTI_Conf.generalTag, FTI_Exec.globalComm);
            MPI_Send(FTI_Exec.meta[0].varID, headInfo->nbVar, MPI_INT, FTI_Topo.headRank, FTI_Conf.generalTag, FTI_Exec.globalComm);
            MPI_Send(FTI_Exec.meta[0].varSize, headInfo->nbVar, MPI_LONG, FTI_Topo.headRank, FTI_Conf.generalTag, FTI_Exec.globalComm);
            free(headInfo);
        }

    }
    else { //If post-processing is inline
        FTI_Exec.wasLastOffline = 0;
        if (FTI_Exec.iCPInfo.status == FTI_ICP_FAIL) { //If Writing checkpoint failed
            FTI_Exec.ckptLvel = FTI_REJW - FTI_BASE; //The same as head call FTI_PostCkpt with reject ckptLvel if not success
        }
        resPP = FTI_Try(FTI_PostCkpt(&FTI_Conf, &FTI_Exec, &FTI_Topo, FTI_Ckpt), "postprocess the checkpoint.");
        if (resPP == FTI_SCES) { //If post-processing succeed
            FTI_Exec.lastCkptLvel = FTI_Exec.ckptLvel; //Store last successful post-processing checkpoint level
        }
    }

    double t3 = MPI_Wtime(); //Time after post-processing

    if( resCP == FTI_SCES ) {
        sprintf(str, "Ckpt. ID %d (L%d) (%.2f MB/proc) taken in %.2f sec. (Wt:%.2fs, Wr:%.2fs, Ps:%.2fs)",
                FTI_Exec.ckptID, FTI_Exec.ckptLvel, FTI_Exec.ckptSize / (1024.0 * 1024.0), t3 - FTI_Exec.iCPInfo.t0, FTI_Exec.iCPInfo.t1 - FTI_Exec.iCPInfo.t0, t2 - FTI_Exec.iCPInfo.t1, t3 - t2);
        FTI_Print(str, FTI_INFO);

        if ( FTI_Conf.dcpEnabled && FTI_Ckpt[4].isDcp ) {

            long norder_data, norder_dcp;
            char corder_data[3], corder_dcp[3];
            long DCP_TB = (1024L*1024L*1024L*1024L);
            long DCP_GB = (1024L*1024L*1024L);
            long DCP_MB = (1024L*1024L);
            if ( FTI_Exec.FTIFFMeta.dataSize > DCP_TB ) {
                norder_data = DCP_TB;
                snprintf( corder_data, 3, "TB" );
            } else if ( FTI_Exec.FTIFFMeta.dataSize > DCP_GB ) {
                norder_data = DCP_GB;
                snprintf( corder_data, 3, "GB" );
            } else {
                norder_data = DCP_MB;
                snprintf( corder_data, 3, "MB" );
            }
            if ( FTI_Exec.FTIFFMeta.dcpSize > DCP_TB ) {
                norder_dcp = DCP_TB;
                snprintf( corder_dcp, 3, "TB" );
            } else if ( FTI_Exec.FTIFFMeta.dcpSize > DCP_GB ) {
                norder_dcp = DCP_GB;
                snprintf( corder_dcp, 3, "GB" );
            } else {
                norder_dcp = DCP_MB;
                snprintf( corder_dcp, 3, "MB" );
            }

            if ( FTI_Topo.splitRank != 0 ) {
                snprintf( str, FTI_BUFS, "Local CP data: %.2lf %s, Local dCP update: %.2lf %s, dCP share: %.2lf%%",
                        (double)FTI_Exec.FTIFFMeta.dataSize/norder_data, corder_data,
                        (double)FTI_Exec.FTIFFMeta.dcpSize/norder_dcp, corder_dcp,
                        ((double)FTI_Exec.FTIFFMeta.dcpSize/FTI_Exec.FTIFFMeta.dataSize)*100 );
                FTI_Print( str, FTI_DBUG );
            } else {
                snprintf( str, FTI_BUFS, "Total CP data: %.2lf %s, Total dCP update: %.2lf %s, dCP share: %.2lf%%",
                        (double)FTI_Exec.FTIFFMeta.dataSize/norder_data, corder_data,
                        (double)FTI_Exec.FTIFFMeta.dcpSize/norder_dcp, corder_dcp,
                        ((double)FTI_Exec.FTIFFMeta.dcpSize/FTI_Exec.FTIFFMeta.dataSize)*100 );
            }

            FTI_Print(str, FTI_IDCP);
        }
        if (FTI_Exec.iCPInfo.isFirstCp && FTI_Topo.splitRank == 0) {
            //Setting recover flag to 1 (to recover from current ckpt level)
            FTI_Try(FTI_UpdateConf(&FTI_Conf, &FTI_Exec, 1), "update configuration file.");
            FTI_Exec.initSCES = 1; //in case FTI couldn't recover all ckpt files in FTI_Init
        }
    } else {
        FTI_Exec.ckptLvel = FTI_Exec.iCPInfo.lastCkptLvel;
        FTI_Exec.ckptID = FTI_Exec.iCPInfo.lastCkptID;
    }
    if ( FTI_Conf.dcpEnabled && FTI_Ckpt[4].isDcp ) {
        FTI_Ckpt[4].isDcp = false;
    }

    FTI_Exec.iCPInfo.status = FTI_ICP_NINI;

    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      It loads the checkpoint data.
  @return     integer         FTI_SCES if successful.

  This function loads the checkpoint data from the checkpoint file and
  it updates some basic checkpoint information.

 **/
/*-------------------------------------------------------------------------*/

static int copyDataToDevice(){
  int i;
  char str[FTI_BUFS]; //For console output
  for (i = 0; i < FTI_Exec.nbVar; i++) {
    if ( FTI_Data[i].isDevicePtr ){
      FTI_copy_to_device( FTI_Data[i].devicePtr, FTI_Data[i].ptr,FTI_Data[i].size, &FTI_Exec);
    }
  }
  return 1;
}

int FTI_Recover()
{
  if ( FTI_Conf.ioMode == FTI_IO_FTIFF ) {
    int ret = FTI_Try(FTIFF_Recover( &FTI_Exec, FTI_Data, FTI_Ckpt ), "Recovering from Checkpoint");
    copyDataToDevice();
    return ret;
  }

    if (FTI_Exec.initSCES == 0) {
        FTI_Print("FTI is not initialized.", FTI_WARN);
        return FTI_NSCS;
    }
    if (FTI_Exec.initSCES == 2) {
        FTI_Print("No checkpoint files to make recovery.", FTI_WARN);
        return FTI_NSCS;
    }

    char fn[FTI_BUFS]; //Path to the checkpoint file
    char str[FTI_BUFS]; //For console output

    //Check if nubmer of protected variables matches
    if (FTI_Exec.nbVar != FTI_Exec.meta[FTI_Exec.ckptLvel].nbVar[0]) {
        sprintf(str, "Checkpoint has %d protected variables, but FTI protects %d.",
                FTI_Exec.meta[FTI_Exec.ckptLvel].nbVar[0], FTI_Exec.nbVar);
        FTI_Print(str, FTI_WARN);
        return FTI_NREC;
    }
    //Check if sizes of protected variables matches
    int i;
    for (i = 0; i < FTI_Exec.nbVar; i++) {
        if (FTI_Data[i].size != FTI_Exec.meta[FTI_Exec.ckptLvel].varSize[i]) {
            sprintf(str, "Cannot recover %ld bytes to protected variable (ID %d) size: %ld",
                    FTI_Exec.meta[FTI_Exec.ckptLvel].varSize[i], FTI_Exec.meta[FTI_Exec.ckptLvel].varID[i],
                    FTI_Data[i].size);
            FTI_Print(str, FTI_WARN);
            return FTI_NREC;
        }
    }

#ifdef ENABLE_HDF5 //If HDF5 is installed
  if (FTI_Conf.ioMode == FTI_IO_HDF5) {
    int ret = FTI_RecoverHDF5(&FTI_Exec, FTI_Ckpt, FTI_Data);
    copyDataToDevice();
    return ret; 
  }
#endif

    //Recovering from local for L4 case in FTI_Recover
    if (FTI_Exec.ckptLvel == 4) {
        //Try from L1
        snprintf(fn, FTI_BUFS, "%s/%s", FTI_Ckpt[1].dir, FTI_Exec.meta[1].ckptFile);
        if (access(fn, F_OK) != 0) {
            //if no L1 files try from L4
            snprintf(fn, FTI_BUFS, "%s/%s", FTI_Ckpt[4].dir, FTI_Exec.meta[4].ckptFile);
        }
    }
    else {
        snprintf(fn, FTI_BUFS, "%s/%s", FTI_Ckpt[FTI_Exec.ckptLvel].dir, FTI_Exec.meta[FTI_Exec.ckptLvel].ckptFile);
    }

    sprintf(str, "Trying to load FTI checkpoint file (%s)...", fn);
    FTI_Print(str, FTI_DBUG);

    FILE* fd = fopen(fn, "rb");
    if (fd == NULL) {
        sprintf(str, "Could not open FTI checkpoint file. (%s)...", fn);
        FTI_Print(str, FTI_EROR);
        return FTI_NREC;
    }

#ifdef GPUSUPPORT
  for (i = 0; i < FTI_Exec.nbVar; i++) {
    if (FTI_Data[i].isDevicePtr){
      FTI_Data[i].ptr = (void *) malloc(FTI_Data[i].size);
      if (!(FTI_Data[i].ptr)){
        FTI_Print("RECOVER:: Could not Allocate memory on host",FTI_EROR );
        return FTI_NREC;
      }
    }


    fread(FTI_Data[i].ptr, 1, FTI_Data[i].size, fd);
    if (ferror(fd)) {
      FTI_Print("Could not read FTI checkpoint file.", FTI_EROR);
      fclose(fd);
      return FTI_NREC;
    }

    if (FTI_Data[i].isDevicePtr){
      int res;
      res = FTI_Try(FTI_copy_to_device(FTI_Data[i].devicePtr, FTI_Data[i].ptr, FTI_Data[i].size, &FTI_Exec), "copying data to GPU");
      if (res == FTI_NSCS) {
        return FTI_NSCS;
      }
      free(FTI_Data[i].ptr);
      FTI_Data[i].ptr = NULL;
    }
  }      
#else
  for (i = 0; i < FTI_Exec.nbVar; i++) {
    fread(FTI_Data[i].ptr, 1, FTI_Data[i].size, fd);
    if (ferror(fd)) {
      FTI_Print("Could not read FTI checkpoint file.", FTI_EROR);
      fclose(fd);
      return FTI_NREC;
    }
  }      
#endif    
  if (fclose(fd) != 0) {
    FTI_Print("Could not close FTI checkpoint file.", FTI_EROR);
    return FTI_NREC;
  }

  FTI_Exec.reco = 0;

  return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Takes an FTI snapshot or recovers the data if it is a restart.
  @return     integer         FTI_SCES if successful.

  This function loads the checkpoint data from the checkpoint file in case
  of restart. Otherwise, it checks if the current iteration requires
  checkpointing, if it does it checks which checkpoint level, write the
  data in the files and it communicates with the head of the node to inform
  that a checkpoint has been taken. Checkpoint ID and counters are updated.

 **/
/*-------------------------------------------------------------------------*/
int FTI_Snapshot()
{
    if (FTI_Exec.initSCES == 0) {
        FTI_Print("FTI is not initialized.", FTI_WARN);
        return FTI_NSCS;
    }

    int i, res, level = -1;

    if (FTI_Exec.reco) { // If this is a recovery load icheckpoint data
        res = FTI_Try(FTI_Recover(), "recover the checkpointed data.");
        if (res == FTI_NREC) {
            return FTI_NREC;
        }
    }
    else { // If it is a checkpoint test
        res = FTI_SCES;
        FTI_UpdateIterTime(&FTI_Exec);
        if (FTI_Exec.ckptNext == FTI_Exec.ckptIcnt) { // If it is time to check for possible ckpt. (every minute)
            FTI_Print("Checking if it is time to checkpoint.", FTI_DBUG);
            if (FTI_Exec.globMeanIter > 60) {
                FTI_Exec.minuteCnt = FTI_Exec.totalIterTime/60;
            }
            else {
                FTI_Exec.minuteCnt++; // Increment minute counter
            }
            for (i = 1; i < 5; i++) { // Check ckpt. level
                if ( (FTI_Ckpt[i].ckptDcpIntv > 0) 
                        && (FTI_Exec.minuteCnt/(FTI_Ckpt[i].ckptDcpCnt*FTI_Ckpt[i].ckptDcpIntv)) ) {
                    // dCP level is level + 4
                    level = i + 4;
                    // counts the passed intervall times (if taken or not...)
                    FTI_Ckpt[i].ckptDcpCnt++;
                }
                if ( (FTI_Ckpt[i].ckptIntv) > 0 
                        && (FTI_Exec.minuteCnt/(FTI_Ckpt[i].ckptCnt*FTI_Ckpt[i].ckptIntv)) ) {
                    level = i;
                    // counts the passed intervall times (if taken or not...)
                    FTI_Ckpt[i].ckptCnt++;
                }
            }
            if (level != -1) {
                res = FTI_Try(FTI_Checkpoint(FTI_Exec.ckptCnt, level), "take checkpoint.");
                if (res == FTI_DONE) {
                    FTI_Exec.ckptCnt++;
                }
            }
            FTI_Exec.ckptLast = FTI_Exec.ckptNext;
            FTI_Exec.ckptNext = FTI_Exec.ckptNext + FTI_Exec.ckptIntv;
            FTI_Exec.iterTime = MPI_Wtime(); // Reset iteration duration timer
        }
    }
    return res;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      It closes FTI properly on the application processes.
  @return     integer         FTI_SCES if successful.

  This function notifies the FTI processes that the execution is over, frees
  some data structures and it closes. If this function is not called on the
  application processes the FTI processes will never finish (deadlock).

 **/
/*-------------------------------------------------------------------------*/
int FTI_Finalize()
{
    if (FTI_Exec.initSCES == 0) {
        FTI_Print("FTI is not initialized.", FTI_WARN);
        return FTI_NSCS;
    }

    if (FTI_Topo.amIaHead) {
        FTI_FreeMeta(&FTI_Exec);
        if ( FTI_Conf.stagingEnabled ) {
            FTI_FinalizeStage( &FTI_Exec, &FTI_Topo, &FTI_Conf );
        }
        MPI_Barrier(FTI_Exec.globalComm);
        if ( !FTI_Conf.keepHeadsAlive ) { 
            MPI_Finalize();
            exit(0);
        } else {
            return FTI_SCES;
        }
    }

    // Notice: The following code is only executed by the application procs
#ifdef GPUSUPPORT
  cudaError_t err;
  char err_str[FTI_BUFS];

  if ((err = cudaStreamSynchronize(FTI_Exec.cStream)) != cudaSuccess) {
    sprintf(err_str, "Cannot synchronize the internal cStream: %s %s\n", cudaGetErrorName(err), cudaGetErrorString(err));
    FTI_Print(err_str, FTI_DBUG);
  }
  if ((err = cudaEventDestroy(FTI_Exec.cEvents[0])) != cudaSuccess) {
    sprintf(err_str, "Cannot destroy cEvents[0]: %s %s\n", cudaGetErrorName(err), cudaGetErrorString(err));
    FTI_Print(err_str, FTI_DBUG);
  }
  if ((err = cudaEventDestroy(FTI_Exec.cEvents[1])) != cudaSuccess) {
    sprintf(err_str, "Cannot destroy cEvents[1]: %s %s\n", cudaGetErrorName(err), cudaGetErrorString(err));
    FTI_Print(err_str, FTI_DBUG);
  }
  if ((err = cudaStreamDestroy(FTI_Exec.cStream)) != cudaSuccess) {
    sprintf(err_str, "Cannot destroy cStream: %s %s\n", cudaGetErrorName(err), cudaGetErrorString(err));
    FTI_Print(err_str, FTI_DBUG);
  }

  if ((err = cudaFreeHost(FTI_Exec.cHostBufs[0])) != cudaSuccess) {
    sprintf(err_str, "Cannot free cHostBufs[0]: %s %s\n", cudaGetErrorName(err), cudaGetErrorString(err));
    FTI_Print(err_str, FTI_DBUG);
  }
  if ((err = cudaFreeHost(FTI_Exec.cHostBufs[1])) != cudaSuccess) {
    sprintf(err_str, "Cannot free cHostBufs[1]: %s %s\n", cudaGetErrorName(err), cudaGetErrorString(err));
    FTI_Print(err_str, FTI_DBUG);
  }
#endif

    // If there is remaining work to do for last checkpoint
    if (FTI_Exec.wasLastOffline == 1) {
        int lastLevel;
        MPI_Recv(&lastLevel, 1, MPI_INT, FTI_Topo.headRank, FTI_Conf.generalTag, FTI_Exec.globalComm, MPI_STATUS_IGNORE);
        if (lastLevel != FTI_NSCS) { //Head sends level of checkpoint if post-processing succeed, FTI_NSCS Otherwise
            FTI_Exec.lastCkptLvel = lastLevel;
        }
    }

    // Send notice to the head to stop listening
    if (FTI_Topo.nbHeads == 1) {
        int value = FTI_ENDW;
        MPI_Send(&value, 1, MPI_INT, FTI_Topo.headRank, FTI_Conf.finalTag, FTI_Exec.globalComm);
    }
    
    // for staging, we have to ensure, that the call to FTI_Clean 
    // comes after the heads have written all the staging files.
    // Thus FTI_FinalizeStage is blocking on global communicator.
    if ( FTI_Conf.stagingEnabled ) {
        FTI_FinalizeStage( &FTI_Exec, &FTI_Topo, &FTI_Conf );
    }

    // If we need to keep the last checkpoint and there was a checkpoint
    if ( FTI_Conf.saveLastCkpt && ( FTI_Exec.ckptID > 0 ) ) {
    //if ((FTI_Conf.saveLastCkpt || FTI_Conf.keepL4Ckpt) && FTI_Exec.ckptID > 0) {
        if (FTI_Exec.lastCkptLvel != 4) {
            FTI_Try(FTI_Flush(&FTI_Conf, &FTI_Exec, &FTI_Topo, FTI_Ckpt, FTI_Exec.lastCkptLvel), "save the last ckpt. in the PFS.");
            MPI_Barrier(FTI_COMM_WORLD);
            if (FTI_Topo.splitRank == 0) {
                if (access(FTI_Ckpt[4].dir, 0) == 0) {
                    FTI_RmDir(FTI_Ckpt[4].dir, 1); //Delete previous L4 checkpoint
                }
                if (rename(FTI_Conf.gTmpDir, FTI_Ckpt[4].dir) == -1) { //Move temporary checkpoint to L4 directory
                    FTI_Print("Cannot rename last ckpt. dir", FTI_EROR);
                }
                if ( FTI_Conf.ioMode != FTI_IO_FTIFF ) {
                    if (access(FTI_Ckpt[4].metaDir, 0) == 0) {
                        FTI_RmDir(FTI_Ckpt[4].metaDir, 1); //Delete previous L4 metadata
                    }
                    if (rename(FTI_Ckpt[FTI_Exec.ckptLvel].metaDir, FTI_Ckpt[4].metaDir) == -1) { //Move temporary metadata to L4 metadata directory
                        FTI_Print("Cannot rename last ckpt. metaDir", FTI_EROR);
                    }
                }
            }
        }
        if (FTI_Topo.splitRank == 0) {
            //Setting recover flag to 2 (to recover from L4, keeped last checkpoint)
            FTI_Try(FTI_UpdateConf(&FTI_Conf, &FTI_Exec, 2), "update configuration file to 2.");
        }
        //Cleaning only local storage
        FTI_Try(FTI_Clean(&FTI_Conf, &FTI_Topo, FTI_Ckpt, 6), "clean local directories");
    } else if ( FTI_Conf.keepL4Ckpt && FTI_Ckpt[4].hasCkpt ) {
        FTI_ArchiveL4Ckpt( &FTI_Conf, &FTI_Exec, FTI_Ckpt, &FTI_Topo );
        MPI_Barrier( FTI_COMM_WORLD );
        FTI_RmDir( FTI_Ckpt[4].dir, FTI_Topo.splitRank == 0 ); 
        MPI_Barrier( FTI_COMM_WORLD );
        //Cleaning only local storage
        FTI_Try(FTI_Clean(&FTI_Conf, &FTI_Topo, FTI_Ckpt, 6), "clean local directories");
    } else {
        if (FTI_Conf.saveLastCkpt) { //if there was no saved checkpoint
            FTI_Print("No checkpoint to keep in PFS.", FTI_INFO);
        }
        if (FTI_Topo.splitRank == 0) {
            //Setting recover flag to 0 (no checkpoint files to recover from means no recovery)
            FTI_Try(FTI_UpdateConf(&FTI_Conf, &FTI_Exec, 0), "update configuration file to 0.");
        }
        //Cleaning everything
        FTI_Try(FTI_Clean(&FTI_Conf, &FTI_Topo, FTI_Ckpt, 5), "do final clean.");
    }
    
    if (FTI_Conf.dcpEnabled) {
        FTI_FinalizeDcp( &FTI_Conf, &FTI_Exec );
    }

    FTI_FreeMeta(&FTI_Exec);
    FTI_FreeTypesAndGroups(&FTI_Exec);
    if( FTI_Conf.ioMode == FTI_IO_FTIFF ) {
        FTIFF_FreeDbFTIFF(FTI_Exec.lastdb);
    }
    MPI_Barrier(FTI_Exec.globalComm);
    FTI_Print("FTI has been finalized.", FTI_INFO);
    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      During the restart, recovers the given variable
  @param      id              Variable to recover
  @return     int             FTI_SCES if successful.

  During a restart process, this function recovers the variable specified
  by the given id. No effect during a regular execution.
  The variable must have already been protected, otherwise, FTI_NSCS is returned.
  Improvements to be done:
  - Open checkpoint file at FTI_Init, close it at FTI_Snapshot
  - Maintain a variable accumulating the offset as variable are protected during
  the restart to avoid doing the loop to calculate the offset in the
  checkpoint file.
 **/
/*-------------------------------------------------------------------------*/
int FTI_RecoverVar(int id)
{
    if (FTI_Conf.ioMode == FTI_IO_FTIFF) {
        return FTIFF_RecoverVar( id, &FTI_Exec, FTI_Data, FTI_Ckpt );
    }
    if (FTI_Exec.initSCES == 0) {
        FTI_Print("FTI is not initialized.", FTI_WARN);
        return FTI_NSCS;
    }

    if(FTI_Exec.reco==0){
        /* This is not a restart: no actions performed */
        return FTI_SCES;
    }

    if (FTI_Exec.initSCES == 2) {
        FTI_Print("No checkpoint files to make recovery.", FTI_WARN);
        return FTI_NSCS;
    }

    //Check if sizes of protected variables matches
    int i;
    for (i = 0; i < FTI_Exec.nbVar; i++) {
        if (id == FTI_Exec.meta[FTI_Exec.ckptLvel].varID[i]) {
            if (FTI_Data[i].size != FTI_Exec.meta[FTI_Exec.ckptLvel].varSize[i]) {
                char str[FTI_BUFS];
                sprintf(str, "Cannot recover %ld bytes to protected variable (ID %d) size: %ld",
                        FTI_Exec.meta[FTI_Exec.ckptLvel].varSize[i], FTI_Exec.meta[FTI_Exec.ckptLvel].varID[i],
                        FTI_Data[i].size);
                FTI_Print(str, FTI_WARN);
                return FTI_NREC;
            }
        }
    }

#ifdef ENABLE_HDF5 //If HDF5 is installed
    if (FTI_Conf.ioMode == FTI_IO_HDF5) {
        return FTI_RecoverVarHDF5(&FTI_Exec, FTI_Ckpt, FTI_Data, id);
    }
#endif

    char fn[FTI_BUFS]; //Path to the checkpoint file

    //Recovering from local for L4 case in FTI_Recover
    if (FTI_Exec.ckptLvel == 4) {
        snprintf(fn, FTI_BUFS, "%s/%s", FTI_Ckpt[1].dir, FTI_Exec.meta[1].ckptFile);
    }
    else {
        snprintf(fn, FTI_BUFS, "%s/%s", FTI_Ckpt[FTI_Exec.ckptLvel].dir, FTI_Exec.meta[FTI_Exec.ckptLvel].ckptFile);
    }

    char str[FTI_BUFS];

    sprintf(str, "Trying to load FTI checkpoint file (%s)...", fn);
    FTI_Print(str, FTI_DBUG);

    FILE* fd = fopen(fn, "rb");
    if (fd == NULL) {
        FTI_Print("Could not open FTI checkpoint file.", FTI_EROR);
        return FTI_NREC;
    }

    long offset = 0;
    for (i = 0; i < FTI_Exec.nbVar; i++) {
        if (id == FTI_Exec.meta[FTI_Exec.ckptLvel].varID[i]) {
            sprintf(str, "Recovering var %d ", id);
            FTI_Print(str, FTI_DBUG);
            fseek(fd, offset, SEEK_SET);
            fread(FTI_Data[i].ptr, 1, FTI_Data[i].size, fd);
            if (ferror(fd)) {
                FTI_Print("Could not read FTI checkpoint file.", FTI_EROR);
                fclose(fd);
                return FTI_NREC;
            }
            break;
        }
        offset += FTI_Exec.meta[FTI_Exec.ckptLvel].varSize[i];
    }

    if (i == FTI_Exec.nbVar) {
        FTI_Print("Variables must be protected before they can be recovered.", FTI_EROR);
        fclose(fd);
        return FTI_NREC;
    }
    if (fclose(fd) != 0) {
        FTI_Print("Could not close FTI checkpoint file.", FTI_EROR);
        return FTI_NREC;
    }

    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Prints FTI messages.
  @param      msg             Message to print.
  @param      priority        Priority of the message to be printed.
  @return     void

  This function prints messages depending on their priority and the
  verbosity level set by the user. DEBUG messages are printed by all
  processes with their rank. INFO messages are printed by one process.
  ERROR messages are printed with errno.

 **/
/*-------------------------------------------------------------------------*/
void FTI_Print(char* msg, int priority)
{
    if (priority >= FTI_Conf.verbosity) {
        if (msg != NULL) {
            switch (priority) {
                case FTI_EROR:
                    fprintf(stderr, "[ " RED "FTI Error - %06d" RESET " ] : %s : %s \n", FTI_Topo.myRank, msg, strerror(errno));
                    break;
                case FTI_WARN:
                    fprintf(stdout, "[ " ORG "FTI Warning %06d" RESET " ] : %s \n", FTI_Topo.myRank, msg);
                    break;
                case FTI_INFO:
                    if (FTI_Topo.splitRank == 0) {
                        fprintf(stdout, "[ " GRN "FTI  Information" RESET " ] : %s \n", msg);
                    }
                    break;
                case FTI_IDCP:
                    if (FTI_Topo.splitRank == 0) {
                        fprintf(stdout, "[ " BLU "FTI  dCP Message" RESET " ] : %s \n", msg);
                    }
                    break;
                case FTI_DBUG:
                    fprintf(stdout, "[FTI Debug - %06d] : %s \n", FTI_Topo.myRank, msg);
                    break;
            }
        }
    }
    fflush(stdout);
}
