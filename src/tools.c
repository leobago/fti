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
 *  @file   tools.c
 *  @date   October, 2017
 *  @brief  Utility functions for the FTI library.
 */

#include "interface.h"
#include <dirent.h>
#include "api_cuda.h"

int FTI_filemetastructsize;		        /**< size of FTIFF_db struct in file    */
int FTI_dbstructsize;		        /**< size of FTIFF_db struct in file    */
int FTI_dbvarstructsize;		        /**< size of FTIFF_db struct in file    */


/*-------------------------------------------------------------------------*/
/**
  @brief      Init of the static variables
  @return     integer         FTI_SCES if successful.

  This function initializes all static variables to zero.

 **/
/*-------------------------------------------------------------------------*/
int FTI_InitExecVars(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
    FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt,
    FTIT_injection* FTI_Inje) {

  // datablock size in file
  FTI_filemetastructsize
    = MD5_DIGEST_STRING_LENGTH
    + MD5_DIGEST_LENGTH
    + 7*sizeof(long);
  // TODO RS L3 only works for even file sizes. This accounts for many but clearly not all cases.
  // This is to fix.
  FTI_filemetastructsize += 2 - FTI_filemetastructsize%2;

  FTI_dbstructsize
    = sizeof(int)               /* numvars */
    + sizeof(long);             /* dbsize */

  FTI_dbvarstructsize
    = 3*sizeof(int)               /* numvars */
    + 2*sizeof(bool)
    + 2*sizeof(uintptr_t)
    + 2*sizeof(long)
    + MD5_DIGEST_LENGTH;

  // +--------- +
  // | FTI_Exec |
  // +--------- +

  /* char[BUFS]       FTI_Exec->id */                 memset(FTI_Exec->id,0x0,FTI_BUFS);
  /* int           */ FTI_Exec->ckpt                  =0;
  /* int           */ FTI_Exec->reco                  =0;
  /* int           */ FTI_Exec->ckptLvel              =0;
  /* int           */ FTI_Exec->ckptIntv              =0;
  /* int           */ FTI_Exec->lastCkptLvel          =0;
  /* int           */ FTI_Exec->wasLastOffline        =0;
  /* double        */ FTI_Exec->iterTime              =0;
  /* double        */ FTI_Exec->lastIterTime          =0;
  /* double        */ FTI_Exec->meanIterTime          =0;
  /* double        */ FTI_Exec->globMeanIter          =0;
  /* double        */ FTI_Exec->totalIterTime         =0;
  /* unsigned int  */ FTI_Exec->syncIter              =0;
  /* int           */ FTI_Exec->syncIterMax           =0;
  /* unsigned int  */ FTI_Exec->minuteCnt             =0;
  /* bool          */ FTI_Exec->hasCkpt               =false;
  /* unsigned int  */ FTI_Exec->ckptCnt               =0;
  /* unsigned int  */ FTI_Exec->ckptIcnt              =0;
  /* unsigned int  */ FTI_Exec->ckptID                =0;
  /* unsigned int  */ FTI_Exec->ckptNext              =0;
  /* unsigned int  */ FTI_Exec->ckptLast              =0;
  /* long          */ FTI_Exec->ckptSize              =0;
  /* unsigned int  */ FTI_Exec->nbVar                 =0;
  /* unsigned int  */ FTI_Exec->nbVarStored           =0;
  /* unsigned int  */ FTI_Exec->nbType                =0;
  /* int           */ FTI_Exec->metaAlloc             =0;
  /* int           */ FTI_Exec->initSCES              =0;
  /* char[BUFS]       FTI_Exec->h5SingleFileLast */   memset(FTI_Exec->h5SingleFileLast,0x0,FTI_BUFS);
  /* FTIT_iCPInfo     FTI_Exec->iCPInfo */            memset(&(FTI_Exec->iCPInfo),0x0,sizeof(FTIT_iCPInfo));
  /* FTIT_metadata[5] FTI_Exec->meta */               memset(FTI_Exec->meta,0x0,5*sizeof(FTIT_metadata));
  /* FTIFF_db      */ FTI_Exec->firstdb               =NULL;
  /* FTIFF_db      */ FTI_Exec->lastdb                =NULL;
  /* FTIT_globalDataset */ FTI_Exec->globalDatasets   =NULL;
  FTI_Exec->stageInfo             =NULL;
  /* FTIFF_metaInfo   FTI_Exec->FTIFFMeta */          memset(&(FTI_Exec->FTIFFMeta),0x0,sizeof(FTIFF_metaInfo));
  FTI_Exec->FTIFFMeta.metaSize                        = FTI_filemetastructsize;
  /* MPI_Comm      */ FTI_Exec->globalComm            =0;
  /* MPI_Comm      */ FTI_Exec->groupComm             =0;
  /* MPI_Comm      */ FTI_Exec->dcpInfoPosix.Counter  =0;
  /* MPI_Comm      */ FTI_Exec->dcpInfoPosix.FileSize =0;
                      memset(FTI_Exec->dcpInfoPosix.LayerSize, 0x0, MAX_STACK_SIZE*sizeof(unsigned long));
                      memset(FTI_Exec->dcpInfoPosix.LayerHash, 0x0, MAX_STACK_SIZE*MD5_DIGEST_STRING_LENGTH);

  // +--------- +
  // | FTI_Conf |
  // +--------- +

  /* char[BUFS]       FTI_Conf->cfgFile */            memset(FTI_Conf->cfgFile,0x0,FTI_BUFS);
  /* int           */ FTI_Conf->saveLastCkpt          =0;
  /* int           */ FTI_Conf->verbosity             =0;
  /* int           */ FTI_Conf->blockSize             =0;
  /* int           */ FTI_Conf->transferSize          =0;
#ifdef LUSTRE
  /* int           */ FTI_Conf->stripeUnit            =0;
  /* int           */ FTI_Conf->stripeOffset          =0;
  /* int           */ FTI_Conf->stripeFactor          =0;
#endif
  /* bool          */ FTI_Conf->keepL4Ckpt            =0;
  /* bool          */ FTI_Conf->h5SingleFileEnable    =0;
  /* int           */ FTI_Conf->ckptTag               =0;
  /* int           */ FTI_Conf->stageTag              =0;
  /* int           */ FTI_Conf->finalTag              =0;
  /* int           */ FTI_Conf->generalTag            =0;
  /* int           */ FTI_Conf->test                  =0;
  /* int           */ FTI_Conf->l3WordSize            =0;
  /* int           */ FTI_Conf->ioMode                =0;
  /* char[BUFS]       FTI_Conf->localDir */           memset(FTI_Conf->localDir,0x0,FTI_BUFS);
  /* char[BUFS]       FTI_Conf->glbalDir */           memset(FTI_Conf->glbalDir,0x0,FTI_BUFS);
  /* char[BUFS]       FTI_Conf->metadDir */           memset(FTI_Conf->metadDir,0x0,FTI_BUFS);
  /* char[BUFS]       FTI_Conf->lTmpDir */            memset(FTI_Conf->lTmpDir,0x0,FTI_BUFS);
  /* char[BUFS]       FTI_Conf->gTmpDir */            memset(FTI_Conf->gTmpDir,0x0,FTI_BUFS);
  /* char[BUFS]       FTI_Conf->mTmpDir */            memset(FTI_Conf->mTmpDir,0x0,FTI_BUFS);

  // +--------- +
  // | FTI_Topo |
  // +--------- +

  /* int           */ FTI_Topo->nbProc                =0;
  /* int           */ FTI_Topo->nbNodes               =0;
  /* int           */ FTI_Topo->myRank                =0;
  /* int           */ FTI_Topo->splitRank             =0;
  /* int           */ FTI_Topo->nodeSize              =0;
  /* int           */ FTI_Topo->nbHeads               =0;
  /* int           */ FTI_Topo->nbApprocs             =0;
  /* int           */ FTI_Topo->groupSize             =0;
  /* int           */ FTI_Topo->sectorID              =0;
  /* int           */ FTI_Topo->nodeID                =0;
  /* int           */ FTI_Topo->groupID               =0;
  /* int           */ FTI_Topo->amIaHead              =0;
  /* int           */ FTI_Topo->headRank              =0;
  /* int           */ FTI_Topo->nodeRank              =0;
  /* int           */ FTI_Topo->groupRank             =0;
  /* int           */ FTI_Topo->right                 =0;
  /* int           */ FTI_Topo->left                  =0;
  /* int[BUFS]        FTI_Topo->body */               memset(FTI_Topo->body,0x0,FTI_BUFS*sizeof(int));

  // +--------- +
  // | FTI_Ckpt |
  // +--------- +

  /* FTIT_Ckpt[5]     FTI_Ckpt Array */               memset(FTI_Ckpt,0x0,sizeof(FTIT_checkpoint)*5);
  /* int           */ FTI_Ckpt[4].isDcp               =false;
  /* int           */ FTI_Ckpt[4].hasDcp              =false;

  // +--------- +
  // | FTI_Injc |
  // +--------- +

  //* int           */ FTI_Injc->rank                  =0;
  //* int           */ FTI_Injc->index                 =0;
  //* int           */ FTI_Injc->position              =0;
  //* int           */ FTI_Injc->number                =0;
  //* int           */ FTI_Injc->frequency             =0;
  //* int           */ FTI_Injc->counter               =0;
  //* double        */ FTI_Injc->timer                 =0;

  return FTI_SCES;

}


/*-------------------------------------------------------------------------*/
/**
  @brief      It calculates checksum of the checkpoint file.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Data        Dataset metadata.
  @param      checksum        Checksum that is calculated.
  @return     integer         FTI_SCES if successful.

  This function calculates checksum of the checkpoint file based on
  MD5 algorithm and saves it in checksum.

 **/
/*-------------------------------------------------------------------------*/
int FTI_Checksum(FTIT_execution* FTI_Exec, FTIT_dataset* FTI_Data,
    FTIT_configuration* FTI_Conf, char* checksum)
{

  MD5_CTX mdContext;
  MD5_Init (&mdContext);
  int i;

  //iterate all variables
  for (i = 0; i < FTI_Exec->nbVar; i++) {
   
#ifdef GPUSUPPORT
    if (FTI_Data[i].isDevicePtr) {
        if (FTI_Conf->ioMode != FTI_IO_FTIFF)
          FTI_Data[i].ptr = malloc(FTI_Data[i].count * FTI_Data[i].eleSize);

      if(FTI_Data[i].ptr == NULL){
        FTI_Print("Failed to allocate FTI scratch buffer", FTI_EROR);
        return FTI_NSCS;
      }
      // TODO: Reuse GPU data on the host memory
      int result = FTI_Try(FTI_copy_from_device(FTI_Data[i].ptr, FTI_Data[i].devicePtr, FTI_Data[i].size,  FTI_Exec), "copying data from GPU");

      if(result == FTI_NSCS) {
        return FTI_NSCS;
      }
    }
#endif
    MD5_Update (&mdContext, FTI_Data[i].ptr, FTI_Data[i].size);

#ifdef GPUSUPPORT    
    if (FTI_Data[i].isDevicePtr) {
        if (FTI_Conf->ioMode != FTI_IO_FTIFF){
          free(FTI_Data[i].ptr);
          FTI_Data[i].ptr = NULL;
        }
    }
#endif
  }

  unsigned char hash[MD5_DIGEST_LENGTH];
  MD5_Final (hash, &mdContext);

  int ii = 0;
  for(i = 0; i < MD5_DIGEST_LENGTH; i++) {
    sprintf(&checksum[ii], "%02x", hash[i]);
    ii += 2;
  }

  return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      It compares checksum of the checkpoint file.
  @param      fileName        Filename of the checkpoint.
  @param      checksumToCmp   Checksum to compare.
  @return     integer         FTI_SCES if successful.

  This function calculates checksum of the checkpoint file based on
  MD5 algorithm. It compares calculated hash value with the one saved
  in the file.

 **/
/*-------------------------------------------------------------------------*/
int FTI_VerifyChecksum(char* fileName, char* checksumToCmp)
{
  FILE *fd = fopen(fileName, "rb");
  if (fd == NULL) {
    char str[FTI_BUFS];
    sprintf(str, "FTI failed to open file %s to calculate checksum.", fileName);
    FTI_Print(str, FTI_WARN);
    return FTI_NSCS;
  }

  MD5_CTX mdContext;
  MD5_Init (&mdContext);

  int bytes;
  unsigned char data[CHUNK_SIZE];
  while ((bytes = fread (data, 1, CHUNK_SIZE, fd)) != 0) {
    MD5_Update (&mdContext, data, bytes);
  }
  unsigned char hash[MD5_DIGEST_LENGTH];
  MD5_Final (hash, &mdContext);

  int i;
  char checksum[MD5_DIGEST_STRING_LENGTH];   //calculated checksum
  int ii = 0;
  for(i = 0; i < MD5_DIGEST_LENGTH; i++) {
    sprintf(&checksum[ii], "%02x", hash[i]);
    ii += 2;
  }

  if (strcmp(checksum, checksumToCmp) != 0) {
    char str[FTI_BUFS];
    sprintf(str, "TOOLS: Checksum do not match. \"%s\" file is corrupted. %s != %s",
        fileName, checksum, checksumToCmp);
    FTI_Print(str, FTI_WARN);

    fclose (fd);

    return FTI_NSCS;
  }

  fclose (fd);

  return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      It receives the return code of a function and prints a message.
  @param      result          Result to check.
  @param      message         Message to print.
  @return     integer         The same result as passed in parameter.

  This function checks the result from a function and then decides to
  print the message either as a debug message or as a warning.

 **/
/*-------------------------------------------------------------------------*/
int FTI_Try(int result, char* message)
{
  char str[FTI_BUFS];
  if (result == FTI_SCES || result == FTI_DONE) {
    sprintf(str, "FTI succeeded to %s", message);
    FTI_Print(str, FTI_DBUG);
  }
  else {
    sprintf(str, "FTI failed to %s", message);
    FTI_Print(str, FTI_WARN);
    sprintf(str, "Error => %s", strerror(errno));
    FTI_Print(str, FTI_WARN);
  }
  return result;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      It mallocs memory for the metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Topo        Topology metadata.

  This function mallocs the memory used for the metadata storage.

 **/
/*-------------------------------------------------------------------------*/
void FTI_MallocMeta(FTIT_execution* FTI_Exec, FTIT_topology* FTI_Topo)
{
  int i;
  if (FTI_Topo->amIaHead) {
    for (i = 0; i < 5; i++) {
      FTI_Exec->meta[i].exists = calloc(FTI_Topo->nodeSize, sizeof(int));
      FTI_Exec->meta[i].maxFs = calloc(FTI_Topo->nodeSize, sizeof(long));
      FTI_Exec->meta[i].fs = calloc(FTI_Topo->nodeSize, sizeof(long));
      FTI_Exec->meta[i].pfs = calloc(FTI_Topo->nodeSize, sizeof(long));
      FTI_Exec->meta[i].ckptFile = calloc(FTI_BUFS * FTI_Topo->nodeSize, sizeof(char));
      FTI_Exec->meta[i].currentL4CkptFile = calloc(FTI_BUFS * FTI_Topo->nodeSize, sizeof(char));
      FTI_Exec->meta[i].nbVar = calloc(FTI_Topo->nodeSize, sizeof(int));
      FTI_Exec->meta[i].varID = calloc(FTI_BUFS * FTI_Topo->nodeSize, sizeof(int));
      FTI_Exec->meta[i].varSize = calloc(FTI_BUFS * FTI_Topo->nodeSize, sizeof(long));
    }
  } else {
    for (i = 0; i < 5; i++) {
      FTI_Exec->meta[i].exists = calloc(1, sizeof(int));
      FTI_Exec->meta[i].maxFs = calloc(1, sizeof(long));
      FTI_Exec->meta[i].fs = calloc(1, sizeof(long));
      FTI_Exec->meta[i].pfs = calloc(1, sizeof(long));
      FTI_Exec->meta[i].ckptFile = calloc(FTI_BUFS, sizeof(char));
      FTI_Exec->meta[i].currentL4CkptFile = calloc(FTI_BUFS, sizeof(char));
      FTI_Exec->meta[i].nbVar = calloc(1, sizeof(int));
      FTI_Exec->meta[i].varID = calloc(FTI_BUFS, sizeof(int));
      FTI_Exec->meta[i].varSize = calloc(FTI_BUFS, sizeof(long));
    }
  }
  FTI_Exec->metaAlloc = 1;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      It frees memory for the metadata.
  @param      FTI_Exec        Execution metadata.

  This function frees the memory used for the metadata storage.

 **/
/*-------------------------------------------------------------------------*/
void FTI_FreeMeta(FTIT_execution* FTI_Exec)
{
  if (FTI_Exec->metaAlloc == 1) {
    int i;
    for (i = 0; i < 5; i++) {
      free(FTI_Exec->meta[i].exists);
      free(FTI_Exec->meta[i].maxFs);
      free(FTI_Exec->meta[i].fs);
      free(FTI_Exec->meta[i].pfs);
      free(FTI_Exec->meta[i].ckptFile);
      free(FTI_Exec->meta[i].nbVar);
      free(FTI_Exec->meta[i].varID);
      free(FTI_Exec->meta[i].varSize);
    }
    FTI_Exec->metaAlloc = 0;
  }
}

/*-------------------------------------------------------------------------*/
/**
  @brief      It mallocs memory for the metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Topo        Topology metadata.

  This function mallocs the memory used for the metadata storage.

 **/
/*-------------------------------------------------------------------------*/
int FTI_InitGroupsAndTypes(FTIT_execution* FTI_Exec) 
{
  FTI_Exec->FTI_Type = malloc(sizeof(FTIT_type*) * FTI_BUFS);
  if (FTI_Exec->FTI_Type == NULL) {
    return FTI_NSCS;
  }

  FTI_Exec->H5groups = malloc(sizeof(FTIT_H5Group*) * FTI_BUFS);
  if (FTI_Exec->H5groups == NULL) {
    return FTI_NSCS;
  }

  FTI_Exec->H5groups[0] = malloc(sizeof(FTIT_H5Group));
  if (FTI_Exec->H5groups[0] == NULL) {
    return FTI_NSCS;
  }

  FTI_Exec->H5groups[0]->id = 0;
  FTI_Exec->H5groups[0]->childrenNo = 0;
  sprintf(FTI_Exec->H5groups[0]->name, "/");
  FTI_Exec->nbGroup = 1;
  return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      It frees memory for the types.
  @param      FTI_Exec        Execution metadata.

  This function frees the memory used for the type storage.

 **/
/*-------------------------------------------------------------------------*/
void FTI_FreeTypesAndGroups(FTIT_execution* FTI_Exec) 
{
  int i;
  for (i = 0; i < FTI_Exec->nbType; i++) {
    if (FTI_Exec->FTI_Type[i]->structure != NULL) {
      //if complex type and have structure
      free(FTI_Exec->FTI_Type[i]->structure);
    }
    free(FTI_Exec->FTI_Type[i]);
  }
  free(FTI_Exec->FTI_Type);
  for (i = 0; i < FTI_Exec->nbGroup; i++) {
    free(FTI_Exec->H5groups[i]);
  }
  free(FTI_Exec->H5groups);
}

#ifdef ENABLE_HDF5
/*-------------------------------------------------------------------------*/
/**
  @brief      It creates h5datatype (hid_t) from definitions in FTI_Types
  @param      ftiType        FTI_Type type

  This function creates (opens) hdf5 compound type. Should be called only
  before saving checkpoint in HDF5 format. Build-in FTI's types are always open.

 **/
/*-------------------------------------------------------------------------*/
void FTI_CreateComplexType(FTIT_type* ftiType, FTIT_type** FTI_Type)
{
  char str[FTI_BUFS];
  if (ftiType->h5datatype > -1) {
    //This type already created
    sprintf(str, "Type [%d] is already created.", ftiType->id);
    FTI_Print(str, FTI_DBUG);
    return;
  }

  if (ftiType->structure == NULL) {
    //Save as array of bytes
    sprintf(str, "Creating type [%d] as array of bytes.", ftiType->id);
    FTI_Print(str, FTI_DBUG);
    ftiType->h5datatype = H5Tcopy(H5T_NATIVE_CHAR);
    H5Tset_size(ftiType->h5datatype, ftiType->size);
    return;
  }

  hid_t partTypes[FTI_BUFS];
  int i;
  //for each field create and rank-dimension array if needed
  for (i = 0; i < ftiType->structure->length; i++) {
    sprintf(str, "Type [%d] trying to create new type [%d].", ftiType->id, ftiType->structure->field[i].typeID);
    FTI_Print(str, FTI_DBUG);
    FTI_CreateComplexType(FTI_Type[ftiType->structure->field[i].typeID], FTI_Type);
    partTypes[i] = FTI_Type[ftiType->structure->field[i].typeID]->h5datatype;
    if (ftiType->structure->field[i].rank > 1) {
      //need to create rank-dimension array type
      hsize_t dims[FTI_BUFS];
      int j;
      for (j = 0; j < ftiType->structure->field[i].rank; j++) {
        dims[j] = ftiType->structure->field[i].dimLength[j];
      }
      sprintf(str, "Type [%d] trying to create %d-D array of type [%d].", ftiType->id, ftiType->structure->field[i].rank, ftiType->structure->field[i].typeID);
      FTI_Print(str, FTI_DBUG);
      partTypes[i] = H5Tarray_create(FTI_Type[ftiType->structure->field[i].typeID]->h5datatype, ftiType->structure->field[i].rank, dims);
    } else {
      if (ftiType->structure->field[i].dimLength[0] > 1) {
        //need to create 1-dimension array type
        sprintf(str, "Type [%d] trying to create 1-D [%d] array of type [%d].", ftiType->id, ftiType->structure->field[i].dimLength[0], ftiType->structure->field[i].typeID);
        FTI_Print(str, FTI_DBUG);
        hsize_t dim = ftiType->structure->field[i].dimLength[0];
        partTypes[i] = H5Tarray_create(FTI_Type[ftiType->structure->field[i].typeID]->h5datatype, 1, &dim);
      }
    }
  }

  //create new HDF5 datatype
  sprintf(str, "Creating type [%d].", ftiType->id);
  FTI_Print(str, FTI_DBUG);
  ftiType->h5datatype = H5Tcreate(H5T_COMPOUND, ftiType->size);
  sprintf(str, "Type [%d] has hid_t %ld.", ftiType->id, (long)ftiType->h5datatype);
  FTI_Print(str, FTI_DBUG);
  if (ftiType->h5datatype < 0) {
    FTI_Print("FTI failed to create HDF5 type.", FTI_WARN);
  }

  //inserting fields into the new type
  for (i = 0; i < ftiType->structure->length; i++) {
    sprintf(str, "Insering type [%d] into new type [%d].", ftiType->structure->field[i].typeID, ftiType->id);
    FTI_Print(str, FTI_DBUG);
    herr_t res = H5Tinsert(ftiType->h5datatype, ftiType->structure->field[i].name, ftiType->structure->field[i].offset, partTypes[i]);
    if (res < 0) {
      FTI_Print("FTI faied to insert type in complex type.", FTI_WARN);
    }
  }

}
#endif

#ifdef ENABLE_HDF5
/*-------------------------------------------------------------------------*/
/**
  @brief      It closes h5datatype
  @param      ftiType        FTI_Type type

  This function destroys (closes) hdf5 compound type. Should be called
  after saving checkpoint in HDF5 format.

 **/
/*-------------------------------------------------------------------------*/
void FTI_CloseComplexType(FTIT_type* ftiType, FTIT_type** FTI_Type)
{
  char str[FTI_BUFS];
  if (ftiType->h5datatype == -1 || ftiType->id < 11) {
    //This type already closed or build-in type
    sprintf(str, "Cannot close type [%d]. Build in or already closed.", ftiType->id);
    FTI_Print(str, FTI_DBUG);
    return;
  }

  if (ftiType->structure != NULL) {
    //array of bytes don't have structure
    int i;
    //close each field
    for (i = 0; i < ftiType->structure->length; i++) {
      sprintf(str, "Closing type [%d] of compound type [%d].", ftiType->structure->field[i].typeID, ftiType->id);
      FTI_Print(str, FTI_DBUG);
      FTI_CloseComplexType(FTI_Type[ftiType->structure->field[i].typeID], FTI_Type);
    }
  }

  //close HDF5 datatype
  sprintf(str, "Closing type [%d].", ftiType->id);
  FTI_Print(str, FTI_DBUG);
  herr_t res = H5Tclose(ftiType->h5datatype);
  if (res < 0) {
    FTI_Print("FTI failed to close HDF5 type.", FTI_WARN);
  }
  ftiType->h5datatype = -1;
}
#endif

#ifdef ENABLE_HDF5
/*-------------------------------------------------------------------------*/
/**
  @brief      It creates a group and all it's children
  @param      ftiGroup        FTI_H5Group to be create
  @param      parentGroup     hid_t of the parent

  This function creates hdf5 group and all it's children. Should be
  called only before saving checkpoint in HDF5 format.

 **/
/*-------------------------------------------------------------------------*/
void FTI_CreateGroup(FTIT_H5Group* ftiGroup, hid_t parentGroup, FTIT_H5Group** FTI_Group)
{
  ftiGroup->h5groupID = H5Gcreate2(parentGroup, ftiGroup->name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  if (ftiGroup->h5groupID < 0) {
    FTI_Print("FTI failed to create HDF5 group.", FTI_WARN);
    return;
  }

  int i;
  for (i = 0; i < ftiGroup->childrenNo; i++) {
    FTI_CreateGroup(FTI_Group[ftiGroup->childrenID[i]], ftiGroup->h5groupID, FTI_Group); //Try to create the child
  }
}
#endif

#ifdef ENABLE_HDF5
/*-------------------------------------------------------------------------*/
/**
  @brief      It opens a group and all it's children
  @param      ftiGroup        FTI_H5Group to be opened
  @param      parentGroup     hid_t of the parent

  This function opens hdf5 group and all it's children. Should be
  called only before recovery in HDF5 format.

 **/
/*-------------------------------------------------------------------------*/
void FTI_OpenGroup(FTIT_H5Group* ftiGroup, hid_t parentGroup, FTIT_H5Group** FTI_Group)
{
  ftiGroup->h5groupID = H5Gopen2(parentGroup, ftiGroup->name, H5P_DEFAULT);
  if (ftiGroup->h5groupID < 0) {
    FTI_Print("FTI failed to open HDF5 group.", FTI_WARN);
    return;
  }

  int i;
  for (i = 0; i < ftiGroup->childrenNo; i++) {
    FTI_OpenGroup(FTI_Group[ftiGroup->childrenID[i]], ftiGroup->h5groupID, FTI_Group); //Try to open the child
  }
}
#endif

#ifdef ENABLE_HDF5
/*-------------------------------------------------------------------------*/
/**
  @brief      It closes a group and all it's children
  @param      ftiGroup        FTI_H5Group to be closed

  This function closes (destoys) hdf5 group and all it's children. Should be
  called only after saving checkpoint in HDF5 format.

 **/
/*-------------------------------------------------------------------------*/
void FTI_CloseGroup(FTIT_H5Group* ftiGroup, FTIT_H5Group** FTI_Group)
{
  char str[FTI_BUFS];
  if (ftiGroup->h5groupID == -1) {
    //This group already closed, in tree this is error
    snprintf(str, FTI_BUFS, "Group %s is already closed?", ftiGroup->name);
    FTI_Print(str, FTI_WARN);
    return;
  }

  int i;
  for (i = 0; i < ftiGroup->childrenNo; i++) {
    FTI_CloseGroup(FTI_Group[ftiGroup->childrenID[i]], FTI_Group); //Try to close the child
  }

  herr_t res = H5Gclose(ftiGroup->h5groupID);
  if (res < 0) {
    FTI_Print("FTI failed to close HDF5 group.", FTI_WARN);
  }
  ftiGroup->h5groupID = -1;
}
#endif

#ifdef ENABLE_HDF5
/*-------------------------------------------------------------------------*/
/**
  @brief      It creates the global dataset in the VPR file.
  @param      FTI_Exec        Execution metadata.
  @return     integer         FTI_SCES if successful.

  Creates global dataset (shared among all ranks) in VPR file. The dataset
  position will be the group assigned to it by calling the FTI API function 
  'FTI_DefineGlobalDataset'.
 **/
/*-------------------------------------------------------------------------*/
int FTI_CreateGlobalDatasets( FTIT_execution* FTI_Exec )
{
    
    FTIT_globalDataset* dataset = FTI_Exec->globalDatasets;
    while( dataset ) {

        // create file space
        dataset->fileSpace = H5Screate_simple( dataset->rank, dataset->dimension, NULL );

        if(dataset->fileSpace < 0) {
            char errstr[FTI_BUFS];
            snprintf( errstr, FTI_BUFS, "Unable to create space for dataset #%d", dataset->id );
            FTI_Print(errstr,FTI_EROR);
            return FTI_NSCS;
        }

        // create dataset
        hid_t loc = dataset->location->h5groupID;
        hid_t tid = FTI_Exec->FTI_Type[dataset->type.id]->h5datatype;
        dataset->hdf5TypeId = tid;
        hid_t fsid = dataset->fileSpace;
        
        // FLETCHER CHECKSUM NOT SUPPORTED FOR PARALLEL I/O IN HDF5
        //hid_t dcpl = H5Pcreate (H5P_DATASET_CREATE);
        //H5Pset_fletcher32 (dcpl);
        //
        //hsize_t *chunk = malloc( sizeof(hsize_t) * dataset->rank );
        //chunk[0] = chunk[1] = 4096;
        //H5Pset_chunk (dcpl, 2, chunk);

        dataset->hid = H5Dcreate( loc, dataset->name, tid, fsid, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT );
        if(dataset->hid < 0) {
            char errstr[FTI_BUFS];
            snprintf( errstr, FTI_BUFS, "Unable to create dataset #%d", dataset->id );
            FTI_Print(errstr,FTI_EROR);
            return FTI_NSCS;
        }

        dataset->initialized = true;

        dataset = dataset->next;

    }

    return FTI_SCES;

}
#endif

#ifdef ENABLE_HDF5
/*-------------------------------------------------------------------------*/
/**
  @brief      Opens global dataset in VPR file.
  @param      FTI_Exec        Execution metadata.
  @return     integer         FTI_SCES if successful.

  This function is the analog to 'FTI_CreateGlobalDatasets' on recovery.
 **/
/*-------------------------------------------------------------------------*/
int FTI_OpenGlobalDatasets( FTIT_execution* FTI_Exec )
{
    
    hsize_t *dims = NULL; 
    hsize_t *maxDims = NULL;

    char errstr[FTI_BUFS];
    FTIT_globalDataset* dataset = FTI_Exec->globalDatasets;
    while( dataset ) {

        dims = (hsize_t*) realloc( dims, sizeof(hsize_t)*dataset->rank ); 
        maxDims = (hsize_t*) realloc( maxDims, sizeof(hsize_t)*dataset->rank );
        
        // open dataset
        hid_t loc = dataset->location->h5groupID;
        hid_t tid = FTI_Exec->FTI_Type[dataset->type.id]->h5datatype;
        dataset->hdf5TypeId = tid;

        dataset->hid = H5Dopen( loc, dataset->name, H5P_DEFAULT );
        if(dataset->hid < 0) {
            snprintf( errstr, FTI_BUFS, "failed to open dataset '%s'", dataset->name );
            FTI_Print( errstr, FTI_WARN );
            return FTI_NSCS;
        }
        
        // get file space and check if rank and dimension coincide for file and execution
        hid_t fsid = H5Dget_space( dataset->hid );
        if( fsid > 0 ) {
            int rank = H5Sget_simple_extent_ndims( fsid );
            if( rank == dataset->rank ) {
                H5Sget_simple_extent_dims( fsid, dims, maxDims );
                if( memcmp( dims, dataset->dimension, sizeof(hsize_t)*rank ) ) {
                    snprintf( errstr, FTI_BUFS, "stored and requested dimensions of dataset '%s' differ!", dataset->name );
                    FTI_Print( errstr, FTI_WARN );
                    return FTI_NSCS;
                }
            } else {
                snprintf( errstr, FTI_BUFS, "stored and requested rank of dataset '%s' differ (stored:%d != requested:%d)!", dataset->name, rank, dataset->rank );
                FTI_Print( errstr, FTI_WARN );
                return FTI_NSCS;
            }
        } else {
            snprintf( errstr, FTI_BUFS, "failed to acquire data space information of dataset '%s'", dataset->name );
            FTI_Print( errstr, FTI_WARN );
            return FTI_NSCS;
        }    
        dataset->fileSpace = fsid;
        
        dataset->initialized = true;

        dataset = dataset->next;

    }

    if( dims ) { free( dims ); }
    if( maxDims ) { free( maxDims ); }

    return FTI_SCES;

}
#endif

#ifdef ENABLE_HDF5
/*-------------------------------------------------------------------------*/
/**
  @brief      Closes global datasets in VPR file 
  @param      FTI_Exec        Execution metadata.
  @return     integer         FTI_SCES if successful.
 **/
/*-------------------------------------------------------------------------*/
int FTI_CloseGlobalDatasets( FTIT_execution* FTI_Exec )
{
    
    FTIT_globalDataset* dataset = FTI_Exec->globalDatasets;
    while( dataset ) {

        H5Sclose(dataset->fileSpace);

        H5Dclose(dataset->hid);

        dataset = dataset->next;

    }

    return FTI_SCES;

}
#endif

#ifdef ENABLE_HDF5
/*-------------------------------------------------------------------------*/
/**
  @brief      Writes global dataset subsets into VPR file.
  @param      FTI_Data        Dataset metadata.
  @return     integer         FTI_SCES if successful.
 **/
/*-------------------------------------------------------------------------*/
herr_t FTI_WriteSharedFileData( FTIT_dataset FTI_Data )
{

    if( FTI_Data.sharedData.dataset ) {
        
        // hdf5 datatype
        hid_t tid = FTI_Data.sharedData.dataset->hdf5TypeId;

        // dataset hdf5-id
        hid_t did = FTI_Data.sharedData.dataset->hid;

        // shared dataset file space
        hid_t fsid = FTI_Data.sharedData.dataset->fileSpace;

        // shared dataset rank
        int ndim = FTI_Data.sharedData.dataset->rank;

        // shared dataset array of nummber of elements in each dimension
        hsize_t *count = FTI_Data.sharedData.count;

        // shared dataset array of the offsets for each dimension
        hsize_t *offset = FTI_Data.sharedData.offset;

        // create dataspace for subset of shared dataset
        hid_t msid = H5Screate_simple( ndim, count, NULL );
        if(msid < 0) {
            char errstr[FTI_BUFS];
            snprintf( errstr, FTI_BUFS, "Unable to create space for var-id %d in dataset #%d", FTI_Data.id, FTI_Data.sharedData.dataset->id );
            FTI_Print(errstr,FTI_EROR);
            return FTI_NSCS;
        }
        // select range in shared dataset in file
        if( H5Sselect_hyperslab(fsid, H5S_SELECT_SET, offset, NULL, count, NULL) < 0 ) {
            char errstr[FTI_BUFS];
            snprintf( errstr, FTI_BUFS, "Unable to select sub-space for var-id %d in dataset #%d", FTI_Data.id, FTI_Data.sharedData.dataset->id );
            FTI_Print(errstr,FTI_EROR);
            return FTI_NSCS;
        }

        // enable collective buffering
        hid_t plid = H5Pcreate( H5P_DATASET_XFER );
        H5Pset_dxpl_mpio(plid, H5FD_MPIO_COLLECTIVE);

        // write data in file
        if( H5Dwrite(did, tid, msid, fsid, plid, FTI_Data.ptr) < 0 ) {
            char errstr[FTI_BUFS];
            snprintf( errstr, FTI_BUFS, "Unable to write var-id %d of dataset #%d", FTI_Data.id, FTI_Data.sharedData.dataset->id );
            FTI_Print(errstr,FTI_EROR);
            return FTI_NSCS;
        }

        H5Sclose( msid );
        H5Pclose( plid );

    }

    return FTI_SCES;

}
#endif

#ifdef ENABLE_HDF5 
/*-------------------------------------------------------------------------*/
/**
  @brief      Checks for matching dimension sizes of sub-sets
  @param      FTI_Data        Dataset metadata.
  @param      FTI_Exec        Execution metadata.
  @return     integer         FTI_SCES if successful.

  This function counts the number of elements of all sub-sets contained in a 
  particular global dataset and accumulates to a total value. If the accu-
  mulated value matches the number of elements defined for the global data-
  set, FTI_SCES is returned. This function is called before the checkpoint
  and before the recovery.

  @todo it would be great to check for region overlapping too.

 **/
/*-------------------------------------------------------------------------*/
int FTI_CheckDimensions( FTIT_dataset * FTI_Data, FTIT_execution * FTI_Exec ) 
{   

    // NOTE checking for overlap is complicated and likely expensive
    // since it requires sorting within all contributing processes.
    // Thus, we check currently only the number of elements.
    FTIT_globalDataset* dataset = FTI_Exec->globalDatasets;
    while( dataset ) {
        int i,j;
        // sum of local elements
        hsize_t numElemLocal = 0, numElemGlobal;
        for( i=0; i<dataset->numSubSets; ++i ) {
            hsize_t numElemSubSet = 1;
            for( j=0; j<dataset->rank; j++ ) {
                numElemSubSet *= FTI_Data[dataset->varIdx[i]].sharedData.count[j];
            }
            numElemLocal += numElemSubSet;
        }
        // number of elements in global dataset
        hsize_t numElem = 1;
        for( i=0; i<dataset->rank; ++i ) {
            numElem *= dataset->dimension[i];
        }
        MPI_Allreduce( &numElemLocal, &numElemGlobal, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, FTI_COMM_WORLD );
        if( numElem != numElemGlobal ) {
            char errstr[FTI_BUFS];
            snprintf( errstr, FTI_BUFS, "Number of elements of subsets (accumulated) do not match number of elements defined for global dataset #%d!", dataset->id ); 
            FTI_Print( errstr, FTI_WARN);
            return FTI_NSCS;
        }
        dataset = dataset->next;
    }
    return FTI_SCES;
}
#endif

#ifdef ENABLE_HDF5
/*-------------------------------------------------------------------------*/
/**
  @brief      Checks if VPR file on restart
  @param      FTI_Conf        Configuration metadata.
  @param      ckptID          Checkpoint ID.
  @return     integer         FTI_SCES if successful.

  Checks if restart is possible for VPR file. 
  1) Checks if file exist for prefix and directory, defined in config file.
  2) Checks if is regular file.
  3) Checks if groups and datasets can be accessed and if datasets can be 
  read

  If file found and sane, ckptID is set and FTI_SCES is returned.

 **/
/*-------------------------------------------------------------------------*/
int FTI_H5CheckSingleFile( FTIT_configuration* FTI_Conf, int *ckptID ) 
{
    char errstr[FTI_BUFS];
    char fn[FTI_BUFS];
    int res = FTI_SCES;
    struct stat st;
   
    struct dirent *entry;
    DIR *dir = opendir( FTI_Conf->h5SingleFileDir );

    if ( dir == NULL ) {
        snprintf( errstr, FTI_BUFS, "VPR directory '%s' could not be accessed.", FTI_Conf->h5SingleFileDir );
        FTI_Print(errstr, FTI_EROR);
        errno = 0;
        return FTI_NSCS;
    }

    *ckptID = -1;

    bool found = false;
    while((entry = readdir(dir)) != NULL) {   
        if(strcmp(entry->d_name,".") && strcmp(entry->d_name,"..")) {
            int len = strlen( entry->d_name ); 
            if( len > 14 ) {
                char fileRoot[FTI_BUFS];
                bzero( fileRoot, FTI_BUFS );
                memcpy( fileRoot, entry->d_name, len - 14 );
                char fileRootExpected[FTI_BUFS];
                snprintf( fileRootExpected, FTI_BUFS, "%s", FTI_Conf->h5SingleFilePrefix );
                if( strncmp( fileRootExpected, fileRoot, FTI_BUFS ) == 0 ) {
                    int id_tmp;
                    sscanf( entry->d_name + len - 14 + 3, "%08d.h5", &id_tmp );
                    if( id_tmp > *ckptID ) {
                        *ckptID = id_tmp;
                        snprintf( fn, FTI_BUFS, "%s/%s-ID%08d.h5", FTI_Conf->h5SingleFileDir, FTI_Conf->h5SingleFilePrefix, *ckptID ); 
                    }
                    found = true;
                }
            }
        }
    }

    if (!found) {
        snprintf( errstr, FTI_BUFS, "unable to find matching VPR file (filename pattern: '%s-ID########.h5')!", FTI_Conf->h5SingleFilePrefix );
        FTI_Print( errstr, FTI_WARN );
        return FTI_NSCS;
    }
    
    stat( fn, &st );
    if( S_ISREG( st.st_mode ) ) {
        hid_t fid = H5Fopen( fn, H5F_ACC_RDONLY, H5P_DEFAULT );
        if( fid > 0 ) {
            hid_t gid = H5Gopen1( fid, "/" );
            if( gid > 0 ) {
                res += FTI_ScanGroup( gid, fn );
                H5Gclose(gid);
            } else {
                snprintf( errstr, FTI_BUFS, "failed to access root group in file '%s'", fn );
                FTI_Print( errstr, FTI_WARN );
                res = FTI_NSCS;
            }
            H5Fclose(fid);
        } else {
            snprintf( errstr, FTI_BUFS, "failed to open file '%s'", fn );
            FTI_Print( errstr, FTI_WARN );
            res = FTI_NSCS;
        }
    } else {
        snprintf( errstr, FTI_BUFS, "'%s', is not a regular file!", fn );
        FTI_Print( errstr, FTI_WARN );
        res = FTI_NSCS;
    }
    return res;
}
#endif

#ifdef ENABLE_HDF5
/*-------------------------------------------------------------------------*/
/**
  @brief      Checks groups and datasets (callable recursively) 
  @param      gid             Parent group ID.
  @param      fn              File name.
  @return     integer         FTI_SCES if successful.

  This function analyses the group structure recursivley starting at group
  'gid'. It steps down in sub groups and checks datasets of consistency.
  The consistency check is performed if the dataset was created with the 
  fletcher32 filter activated. A dataset read will return a negative value 
  in that case if dataset corrupted.

 **/
/*-------------------------------------------------------------------------*/
int FTI_ScanGroup( hid_t gid, char* fn ) 
{
    int res = FTI_SCES;
    char errstr[FTI_BUFS];
    hsize_t nobj;
    if( H5Gget_num_objs( gid, &nobj ) >= 0 ) {
        int i;
        for(i=0; i<nobj; i++) {
            int objtype;
            char dname[FTI_BUFS];
            char gname[FTI_BUFS];
            // determine if element is group or dataset
            objtype = H5Gget_objtype_by_idx(gid, (size_t)i );
            if( objtype == H5G_DATASET ) {
                H5Gget_objname_by_idx(gid, (hsize_t)i, dname, (size_t) FTI_BUFS); 
                // open dataset
                hid_t did = H5Dopen1( gid, dname );
                if( did > 0 ) {
                    hid_t sid = H5Dget_space(did);
                    hid_t tid = H5Dget_type(did);
                    int drank = H5Sget_simple_extent_ndims( sid );
                    size_t typeSize = H5Tget_size( tid );
                    hsize_t *count = (hsize_t*) calloc( drank, sizeof(hsize_t) );
                    hsize_t *offset = (hsize_t*) calloc( drank, sizeof(hsize_t) );
                    count[0] = 1;
                    char* buffer = (char*) malloc( typeSize );
                    hid_t msid = H5Screate_simple( drank, count, NULL );
                    H5Sselect_hyperslab(sid, H5S_SELECT_SET, offset, NULL, count, NULL);
                    // read element to trigger checksum comparison
                    herr_t status = H5Dread(did, tid, msid, sid, H5P_DEFAULT, buffer);
                    if( status < 0 ) {
                        snprintf( errstr, FTI_BUFS, "unable to read from dataset '%s' in file '%s'!", dname, fn );
                        FTI_Print( errstr, FTI_WARN );
                        res += FTI_NSCS;
                    }
                    H5Dclose(did);
                    H5Sclose(msid);
                    H5Sclose(sid);
                    H5Tclose(tid);
                    free( count );
                    free( offset );
                    free( buffer );
                } else {
                    snprintf( errstr, FTI_BUFS, "failed to open dataset '%s' in file '%s'", dname, fn );
                    FTI_Print( errstr, FTI_WARN );
                    res += FTI_NSCS;
                }
            }
            // step down other group
            if( objtype == H5G_GROUP ) {
                H5Gget_objname_by_idx(gid, (hsize_t)i, gname, (size_t) FTI_BUFS); 
                hid_t sgid = H5Gopen1( gid, gname );
                if( sgid > 0 ) {
                    res += FTI_ScanGroup( sgid, fn );
                    H5Gclose(sgid);
                } else {
                    snprintf( errstr, FTI_BUFS, "failed to open group '%s' in file '%s'", gname, fn );
                    FTI_Print( errstr, FTI_WARN );
                    res += FTI_NSCS;
                }
            }
        }
    } else {
        snprintf( errstr, FTI_BUFS, "failed to get number of elements in file '%s'", fn );
        FTI_Print( errstr, FTI_WARN );
        res += FTI_NSCS;
    }
    return FTI_SCES;
}
#endif

#ifdef ENABLE_HDF5
/*-------------------------------------------------------------------------*/
/**
  @brief      Reads a sub-set of a global dataset on recovery.
  @param      FTI_Data        Dataset metadata.
  @return     integer         FTI_SCES if successful.
 **/
/*-------------------------------------------------------------------------*/
herr_t FTI_ReadSharedFileData( FTIT_dataset FTI_Data )
{

    // hdf5 datatype
    hid_t tid = FTI_Data.sharedData.dataset->hdf5TypeId;
    
    // dataset hdf5-id
    hid_t did = FTI_Data.sharedData.dataset->hid;

    // shared dataset file space
    hid_t fsid = FTI_Data.sharedData.dataset->fileSpace;
    
    // shared dataset rank
    int ndim = FTI_Data.sharedData.dataset->rank;

    // shared dataset array of nummber of elements in each dimension
    hsize_t *count = FTI_Data.sharedData.count;

    // shared dataset array of the offsets for each dimension
    hsize_t *offset = FTI_Data.sharedData.offset;

    // create dataspace for subset of shared dataset
    hid_t msid = H5Screate_simple( ndim, count, NULL );
    if(msid < 0) {
        char errstr[FTI_BUFS];
        snprintf( errstr, FTI_BUFS, "Unable to create space for var-id %d in dataset #%d", FTI_Data.id, FTI_Data.sharedData.dataset->id );
        FTI_Print(errstr,FTI_EROR);
        return FTI_NSCS;
    }

    // select range in shared dataset in file
    H5Sselect_hyperslab(fsid, H5S_SELECT_SET, offset, NULL, count, NULL);

    // enable collective buffering
    hid_t plid = H5Pcreate( H5P_DATASET_XFER );
    H5Pset_dxpl_mpio(plid, H5FD_MPIO_COLLECTIVE);

    // write data in file
    herr_t status = H5Dread(did, tid, msid, fsid, plid, FTI_Data.ptr);
    if(status < 0) {
        char errstr[FTI_BUFS];
        snprintf( errstr, FTI_BUFS, "Unable to read var-id %d from dataset #%d", FTI_Data.id, FTI_Data.sharedData.dataset->id );
        FTI_Print(errstr,FTI_EROR);
        return FTI_NSCS;
    }

    H5Sclose( msid );
    H5Pclose( plid );

    return status;

}
#endif

#ifdef ENABLE_HDF5 
void FTI_FreeVPRMem( FTIT_execution* FTI_Exec, FTIT_dataset* FTI_Data ) 
{
    FTIT_globalDataset * dataset = FTI_Exec->globalDatasets;
    while( dataset ) {
        if( dataset->dimension ) { free( dataset->dimension ); }
        if( dataset->varIdx ) { free( dataset->varIdx ); }
        FTIT_globalDataset * curr = dataset;
        dataset = dataset->next;
        free( curr );
    }

    int i=0;
    for( ; i<FTI_Exec->nbVar; i++ ) {
        if( FTI_Data[i].sharedData.offset ) {
            free( FTI_Data[i].sharedData.offset );
        }
        if( FTI_Data[i].sharedData.count ) {
            free( FTI_Data[i].sharedData.count );
        }
    }
}
#endif

/*-------------------------------------------------------------------------*/
/**
  @brief      It creates the basic datatypes and the dataset array.
  @param      FTI_Data        Dataset metadata.
  @return     integer         FTI_SCES if successful.

  This function creates the basic data types using FTIT_Type.

 **/
/*-------------------------------------------------------------------------*/
int FTI_InitBasicTypes(FTIT_dataset* FTI_Data)
{
  int i;
  for (i = 0; i < FTI_BUFS; i++) {
    FTI_Data[i].id = -1;
  }
  FTI_InitType(&FTI_CHAR, sizeof(char));
  FTI_InitType(&FTI_SHRT, sizeof(short));
  FTI_InitType(&FTI_INTG, sizeof(int));
  FTI_InitType(&FTI_LONG, sizeof(long));
  FTI_InitType(&FTI_UCHR, sizeof(unsigned char));
  FTI_InitType(&FTI_USHT, sizeof(unsigned short));
  FTI_InitType(&FTI_UINT, sizeof(unsigned int));
  FTI_InitType(&FTI_ULNG, sizeof(unsigned long));
  FTI_InitType(&FTI_SFLT, sizeof(float));
  FTI_InitType(&FTI_DBLE, sizeof(double));
  FTI_InitType(&FTI_LDBE, sizeof(long double));

  return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      It erases a directory and all its files.
  @param      path            Path to the directory we want to erase.
  @param      flag            Set to 1 to activate.
  @return     integer         FTI_SCES if successful.

  This function erases a directory and all its files. It focusses on the
  checkpoint directories created by FTI so it does NOT handle recursive
  erasing if the given directory includes other directories.

 **/
/*-------------------------------------------------------------------------*/
int FTI_RmDir(char path[FTI_BUFS], int flag)
{
  if (flag) {
    char str[FTI_BUFS];
    sprintf(str, "Removing directory %s and its files.", path);
    FTI_Print(str, FTI_DBUG);

    DIR* dp = opendir(path);
    if (dp != NULL) {
      struct dirent* ep = NULL;
      while ((ep = readdir(dp)) != NULL) {
        char fil[FTI_BUFS];
        sprintf(fil, "%s", ep->d_name);
        FTI_Print(fil, FTI_DBUG);
        if ((strcmp(fil, ".") != 0) && (strcmp(fil, "..") != 0)) {
          char fn[FTI_BUFS];
          sprintf(fn, "%s/%s", path, fil);
          sprintf(str, "File %s will be removed.", fn);
          FTI_Print(str, FTI_DBUG);
          if (remove(fn) == -1) {
            if (errno != ENOENT) {
              snprintf(str, FTI_BUFS, "Error removing target file (%s).", fn);
              FTI_Print(str, FTI_EROR);
            }
          }
        }
      }
    }
    else {
      if (errno != ENOENT) {
        FTI_Print("Error with opendir.", FTI_EROR);
      }
    }
    if (dp != NULL) {
      closedir(dp);
    }

    if (remove(path) == -1) {
      if (errno != ENOENT) {
        FTI_Print("Error removing target directory.", FTI_EROR);
      }
    }
  }
  return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      It erases the previous checkpoints and their metadata.
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Topo        Topology metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @param      level           Level of cleaning.
  @return     integer         FTI_SCES if successful.

  This function erases previous checkpoint depending on the level of the
  current checkpoint. Level 5 means complete clean up. Level 6 means clean
  up local nodes but keep last checkpoint data and metadata in the PFS.

 **/
/*-------------------------------------------------------------------------*/
int FTI_Clean(FTIT_configuration* FTI_Conf, FTIT_topology* FTI_Topo,
    FTIT_checkpoint* FTI_Ckpt, int level)
{
  int nodeFlag; //only one process in the node has set it to 1
  int globalFlag = !FTI_Topo->splitRank; //only one process in the FTI_COMM_WORLD has set it to 1

  nodeFlag = (((!FTI_Topo->amIaHead) && ((FTI_Topo->nodeRank - FTI_Topo->nbHeads) == 0)) || (FTI_Topo->amIaHead)) ? 1 : 0;
  
  bool notDcpFtiff = !(FTI_Ckpt[4].isDcp && FTI_Conf->dcpFtiff); 
  bool notDcp = !FTI_Ckpt[4].isDcp;   

  if (level == 0) {
    FTI_RmDir(FTI_Conf->mTmpDir, globalFlag && notDcpFtiff );
    FTI_RmDir(FTI_Conf->gTmpDir, globalFlag && notDcp );
    FTI_RmDir(FTI_Conf->lTmpDir, nodeFlag && notDcp );
  }

  // Clean last checkpoint level 1
  if (level >= 1) {
    FTI_RmDir(FTI_Ckpt[1].metaDir, globalFlag && notDcpFtiff );
    FTI_RmDir(FTI_Ckpt[1].dir, nodeFlag && notDcp );
  }

  // Clean last checkpoint level 2
  if (level >= 2) {
    FTI_RmDir(FTI_Ckpt[2].metaDir, globalFlag && notDcpFtiff );
    FTI_RmDir(FTI_Ckpt[2].dir, nodeFlag && notDcp );
  }

  // Clean last checkpoint level 3
  if (level >= 3) {
    FTI_RmDir(FTI_Ckpt[3].metaDir, globalFlag && notDcpFtiff );
    FTI_RmDir(FTI_Ckpt[3].dir, nodeFlag && notDcp );
  }

  // Clean last checkpoint level 4
  if ( level == 4 || level == 5 ) {
    FTI_RmDir(FTI_Ckpt[4].metaDir, globalFlag && notDcpFtiff );
    FTI_RmDir(FTI_Ckpt[4].dir, globalFlag && notDcp );
    rmdir(FTI_Conf->gTmpDir);
  }
  if ( (FTI_Conf->dcpPosix || FTI_Conf->dcpFtiff) && level == 5 ) {
    FTI_RmDir(FTI_Ckpt[4].dcpDir, !FTI_Topo->splitRank);
  }

  // If it is the very last cleaning and we DO NOT keep the last checkpoint
  if (level == 5) {
    rmdir(FTI_Conf->lTmpDir);
    rmdir(FTI_Conf->localDir);
    rmdir(FTI_Conf->glbalDir);
    char buf[FTI_BUFS];
    snprintf(buf, FTI_BUFS, "%s/Topology.fti", FTI_Conf->metadDir);
    if (remove(buf) == -1) {
      if (errno != ENOENT) {
        FTI_Print("Cannot remove Topology.fti", FTI_EROR);
      }
    }
    snprintf(buf, FTI_BUFS, "%s/Checkpoint.fti", FTI_Conf->metadDir);
    if (remove(buf) == -1) {
      if (errno != ENOENT) {
        FTI_Print("Cannot remove Checkpoint.fti", FTI_EROR);
      }
    }
    rmdir(FTI_Conf->metadDir);
  }

  // If it is the very last cleaning and we DO keep the last checkpoint
  if (level == 6) {
    rmdir(FTI_Conf->lTmpDir);
    rmdir(FTI_Conf->localDir);
  }

  return FTI_SCES;
}

char* hashHex( const unsigned char* hash, int digestWidth, char* hashHexStr )
{       
    static unsigned char hashHexStatic[MD5_DIGEST_STRING_LENGTH];
    if( hashHexStr == NULL ) {
        hashHexStr = hashHexStatic;
    }

    int i;
    for(i = 0; i < digestWidth; i++) {
        sprintf(&hashHexStr[2*i], "%02x", hash[i]);
    }

    return hashHexStr;
}
