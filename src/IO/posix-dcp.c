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
 *  @file   utility.c
 *  @date   October, 2017
 *  @brief  API functions for the FTI library.
 */

#include "../interface.h"
#include "../api-cuda.h"
#include "cuda-md5/md5Opt.h"
#include "../profiler/profiler.h"



/*-------------------------------------------------------------------------*/
/**
  @brief      Returns the file positio.
  @param      fileDesc        The file descriptor.
  @return     integer         The position in the file.
 **/
/*-------------------------------------------------------------------------*/
size_t FTI_GetDCPPosixFilePos(void *fileDesc){
  WriteDCPPosixInfo_t *fd = (WriteDCPPosixInfo_t*) fileDesc;
  return ftell(fd->write_info.f);
}


/*-------------------------------------------------------------------------*/
/**
  @brief      Initializes dCP POSIX I/O.
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Topo        Topology metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @param      FTI_Data        Dataset metadata.
  @return     integer         FTI_SCES if successful.

  This function takes care of the I/O specific actions needed before
  protected variables may be added to the checkpoint files.
 **/
/*-------------------------------------------------------------------------*/
void *FTI_InitDCPPosix(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec, FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt, FTIT_dataset* FTI_Data)
{

  FTI_Print("I/O mode: Posix.", FTI_IDCP);

  char fn[FTI_BUFS];

  WriteDCPPosixInfo_t *write_DCPinfo = (WriteDCPPosixInfo_t*) malloc (sizeof(WriteDCPPosixInfo_t));
  WritePosixInfo_t *write_info = &(write_DCPinfo->write_info); 
  write_DCPinfo->FTI_Exec = FTI_Exec;
  write_DCPinfo->FTI_Conf = FTI_Conf;
  write_DCPinfo->FTI_Ckpt = FTI_Ckpt;
  write_DCPinfo->FTI_Topo = FTI_Topo;
  write_DCPinfo->FTI_Data = FTI_Data;
  write_DCPinfo->layerSize = 0;


  FTI_Exec->dcpInfoPosix.dcpSize = 0;
  FTI_Exec->dcpInfoPosix.dataSize = 0;

  // dcpFileId increments every dcpStackSize checkpoints.
  int dcpFileId = FTI_Exec->dcpInfoPosix.Counter / FTI_Conf->dcpInfoPosix.StackSize;

  // dcpLayer corresponds to the additional layers towards the base layer.
  int dcpLayer = FTI_Exec->dcpInfoPosix.Counter % FTI_Conf->dcpInfoPosix.StackSize;
  FTI_Exec->dcpInfoPosix.currentCounter = dcpLayer;

  int i = 0;
  // if first layer, make sure that we write all data by setting hashdatasize = 0
  if( dcpLayer == 0 ) {
    for(; i<FTI_Exec->nbVar; i++){
      FTI_Data[i].dcpInfoPosix.hashDataSize = 0;
    }
  }

  for(i = 0; i<FTI_Exec->nbVar; i++) {
    if (FTI_Data[i].isDevicePtr){
      FTI_MD5GPU(&FTI_Data[i]);
    }
    else{
      FTI_MD5CPU(&FTI_Data[i]);
    }
    char str[FTI_BUFS];
  }
  FTI_startMD5();

  snprintf( FTI_Exec->meta[0].ckptFile, FTI_BUFS, "dcp-id%d-rank%d.fti", dcpFileId, FTI_Topo->myRank );
  int level = FTI_Exec->ckptLvel;
  if (level == 4 && FTI_Ckpt[4].isInline) { //If inline L4 save directly to global directory
    snprintf( fn, FTI_BUFS, "%s/%s", FTI_Ckpt[4].dcpDir, FTI_Exec->meta[0].ckptFile );
  } else {
    snprintf( fn, FTI_BUFS, "%s/%s", FTI_Ckpt[1].dcpDir, FTI_Exec->meta[0].ckptFile );
  }
  if( dcpLayer == 0 ) 
    write_info->flag = 'w';
  else 
    write_info->flag = 'a';

  FTI_PosixOpen(fn, write_info);

  if( dcpLayer == 0 ) FTI_Exec->dcpInfoPosix.FileSize = 0;

  // write constant meta data in the beginning of file
  // - blocksize
  // - stacksize
  // write actual amount of variables at the beginning of each layer
  return write_DCPinfo;
}




/*-------------------------------------------------------------------------*/
/**
  @brief      Writes dataset into dCP ckpt file using POSIX.
  @param      FTI_Data          Dataset metadata for a specific variable.
  @param      file descrriptor  FIle descriptor.
  @return     integer           FTI_SCES if successful.
 **/
/*-------------------------------------------------------------------------*/
int FTI_WritePosixDCPData(FTIT_dataset *FTI_DataVar, void *fd){

  // dcpLayer corresponds to the additional layers towards the base layer.
  WriteDCPPosixInfo_t *write_DCPinfo = (WriteDCPPosixInfo_t *) fd;
  WritePosixInfo_t *write_info = &(write_DCPinfo->write_info); 
  FTIT_configuration *FTI_Conf = write_DCPinfo -> FTI_Conf; 
  FTIT_execution *FTI_Exec = write_DCPinfo->FTI_Exec;

  int dcpLayer = FTI_Exec->dcpInfoPosix.Counter % FTI_Conf->dcpInfoPosix.StackSize;
  char errstr[FTI_BUFS];
  size_t bytes =0;
  long varId = FTI_DataVar->id;

  FTI_Exec->dcpInfoPosix.dataSize += FTI_DataVar->size;
  unsigned long dataSize = FTI_DataVar->size;
  unsigned long nbHashes = dataSize/FTI_Conf->dcpInfoPosix.BlockSize + (bool)(dataSize%FTI_Conf->dcpInfoPosix.BlockSize);

  if( dataSize > (MAX_BLOCK_IDX*FTI_Conf->dcpInfoPosix.BlockSize) ) {
    snprintf( errstr, FTI_BUFS, "overflow in size of dataset with id: %d (datasize: %lu > MAX_DATA_SIZE: %lu)", 
        FTI_DataVar->id, dataSize, ((unsigned long)MAX_BLOCK_IDX)*((unsigned long)FTI_Conf->dcpInfoPosix.BlockSize) );
    FTI_Print( errstr, FTI_EROR );
    return FTI_NSCS;
  }
  if( varId > MAX_VAR_ID ) {
    snprintf( errstr, FTI_BUFS, "overflow in ID (id: %d > MAX_ID: %d)!", FTI_DataVar->id, (int)MAX_VAR_ID );
    FTI_Print( errstr, FTI_EROR );
    return FTI_NSCS;
  }

  // create meta data buffer

  if( dcpLayer == 0 ) {
    memset(FTI_DataVar->storeFileLocations,0, sizeof(size_t)*nbHashes);
  }

  unsigned long pos = 0;

  FTIT_data_prefetch prefetcher;
  size_t totalBytes = 0;
  unsigned char * ptr;
#ifdef GPUSUPPORT    
  prefetcher.fetchSize = ((FTI_Conf->cHostBufSize) / FTI_Conf->dcpInfoPosix.BlockSize ) * FTI_Conf->dcpInfoPosix.BlockSize;
#else
  prefetcher.fetchSize =  FTI_DataVar->size;
#endif
  prefetcher.totalBytesToFetch = FTI_DataVar->size;
  prefetcher.isDevice = FTI_DataVar->isDevicePtr;

  if ( prefetcher.isDevice ){ 
    prefetcher.dptr = FTI_DataVar->devicePtr;
  }
  else{
    prefetcher.dptr = FTI_DataVar->ptr;
  }



  FTI_InitPrefetcher(&prefetcher);

  if ( FTI_Try(FTI_getPrefetchedData ( &prefetcher, &totalBytes, &ptr), " Fetching Next Memory block from memory") != FTI_SCES ){
    return FTI_NSCS;
  }
  size_t offset = 0;
  size_t startFilePos= ftell(write_info->f);

  startCount("waitIntegrity");
  FTI_waitVariable(FTI_DataVar->threadIndex);
  stopCount("waitIntegrity");

  while ( ptr ){
    pos = 0;
    unsigned char *startWritePtr = NULL;
    size_t bytesToCommit = 0;
    bool success;
    while( pos < totalBytes ) {
      // hash index
      unsigned int blockId = offset/FTI_Conf->dcpInfoPosix.BlockSize;
      unsigned int hashIdx = blockId*FTI_Conf->dcpInfoPosix.digestWidth;
      unsigned int chunkSize = ( (dataSize-offset) < FTI_Conf->dcpInfoPosix.BlockSize ) ? dataSize-offset : FTI_Conf->dcpInfoPosix.BlockSize;
      unsigned int dcpChunkSize = chunkSize;
      bool commitBlock = false;
      if( offset < FTI_DataVar->dcpInfoPosix.hashDataSize ) {
        int tmp = memcmp( &(FTI_DataVar->dcpInfoPosix.currentHashArray[hashIdx]), &(FTI_DataVar->dcpInfoPosix.oldHashArray[hashIdx]), FTI_Conf->dcpInfoPosix.digestWidth );
        if ( tmp != 0 ){
          FTI_DataVar->storeFileLocations[blockId] = startFilePos;
          startFilePos += chunkSize;
          if ( bytesToCommit == 0 )
            startWritePtr = ptr;
          bytesToCommit += chunkSize;
        }
        else{
          if (bytesToCommit != 0 )
            commitBlock = true;
        }
      } 
      else {
        if (bytesToCommit == 0)
          startWritePtr = ptr;
        bytesToCommit += chunkSize;
        FTI_DataVar->storeFileLocations[blockId] = startFilePos;
        startFilePos += chunkSize;
      }

      success = true;

      if( commitBlock ) {
        FWRITE(FTI_NSCS, success,startWritePtr, bytesToCommit,1,write_info->f,"",NULL);
        FTI_Exec->dcpInfoPosix.FileSize += bytesToCommit;
        write_DCPinfo->layerSize += bytesToCommit;
        FTI_Exec->dcpInfoPosix.dcpSize += bytesToCommit;
        bytesToCommit = 0;
      }

      offset += dcpChunkSize*success;
      pos += dcpChunkSize*success;
      ptr = ptr + dcpChunkSize ; //chunkSize*success;
    }

    success = true;

    if ( bytesToCommit ){
      FWRITE(FTI_NSCS, success,startWritePtr, bytesToCommit,1,write_info->f,"",NULL);
      FTI_Exec->dcpInfoPosix.FileSize += bytesToCommit;
      write_DCPinfo->layerSize += bytesToCommit;
      FTI_Exec->dcpInfoPosix.dcpSize += bytesToCommit;
      bytesToCommit = 0;
    }

    if ( FTI_Try(FTI_getPrefetchedData ( &prefetcher, &totalBytes, &ptr), " Fetching Next Memory block from memory") != FTI_SCES ){
      return FTI_NSCS;
    }

  }
  FTI_DataVar->dcpInfoPosix.hashDataSize = dataSize;



  return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Finalizes for dCP POSIX I/O.
  @param      fileDesc  file descriptor for dcp POSIX.
  @return     integer         FTI_SCES if successful.

  This function takes care of the I/O specific actions needed to
  finalize iCP.
 **/
/*-------------------------------------------------------------------------*/


int FTI_AsyncDCPClose(void *fileDesc)
{
  WriteDCPPosixInfo_t *write_dcpInfo = (WriteDCPPosixInfo_t *) fileDesc;
  FTIT_execution *FTI_Exec = write_dcpInfo->FTI_Exec;
  FTIT_configuration *FTI_Conf = write_dcpInfo->FTI_Conf;
  FTIT_checkpoint *FTI_Ckpt = write_dcpInfo->FTI_Ckpt;
  FTIT_topology *FTI_Topo = write_dcpInfo->FTI_Topo;
  FTIT_dataset *FTI_Data = write_dcpInfo->FTI_Data;
  int i;
  size_t bytes; 
  size_t sizeOfMeta = 0;
  FWRITE(FTI_NSCS, bytes , &(FTI_Exec->nbVar),sizeof(int), 1,write_dcpInfo->write_info.f, "", NULL);
  sizeOfMeta+= sizeof(int);

  unsigned char *hashes = (unsigned char *) malloc (sizeof(unsigned char)*FTI_Exec->nbVar*MD5_DIGEST_LENGTH);

  for ( i = 0; i < FTI_Exec->nbVar; i++){
    FWRITE(FTI_NSCS, bytes, &(FTI_Data[i].id),sizeof(int), 1,write_dcpInfo->write_info.f, "", NULL);
    sizeOfMeta+=sizeof(int);
    FWRITE(FTI_NSCS, bytes , &(FTI_Data[i].size),sizeof(long), 1,write_dcpInfo->write_info.f, "", NULL);

    sizeOfMeta+=sizeof(long);
    unsigned long nbHashes = FTI_Data[i].size /FTI_Conf->dcpInfoPosix.BlockSize + (bool)(FTI_Data[i].size %FTI_Conf->dcpInfoPosix.BlockSize);
    FWRITE(FTI_NSCS, bytes, FTI_Data[i].storeFileLocations,sizeof(size_t), nbHashes,write_dcpInfo->write_info.f, "", NULL);
    sizeOfMeta+= (sizeof(size_t)*nbHashes);
    MD5(FTI_Data[i].dcpInfoPosix.currentHashArray, nbHashes*MD5_DIGEST_LENGTH, &hashes[i*MD5_DIGEST_LENGTH]); 
  }
  unsigned char finalHash[MD5_DIGEST_LENGTH];

  MD5(hashes,FTI_Exec->nbVar*MD5_DIGEST_LENGTH, finalHash);
  FWRITE(FTI_NSCS, bytes, finalHash ,sizeof(char), MD5_DIGEST_LENGTH ,write_dcpInfo->write_info.f, "", NULL);

  char errstr[FTI_BUFS];

  int dcpFileId = FTI_Exec->dcpInfoPosix.Counter / FTI_Conf->dcpInfoPosix.StackSize;

  int dcpLayer = FTI_Exec->dcpInfoPosix.Counter % FTI_Conf->dcpInfoPosix.StackSize;

  FTI_PosixSync(&(write_dcpInfo->write_info));
  FTI_PosixClose(&(write_dcpInfo->write_info));

  // create final dcp layer hash
  unsigned char LayerHash[MD5_DIGEST_LENGTH];
  MD5_Final( LayerHash, &(write_dcpInfo->write_info.integrity) );
  FTI_Exec->dcpInfoPosix.Counter++;
  if( (dcpLayer == 0) ) {
    char ofn[512];
    snprintf( ofn, FTI_BUFS, "%s/dcp-id%d-rank%d.fti", FTI_Ckpt[FTI_Exec->ckptLvel].dcpDir, dcpFileId-1, FTI_Topo->myRank );
    if( (remove(ofn) < 0) && (errno != ENOENT) ) {
      snprintf(errstr, FTI_BUFS, "cannot delete file '%s'", ofn );
      FTI_Print( errstr, FTI_WARN ); 
    }
  }

  for ( i = 0; i < FTI_Exec->nbVar; i++){
    FTIT_dataset *FTI_DataVar = &FTI_Data[i];
    unsigned char *tmp = FTI_DataVar->dcpInfoPosix.currentHashArray;
    FTI_DataVar->dcpInfoPosix.currentHashArray = FTI_DataVar->dcpInfoPosix.oldHashArray;
    FTI_DataVar->dcpInfoPosix.oldHashArray = tmp;
  }

  return FTI_SCES;
}

int FTI_PosixDCPClose(void *fileDesc)
{
  WriteDCPPosixInfo_t *write_dcpInfo = (WriteDCPPosixInfo_t *) fileDesc;
  FTIT_execution *FTI_Exec = write_dcpInfo->FTI_Exec;
  FTIT_configuration *FTI_Conf = write_dcpInfo->FTI_Conf;
  FTIT_checkpoint *FTI_Ckpt = write_dcpInfo->FTI_Ckpt;
  FTIT_topology *FTI_Topo = write_dcpInfo->FTI_Topo;
  FTIT_dataset *FTI_Data = write_dcpInfo->FTI_Data;
  size_t startOfMeta = ftell(write_dcpInfo->write_info.f);
  int dcpLayer = FTI_Exec->dcpInfoPosix.Counter % FTI_Conf->dcpInfoPosix.StackSize;
  FTI_Exec->dcpInfoPosix.LayerMetaPos[dcpLayer] = startOfMeta;
  FTI_CLOSE_ASYNC(fileDesc);
  return FTI_SCES;
}



/*-------------------------------------------------------------------------*/
/**
  @brief      It loads the checkpoint data for dcpPosix.
  @return     integer         FTI_SCES if successful.

  dCP POSIX implementation of FTI_Recover().
 **/
/*-------------------------------------------------------------------------*/
int FTI_RecoverDcpPosix
( 
 FTIT_configuration* FTI_Conf, 
 FTIT_execution* FTI_Exec, 
 FTIT_checkpoint* FTI_Ckpt, 
 FTIT_dataset* FTI_Data 
 )
{
  FTIT_execution* exec = FTI_Exec;
  FTIT_configuration* conf = FTI_Conf; 
  char fn[FTI_BUFS];
  int LastLayer = exec->dcpInfoPosix.nbLayerReco;


  snprintf( fn, FTI_BUFS, "%s/%s", FTI_Ckpt[FTI_Exec->ckptLvel].dcpDir, FTI_Exec->meta[FTI_Exec->ckptLvel].ckptFile );
  FILE *fd = fopen(fn, "rb");

  if (fd == NULL) {
    char str[FTI_BUFS];
    sprintf(str, "FTI failed to open file %s to recover", fn);
    FTI_Print(str, FTI_WARN);
    return FTI_NSCS;
  }

  size_t filePos = exec->dcpInfoPosix.LayerMetaPos[LastLayer];
  fseek(fd, filePos, SEEK_SET);
  int nbVars ;
  size_t bytes;
  FREAD(FTI_NSCS, bytes,&nbVars, sizeof(int),1,fd, "", NULL);
  if (nbVars != FTI_Exec->nbVar ){
    FTI_Print("Number of Variables mismatch",FTI_EROR);
    fclose(fd);
  }

  FTI_startMD5();
  int j;
  for ( j = 0; j < nbVars; j++){
    int varId = 0;
    size_t varSize;
    FREAD(FTI_NSCS, bytes,&varId, sizeof(int),1,fd, "", NULL);
    int newPosistion = FTI_DataGetIdx( varId, FTI_Exec, FTI_Data ); 
    FREAD(FTI_NSCS, bytes,&varSize, sizeof(long),1,fd, "", NULL);

    if (varSize != FTI_Data[newPosistion].size ){
      FTI_Print("Size of variable mismatch", FTI_EROR);
      return FTI_NSCS;
    }
    long nbHashes = varSize /conf->dcpInfoPosix.BlockSize + (bool)(varSize %conf->dcpInfoPosix.BlockSize);
    FREAD(FTI_NSCS, bytes,FTI_Data[newPosistion].storeFileLocations, sizeof(size_t),nbHashes,fd, "", NULL);
    long curPos = ftell(fd);
    if ( FTI_Data[newPosistion].isDevicePtr ){
      FTI_ReadRandomFileToGPU(fd, FTI_Data[newPosistion].devicePtr, FTI_Data[newPosistion].storeFileLocations, nbHashes, varSize);
      FTI_MD5GPU(&FTI_Data[newPosistion]);
    }
    else{
      size_t remainingBytes = varSize;
      size_t k;
      char *buff = FTI_Data[newPosistion].ptr;
      size_t *locations = FTI_Data[newPosistion].storeFileLocations;
      startCount("ReadFile");
      for ( k= 0; k < nbHashes; k++){
        fseek(fd, locations[k], SEEK_SET);
        size_t dataToRead = MIN(16384,remainingBytes);
        FREAD(FTI_NSCS, bytes,&buff[k*16384], sizeof(char), dataToRead ,fd, "", NULL);
        remainingBytes -= dataToRead;
      }
      stopCount("ReadFile");
      FTI_MD5CPU(&FTI_Data[newPosistion]);
    }

    FTI_Data[newPosistion].dcpInfoPosix.hashDataSize = varSize;
    fseek(fd, curPos, SEEK_SET);
  }
  int i;
  for (i = 0; i < FTI_Exec->nbVar; i++){
    FTIT_dataset *FTI_DataVar = &FTI_Data[i];
    FTI_waitVariable(FTI_DataVar->threadIndex);
    unsigned char *tmp = FTI_DataVar->dcpInfoPosix.currentHashArray;
    FTI_DataVar->dcpInfoPosix.currentHashArray = FTI_DataVar->dcpInfoPosix.oldHashArray;
    FTI_DataVar->dcpInfoPosix.oldHashArray = tmp;
  }
  startCount("Integrity");
  FTI_SyncMD5();
  stopCount("Integrity");
  FTI_SleepWorker();
  FTI_Exec->dcpInfoPosix.Counter = LastLayer+1; 

  return FTI_SCES;

}

/*-------------------------------------------------------------------------*/
/**
  @brief      Recovers the given variable for dcpPosix
  @param      id              Variable to recover
  @return     int             FTI_SCES if successful.

  dCP POSIX implementation of FTI_RecoverVar().
 **/
/*-------------------------------------------------------------------------*/
int FTI_RecoverVarDcpPosix
( 
 FTIT_configuration* FTI_Conf, 
 FTIT_execution* FTI_Exec, 
 FTIT_checkpoint* FTI_Ckpt, 
 FTIT_dataset* FTI_Data,
 int id
 )

{
#ifdef ZERO
  unsigned long blockSize;
  unsigned int stackSize;
  int nbVarLayer;
  int ckptID;

  char errstr[FTI_BUFS];
  char fn[FTI_BUFS];

  snprintf( fn, FTI_BUFS, "%s/%s", FTI_Ckpt[FTI_Exec->ckptLvel].dcpDir, FTI_Exec->meta[4].ckptFile );

  // read base part of file
  FILE* fd = fopen( fn, "rb" );
  fread( &blockSize, sizeof(unsigned long), 1, fd );
  if(ferror(fd)) {
    snprintf( errstr, FTI_BUFS, "unable to read in file %s", fn );
    FTI_Print( errstr, FTI_EROR );
    return FTI_NSCS;
  }
  fread( &stackSize, sizeof(unsigned int), 1, fd );
  if(ferror(fd)) {
    snprintf( errstr, FTI_BUFS, "unable to read in file %s", fn );
    FTI_Print( errstr, FTI_EROR );
    return FTI_NSCS;
  }

  // check if settings are correct. If not correct them
  if( blockSize != FTI_Conf->dcpInfoPosix.BlockSize )
  {
    char str[FTI_BUFS];
    snprintf( str, FTI_BUFS, "dCP blocksize differ between configuration settings ('%lu') and checkpoint file ('%lu')", FTI_Conf->dcpInfoPosix.BlockSize, blockSize );
    FTI_Print( str, FTI_WARN );
    return FTI_NREC;
  }
  if( stackSize != FTI_Conf->dcpInfoPosix.StackSize )
  {
    char str[FTI_BUFS];
    snprintf( str, FTI_BUFS, "dCP stacksize differ between configuration settings ('%u') and checkpoint file ('%u')", FTI_Conf->dcpInfoPosix.StackSize, stackSize );
    FTI_Print( str, FTI_WARN );
    return FTI_NREC;
  }


  void *buffer = (void*) malloc( blockSize ); 
  if( !buffer ) {
    FTI_Print("unable to allocate memory!", FTI_EROR);
    return FTI_NSCS;
  }

  int i;

  // treat Layer 0 first
  fread( &ckptID, 1, sizeof(int), fd );
  if(ferror(fd)) {
    snprintf( errstr, FTI_BUFS, "unable to read in file %s", fn );
    FTI_Print( errstr, FTI_EROR );
    return FTI_NSCS;
  }
  fread( &nbVarLayer, 1, sizeof(int), fd );
  if(ferror(fd)) {
    snprintf( errstr, FTI_BUFS, "unable to read in file %s", fn );
    FTI_Print( errstr, FTI_EROR );
    return FTI_NSCS;
  }
  for(i=0; i<nbVarLayer; i++) {
    unsigned int varId;
    unsigned long locDataSize;
    fread( &varId, sizeof(int), 1, fd );
    if(ferror(fd)) {
      snprintf( errstr, FTI_BUFS, "unable to read in file %s", fn );
      FTI_Print( errstr, FTI_EROR );
      return FTI_NSCS;
    }
    fread( &locDataSize, sizeof(unsigned long), 1, fd );
    if(ferror(fd)) {
      snprintf( errstr, FTI_BUFS, "unable to read in file %s", fn );
      FTI_Print( errstr, FTI_EROR );
      return FTI_NSCS;
    }
    // if requested id load else skip dataSize
    if( varId == id ) {
      int idx = FTI_DataGetIdx(varId, FTI_Exec, FTI_Data);
      if( idx < 0 ) {
        snprintf(errstr, FTI_BUFS, "id '%d' does not exist!", varId);
        FTI_Print( errstr, FTI_EROR );
        return FTI_NSCS;
      }
      fread( FTI_Data[idx].ptr, locDataSize, 1, fd );
      if(ferror(fd)) {
        snprintf( errstr, FTI_BUFS, "unable to read in file %s", fn );
        FTI_Print( errstr, FTI_EROR );
        return FTI_NSCS;
      }

      int overflow;
      if( (overflow=locDataSize%blockSize) != 0 ) {
        fread( buffer, blockSize - overflow, 1, fd );
        if(ferror(fd)) {
          snprintf( errstr, FTI_BUFS, "unable to read in file %s", fn );
          FTI_Print( errstr, FTI_EROR );
          return FTI_NSCS;
        }
      }
    } else {
      unsigned long skip = ( locDataSize%blockSize == 0 ) ? locDataSize : (locDataSize/blockSize + 1)*blockSize;
      if( fseek( fd, skip, SEEK_CUR ) == -1 ) {
        snprintf( errstr, FTI_BUFS, "unable to seek in file %s", fn );
        FTI_Print( errstr, FTI_EROR );
        return FTI_NSCS;
      }
    }
  }


  unsigned long offset;

  blockMetaInfo_t blockMeta;
  unsigned char *block = (unsigned char*) malloc( blockSize );
  if( !block ) {
    FTI_Print("unable to allocate memory!", FTI_EROR);
    return FTI_NSCS;
  }

  int nbLayer = FTI_Exec->dcpInfoPosix.nbLayerReco;
  for( i=1; i<nbLayer; i++) {

    unsigned long pos = 0;
    pos += fread( &ckptID, 1, sizeof(int), fd );
    if(ferror(fd)) {
      snprintf( errstr, FTI_BUFS, "unable to read in file %s", fn );
      FTI_Print( errstr, FTI_EROR );
      return FTI_NSCS;
    }
    pos += fread( &nbVarLayer, 1, sizeof(int), fd );
    if(ferror(fd)) {
      snprintf( errstr, FTI_BUFS, "unable to read in file %s", fn );
      FTI_Print( errstr, FTI_EROR );
      return FTI_NSCS;
    }

    while( pos < FTI_Exec->dcpInfoPosix.LayerSize[i] ) {

      fread( &blockMeta, 1, 6, fd );
      if(ferror(fd)) {
        snprintf( errstr, FTI_BUFS, "unable to read in file %s", fn );
        FTI_Print( errstr, FTI_EROR );
        return FTI_NSCS;
      }
      if( blockMeta.varId == id ) {
        int idx = FTI_DataGetIdx(blockMeta.varId, FTI_Exec, FTI_Data);
        if( idx < 0 ) {
          snprintf(errstr, FTI_BUFS, "id '%d' does not exist!", blockMeta.varId);
          FTI_Print( errstr, FTI_EROR );
          return FTI_NSCS;
        }


        offset = blockMeta.blockId * blockSize;
        void* ptr = FTI_Data[idx].ptr + offset;
        unsigned int chunkSize = ( (FTI_Data[idx].size-offset) < blockSize ) ? FTI_Data[idx].size-offset : blockSize; 

        fread( ptr, 1, chunkSize, fd );
        if(ferror(fd)) {
          snprintf( errstr, FTI_BUFS, "unable to read in file %s", fn );
          FTI_Print( errstr, FTI_EROR );
          return FTI_NSCS;
        }
        fread( buffer, 1, blockSize - chunkSize, fd ); 
        if(ferror(fd)) {
          snprintf( errstr, FTI_BUFS, "unable to read in file %s", fn );
          FTI_Print( errstr, FTI_EROR );
          return FTI_NSCS;
        }

      } else {
        if( fseek( fd, blockSize, SEEK_CUR ) == -1 ) {
          snprintf( errstr, FTI_BUFS, "unable to seek in file %s", fn );
          FTI_Print( errstr, FTI_EROR );
          return FTI_NSCS;
        }
      }
      pos += (blockSize+6);
    }

  }


  i = FTI_DataGetIdx( id, FTI_Exec, FTI_Data );
  FTIT_dataset *FTI_DataVar = &FTI_Data[i];
  FTIT_data_prefetch prefetcher;
  size_t totalBytes = 0;
  unsigned char * ptr = NULL,*startPtr = NULL;

#ifdef GPUSUPPORT    
  prefetcher.fetchSize = ((FTI_Conf->cHostBufSize) / FTI_Conf->dcpInfoPosix.BlockSize ) * FTI_Conf->dcpInfoPosix.BlockSize;
#else
  prefetcher.fetchSize =  FTI_DataVar->size;
#endif
  prefetcher.totalBytesToFetch = FTI_DataVar->size;
  prefetcher.isDevice = FTI_DataVar->isDevicePtr;

  if ( prefetcher.isDevice ){ 
    prefetcher.dptr = FTI_DataVar->devicePtr;
  }
  else{
    prefetcher.dptr = FTI_DataVar->ptr;
  }

  FTI_InitPrefetcher(&prefetcher);
  if ( FTI_Try(FTI_getPrefetchedData ( &prefetcher, &totalBytes, &startPtr), " Fetching Next Memory block from memory") != FTI_SCES ){
    return FTI_NSCS;
  }

  unsigned long nbBlocks = (FTI_DataVar->size % blockSize) ? FTI_DataVar->size/blockSize + 1 : FTI_DataVar->size/blockSize;
  FTI_DataVar->dcpInfoPosix.hashDataSize = FTI_DataVar->size;
  int j =0 ;
  while (startPtr){
    ptr = startPtr;
    int currentBlocks = (totalBytes % blockSize) ?  totalBytes/blockSize + 1 : totalBytes/blockSize;
    int k;
    for ( k = 0 ; k < currentBlocks && j<nbBlocks-1; k++){
      unsigned long hashIdx = j*MD5_DIGEST_LENGTH;
      FTI_Conf->dcpInfoPosix.hashFunc( ptr, blockSize, &FTI_DataVar->dcpInfoPosix.oldHashArray[hashIdx] );
      ptr = ptr+blockSize;
      j++;
    }
    if ( FTI_Try(FTI_getPrefetchedData ( &prefetcher, &totalBytes, &startPtr ), " Fetching Next Memory block from memory") != FTI_SCES ){
      return FTI_NSCS;
    }
  }

  if( FTI_DataVar->size%blockSize ) {
    unsigned char* buffer = calloc( 1, blockSize );
    if( !buffer ) {
      FTI_Print("unable to allocate memory!", FTI_EROR);
      return FTI_NSCS;
    }
    unsigned long dataOffset = blockSize * (nbBlocks - 1);
    unsigned long dataSize = FTI_DataVar->size - dataOffset;
    memcpy( buffer, ptr , dataSize ); 
    FTI_Conf->dcpInfoPosix.hashFunc( buffer, blockSize, &FTI_DataVar->dcpInfoPosix.oldHashArray[(nbBlocks-1)*MD5_DIGEST_LENGTH] );
  }

  /*
  // create hasharray for id
  i = FTI_DataGetIdx( id, FTI_Exec, FTI_Data );
  unsigned long nbBlocks = (FTI_Data[i].size % blockSize) ? FTI_Data[i].size/blockSize + 1 : FTI_Data[i].size/blockSize;
  FTI_Data[i].dcpInfoPosix.hashDataSize = FTI_Data[i].size;

  int j;
  for(j=0; j<nbBlocks-1; j++) {
  unsigned long offset = j*blockSize;
  unsigned long hashIdx = j*MD5_DIGEST_LENGTH;
  MD5( FTI_Data[i].ptr+offset, blockSize, &FTI_Data[i].dcpInfoPosix.hashArray[hashIdx] );
  }
  if( FTI_Data[i].size%blockSize ) {
  unsigned char* buffer = calloc( 1, blockSize );
  if( !buffer ) {
  FTI_Print("unable to allocate memory!", FTI_EROR);
  return FTI_NSCS;
  }
  unsigned long dataOffset = blockSize * (nbBlocks - 1);
  unsigned long dataSize = FTI_Data[i].size - dataOffset;
  memcpy( buffer, FTI_Data[i].ptr + dataOffset, dataSize ); 
  MD5( buffer, blockSize, &FTI_Data[i].dcpInfoPosix.hashArray[(nbBlocks-1)*MD5_DIGEST_LENGTH] );
  free(buffer);
  }
  */
  free(buffer);
  fclose(fd);
#endif
  return FTI_SCES;

}

/*-------------------------------------------------------------------------*/
/**
  @brief      It checks if a file exist and that its size is 'correct'.
  @param      fn              The ckpt. file name to check.
  @param      fs              The ckpt. file size to check.
  @param      checksum        The file checksum to check.
  @return     integer         0 if file exists, 1 if not or wrong size.

  dCP POSIX implementation of FTI_CheckFile().
 **/
/*-------------------------------------------------------------------------*/
  int FTI_CheckFileDcpPosix
(
 char* fn, 
 long fs, 
 char* checksum
 )

{
  if (access(fn, F_OK) == 0) {
    struct stat fileStatus;
    if (stat(fn, &fileStatus) == 0) {
      if (strlen(checksum)) {
        int res = FTI_VerifyChecksumDcpPosix(fn);
        if (res != FTI_SCES) {
          return 1;
        }
        return 0;
      }
      return 0;
    }
    else {
      FTI_Print("Stat return wrong error code",FTI_WARN);
      return 1;
    }
  }
  else {
    char str[FTI_BUFS];
    sprintf(str, "Missing file: \"%s\"", fn);
    FTI_Print(str, FTI_WARN);
    return 1;
  }
}

void FTI_dcpMD5(unsigned char *dest, void *md5)
{
  FTI_SyncMD5();
}

/*-------------------------------------------------------------------------*/
/**
  @brief      It compares checksum of the checkpoint file.
  @param      fileName        Filename of the checkpoint.
  @param      checksumToCmp   Checksum to compare.
  @return     integer         FTI_SCES if successful.

  dCP POSIX implementation of FTI_VerifyChecksum().
 **/
/*-------------------------------------------------------------------------*/
int FTI_VerifyChecksumDcpPosix (  char* fileName  )
{
  FTIT_execution* exec = FTI_DcpPosixRecoverRuntimeInfo( DCP_POSIX_EXEC_TAG, NULL, NULL );
  FTIT_configuration* conf = FTI_DcpPosixRecoverRuntimeInfo( DCP_POSIX_CONF_TAG, NULL, NULL ); 
  char str[FTI_BUFS];
  int LastLayer = exec->dcpInfoPosix.nbLayerReco;

  int dcpFileId;
  int dummy;
  int counter;
  sscanf( exec->meta[exec->ckptLvel].ckptFile, "dcp-id%d-rank%d.fti", &dcpFileId, &dummy );
  counter = dcpFileId * conf->dcpInfoPosix.StackSize; 

  FILE *fd = fopen(fileName, "rb");
  if (fd == NULL) {
    sprintf(str, "FTI failed to open file %s to calculate checksum.", fileName);
    FTI_Print(str, FTI_WARN);
    return FTI_NSCS;
  }
  int i;
  for (i = LastLayer; i >= 0; i--){
    size_t filePos = exec->dcpInfoPosix.LayerMetaPos[i];
    fseek(fd, filePos, SEEK_SET);
    int nbVars ;
    size_t bytes;
    FREAD(FTI_NSCS, bytes,&nbVars, sizeof(int),1,fd, "", NULL);

    int j;

    unsigned char *hashes = (unsigned char *) malloc (sizeof(unsigned char)*nbVars*MD5_DIGEST_LENGTH);
    cudaStream_t stream;
    CUDA_ERROR_CHECK(cudaStreamCreate(&stream));
    for ( j = 0; j < nbVars; j++){
      int varId = 0;
      FREAD(FTI_NSCS, bytes,&varId, sizeof(int),1,fd, "", NULL);
      long varSize= 0;
      FREAD(FTI_NSCS, bytes,&varSize, sizeof(long),1,fd, "", NULL);
      long nbHashes = varSize /conf->dcpInfoPosix.BlockSize + (bool)(varSize %conf->dcpInfoPosix.BlockSize);
      size_t *filePos = (size_t *) malloc (sizeof(size_t) * nbHashes);
      FREAD(FTI_NSCS, bytes,filePos, sizeof(size_t),nbHashes,fd, "", NULL);
      unsigned char *buff = (unsigned char *) malloc (sizeof(char)*varSize);
      unsigned char *interMediateHashes;

      CUDA_ERROR_CHECK(cudaMallocManaged((void **) &interMediateHashes, sizeof(char)*nbHashes*MD5_DIGEST_LENGTH,cudaMemAttachGlobal));
      long curPos = ftell(fd);

      size_t remainingBytes = varSize;

      size_t hashesPerStep = 8192;
      size_t numSteps = nbHashes/(hashesPerStep) + (nbHashes%(hashesPerStep)>0);
      size_t k = 0;
      size_t l;
      for ( l = 0; l < numSteps; l++){
        size_t end = MIN(nbHashes, (l+1)*hashesPerStep);
        size_t start = k;
        size_t bytesRead = 0;

        for ( ; k < end; k++){
          fseek(fd, filePos[k], SEEK_SET);
          size_t dataToRead = MIN(16384,remainingBytes);
          FREAD(FTI_NSCS, bytes,&buff[k*16384], sizeof(char), dataToRead ,fd, "", NULL);
          remainingBytes -= dataToRead;
          bytesRead += dataToRead;
        }
        CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));
        md5Level(&buff[start*16384] , (MD5_u32plus*)&interMediateHashes[start*MD5_DIGEST_LENGTH], bytesRead, stream );
      }
      CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));
      fseek(fd, curPos, SEEK_SET);
      sprintf(str, "(%p %ld, %p)Hashes Per Step %d, Num Steps %ld, VarId %d VarSize %ld", interMediateHashes, nbHashes*MD5_DIGEST_LENGTH,&hashes[j*MD5_DIGEST_LENGTH], hashesPerStep, numSteps, varId, varSize);
      FTI_Print(str,FTI_DBUG);

      MD5(interMediateHashes,nbHashes*MD5_DIGEST_LENGTH,&hashes[j*MD5_DIGEST_LENGTH]);
      free(filePos);
      free(buff);
      CUDA_ERROR_CHECK(cudaFree(interMediateHashes));

      filePos = NULL;
      buff = NULL;
      interMediateHashes = NULL;

    }

    CUDA_ERROR_CHECK(cudaStreamDestroy(stream));

    char layerHash[MD5_DIGEST_LENGTH];
    FREAD(FTI_NSCS, bytes,layerHash, sizeof(char), 16 ,fd, "", NULL);
    char checkpointHash[MD5_DIGEST_STRING_LENGTH];
    char *tmp = FTI_GetHashHexStr( (unsigned char*) layerHash , conf->dcpInfoPosix.digestWidth, NULL );
    strcpy(checkpointHash, tmp);

    char finalHash[MD5_DIGEST_LENGTH];
    MD5((unsigned char*) hashes,nbVars*MD5_DIGEST_LENGTH,(unsigned char*) finalHash);

    tmp = FTI_GetHashHexStr((unsigned char*) finalHash , conf->dcpInfoPosix.digestWidth, NULL );

    free(hashes);
    hashes = NULL;

    if( strcmp( checkpointHash, tmp ) ) {
      sprintf(str,"hashes differ %s %s in Layer %ld",checkpointHash, tmp,i);
      FTI_Print(str,FTI_IDCP);
    }
    else{
      exec->dcpInfoPosix.nbLayerReco = i ;
      exec->dcpInfoPosix.Counter = counter + i ;
      exec->dcpInfoPosix.nbVarReco = nbVars;
      sprintf(str,"Successfull recovered from checkpoint (Layer %d)",i);
      FTI_Print(str,FTI_INFO);
      fclose(fd);
      return FTI_SCES;
    }
  }
  fclose(fd);
  return FTI_NSCS;
}

// HELPER FUNCTIONS

void* FTI_DcpPosixRecoverRuntimeInfo( int tag, void* exec_, void* conf_ ) {

  static void* exec = NULL;
  static void* conf = NULL;

  void* ret = NULL;

  switch( tag ) {
    case DCP_POSIX_EXEC_TAG:
      ret = exec;
      break;
    case DCP_POSIX_CONF_TAG:
      ret = conf;
      break;
    case DCP_POSIX_INIT_TAG:
      ret = NULL;
      exec = exec_;
      conf = conf_;
      break;
  }

  return ret;
}

// have the same for for MD5 and CRC32
unsigned char* CRC32( const unsigned char *d, unsigned long nBytes, unsigned char *hash )
{
  static unsigned char hash_[CRC32_DIGEST_LENGTH];
  if( hash == NULL ) {
    hash = hash_;
  }

#ifdef FTI_NOZLIB
  uint32_t digest = crc32( d, nBytes );
#else
  uint32_t digest = crc32( 0L, Z_NULL, 0 );
  digest = crc32( digest, d, nBytes );
#endif

  memcpy( hash, &digest, CRC32_DIGEST_LENGTH );

  return hash;
}

int FTI_DataGetIdx( int varId, FTIT_execution* FTI_Exec, FTIT_dataset* FTI_Data )
{
  int i=0;
  for(; i<FTI_Exec->nbVar; i++) {
    if(FTI_Data[i].id == varId) break;
  }
  if( i==FTI_Exec->nbVar ) {
    return -1;
  }
  return i;
}


