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
#include "posix-dcp.h"
#include "../api-cuda.h"
#include "cuda-md5/md5Opt.h"
#include <ieee754.h>
#include <math.h>

//PBDC-25

float converttoIeeeFlt(double value,unsigned int precision){
    union ieee754_float _f = {0};
    _f.f=value;
    int m=23-precision;
    struct {unsigned int mantissa:23;} a={0};
    a.mantissa=~((1<<m)-1);
    _f.ieee.mantissa &= a.mantissa;
    return _f.f;
}

double converttoIeeeDbl(double value,unsigned int precision){
    union ieee754_double _d = {0};
    _d.d=value;
    int m=52-precision;
    struct {unsigned int mantissa0:20; unsigned int mantissa1:32;} a={0};
    if(m>=32){
        a.mantissa0=~((1<<(m-32))-1);
        _d.ieee.mantissa0 &= a.mantissa0;
        _d.ieee.mantissa1=0;
    }
    else{
        a.mantissa1=~((1<<m)-1);
        _d.ieee.mantissa1 &= a.mantissa1;
    }
    return _d.d;
}

double * FTI_TruncateMantissa(void *block, uint64_t nBytes, FTIT_Datatype* type, unsigned int precision,int *nbValues,double *error){
    double errorSum=0;
    int nValues=0;
    int i;
    void *block_;
    block_ = malloc(nBytes);
    if(type->id == FTI_DBLE){
        for(i=0;i<nBytes/sizeof(double);i++){
            if(((double *)block)[i]!=0){
                ((double *)block_)[i]=converttoIeeeDbl(((double *)block)[i],precision);
                errorSum+=pow((((double *)block)[i]-((double *)block_)[i]),2);
                nValues++;
            }
            else
                ((double *)block_)[i]=((double *)block)[i];
        }
    }
    else if(type->id == FTI_SFLT){
        for(i=0;i<nBytes/sizeof(float);i++){
            if(((float *)block)[i]!=0){
                ((float *)block_)[i]=converttoIeeeFlt(((float *)block)[i],precision);
                errorSum+=pow((((float *)block)[i]-((float *)block_)[i]),2);
                nValues++;
            }
            else
                ((float *)block_)[i]=((float *)block)[i];
        }
    }
    memcpy(block, block_, nBytes);
    free(block_);

    *error=errorSum;
    *nbValues=nValues;
    return FTI_SCES;
}

/***************************************************************************/
/**
  @brief      Precision based DCP function.
  @param      FTI_Conf        Configuration metadata
  @param      FTI_Exec        Execution metadata
  @param      FTI_Data        Dataset metadata
  @param      block           Block of float or double values
  @param      nBytes          Size of block
  @param      hash            Pointer to hash            
  @return     integer         FTI_SCES if successful.

  This function performs precision based dcp operation if conditions are met
**/
/*-------------------------------------------------------------------------*/
int FTI_BlockHashDcp (FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec, 
  FTIT_dataset* FTI_Data, void *block, uint64_t nBytes, unsigned char *hash)
{
  void* block_;
  bool allocBlock = false;
  int nVals;
  double error;
  if ( FTI_Conf->pbdcpEnabled && (FTI_Exec->ckptLvel == FTI_Exec->isPbdcp)) {
    if ( (FTI_Data->type != FTI_GetType(FTI_DBLE)) && (FTI_Data->type != FTI_GetType(FTI_SFLT)) ) {
      FTI_Print ( "Only float and double types supported in PBDCP", FTI_WARN );
      block_ = block;
    } else {
      block_ = malloc(nBytes);
      memcpy(block_, block, nBytes);
      allocBlock = true;
      FTI_TruncateMantissa ( block_, nBytes, FTI_Data->type, FTI_Conf->pbdcp_precision ,&nVals,&error);
      FTI_Exec->dcpInfoPosix.errorSum += error;
      FTI_Exec->dcpInfoPosix.nbValues += nVals;
    }
  } else {
    block_ = block;
  }
  
  FTI_Conf->dcpInfoPosix.hashFunc(block_, nBytes, hash);
  
  if (allocBlock) free(block_);
  return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Returns the file positio.
  @param      fileDesc        The file descriptor.
  @return     integer         The position in the file.
 **/
/*-------------------------------------------------------------------------*/
size_t FTI_GetDCPPosixFilePos(void *fileDesc) {
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
void *FTI_InitDCPPosix(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
 FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt, FTIT_keymap* FTI_Data) {
    FTI_Print("I/O mode: Posix.", FTI_DBUG);

    char fn[FTI_BUFS];
    size_t bytes;

    WriteDCPPosixInfo_t *write_DCPinfo = (WriteDCPPosixInfo_t*)
    malloc(sizeof(WriteDCPPosixInfo_t));
    WritePosixInfo_t *write_info = &(write_DCPinfo->write_info);
    write_DCPinfo->FTI_Exec = FTI_Exec;
    write_DCPinfo->FTI_Conf = FTI_Conf;
    write_DCPinfo->FTI_Ckpt = FTI_Ckpt;
    write_DCPinfo->FTI_Topo = FTI_Topo;
    write_DCPinfo->layerSize = 0;


    FTI_Exec->dcpInfoPosix.dcpSize = 0;
    FTI_Exec->dcpInfoPosix.dataSize = 0;

    // dcpFileId increments every dcpStackSize checkpoints.
    int dcpFileId = FTI_Exec->dcpInfoPosix.Counter /
     FTI_Conf->dcpInfoPosix.StackSize;

    // dcpLayer corresponds to the additional layers towards the base layer.
    int dcpLayer = FTI_Exec->dcpInfoPosix.Counter %
     FTI_Conf->dcpInfoPosix.StackSize;

    // if first layer, make sure that we write all
    // data by setting hashdatasize = 0
    if (dcpLayer == 0) {
        FTIT_dataset* data;
        if ((FTI_Data->data(&data, FTI_Exec->nbVar) != FTI_SCES) || !data)
            return write_DCPinfo;

        int i = 0; for (; i < FTI_Exec->nbVar; i++) {
            //            free(FTI_Data[i].dcpInfoPosix.hashArray);
            //            FTI_Data[i].dcpInfoPosix.hashArray = NULL;
            data[i].dcpInfoPosix.hashDataSize = 0;
        }
    }

    snprintf(FTI_Exec->ckptMeta.ckptFile, FTI_BUFS, "dcp-id%d-rank%d.fti",
     dcpFileId, FTI_Topo->myRank);
    if (FTI_Ckpt[4].isInline) {
        // If inline L4 save directly to global directory
        snprintf(fn, FTI_BUFS, "%s/%s", FTI_Ckpt[4].dcpDir,
         FTI_Exec->ckptMeta.ckptFile);
    } else {
        snprintf(fn, FTI_BUFS, "%s/%s", FTI_Ckpt[1].dcpDir,
         FTI_Exec->ckptMeta.ckptFile);
    }

    if (dcpLayer == 0)
        write_info->flag = 'w';
    else
        write_info->flag = 'a';

    FTI_PosixOpen(fn, write_info);

    if (dcpLayer == 0) FTI_Exec->dcpInfoPosix.FileSize = 0;

    // write constant meta data in the beginning of file
    // - blocksize
    // - stacksize
    if (dcpLayer == 0) {
        DFTI_EH_FWRITE(NULL, bytes, &FTI_Conf->dcpInfoPosix.BlockSize,
         sizeof(uint32_t), 1, write_info->f, "p", write_info);
        DFTI_EH_FWRITE(NULL, bytes, &FTI_Conf->dcpInfoPosix.StackSize,
         sizeof(unsigned int), 1, write_info->f, "p", write_info);
        FTI_Exec->dcpInfoPosix.FileSize += sizeof(uint32_t) +
         sizeof(unsigned int);
        write_DCPinfo->layerSize += sizeof(uint32_t) +
         sizeof(unsigned int);
    }

    // write actual amount of variables at the beginning of each layer
    DFTI_EH_FWRITE(NULL, bytes, &FTI_Exec->ckptId, sizeof(int), 1, write_info->f,
     "p", write_info);
    DFTI_EH_FWRITE(NULL, bytes, &FTI_Exec->nbVar, sizeof(int), 1, write_info->f,
     "p", write_info);
    // + sizeof(unsigned int);
    FTI_Exec->dcpInfoPosix.FileSize += 2*sizeof(int);
    // + sizeof(unsigned int);
    write_DCPinfo->layerSize += 2*sizeof(int);

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
int FTI_WritePosixDCPData(FTIT_dataset *data, void *fd) {
    // dcpLayer corresponds to the additional layers towards the base layer.
    WriteDCPPosixInfo_t *write_DCPinfo = (WriteDCPPosixInfo_t *) fd;
    WritePosixInfo_t *write_info = &(write_DCPinfo->write_info);
    FTIT_configuration *FTI_Conf = write_DCPinfo -> FTI_Conf;
    FTIT_execution *FTI_Exec = write_DCPinfo->FTI_Exec;

    int dcpLayer = FTI_Exec->dcpInfoPosix.Counter %
     FTI_Conf->dcpInfoPosix.StackSize;
    char errstr[FTI_BUFS];
    unsigned char * block = (unsigned char*)malloc
    (FTI_Conf->dcpInfoPosix.BlockSize);
    size_t bytes;
    int32_t varId = data->id;

    FTI_Exec->dcpInfoPosix.dataSize += data->size;
    uint32_t dataSize = data->size;
    // uint32_t nbHashes = dataSize/FTI_Conf->dcpInfoPosix.BlockSize +
    // (bool)(dataSize%FTI_Conf->dcpInfoPosix.BlockSize);

    if (dataSize > (MAX_BLOCK_IDX*FTI_Conf->dcpInfoPosix.BlockSize)) {
        snprintf(errstr, FTI_BUFS, "overflow in size of dataset with id:"
            " %d (datasize: %u > MAX_DATA_SIZE: %u)", data->id, dataSize,
             ((uint32_t)MAX_BLOCK_IDX)*
             ((uint32_t)FTI_Conf->dcpInfoPosix.BlockSize));
        FTI_Print(errstr, FTI_EROR);
        return FTI_NSCS;
    }
    if (varId > MAX_VAR_ID) {
        snprintf(errstr, FTI_BUFS, "overflow in ID (id: %d > MAX_ID: %d)!",
         data->id, (int)MAX_VAR_ID);
        FTI_Print(errstr, FTI_EROR);
        return FTI_NSCS;
    }

    // allocate tmp hash array
    // data->dcpInfoPosix.hashArrayTmp = (unsigned char*) malloc(sizeof
    // (unsigned char)*nbHashes*FTI_Conf->dcpInfoPosix.digestWidth);

    // create meta data buffer
    blockMetaInfo_t blockMeta;
    blockMeta.varId = data->id;

    if (dcpLayer == 0) {
        DFTI_EH_FWRITE(FTI_NSCS, bytes, &data->id, sizeof(int), 1,
         write_info->f, "p", block);
        DFTI_EH_FWRITE(FTI_NSCS, bytes, &dataSize, sizeof(uint32_t), 1,
         write_info->f, "p", block);
        FTI_Exec->dcpInfoPosix.FileSize += (sizeof(int) +
         sizeof(uint32_t));
        write_DCPinfo->layerSize += sizeof(int) + sizeof(uint32_t);
    }
    uint32_t pos = 0;

    FTIT_data_prefetch prefetcher;
    size_t totalBytes = 0;
    unsigned char * ptr;
#ifdef GPUSUPPORT
    prefetcher.fetchSize = ((FTI_Conf->cHostBufSize) /
     FTI_Conf->dcpInfoPosix.BlockSize) * FTI_Conf->dcpInfoPosix.BlockSize;
#else
    prefetcher.fetchSize =  data->size;
#endif
    prefetcher.totalBytesToFetch = data->size;
    prefetcher.isDevice = data->isDevicePtr;

    if (prefetcher.isDevice) {
        FTI_MD5GPU(data);
        prefetcher.dptr = data->devicePtr;
    } else {
        FTI_MD5CPU(FTI_Conf,FTI_Exec,data);
        prefetcher.dptr = data->ptr;
    }
    FTI_startMD5();
    FTI_InitPrefetcher(&prefetcher);

    if (FTI_Try(FTI_getPrefetchedData (&prefetcher, &totalBytes, &ptr),
     " Fetching Next Memory block from memory") != FTI_SCES) {
        return FTI_NSCS;
    }
    size_t offset = 0;
    FTI_SyncMD5();
    while (ptr) {
        pos = 0;
        while (pos < totalBytes) {
            // hash index
            unsigned int blockId = offset/FTI_Conf->dcpInfoPosix.BlockSize;
            unsigned int hashIdx = blockId*FTI_Conf->dcpInfoPosix.digestWidth;

            blockMeta.blockId = blockId;

            unsigned int chunkSize = ((dataSize-offset) <
             FTI_Conf->dcpInfoPosix.BlockSize) ? dataSize-offset :
             FTI_Conf->dcpInfoPosix.BlockSize;
            unsigned int dcpChunkSize = chunkSize;
            if (chunkSize < FTI_Conf->dcpInfoPosix.BlockSize) {
                // if block smaller pad with zeros
                memset(block, 0x0, FTI_Conf->dcpInfoPosix.BlockSize);
                memcpy(block, ptr, chunkSize);

                // FTI_Conf->dcpInfoPosix.hashFunc(block,
                // FTI_Conf->dcpInfoPosix.BlockSize,
                // &data->dcpInfoPosix.currentHashArray[hashIdx]);
                ptr = block;
                chunkSize = FTI_Conf->dcpInfoPosix.BlockSize;
            }
            /*else {
              FTI_Conf->dcpInfoPosix.hashFunc(ptr, FTI_Conf->dcpInfoPosix.BlockSize, &data->dcpInfoPosix.currentHashArray[hashIdx]);
              }*/

            bool commitBlock;
            // if old hash exists, compare. If datasize increased, there
            // wont be an old hash to compare with.
            if (offset < data->dcpInfoPosix.hashDataSize) {
                commitBlock = memcmp(&(data->
                 dcpInfoPosix.currentHashArray[hashIdx]),
                 &(data->dcpInfoPosix.oldHashArray[hashIdx]),
                 FTI_Conf->dcpInfoPosix.digestWidth);
            } else {
                commitBlock = true;
            }

            bool success = true;
            int fileUpdate = 0;
            if (commitBlock) {
                if (dcpLayer > 0) {
                    DFTI_EH_FWRITE(FTI_NSCS, success, &blockMeta, 6, 1, write_info->f,
                     "p", block);
                    if (success) fileUpdate += 6;
                }
                if (success) {
                    DFTI_EH_FWRITE(FTI_NSCS, success, ptr, chunkSize, 1, write_info->f,
                     "p", block);
                    if (success) fileUpdate += chunkSize;
                }
                FTI_Exec->dcpInfoPosix.FileSize += success*fileUpdate;
                write_DCPinfo->layerSize += success*fileUpdate;

                FTI_Exec->dcpInfoPosix.dcpSize += success*dcpChunkSize;
                if (success) {
                    MD5_Update(&write_info->integrity,
                     &data->dcpInfoPosix.currentHashArray[hashIdx],
                      MD5_DIGEST_LENGTH);
                }
            }
            offset += dcpChunkSize*success;
            pos += dcpChunkSize*success;
            ptr = ptr + dcpChunkSize;  // chunkSize*success;
        }
        if (FTI_Try(FTI_getPrefetchedData (&prefetcher, &totalBytes, &ptr),
         " Fetching Next Memory block from memory") != FTI_SCES) {
            return FTI_NSCS;
        }
    }
    // swap hash arrays and free old one
    //    free(data->dcpInfoPosix.hashArray);
    data->dcpInfoPosix.hashDataSize = dataSize;
    unsigned char *tmp = data->dcpInfoPosix.currentHashArray;
    data->dcpInfoPosix.currentHashArray = data->dcpInfoPosix.oldHashArray;
    data->dcpInfoPosix.oldHashArray = tmp;
    //    data->dcpInfoPosix.hashArray = data->dcpInfoPosix.hashArrayTmp;

    free(block);

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

int FTI_PosixDCPClose(void *fileDesc) {
    WriteDCPPosixInfo_t *write_dcpInfo = (WriteDCPPosixInfo_t *) fileDesc;
    FTIT_execution *FTI_Exec = write_dcpInfo->FTI_Exec;
    FTIT_configuration *FTI_Conf = write_dcpInfo->FTI_Conf;
    FTIT_checkpoint *FTI_Ckpt = write_dcpInfo->FTI_Ckpt;
    FTIT_topology *FTI_Topo = write_dcpInfo->FTI_Topo;


    char errstr[FTI_BUFS];

    int dcpFileId = FTI_Exec->dcpInfoPosix.Counter /
     FTI_Conf->dcpInfoPosix.StackSize;
    // dcpLayer corresponds to the additional layers towards the base layer.
    int dcpLayer = FTI_Exec->dcpInfoPosix.Counter %
     FTI_Conf->dcpInfoPosix.StackSize;

    if (FTI_Conf->dcpInfoPosix.cachedCkpt) {
        FTI_CLOSE_ASYNC((write_dcpInfo->write_info.f));
    } else {
        FTI_PosixSync(&(write_dcpInfo->write_info));
        FTI_PosixClose(&(write_dcpInfo->write_info));
    }

    // create final dcp layer hash
    unsigned char LayerHash[MD5_DIGEST_LENGTH];
    MD5_Final(LayerHash, &(write_dcpInfo->write_info.integrity));
    FTI_GetHashHexStr(LayerHash, MD5_DIGEST_LENGTH,
     &FTI_Exec->dcpInfoPosix.LayerHash[dcpLayer*MD5_DIGEST_STRING_LENGTH]);
    // layer size is needed in order to create layer hash during recovery
    FTI_Exec->dcpInfoPosix.LayerSize[dcpLayer] = write_dcpInfo->layerSize;
    FTI_Exec->dcpInfoPosix.Counter++;
    if (dcpLayer == 0) {
        char ofn[512];
        snprintf(ofn, FTI_BUFS, "%s/dcp-id%d-rank%d.fti", FTI_Ckpt[4].dcpDir,
         dcpFileId-1, FTI_Topo->myRank);
        if ((remove(ofn) < 0) && (errno != ENOENT)) {
            snprintf(errstr, FTI_BUFS, "cannot delete file '%s'", ofn);
            FTI_Print(errstr, FTI_WARN);
        }
    }
    return FTI_SCES;
}



/*-------------------------------------------------------------------------*/
/**
  @brief      It loads the checkpoint data for dcpPosix.
  @return     integer         FTI_SCES if successful.

  dCP POSIX implementation of FTI_Recover().
 **/
/*-------------------------------------------------------------------------*/
int FTI_RecoverDcpPosix(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
    FTIT_checkpoint* FTI_Ckpt, FTIT_keymap* FTI_Data) {
    uint32_t blockSize;
    unsigned int stackSize;
    int nbVarLayer;
    int ckptId;

    char errstr[FTI_BUFS];
    char fn[FTI_BUFS];

    void* ptr;

    FTIT_dataset* data;

    snprintf(fn, FTI_BUFS, "%s/%s", FTI_Ckpt[FTI_Exec->ckptLvel].dcpDir,
     FTI_Exec->ckptMeta.ckptFile);

    // read base part of file
    FILE* fd = fopen(fn, "rb");
    fread(&blockSize, sizeof(uint32_t), 1, fd);
    if (ferror(fd)) {
        snprintf(errstr, FTI_BUFS, "unable to read in file %s", fn);
        FTI_Print(errstr, FTI_EROR);
        return FTI_NSCS;
    }
    fread(&stackSize, sizeof(unsigned int), 1, fd);
    if (ferror(fd)) {
        snprintf(errstr, FTI_BUFS, "unable to read in file %s", fn);
        FTI_Print(errstr, FTI_EROR);
        return FTI_NSCS;
    }

    // check if settings are correct. If not correct them
    if (blockSize != FTI_Conf->dcpInfoPosix.BlockSize) {
        char str[FTI_BUFS];
        snprintf(str, FTI_BUFS, "dCP blocksize differ between configuration"
        " settings ('%u') and checkpoint file ('%u')",
         FTI_Conf->dcpInfoPosix.BlockSize, blockSize);
        FTI_Print(str, FTI_WARN);
        return FTI_NREC;
    }
    if (stackSize != FTI_Conf->dcpInfoPosix.StackSize) {
        char str[FTI_BUFS];
        snprintf(str, FTI_BUFS, "dCP stacksize differ between configuration"
        " settings ('%u') and checkpoint file ('%u')",
         FTI_Conf->dcpInfoPosix.StackSize, stackSize);
        FTI_Print(str, FTI_WARN);
        return FTI_NREC;
    }


    void *buffer = malloc(blockSize);
    if (!buffer) {
        FTI_Print("unable to allocate memory!", FTI_EROR);
        return FTI_NSCS;
    }

    int i;
    // treat Layer 0 first
    fread(&ckptId, 1, sizeof(int), fd);
    if (ferror(fd)) {
        snprintf(errstr, FTI_BUFS, "unable to read in file %s", fn);
        FTI_Print(errstr, FTI_EROR);
        return FTI_NSCS;
    }
    fread(&nbVarLayer, 1, sizeof(int), fd);
    if (ferror(fd)) {
        snprintf(errstr, FTI_BUFS, "unable to read in file %s", fn);
        FTI_Print(errstr, FTI_EROR);
        return FTI_NSCS;
    }
    for (i = 0; i < nbVarLayer; i++) {
        unsigned int varId;
        uint32_t locDataSize;
        fread(&varId, sizeof(int), 1, fd);
        if (ferror(fd)) {
            snprintf(errstr, FTI_BUFS, "unable to read in file %s", fn);
            FTI_Print(errstr, FTI_EROR);
            return FTI_NSCS;
        }
        fread(&locDataSize, sizeof(uint32_t), 1, fd);
        if (ferror(fd)) {
            snprintf(errstr, FTI_BUFS, "unable to read in file %s", fn);
            FTI_Print(errstr, FTI_EROR);
            return FTI_NSCS;
        }

        if (FTI_Data->get(&data, varId) != FTI_SCES) return FTI_NSCS;

        if (!data) {
            snprintf(errstr, FTI_BUFS, "id '%d' does not exist!", varId);
            FTI_Print(errstr, FTI_EROR);
            return FTI_NSCS;
        }
#ifdef GPUSUPPORT
        if (data->isDevicePtr) {
            FTI_TransferFileToDeviceAsync(fd, data->devicePtr, data->size);
        } else {
            fread(data->ptr, locDataSize, 1, fd);
            if (ferror(fd)) {
                snprintf(errstr, FTI_BUFS, "unable to read in file %s", fn);
                FTI_Print(errstr, FTI_EROR);
                return FTI_NSCS;
            }
        }

#else
        fread(data->ptr, locDataSize, 1, fd);
        if (ferror(fd)) {
            snprintf(errstr, FTI_BUFS, "unable to read in file %s", fn);
            FTI_Print(errstr, FTI_EROR);
            return FTI_NSCS;
        }
#endif
        int overflow;
        if ((overflow=locDataSize%blockSize) != 0) {
            fread(buffer, blockSize - overflow, 1, fd);
            if (ferror(fd)) {
                snprintf(errstr, FTI_BUFS, "unable to read in file %s", fn);
                FTI_Print(errstr, FTI_EROR);
                return FTI_NSCS;
            }
        }
    }


    uint32_t offset;
    blockMetaInfo_t blockMeta;
    unsigned char *block = (unsigned char*) malloc(blockSize);
    if (!block) {
        FTI_Print("unable to allocate memory!", FTI_EROR);
        return FTI_NSCS;
    }

    int nbLayer = FTI_Exec->dcpInfoPosix.nbLayerReco;


    for (i = 1; i < nbLayer; i++) {
        uint32_t pos = 0;
        pos += fread(&ckptId, 1, sizeof(int), fd);
        if (ferror(fd)) {
            snprintf(errstr, FTI_BUFS, "unable to read in file %s", fn);
            FTI_Print(errstr, FTI_EROR);
            return FTI_NSCS;
        }
        pos += fread(&nbVarLayer, 1, sizeof(int), fd);
        if (ferror(fd)) {
            snprintf(errstr, FTI_BUFS, "unable to read in file %s", fn);
            FTI_Print(errstr, FTI_EROR);
            return FTI_NSCS;
        }

        while (pos < FTI_Exec->dcpInfoPosix.LayerSize[i]) {
            fread(&blockMeta, 1, 6, fd);
            if (ferror(fd)) {
                snprintf(errstr, FTI_BUFS, "unable to read in file %s", fn);
                FTI_Print(errstr, FTI_EROR);
                return FTI_NSCS;
            }

            if (FTI_Data->get(&data, blockMeta.varId) != FTI_SCES)
                return FTI_NSCS;

            if (!data) {
                snprintf(errstr, FTI_BUFS, "id '%d' does not exist!",
                 blockMeta.varId);
                FTI_Print(errstr, FTI_EROR);
                return FTI_NSCS;
            }

            offset = blockMeta.blockId * blockSize;
            unsigned int chunkSize = ((data->size-offset) < blockSize) ?
             data->size-offset : blockSize;

#ifdef GPUSUPPORT
            if (data->isDevicePtr) {
                FTI_device_sync();
                fread(block, 1, chunkSize, fd);
                FTI_copy_to_device_async(data->devicePtr + offset, block,
                 chunkSize);
            } else {
                ptr = data->ptr + offset;
                fread(ptr, 1, chunkSize, fd);
            }
#else
            ptr = data->ptr + offset;
            fread(ptr, 1, chunkSize, fd);
#endif
            if (ferror(fd)) {
                snprintf(errstr, FTI_BUFS, "unable to read in file %s", fn);
                FTI_Print(errstr, FTI_EROR);
                return FTI_NSCS;
            }
            fread(buffer, 1, blockSize - chunkSize, fd);
            if (ferror(fd)) {
                snprintf(errstr, FTI_BUFS, "unable to read in file %s", fn);
                FTI_Print(errstr, FTI_EROR);
                return FTI_NSCS;
            }

            pos += (blockSize+6);
        }
    }

    // create hasharray
    if ((FTI_Data->data(&data, FTI_Exec->nbVarStored) != FTI_SCES) || !data)
        return FTI_NSCS;

    for (i = 0; i < FTI_Exec->nbVarStored; i++) {
        FTIT_data_prefetch prefetcher;
        size_t totalBytes = 0;
        unsigned char * ptr = NULL, *startPtr = NULL;

#ifdef GPUSUPPORT
        prefetcher.fetchSize = ((FTI_Conf->cHostBufSize) /
         FTI_Conf->dcpInfoPosix.BlockSize) * FTI_Conf->dcpInfoPosix.BlockSize;
#else
        prefetcher.fetchSize =  data[i].size;
#endif
        prefetcher.totalBytesToFetch = data[i].size;
        prefetcher.isDevice = data[i].isDevicePtr;

        if (prefetcher.isDevice) {
            prefetcher.dptr = data[i].devicePtr;
        } else {
            prefetcher.dptr = data[i].ptr;
        }

        FTI_InitPrefetcher(&prefetcher);
        if (FTI_Try(FTI_getPrefetchedData (&prefetcher, &totalBytes,
         &startPtr), " Fetching Next Memory block from memory") != FTI_SCES) {
            return FTI_NSCS;
        }

        uint32_t nbBlocks = (data[i].size % blockSize) ?
         data[i].size/blockSize + 1 : data[i].size/blockSize;
        data[i].dcpInfoPosix.hashDataSize = data[i].size;
        int j = 0;
        while (startPtr) {
            ptr = startPtr;
            int currentBlocks = (totalBytes % blockSize) ?
              totalBytes/blockSize + 1 : totalBytes/blockSize;
            int k;
            for (k = 0 ; k < currentBlocks && j < nbBlocks-1; k++) {
                uint32_t hashIdx = j*MD5_DIGEST_LENGTH;
                FTI_Conf->dcpInfoPosix.hashFunc(ptr, blockSize,
                 &data[i].dcpInfoPosix.oldHashArray[hashIdx]);
                ptr = ptr+blockSize;
                j++;
            }
            if (FTI_Try(FTI_getPrefetchedData(&prefetcher,
             &totalBytes, &startPtr),
             " Fetching Next Memory block from memory") != FTI_SCES) {
                return FTI_NSCS;
            }
        }

        if (data[i].size%blockSize) {
            unsigned char* buffer = calloc(1, blockSize);
            if (!buffer) {
                FTI_Print("unable to allocate memory!", FTI_EROR);
                return FTI_NSCS;
            }
            uint32_t dataOffset = blockSize * (nbBlocks - 1);
            uint32_t dataSize = data[i].size - dataOffset;
            memcpy(buffer, ptr, dataSize);
            FTI_Conf->dcpInfoPosix.hashFunc(buffer, blockSize,
            &data[i].dcpInfoPosix.oldHashArray[(nbBlocks-1)*MD5_DIGEST_LENGTH]);
        }
    }


    FTI_Exec->reco = 0;

    free(buffer);
    fclose(fd);

    return FTI_SCES;
}

int FTI_RecoverVarDcpPosixInit() { /*TODO*/ return FTI_SCES; }
int FTI_RecoverVarDcpPosixFinalize() { /*TODO*/ return FTI_SCES; }

/*-------------------------------------------------------------------------*/
/**
  @brief      Recovers the given variable for dcpPosix
  @param      id              Variable to recover
  @return     int             FTI_SCES if successful.

  dCP POSIX implementation of FTI_RecoverVar().
 **/
/*-------------------------------------------------------------------------*/
int FTI_RecoverVarDcpPosix(FTIT_configuration* FTI_Conf,
    FTIT_execution* FTI_Exec, FTIT_checkpoint* FTI_Ckpt, FTIT_keymap* FTI_Data,
    int id) {
    uint32_t blockSize;
    unsigned int stackSize;
    int nbVarLayer;
    int ckptId;

    char errstr[FTI_BUFS];
    char fn[FTI_BUFS];

    FTIT_dataset* data;

    snprintf(fn, FTI_BUFS, "%s/%s", FTI_Ckpt[FTI_Exec->ckptLvel].dcpDir,
     FTI_Exec->ckptMeta.ckptFile);

    // read base part of file
    FILE* fd = fopen(fn, "rb");
    fread(&blockSize, sizeof(uint32_t), 1, fd);
    if (ferror(fd)) {
        snprintf(errstr, FTI_BUFS, "unable to read in file %s", fn);
        FTI_Print(errstr, FTI_EROR);
        return FTI_NSCS;
    }
    fread(&stackSize, sizeof(unsigned int), 1, fd);
    if (ferror(fd)) {
        snprintf(errstr, FTI_BUFS, "unable to read in file %s", fn);
        FTI_Print(errstr, FTI_EROR);
        return FTI_NSCS;
    }

    // check if settings are correct. If not correct them
    if (blockSize != FTI_Conf->dcpInfoPosix.BlockSize) {
        char str[FTI_BUFS];
        snprintf(str, FTI_BUFS, "dCP blocksize differ between configuration"
        " settings ('%u') and checkpoint file ('%u')",
         FTI_Conf->dcpInfoPosix.BlockSize, blockSize);
        FTI_Print(str, FTI_WARN);
        return FTI_NREC;
    }
    if (stackSize != FTI_Conf->dcpInfoPosix.StackSize) {
        char str[FTI_BUFS];
        snprintf(str, FTI_BUFS, "dCP stacksize differ between configuration"
        " settings ('%u') and checkpoint file ('%u')",
         FTI_Conf->dcpInfoPosix.StackSize, stackSize);
        FTI_Print(str, FTI_WARN);
        return FTI_NREC;
    }


    void *buffer = malloc(blockSize);
    if (!buffer) {
        FTI_Print("unable to allocate memory!", FTI_EROR);
        return FTI_NSCS;
    }

    int i;

    // treat Layer 0 first
    fread(&ckptId, 1, sizeof(int), fd);
    if (ferror(fd)) {
        snprintf(errstr, FTI_BUFS, "unable to read in file %s", fn);
        FTI_Print(errstr, FTI_EROR);
        return FTI_NSCS;
    }
    fread(&nbVarLayer, 1, sizeof(int), fd);
    if (ferror(fd)) {
        snprintf(errstr, FTI_BUFS, "unable to read in file %s", fn);
        FTI_Print(errstr, FTI_EROR);
        return FTI_NSCS;
    }
    for (i = 0; i < nbVarLayer; i++) {
        unsigned int varId;
        uint32_t locDataSize;
        fread(&varId, sizeof(int), 1, fd);
        if (ferror(fd)) {
            snprintf(errstr, FTI_BUFS, "unable to read in file %s", fn);
            FTI_Print(errstr, FTI_EROR);
            return FTI_NSCS;
        }
        fread(&locDataSize, sizeof(uint32_t), 1, fd);
        if (ferror(fd)) {
            snprintf(errstr, FTI_BUFS, "unable to read in file %s", fn);
            FTI_Print(errstr, FTI_EROR);
            return FTI_NSCS;
        }
        // if requested id load else skip dataSize
        if (varId == id) {
            if (FTI_Data->get(&data, varId) != FTI_SCES) return FTI_NSCS;

            if (!data) {
                snprintf(errstr, FTI_BUFS, "id '%d' does not exist!", varId);
                FTI_Print(errstr, FTI_EROR);
                return FTI_NSCS;
            }

            fread(data->ptr, locDataSize, 1, fd);
            if (ferror(fd)) {
                snprintf(errstr, FTI_BUFS, "unable to read in file %s", fn);
                FTI_Print(errstr, FTI_EROR);
                return FTI_NSCS;
            }

            int overflow;
            if ((overflow=locDataSize%blockSize) != 0) {
                fread(buffer, blockSize - overflow, 1, fd);
                if (ferror(fd)) {
                    snprintf(errstr, FTI_BUFS, "unable to read in file %s", fn);
                    FTI_Print(errstr, FTI_EROR);
                    return FTI_NSCS;
                }
            }
        } else {
            uint32_t skip = (locDataSize%blockSize == 0) ?
             locDataSize : (locDataSize/blockSize + 1)*blockSize;
            if (fseek(fd, skip, SEEK_CUR) == -1) {
                snprintf(errstr, FTI_BUFS, "unable to seek in file %s", fn);
                FTI_Print(errstr, FTI_EROR);
                return FTI_NSCS;
            }
        }
    }


    uint32_t offset;

    blockMetaInfo_t blockMeta;
    unsigned char *block = (unsigned char*) malloc(blockSize);
    if (!block) {
        FTI_Print("unable to allocate memory!", FTI_EROR);
        return FTI_NSCS;
    }

    int nbLayer = FTI_Exec->dcpInfoPosix.nbLayerReco;
    for (i = 1; i < nbLayer; i++) {
        uint32_t pos = 0;
        pos += fread(&ckptId, 1, sizeof(int), fd);
        if (ferror(fd)) {
            snprintf(errstr, FTI_BUFS, "unable to read in file %s", fn);
            FTI_Print(errstr, FTI_EROR);
            return FTI_NSCS;
        }
        pos += fread(&nbVarLayer, 1, sizeof(int), fd);
        if (ferror(fd)) {
            snprintf(errstr, FTI_BUFS, "unable to read in file %s", fn);
            FTI_Print(errstr, FTI_EROR);
            return FTI_NSCS;
        }

        while (pos < FTI_Exec->dcpInfoPosix.LayerSize[i]) {
            fread(&blockMeta, 1, 6, fd);
            if (ferror(fd)) {
                snprintf(errstr, FTI_BUFS, "unable to read in file %s", fn);
                FTI_Print(errstr, FTI_EROR);
                return FTI_NSCS;
            }
            if (blockMeta.varId == id) {
                if (FTI_Data->get(&data, blockMeta.varId) != FTI_SCES)
                    return FTI_NSCS;

                if (!data) {
                    snprintf(errstr, FTI_BUFS, "id '%d' does not exist!",
                     blockMeta.varId);
                    FTI_Print(errstr, FTI_EROR);
                    return FTI_NSCS;
                }

                offset = blockMeta.blockId * blockSize;
                void* ptr = data->ptr + offset;
                unsigned int chunkSize = ((data->size-offset) < blockSize) ?
                 data->size-offset : blockSize;

                fread(ptr, 1, chunkSize, fd);
                if (ferror(fd)) {
                    snprintf(errstr, FTI_BUFS, "unable to read in file %s", fn);
                    FTI_Print(errstr, FTI_EROR);
                    return FTI_NSCS;
                }
                fread(buffer, 1, blockSize - chunkSize, fd);
                if (ferror(fd)) {
                    snprintf(errstr, FTI_BUFS, "unable to read in file %s", fn);
                    FTI_Print(errstr, FTI_EROR);
                    return FTI_NSCS;
                }

            } else {
                if (fseek(fd, blockSize, SEEK_CUR) == -1) {
                    snprintf(errstr, FTI_BUFS, "unable to seek in file %s", fn);
                    FTI_Print(errstr, FTI_EROR);
                    return FTI_NSCS;
                }
            }
            pos += (blockSize+6);
        }
    }

    if (FTI_Data->get(&data, id) != FTI_SCES) return FTI_NSCS;

    if (!data) {
        snprintf(errstr, FTI_BUFS, "id '%d' does not exist!", blockMeta.varId);
        FTI_Print(errstr, FTI_EROR);
        return FTI_NSCS;
    }
    FTIT_data_prefetch prefetcher;
    size_t totalBytes = 0;
    unsigned char * ptr = NULL, *startPtr = NULL;

#ifdef GPUSUPPORT
    prefetcher.fetchSize = ((FTI_Conf->cHostBufSize) /
     FTI_Conf->dcpInfoPosix.BlockSize) * FTI_Conf->dcpInfoPosix.BlockSize;
#else
    prefetcher.fetchSize =  data->size;
#endif
    prefetcher.totalBytesToFetch = data->size;
    prefetcher.isDevice = data->isDevicePtr;

    if (prefetcher.isDevice) {
        prefetcher.dptr = data->devicePtr;
    } else {
        prefetcher.dptr = data->ptr;
    }

    FTI_InitPrefetcher(&prefetcher);
    if (FTI_Try(FTI_getPrefetchedData (&prefetcher, &totalBytes, &startPtr),
     " Fetching Next Memory block from memory") != FTI_SCES) {
        return FTI_NSCS;
    }

    uint32_t nbBlocks = (data->size % blockSize) ?
     data->size/blockSize + 1 : data->size/blockSize;
    data->dcpInfoPosix.hashDataSize = data->size;
    int j = 0;
    while (startPtr) {
        ptr = startPtr;
        int currentBlocks = (totalBytes % blockSize) ?
          totalBytes/blockSize + 1 : totalBytes/blockSize;
        int k;
        for (k = 0 ; k < currentBlocks && j < nbBlocks-1; k++) {
            uint32_t hashIdx = j*MD5_DIGEST_LENGTH;
            FTI_Conf->dcpInfoPosix.hashFunc(ptr, blockSize,
             &data->dcpInfoPosix.oldHashArray[hashIdx]);
            ptr = ptr+blockSize;
            j++;
        }
        if (FTI_Try(FTI_getPrefetchedData (&prefetcher, &totalBytes,
         &startPtr), " Fetching Next Memory block from memory") != FTI_SCES) {
            return FTI_NSCS;
        }
    }

    if (data->size%blockSize) {
        unsigned char* buffer = calloc(1, blockSize);
        if (!buffer) {
            FTI_Print("unable to allocate memory!", FTI_EROR);
            return FTI_NSCS;
        }
        uint32_t dataOffset = blockSize * (nbBlocks - 1);
        uint32_t dataSize = data->size - dataOffset;
        memcpy(buffer, ptr, dataSize);
        FTI_Conf->dcpInfoPosix.hashFunc(buffer, blockSize,
         &data->dcpInfoPosix.oldHashArray[(nbBlocks-1)*MD5_DIGEST_LENGTH]);
    }

    /*
    // create hasharray for id
    i = FTI_DataGetIdx(id, FTI_Exec, FTI_Data);
    uint32_t nbBlocks = (FTI_Data[i].size % blockSize) ? FTI_Data[i].size/blockSize + 1 : FTI_Data[i].size/blockSize;
    FTI_Data[i].dcpInfoPosix.hashDataSize = FTI_Data[i].size;

    int j;
    for(j=0; j<nbBlocks-1; j++) {
    uint32_t offset = j*blockSize;
    uint32_t hashIdx = j*MD5_DIGEST_LENGTH;
    MD5(FTI_Data[i].ptr+offset, blockSize, &FTI_Data[i].dcpInfoPosix.hashArray[hashIdx]);
    }
    if (FTI_Data[i].size%blockSize) {
    unsigned char* buffer = calloc(1, blockSize);
    if (!buffer) {
    FTI_Print("unable to allocate memory!", FTI_EROR);
    return FTI_NSCS;
    }
    uint32_t dataOffset = blockSize * (nbBlocks - 1);
    uint32_t dataSize = FTI_Data[i].size - dataOffset;
    memcpy(buffer, FTI_Data[i].ptr + dataOffset, dataSize); 
    MD5(buffer, blockSize, &FTI_Data[i].dcpInfoPosix.hashArray[(nbBlocks-1)*MD5_DIGEST_LENGTH]);
    free(buffer);
    }
    */
    free(buffer);
    fclose(fd);


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
int FTI_CheckFileDcpPosix(char* fn, int32_t fs, char* checksum) {
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
        } else {
            FTI_Print("Stat return wrong error code", FTI_WARN);
            return 1;
        }
    } else {
        char str[FTI_BUFS];
        snprintf(str, sizeof(str), "Missing file: \"%s\"", fn);
        FTI_Print(str, FTI_WARN);
        return 1;
    }
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
int FTI_VerifyChecksumDcpPosix(char* fileName) {
    FTIT_execution* exec = FTI_DcpPosixRecoverRuntimeInfo(DCP_POSIX_EXEC_TAG,
     NULL, NULL);
    FTIT_configuration* conf = FTI_DcpPosixRecoverRuntimeInfo(
        DCP_POSIX_CONF_TAG, NULL, NULL);
    int *nbVarLayers = NULL;
    size_t *layerSizes = NULL;
    int *ckptIds  = NULL;
    char errstr[FTI_BUFS];
    char dummyBuffer[FTI_BUFS];
    uint32_t blockSize;
    unsigned int stackSize;
    unsigned int counter = 0;
    unsigned int dcpFileId;
    int lastCorrectLayer = -1;
    int minLayer = 0;

    FILE *fd = fopen(fileName, "rb");
    if (fd == NULL) {
        char str[FTI_BUFS];
        snprintf(str, sizeof(str),
         "FTI failed to open file %s to calculate checksum.", fileName);
        FTI_Print(str, FTI_WARN);
        goto FINALIZE;
    }

    unsigned char md5_tmp[MD5_DIGEST_LENGTH];
    unsigned char md5_final[MD5_DIGEST_LENGTH];

    MD5_CTX mdContext;

    // position in file
    size_t fs = 0;

    // get blocksize
    fs += fread(&blockSize, 1, sizeof(uint32_t), fd);
    if (ferror(fd)) {
        snprintf(errstr, FTI_BUFS, "unable to read in file %s", fileName);
        FTI_Print(errstr, FTI_EROR);
        goto FINALIZE;
    }
    fs += fread(&stackSize, 1, sizeof(unsigned int), fd);
    if (ferror(fd)) {
        snprintf(errstr, FTI_BUFS, "unable to read in file %s", fileName);
        FTI_Print(errstr, FTI_EROR);
        goto FINALIZE;
    }

    // check if settings are correckt. If not correct them
    if (blockSize != conf->dcpInfoPosix.BlockSize) {
        char str[FTI_BUFS];
        snprintf(str, FTI_BUFS, "dCP blocksize differ between configuration"
        " settings ('%u') and checkpoint file ('%u')",
         conf->dcpInfoPosix.BlockSize, blockSize);
        FTI_Print(str, FTI_WARN);
        conf->dcpInfoPosix.BlockSize = blockSize;
    }
    if (stackSize != conf->dcpInfoPosix.StackSize) {
        char str[FTI_BUFS];
        snprintf(str, FTI_BUFS, "dCP stacksize differ between configuration"
        " settings ('%u') and checkpoint file ('%u')",
         conf->dcpInfoPosix.StackSize, stackSize);
        FTI_Print(str, FTI_WARN);
        conf->dcpInfoPosix.StackSize = stackSize;
    }

    // get dcpFileId from filename
    int dummy;
    sscanf(exec->ckptMeta.ckptFile, "dcp-id%d-rank%d.fti", &dcpFileId, &dummy);
    counter = dcpFileId * stackSize;

    int i;
    int layer = 0;
    int nbVarLayer;
    int ckptId;

    // set number of recovered layers to 0
    exec->dcpInfoPosix.nbLayerReco = 0;

    // data buffer
    void* buffer = malloc(blockSize);
    if (!buffer) {
        FTI_Print("unable to allocate memory!", FTI_EROR);
        goto FINALIZE;
    }

    // check layer 0 first
    // get number of variables stored in layer
    MD5_Init(&mdContext);
    fs += fread(&ckptId, 1, sizeof(int), fd);
    if (ferror(fd)|| feof(fd)) {
        snprintf(errstr, FTI_BUFS, "unable to read in file %s", fileName);
        FTI_Print(errstr, FTI_EROR);
        goto FINALIZE;
    }
    fs += fread(&nbVarLayer, 1, sizeof(int), fd);
    if (ferror(fd) || feof(fd)) {
        snprintf(errstr, FTI_BUFS, "unable to read in file %s", fileName);
        FTI_Print(errstr, FTI_EROR);
        goto FINALIZE;
    }
    for (i = 0; i < nbVarLayer; i++) {
        uint32_t dataSize;
        uint32_t pos = 0;
        fs += fread(dummyBuffer, 1, sizeof(int), fd);
        if (ferror(fd)|| feof(fd)) {
            snprintf(errstr, FTI_BUFS, "unable to read in file %s", fileName);
            FTI_Print(errstr, FTI_EROR);
            goto FINALIZE;
        }
        fs += fread(&dataSize, 1, sizeof(uint32_t), fd);
        if (ferror(fd)|| feof(fd)) {
            snprintf(errstr, FTI_BUFS, "unable to read in file %s", fileName);
            FTI_Print(errstr, FTI_EROR);
            goto FINALIZE;
        }
        while (pos < dataSize) {
            pos += fread(buffer, 1, blockSize, fd);
            if (ferror(fd)|| feof(fd)) {
                snprintf(errstr, FTI_BUFS, "unable to read in file %s",
                 fileName);
                FTI_Print(errstr, FTI_EROR);
                goto FINALIZE;
            }
            MD5(buffer, blockSize, md5_tmp);
            MD5_Update(&mdContext, md5_tmp, MD5_DIGEST_LENGTH);
        }
        fs += pos;
    }
    MD5_Final(md5_final, &mdContext);
    // compare hashes
    if (strcmp(FTI_GetHashHexStr(md5_final, conf->dcpInfoPosix.digestWidth,
     NULL), &exec->dcpInfoPosix.LayerHash[layer*MD5_DIGEST_STRING_LENGTH]) && !conf->pbdcpEnabled)  {
        FTI_Print("hashes differ in base", FTI_WARN);
        goto FINALIZE;
    }
    layerSizes = talloc(size_t, stackSize);
    ckptIds  = talloc(int, stackSize);
    nbVarLayers = talloc(int, stackSize);

    layerSizes[layer] = fs;
    ckptIds[layer] = ckptId;
    nbVarLayers[layer] = nbVarLayer;

    lastCorrectLayer++;
    layer++;
    exec->dcpInfoPosix.nbLayerReco = layer;
    exec->dcpInfoPosix.nbVarReco = nbVarLayer;
    exec->ckptId = ckptId;

    // exec->dcpInfoPosix.Counter = counter;
    bool readLayer = true;
    size_t bytes;
    size_t layerSize;
    // now treat other layers

    for (; layer < stackSize && readLayer; layer++) {
        readLayer = true;
        layerSize = 0;
        MD5_Init(&mdContext);
        bytes = fread(&ckptId, 1, sizeof(int), fd);
        if (feof(fd)) {
            readLayer = false;
            break;
        }
        if (ferror(fd)) {
            snprintf(errstr, FTI_BUFS, "unable to read in file %s", fileName);
            FTI_Print(errstr, FTI_EROR);
            goto FINALIZE;
        }

        layerSize += bytes;

        bytes = fread(&nbVarLayer, 1, sizeof(int), fd);

        if (feof(fd)) {
            readLayer = false;
            break;
        }


        if (ferror(fd)) {
            snprintf(errstr, FTI_BUFS, "unable to read in file %s", fileName);
            FTI_Print(errstr, FTI_EROR);
            goto FINALIZE;
        }

        layerSize += bytes;

        while ((readLayer) && layerSize < exec->dcpInfoPosix.LayerSize[layer]) {
            bytes = fread(dummyBuffer, 1, 6, fd);
            if (feof(fd)) {
                readLayer = false;
                break;
            }
            if (ferror(fd)) {
                snprintf(errstr, FTI_BUFS, "unable to read in file %s",
                 fileName);
                FTI_Print(errstr, FTI_EROR);
                goto FINALIZE;
            }
            layerSize += bytes;

            bytes = fread(buffer, 1, blockSize, fd);
            if (feof(fd)) {
                readLayer = false;
                break;
            }

            if (ferror(fd)) {
                snprintf(errstr, FTI_BUFS, "unable to read in file %s",
                 fileName);
                FTI_Print(errstr, FTI_EROR);
                goto FINALIZE;
            }
            layerSize += bytes;

            MD5(buffer, blockSize, md5_tmp);
            MD5_Update(&mdContext, md5_tmp, MD5_DIGEST_LENGTH);
        }
        MD5_Final(md5_final, &mdContext);
        // compare hashes
        if (readLayer && strcmp(FTI_GetHashHexStr(md5_final,
         conf->dcpInfoPosix.digestWidth, NULL),
         &exec->dcpInfoPosix.LayerHash[layer*MD5_DIGEST_STRING_LENGTH]) && !conf->pbdcpEnabled) {
            readLayer = false;
        }

        if (readLayer) {
            fs += layerSize;
            layerSizes[layer] = fs;
            ckptIds[layer] = ckptId;
            nbVarLayers[layer] = nbVarLayer;
            exec->dcpInfoPosix.nbLayerReco = layer+1;
            exec->dcpInfoPosix.nbVarReco = nbVarLayer;
            lastCorrectLayer++;
        }
    }

FINALIZE:
         MPI_Allreduce(&lastCorrectLayer, &minLayer, 1, MPI_INT, MPI_MIN,
          FTI_COMM_WORLD);
         if (minLayer == -1) {
             exec->dcpInfoPosix.Counter = 0;
             fclose(fd);
             return FTI_NSCS;
         } else {
             exec->dcpInfoPosix.nbLayerReco = minLayer+1;
             exec->dcpInfoPosix.nbVarReco = nbVarLayers[minLayer];
             exec->dcpInfoPosix.Counter = minLayer +counter + 1;
             fclose(fd);
             exec->ckptId = ckptIds[minLayer];
             if (truncate(fileName, layerSizes[minLayer]) != 0) {
                 FTI_Print("Error On Truncating the file", FTI_EROR);
                 return FTI_EROR;
             }

             free(layerSizes);
             free(ckptIds);
             free(nbVarLayers);
             return FTI_SCES;
         }
         // truncate file if some layer were not possible to recover.
}

// HELPER FUNCTIONS

void* FTI_DcpPosixRecoverRuntimeInfo(int tag, void* exec_, void* conf_) {
    static void* exec = NULL;
    static void* conf = NULL;

    void* ret = NULL;

    switch (tag) {
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
unsigned char* CRC32(const unsigned char *d, uint64_t nBytes,
 unsigned char *hash) {
    static unsigned char hash_[CRC32_DIGEST_LENGTH];
    if (hash == NULL) {
        hash = hash_;
    }

#ifdef FTI_NOZLIB
    uint32_t digest = crc32(d, nBytes);
#else
    uint32_t digest = crc32(0L, Z_NULL, 0);
    digest = crc32(digest, d, nBytes);
#endif

    memcpy(hash, &digest, CRC32_DIGEST_LENGTH);

    return hash;
}

