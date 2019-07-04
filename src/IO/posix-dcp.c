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

    FTI_Print("I/O mode: Posix.", FTI_DBUG);

    char fn[FTI_BUFS];
    size_t bytes;

    WriteDCPPosixInfo_t *write_DCPinfo = (WriteDCPPosixInfo_t*) malloc (sizeof(WriteDCPPosixInfo_t));
    WritePosixInfo_t *write_info = &(write_DCPinfo->write_info); 
    write_DCPinfo->FTI_Exec = FTI_Exec;
    write_DCPinfo->FTI_Conf = FTI_Conf;
    write_DCPinfo->FTI_Ckpt = FTI_Ckpt;
    write_DCPinfo->FTI_Topo = FTI_Topo;
    write_DCPinfo->layerSize = 0;


    FTI_Exec->dcpInfoPosix.dcpSize = 0;
    FTI_Exec->dcpInfoPosix.dataSize = 0;

    // dcpFileId increments every dcpStackSize checkpoints.
    int dcpFileId = FTI_Exec->dcpInfoPosix.Counter / FTI_Conf->dcpInfoPosix.StackSize;

    // dcpLayer corresponds to the additional layers towards the base layer.
    int dcpLayer = FTI_Exec->dcpInfoPosix.Counter % FTI_Conf->dcpInfoPosix.StackSize;

    // if first layer, make sure that we write all data by setting hashdatasize = 0
    if( dcpLayer == 0 ) {
        int i = 0;
        for(; i<FTI_Exec->nbVar; i++) {
            //            free(FTI_Data[i].dcpInfoPosix.hashArray);
            //            FTI_Data[i].dcpInfoPosix.hashArray = NULL;
            FTI_Data[i].dcpInfoPosix.hashDataSize = 0;
        }
    }

    snprintf( FTI_Exec->meta[0].ckptFile, FTI_BUFS, "dcp-id%d-rank%d.fti", dcpFileId, FTI_Topo->myRank );
    if (FTI_Ckpt[4].isInline) { //If inline L4 save directly to global directory
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
    if( dcpLayer == 0 ) {
        FWRITE(NULL, bytes, &FTI_Conf->dcpInfoPosix.BlockSize, sizeof(unsigned long), 1, write_info->f, "p", write_info);
        FWRITE(NULL, bytes, &FTI_Conf->dcpInfoPosix.StackSize, sizeof(unsigned int), 1, write_info->f, "p", write_info);
        FTI_Exec->dcpInfoPosix.FileSize += sizeof(unsigned long) + sizeof(unsigned int);
        write_DCPinfo->layerSize += sizeof(unsigned long) + sizeof(unsigned int);
    }

    // write actual amount of variables at the beginning of each layer
    FWRITE(NULL, bytes, &FTI_Exec->ckptID, sizeof(int), 1, write_info->f, "p", write_info);
    FWRITE(NULL, bytes, &FTI_Exec->nbVar, sizeof(int), 1, write_info->f, "p", write_info);
    FTI_Exec->dcpInfoPosix.FileSize += 2*sizeof(int);// + sizeof(unsigned int);
    write_DCPinfo->layerSize += 2*sizeof(int);// + sizeof(unsigned int);

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
    unsigned char * block = (unsigned char*) malloc( FTI_Conf->dcpInfoPosix.BlockSize );
    size_t bytes;
    long varId = FTI_DataVar->id;

    FTI_Exec->dcpInfoPosix.dataSize += FTI_DataVar->size;
    unsigned long dataSize = FTI_DataVar->size;
    //    unsigned long nbHashes = dataSize/FTI_Conf->dcpInfoPosix.BlockSize + (bool)(dataSize%FTI_Conf->dcpInfoPosix.BlockSize);

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

    // allocate tmp hash array
    //    FTI_DataVar->dcpInfoPosix.hashArrayTmp = (unsigned char*) malloc( sizeof(unsigned char)*nbHashes*FTI_Conf->dcpInfoPosix.digestWidth );

    // create meta data buffer
    blockMetaInfo_t blockMeta;
    blockMeta.varId = FTI_DataVar->id;

    if( dcpLayer == 0 ) {
        FWRITE(FTI_NSCS,bytes,&FTI_DataVar->id, sizeof(int), 1,write_info->f, "p",block);
        FWRITE(FTI_NSCS, bytes,&dataSize, sizeof(unsigned long ), 1,write_info->f, "p",block);
        FTI_Exec->dcpInfoPosix.FileSize += (sizeof(int) + sizeof(unsigned long));
        write_DCPinfo->layerSize += sizeof(int) + sizeof(unsigned long);
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
        FTI_MD5GPU(FTI_DataVar);
        prefetcher.dptr = FTI_DataVar->devicePtr;
    }
    else{
        FTI_MD5CPU(FTI_DataVar);
        prefetcher.dptr = FTI_DataVar->ptr;
    }
    FTI_startMD5();
    FTI_InitPrefetcher(&prefetcher);

    if ( FTI_Try(FTI_getPrefetchedData ( &prefetcher, &totalBytes, &ptr), " Fetching Next Memory block from memory") != FTI_SCES ){
        return FTI_NSCS;
    }
    size_t offset = 0;
    FTI_SyncMD5();
    while ( ptr ){
        pos = 0;
        while( pos < totalBytes ) {
            // hash index
            unsigned int blockId = offset/FTI_Conf->dcpInfoPosix.BlockSize;
            unsigned int hashIdx = blockId*FTI_Conf->dcpInfoPosix.digestWidth;

            blockMeta.blockId = blockId;

            unsigned int chunkSize = ( (dataSize-offset) < FTI_Conf->dcpInfoPosix.BlockSize ) ? dataSize-offset : FTI_Conf->dcpInfoPosix.BlockSize;
            unsigned int dcpChunkSize = chunkSize;
            if( chunkSize < FTI_Conf->dcpInfoPosix.BlockSize ) {
                // if block smaller pad with zeros
                memset( block, 0x0, FTI_Conf->dcpInfoPosix.BlockSize );
                memcpy( block, ptr, chunkSize );

                //                FTI_Conf->dcpInfoPosix.hashFunc( block, FTI_Conf->dcpInfoPosix.BlockSize, &FTI_DataVar->dcpInfoPosix.currentHashArray[hashIdx] );
                ptr = block;
                chunkSize = FTI_Conf->dcpInfoPosix.BlockSize;
            }
            /*else {
              FTI_Conf->dcpInfoPosix.hashFunc( ptr, FTI_Conf->dcpInfoPosix.BlockSize, &FTI_DataVar->dcpInfoPosix.currentHashArray[hashIdx] );
              }*/

            bool commitBlock;
            // if old hash exists, compare. If datasize increased, there wont be an old hash to compare with.
            if( offset < FTI_DataVar->dcpInfoPosix.hashDataSize ) {
                commitBlock = memcmp( &(FTI_DataVar->dcpInfoPosix.currentHashArray[hashIdx]), &(FTI_DataVar->dcpInfoPosix.oldHashArray[hashIdx]), FTI_Conf->dcpInfoPosix.digestWidth );
            } else {
                commitBlock = true;
            }

            bool success = true;
            int fileUpdate = 0;
            if( commitBlock ) {
                if( dcpLayer > 0 ) {
                    FWRITE(FTI_NSCS, success,&blockMeta, 6,1,write_info->f,"p",block);
                    if( success) fileUpdate += 6;
                }
                if( success ) {
                    FWRITE(FTI_NSCS, success,ptr, chunkSize,1,write_info->f,"p",block);
                    if( success ) fileUpdate += chunkSize;
                }
                FTI_Exec->dcpInfoPosix.FileSize += success*fileUpdate;
                write_DCPinfo->layerSize += success*fileUpdate;

                FTI_Exec->dcpInfoPosix.dcpSize += success*dcpChunkSize;
                if(success) {
                    MD5_Update( &write_info->integrity, &FTI_DataVar->dcpInfoPosix.currentHashArray[hashIdx], MD5_DIGEST_LENGTH ); 
                }
            }
            offset += dcpChunkSize*success;
            pos += dcpChunkSize*success;
            ptr = ptr + dcpChunkSize ; //chunkSize*success;
        }
        if ( FTI_Try(FTI_getPrefetchedData ( &prefetcher, &totalBytes, &ptr), " Fetching Next Memory block from memory") != FTI_SCES ){
            return FTI_NSCS;
        }

    }
    // swap hash arrays and free old one
    //    free(FTI_DataVar->dcpInfoPosix.hashArray);
    FTI_DataVar->dcpInfoPosix.hashDataSize = dataSize;
    unsigned char *tmp = FTI_DataVar->dcpInfoPosix.currentHashArray;
    FTI_DataVar->dcpInfoPosix.currentHashArray = FTI_DataVar->dcpInfoPosix.oldHashArray;
    FTI_DataVar->dcpInfoPosix.oldHashArray = tmp;
    //    FTI_DataVar->dcpInfoPosix.hashArray = FTI_DataVar->dcpInfoPosix.hashArrayTmp;

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

int FTI_PosixDCPClose(void *fileDesc)
{
    WriteDCPPosixInfo_t *write_dcpInfo = (WriteDCPPosixInfo_t *) fileDesc;
    FTIT_execution *FTI_Exec = write_dcpInfo->FTI_Exec;
    FTIT_configuration *FTI_Conf = write_dcpInfo->FTI_Conf;
    FTIT_checkpoint *FTI_Ckpt = write_dcpInfo->FTI_Ckpt;
    FTIT_topology *FTI_Topo = write_dcpInfo->FTI_Topo;


    char errstr[FTI_BUFS];

    int dcpFileId = FTI_Exec->dcpInfoPosix.Counter / FTI_Conf->dcpInfoPosix.StackSize;
    // dcpLayer corresponds to the additional layers towards the base layer.
    int dcpLayer = FTI_Exec->dcpInfoPosix.Counter % FTI_Conf->dcpInfoPosix.StackSize;

    if (FTI_Conf->dcpInfoPosix.cachedCkpt){ 
        FTI_CLOSE_ASYNC((write_dcpInfo->write_info.f));
    }
    else{
        FTI_PosixSync(&(write_dcpInfo->write_info));
        FTI_PosixClose(&(write_dcpInfo->write_info));
    }

    // create final dcp layer hash
    unsigned char LayerHash[MD5_DIGEST_LENGTH];
    MD5_Final( LayerHash, &(write_dcpInfo->write_info.integrity) );
    FTI_GetHashHexStr( LayerHash, MD5_DIGEST_LENGTH, &FTI_Exec->dcpInfoPosix.LayerHash[dcpLayer*MD5_DIGEST_STRING_LENGTH] );
    // layer size is needed in order to create layer hash during recovery
    FTI_Exec->dcpInfoPosix.LayerSize[dcpLayer] = write_dcpInfo->layerSize;
    FTI_Exec->dcpInfoPosix.Counter++;
    if( (dcpLayer == 0) ) {
        char ofn[512];
        snprintf( ofn, FTI_BUFS, "%s/dcp-id%d-rank%d.fti", FTI_Ckpt[4].dcpDir, dcpFileId-1, FTI_Topo->myRank );
        if( (remove(ofn) < 0) && (errno != ENOENT) ) {
            snprintf(errstr, FTI_BUFS, "cannot delete file '%s'", ofn );
            FTI_Print( errstr, FTI_WARN ); 
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
int FTI_RecoverDcpPosix
( 
 FTIT_configuration* FTI_Conf, 
 FTIT_execution* FTI_Exec, 
 FTIT_checkpoint* FTI_Ckpt, 
 FTIT_dataset* FTI_Data 
 )

{
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
        int idx = FTI_DataGetIdx(varId, FTI_Exec, FTI_Data);
        if( idx < 0 ) {
            snprintf(errstr, FTI_BUFS, "id '%d' does not exist!", varId);
            FTI_Print( errstr, FTI_EROR );
            return FTI_NSCS;
        }
#ifdef GPUSUPPORT
        if (FTI_Data[idx].isDevicePtr){
            FTI_TransferFileToDeviceAsync(fd,FTI_Data[idx].devicePtr, FTI_Data[i].size); 
        }
        else{ 
            fread( FTI_Data[idx].ptr, locDataSize, 1, fd );
            if(ferror(fd)) {
                snprintf( errstr, FTI_BUFS, "unable to read in file %s", fn );
                FTI_Print( errstr, FTI_EROR );
                return FTI_NSCS;
            }
        }

#else            
        fread( FTI_Data[idx].ptr, locDataSize, 1, fd );
        if(ferror(fd)) {
            snprintf( errstr, FTI_BUFS, "unable to read in file %s", fn );
            FTI_Print( errstr, FTI_EROR );
            return FTI_NSCS;
        }
#endif
        int overflow;
        if( (overflow=locDataSize%blockSize) != 0 ) {
            fread( buffer, blockSize - overflow, 1, fd );
            if(ferror(fd)) {
                snprintf( errstr, FTI_BUFS, "unable to read in file %s", fn );
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
            int idx = FTI_DataGetIdx(blockMeta.varId, FTI_Exec, FTI_Data);
            if( idx < 0 ) {
                snprintf(errstr, FTI_BUFS, "id '%d' does not exist!", blockMeta.varId);
                FTI_Print( errstr, FTI_EROR );
                return FTI_NSCS;
            }


            offset = blockMeta.blockId * blockSize;
            unsigned int chunkSize = ( (FTI_Data[idx].size-offset) < blockSize ) ? FTI_Data[idx].size-offset : blockSize; 

#ifdef GPUSUPPORT
            if ( FTI_Data[idx].isDevicePtr){
                FTI_device_sync();
                fread( block, 1, chunkSize, fd );
                FTI_copy_to_device_async(FTI_Data[idx].devicePtr + offset ,block, chunkSize); 
            }
            else{
                void* ptr = FTI_Data[idx].ptr + offset;
                fread( ptr, 1, chunkSize, fd );
            }
#else
            void* ptr = FTI_Data[idx].ptr + offset;
            fread( ptr, 1, chunkSize, fd );
#endif
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

            pos += (blockSize+6);
        }

    }

    // create hasharray
    for(i=0; i<FTI_Exec->nbVar; i++) {
        FTIT_data_prefetch prefetcher;
        size_t totalBytes = 0;
        unsigned char * ptr,*startPtr;
        FTIT_dataset *FTI_DataVar = &FTI_Data[i];

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
    }


    FTI_Exec->reco = 0;

    free(buffer);
    fclose(fd);

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
    /*
    // create hasharray for id
    i = FTI_DataGetIdx( id, FTI_Exec, FTI_Data );
    unsigned long nbBlocks = (FTI_Data[i].size % blockSize) ? FTI_Data[i].size/blockSize + 1 : FTI_Data[i].size/blockSize;
    FTI_Data[i].dcpInfoPosix.hashDataSize = FTI_Data[i].size;
    //    FTI_Data[i].dcpInfoPosix.hashArray = realloc(FTI_Data[i].dcpInfoPosix.hashArray, nbBlocks*MD5_DIGEST_LENGTH);
    int j;
    for(j=0; j<nbBlocks-1; j++) {
    unsigned long offset = j*blockSize;
    unsigned long hashIdx = j*MD5_DIGEST_LENGTH;
//        MD5( FTI_Data[i].ptr+offset, blockSize, &FTI_Data[i].dcpInfoPosix.hashArray[hashIdx] );
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
//        MD5( buffer, blockSize, &FTI_Data[i].dcpInfoPosix.hashArray[(nbBlocks-1)*MD5_DIGEST_LENGTH] );
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

/*-------------------------------------------------------------------------*/
/**
  @brief      It compares checksum of the checkpoint file.
  @param      fileName        Filename of the checkpoint.
  @param      checksumToCmp   Checksum to compare.
  @return     integer         FTI_SCES if successful.

  dCP POSIX implementation of FTI_VerifyChecksum().
 **/
/*-------------------------------------------------------------------------*/
    int FTI_VerifyChecksumDcpPosix
(
 char* fileName
 )

{
    FTIT_execution* exec = FTI_DcpPosixRecoverRuntimeInfo( DCP_POSIX_EXEC_TAG, NULL, NULL );
    FTIT_configuration* conf = FTI_DcpPosixRecoverRuntimeInfo( DCP_POSIX_CONF_TAG, NULL, NULL ); 

    char errstr[FTI_BUFS];
    char dummyBuffer[FTI_BUFS];
    unsigned long blockSize;
    unsigned int stackSize;
    unsigned int counter;
    unsigned int dcpFileId;
    int lastCorrectLayer=-1;

    FILE *fd = fopen(fileName, "rb");
    if (fd == NULL) {
        char str[FTI_BUFS];
        sprintf(str, "FTI failed to open file %s to calculate checksum.", fileName);
        FTI_Print(str, FTI_WARN);
        goto FINALIZE;
    }

    unsigned char md5_tmp[MD5_DIGEST_LENGTH];
    unsigned char md5_final[MD5_DIGEST_LENGTH];

    MD5_CTX mdContext;

    // position in file
    size_t fs = 0;

    // get blocksize
    fs += fread( &blockSize, 1, sizeof(unsigned long), fd );
    if(ferror(fd)) {
        snprintf( errstr, FTI_BUFS, "unable to read in file %s", fileName );
        FTI_Print( errstr, FTI_EROR );
        goto FINALIZE;
    }
    fs += fread( &stackSize, 1, sizeof(unsigned int), fd );
    if(ferror(fd)) {
        snprintf( errstr, FTI_BUFS, "unable to read in file %s", fileName );
        FTI_Print( errstr, FTI_EROR );
        goto FINALIZE;
    }

    // check if settings are correckt. If not correct them
    if( blockSize != conf->dcpInfoPosix.BlockSize )
    {
        char str[FTI_BUFS];
        snprintf( str, FTI_BUFS, "dCP blocksize differ between configuration settings ('%lu') and checkpoint file ('%lu')", conf->dcpInfoPosix.BlockSize, blockSize );
        FTI_Print( str, FTI_WARN );
        conf->dcpInfoPosix.BlockSize = blockSize;
    }
    if( stackSize != conf->dcpInfoPosix.StackSize )
    {
        char str[FTI_BUFS];
        snprintf( str, FTI_BUFS, "dCP stacksize differ between configuration settings ('%u') and checkpoint file ('%u')", conf->dcpInfoPosix.StackSize, stackSize );
        FTI_Print( str, FTI_WARN );
        conf->dcpInfoPosix.StackSize = stackSize;
    }

    // get dcpFileId from filename
    int dummy;
    sscanf( exec->meta[4].ckptFile, "dcp-id%d-rank%d.fti", &dcpFileId, &dummy );
    counter = dcpFileId * stackSize;

    int i;
    int layer = 0;
    int nbVarLayer;
    int ckptID;

    // set number of recovered layers to 0
    exec->dcpInfoPosix.nbLayerReco = 0;

    // data buffer
    void* buffer = malloc( blockSize );
    if( !buffer ) {
        FTI_Print("unable to allocate memory!", FTI_EROR);
        goto FINALIZE;
    }

    // check layer 0 first
    // get number of variables stored in layer
    MD5_Init( &mdContext );
    fs += fread( &ckptID, 1, sizeof(int), fd );
    if(ferror(fd)|| feof(fd)) {
        snprintf( errstr, FTI_BUFS, "unable to read in file %s", fileName );
        FTI_Print( errstr, FTI_EROR );
        goto FINALIZE;
    }
    fs += fread( &nbVarLayer, 1, sizeof(int), fd );
    if(ferror(fd) || feof(fd) ) {
        snprintf( errstr, FTI_BUFS, "unable to read in file %s", fileName );
        FTI_Print( errstr, FTI_EROR );
        goto FINALIZE;
    }
    for(i=0; i<nbVarLayer; i++) {
        unsigned long dataSize;
        unsigned long pos = 0;
        fs += fread( dummyBuffer, 1, sizeof(int), fd );
        if(ferror(fd)|| feof(fd)) {
            snprintf( errstr, FTI_BUFS, "unable to read in file %s", fileName );
            FTI_Print( errstr, FTI_EROR );
            goto FINALIZE;
        }
        fs += fread( &dataSize, 1, sizeof(unsigned long), fd );
        if(ferror(fd)|| feof(fd)) {
            snprintf( errstr, FTI_BUFS, "unable to read in file %s", fileName );
            FTI_Print( errstr, FTI_EROR );
            goto FINALIZE;
        }
        while( pos < dataSize ) {
            pos += fread( buffer, 1, blockSize, fd );
            if(ferror(fd)|| feof(fd)) {
                snprintf( errstr, FTI_BUFS, "unable to read in file %s", fileName );
                FTI_Print( errstr, FTI_EROR );
                goto FINALIZE;
            }
            MD5( buffer, blockSize, md5_tmp );
            MD5_Update( &mdContext, md5_tmp, MD5_DIGEST_LENGTH );
        }
        fs += pos;
    }
    MD5_Final( md5_final, &mdContext );
    // compare hashes
    if( strcmp( FTI_GetHashHexStr( md5_final, conf->dcpInfoPosix.digestWidth, NULL ), &exec->dcpInfoPosix.LayerHash[layer*MD5_DIGEST_STRING_LENGTH] ) ) {
        FTI_Print("hashes differ in base", FTI_WARN);
        goto FINALIZE;
    }
    size_t *layerSizes = (size_t*) malloc (sizeof(size_t)*stackSize);
    int *ckptIDs  = (int*) malloc (sizeof(int)*stackSize);
    int *nbVarLayers = (int *) malloc (sizeof(int)*stackSize);

    layerSizes[layer] = fs;
    ckptIDs[layer] = ckptID;
    nbVarLayers[layer] = nbVarLayer;

    lastCorrectLayer++;
    layer++;
    exec->dcpInfoPosix.nbLayerReco = layer;
    exec->dcpInfoPosix.nbVarReco = nbVarLayer;
    exec->ckptID = ckptID;

    //exec->dcpInfoPosix.Counter = counter;
    bool readLayer = true;
    size_t bytes;
    size_t layerSize;
    // now treat other layers

    for(; layer<stackSize && readLayer; layer++) {
        readLayer = true;
        layerSize = 0;
        MD5_Init( &mdContext );
        bytes = fread( &ckptID, 1, sizeof(int), fd );
        if (feof(fd) ){
            readLayer = false;
            break;
        }
        if(ferror(fd)) {
            snprintf( errstr, FTI_BUFS, "unable to read in file %s", fileName );
            FTI_Print( errstr, FTI_EROR );
            goto FINALIZE;
        }

        layerSize += bytes;

        bytes = fread( &nbVarLayer, 1, sizeof(int), fd );

        if (feof(fd) ){
            readLayer = false;
            break;
        }


        if(ferror(fd)) {
            snprintf( errstr, FTI_BUFS, "unable to read in file %s", fileName );
            FTI_Print( errstr, FTI_EROR );
            goto FINALIZE;
        }

        layerSize += bytes;

        while( (readLayer) && layerSize < exec->dcpInfoPosix.LayerSize[layer] ) {
            bytes = fread( dummyBuffer, 1, 6, fd );
            if (feof(fd) ){
                readLayer = false;
                break;
            }
            if(ferror(fd)) {
                snprintf( errstr, FTI_BUFS, "unable to read in file %s", fileName );
                FTI_Print( errstr, FTI_EROR );
                goto FINALIZE;
            }
            layerSize += bytes;

            bytes = fread( buffer, 1, blockSize, fd );
            if (feof(fd) ){
                readLayer = false;
                break;
            }

            if(ferror(fd)) {
                snprintf( errstr, FTI_BUFS, "unable to read in file %s", fileName );
                FTI_Print( errstr, FTI_EROR );
                goto FINALIZE;
            }
            layerSize += bytes;

            MD5( buffer, blockSize, md5_tmp );
            MD5_Update( &mdContext, md5_tmp, MD5_DIGEST_LENGTH ); 
        }
        MD5_Final( md5_final, &mdContext );
        // compare hashes
        if( readLayer && strcmp( FTI_GetHashHexStr( md5_final, conf->dcpInfoPosix.digestWidth, NULL ), &exec->dcpInfoPosix.LayerHash[layer*MD5_DIGEST_STRING_LENGTH] ) ) {
            readLayer = false;
        }

        if (readLayer){
            fs += layerSize;
            layerSizes[layer] = fs;
            ckptIDs[layer] = ckptID;
            nbVarLayers[layer] = nbVarLayer;
            exec->dcpInfoPosix.nbLayerReco = layer+1;
            exec->dcpInfoPosix.nbVarReco = nbVarLayer;
            lastCorrectLayer++;
        }
    }


FINALIZE:;
         int minLayer =0;

         MPI_Allreduce(&lastCorrectLayer, &minLayer, 1, MPI_INT, MPI_MIN, FTI_COMM_WORLD); 
         if ( minLayer == -1 ){
             exec->dcpInfoPosix.Counter= 0;
             fclose (fd);
             return FTI_NSCS;
         }
         else{
             exec->dcpInfoPosix.nbLayerReco = minLayer+1;
             exec->dcpInfoPosix.nbVarReco = nbVarLayers[minLayer];
             exec->dcpInfoPosix.Counter= minLayer +counter + 1;
             fclose (fd);
             exec->ckptID = ckptIDs[minLayer];
             if ( truncate(fileName, layerSizes[minLayer]) != 0 ){
                 FTI_Print("Error On Truncating the file",FTI_EROR);
                 return FTI_EROR;
             }

             free(layerSizes);
             free(ckptIDs);
             free(nbVarLayers);
             return FTI_SCES;
         }
         // truncate file if some layer were not possible to recover.


}

// HELPER FUNCTIONS

void* FTI_DcpPosixRecoverRuntimeInfo( int tag, void* exec_, void* conf_ ) {

    static void* exec = NULL;
    static void* conf = NULL;

    void* ret;

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


