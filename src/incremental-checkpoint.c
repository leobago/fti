#include <fti-int/incremental_checkpoint.h>
#include "interface.h"
#include "utility.h"
#include "FTI_IO.h"

int FTI_startICP(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt,
        FTIT_dataset* FTI_Data, FTIT_IO *io)
{
    void *ret = io->initCKPT(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt, FTI_Data);
    FTI_Exec->iCPInfo.fd = ret;
    return FTI_SCES;
}


int FTI_WriteVar(int varID, FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt,
        FTIT_dataset* FTI_Data, FTIT_IO *io)
{
    void *write_info = (void *) FTI_Exec->iCPInfo.fd;
    int res;
    int i;
    for (i = 0; i < FTI_Exec->nbVar; i++) {
        if ( FTI_Data[i].id == varID ) {
            FTI_Data[i].filePos = io->getPos(write_info);
            res = io->WriteData(&FTI_Data[i],write_info);
        }
    }
    FTI_Exec->iCPInfo.result = res;
    return res;
}


int FTI_FinishICP(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec, FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt, FTIT_dataset* FTI_Data, FTIT_IO *io){
    if ( FTI_Exec->iCPInfo.status == FTI_ICP_FAIL ) {
        return FTI_NSCS;
    }
    void *write_info = FTI_Exec->iCPInfo.fd;
    io->finCKPT(write_info);
    io->finIntegrity(FTI_Exec->integrity, write_info);
    free(write_info);
    FTI_Exec->iCPInfo.fd = NULL;
    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Initializes iCP for dCP POSIX I/O.
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
int FTI_InitPosixIcpDcp(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt,
        FTIT_dataset* FTI_Data)
{
    /*
    char errstr[FTI_BUFS];
    
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
            free(FTI_Data[i].dcpInfoPosix.hashArray);
            FTI_Data[i].dcpInfoPosix.hashArray = NULL;
            FTI_Data[i].dcpInfoPosix.hashDataSize = 0;
        }
    }
    
    // for file hash create hash only from data block hashes
    MD5_Init(&FTI_Exec->iCPInfo.ctx[dcpLayer]); 
   
    char fn[FTI_BUFS];

    snprintf( FTI_Exec->meta[0].ckptFile, FTI_BUFS, "dcp-id%d-rank%d.fti", dcpFileId, FTI_Topo->myRank );
    if (FTI_Ckpt[4].isInline) { //If inline L4 save directly to global directory
        snprintf( fn, FTI_BUFS, "%s/%s", FTI_Ckpt[4].dcpDir, FTI_Exec->meta[0].ckptFile );
    } else {
        snprintf( fn, FTI_BUFS, "%s/%s", FTI_Ckpt[1].dcpDir, FTI_Exec->meta[0].ckptFile );
    }

    FILE *fd;
    if( dcpLayer == 0 ) {
        fd = fopen( fn, "wb" );
        if( fd == NULL ) {
            snprintf( errstr, FTI_BUFS, "Cannot create file '%s'!", fn ); 
            FTI_Print( errstr, FTI_EROR );
            return FTI_NSCS;
        }
    } else {
        fd = fopen( fn, "ab" );
        if( fd == NULL ) {
            snprintf( errstr, FTI_BUFS, "Cannot open file '%s' in append mode!", fn );
            FTI_Print( errstr, FTI_EROR );
            return FTI_NSCS;
        }
    }
    
    if( dcpLayer == 0 ) FTI_Exec->dcpInfoPosix.FileSize = 0;
    
    // write constant meta data in the beginning of file
    // - blocksize
    // - stacksize
    if( dcpLayer == 0 ) {
        while( !fwrite( &FTI_Conf->dcpInfoPosix.BlockSize, sizeof(unsigned long), 1, fd ) ) {
            if(ferror(fd)) {
                snprintf( errstr, FTI_BUFS, "unable to write in file %s", FTI_Exec->iCPInfo.fn );
                FTI_Print( errstr, FTI_EROR );
                return FTI_NSCS;
            }
        }
        while( !fwrite( &FTI_Conf->dcpInfoPosix.StackSize, sizeof(unsigned int), 1, fd ) ) {
            if(ferror(fd)) {
                snprintf( errstr, FTI_BUFS, "unable to write in file %s", FTI_Exec->iCPInfo.fn );
                FTI_Print( errstr, FTI_EROR );
                return FTI_NSCS;
            }
        }
        FTI_Exec->dcpInfoPosix.FileSize += sizeof(unsigned long) + sizeof(unsigned int);
        FTI_Exec->iCPInfo.layerSize += sizeof(unsigned long) + sizeof(unsigned int);
    }
    
    // write actual amount of variables at the beginning of each layer
    while( !fwrite( &FTI_Exec->ckptID, sizeof(int), 1, fd ) ) {
        if(ferror(fd)) {
            snprintf( errstr, FTI_BUFS, "unable to write in file %s", FTI_Exec->iCPInfo.fn );
            FTI_Print( errstr, FTI_EROR );
            return FTI_NSCS;
        }
    }
    while( !fwrite( &FTI_Exec->nbVar, sizeof(int), 1, fd ) ) {
        if(ferror(fd)) {
            snprintf( errstr, FTI_BUFS, "unable to write in file %s", FTI_Exec->iCPInfo.fn );
            FTI_Print( errstr, FTI_EROR );
            return FTI_NSCS;
        }
    }
    FTI_Exec->dcpInfoPosix.FileSize += 2*sizeof(int);// + sizeof(unsigned int);
    FTI_Exec->iCPInfo.layerSize += 2*sizeof(int);// + sizeof(unsigned int);
    
    memcpy( FTI_Exec->iCPInfo.fh, &fd, sizeof(FTI_PO_FH) );

*/
    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Writes dataset into dCP ckpt file using POSIX.
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Topo        Topology metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @param      FTI_Data        Dataset metadata.
  @return     integer         FTI_SCES if successful.
 **/
/*-------------------------------------------------------------------------*/
int FTI_WritePosixDcpVar(int id, FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt, FTIT_dataset* FTI_Data)
{
    /*
    // dcpFileId increments every dcpStackSize checkpoints.
    int dcpFileId = FTI_Exec->dcpInfoPosix.Counter / FTI_Conf->dcpInfoPosix.StackSize;

    // dcpLayer corresponds to the additional layers towards the base layer.
    int dcpLayer = FTI_Exec->dcpInfoPosix.Counter % FTI_Conf->dcpInfoPosix.StackSize;
   
    FILE *fd;
    int res;
    memcpy( &fd, FTI_Exec->iCPInfo.fh, sizeof(FTI_PO_FH) );

    char errstr[FTI_BUFS];
    

    unsigned char * block = (unsigned char*) malloc( FTI_Conf->dcpInfoPosix.BlockSize );
    int i = 0;
    
    for(; i<FTI_Exec->nbVar; i++) {
        
        unsigned int varId = FTI_Data[i].id;
        
        if( varId == id ) {
            FTI_Exec->dcpInfoPosix.dataSize += FTI_Data[i].size;
            unsigned long dataSize = FTI_Data[i].size;
            unsigned long nbHashes = dataSize/FTI_Conf->dcpInfoPosix.BlockSize + (bool)(dataSize%FTI_Conf->dcpInfoPosix.BlockSize);

            if( dataSize > (MAX_BLOCK_IDX*FTI_Conf->dcpInfoPosix.BlockSize) ) {
                snprintf( errstr, FTI_BUFS, "overflow in size of dataset with id: %d (datasize: %lu > MAX_DATA_SIZE: %lu)", 
                        FTI_Data[i].id, dataSize, ((unsigned long)MAX_BLOCK_IDX)*((unsigned long)FTI_Conf->dcpInfoPosix.BlockSize) );
                FTI_Print( errstr, FTI_EROR );
                return FTI_NSCS;
            }
            if( varId > MAX_VAR_ID ) {
                snprintf( errstr, FTI_BUFS, "overflow in ID (id: %d > MAX_ID: %d)!", FTI_Data[i].id, (int)MAX_VAR_ID );
                FTI_Print( errstr, FTI_EROR );
                return FTI_NSCS;
            }

            // allocate tmp hash array
            FTI_Data[i].dcpInfoPosix.hashArrayTmp = (unsigned char*) malloc( sizeof(unsigned char)*nbHashes*FTI_Conf->dcpInfoPosix.digestWidth );

            // create meta data buffer
            blockMetaInfo_t blockMeta;
            blockMeta.varId = FTI_Data[i].id;

            if( dcpLayer == 0 ) {
                while( !fwrite( &FTI_Data[i].id, sizeof(int), 1, fd ) ) {
                    if(ferror(fd)) {
                        snprintf( errstr, FTI_BUFS, "unable to write in file %s", FTI_Exec->iCPInfo.fn );
                        FTI_Print( errstr, FTI_EROR );
                        return FTI_NSCS;
                    }
                }
                while( !fwrite( &dataSize, sizeof(unsigned long), 1, fd ) ) {
                    if(ferror(fd)) {
                        snprintf( errstr, FTI_BUFS, "unable to write in file %s", FTI_Exec->iCPInfo.fn );
                        FTI_Print( errstr, FTI_EROR );
                    }
                }
                FTI_Exec->dcpInfoPosix.FileSize += (sizeof(int) + sizeof(unsigned long));
                FTI_Exec->iCPInfo.layerSize += sizeof(int) + sizeof(unsigned long);
            }
            unsigned long pos = 0;
            unsigned char * ptr = FTI_Data[i].ptr;

            while( pos < dataSize ) {

                // hash index
                unsigned int blockId = pos/FTI_Conf->dcpInfoPosix.BlockSize;
                unsigned int hashIdx = blockId*FTI_Conf->dcpInfoPosix.digestWidth;

                blockMeta.blockId = blockId;

                unsigned int chunkSize = ( (dataSize-pos) < FTI_Conf->dcpInfoPosix.BlockSize ) ? dataSize-pos : FTI_Conf->dcpInfoPosix.BlockSize;
                unsigned int dcpChunkSize = chunkSize;

                // compute hashes
                if( chunkSize < FTI_Conf->dcpInfoPosix.BlockSize ) {
                    // if block smaller pad with zeros
                    memset( block, 0x0, FTI_Conf->dcpInfoPosix.BlockSize );
                    memcpy( block, ptr, chunkSize );
                    FTI_Conf->dcpInfoPosix.hashFunc( block, FTI_Conf->dcpInfoPosix.BlockSize, &FTI_Data[i].dcpInfoPosix.hashArrayTmp[hashIdx] );
                    ptr = block;
                    chunkSize = FTI_Conf->dcpInfoPosix.BlockSize;
                } else {
                    FTI_Conf->dcpInfoPosix.hashFunc( ptr, FTI_Conf->dcpInfoPosix.BlockSize, &FTI_Data[i].dcpInfoPosix.hashArrayTmp[hashIdx] );
                }

                bool commitBlock;
                // if old hash exists, compare. If datasize increased, there wont be an old hash to compare with.
                if( pos < FTI_Data[i].dcpInfoPosix.hashDataSize ) {
                    commitBlock = memcmp( &FTI_Data[i].dcpInfoPosix.hashArray[hashIdx], &FTI_Data[i].dcpInfoPosix.hashArrayTmp[hashIdx], FTI_Conf->dcpInfoPosix.digestWidth );
                } else {
                    commitBlock = true;
                }

                bool success = true;
                int fileUpdate = 0;
                if( commitBlock ) {
                    if( dcpLayer > 0 ) {
                        success = (bool)fwrite( &blockMeta, 6, 1, fd );
                        if( success) fileUpdate += 6;
                    }
                    if( success ) {
                        success = (bool)fwrite( ptr, chunkSize, 1, fd );
                        if( success ) fileUpdate += chunkSize;
                    }
                    FTI_Exec->dcpInfoPosix.FileSize += success*fileUpdate;
                    FTI_Exec->iCPInfo.layerSize += success*fileUpdate;
                    FTI_Exec->dcpInfoPosix.dcpSize += success*dcpChunkSize;
                    if(success) {
                        MD5_Update( &FTI_Exec->iCPInfo.ctx[dcpLayer], &FTI_Data[i].dcpInfoPosix.hashArrayTmp[hashIdx], MD5_DIGEST_LENGTH ); 
                    }
                }

                pos += chunkSize*success;
                ptr = FTI_Data[i].ptr + pos; //chunkSize*success;

            }

            // swap hash arrays and free old one
            free(FTI_Data[i].dcpInfoPosix.hashArray);
            FTI_Data[i].dcpInfoPosix.hashDataSize = dataSize;
            FTI_Data[i].dcpInfoPosix.hashArray = FTI_Data[i].dcpInfoPosix.hashArrayTmp;
        }
    }

    free(block);
    
    FTI_Exec->iCPInfo.result = FTI_SCES;

    return FTI_SCES;
  */ 
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Finalizes iCP for dCP POSIX I/O.
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Topo        Topology metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @param      FTI_Data        Dataset metadata.
  @return     integer         FTI_SCES if successful.

  This function takes care of the I/O specific actions needed to
  finalize iCP.
 **/
/*-------------------------------------------------------------------------*/
int FTI_FinalizePosixDcpICP(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt,
        FTIT_dataset* FTI_Data)
{
    /*
    if ( FTI_Exec->iCPInfo.status == FTI_ICP_FAIL ) {
        return FTI_NSCS;
    }
    
    char errstr[FTI_BUFS];
    
    int dcpFileId = FTI_Exec->dcpInfoPosix.Counter / FTI_Conf->dcpInfoPosix.StackSize;

    // dcpLayer corresponds to the additional layers towards the base layer.
    int dcpLayer = FTI_Exec->dcpInfoPosix.Counter % FTI_Conf->dcpInfoPosix.StackSize;
    

    FILE *fd;
    memcpy( &fd, FTI_Exec->iCPInfo.fh, sizeof(FTI_PO_FH) );
    
    fsync(fileno(fd));
    fclose( fd );

    // create final dcp layer hash
    unsigned char LayerHash[MD5_DIGEST_LENGTH];
    MD5_Final( LayerHash, &FTI_Exec->iCPInfo.ctx[dcpLayer] );
    FTI_GetHashHexStr( LayerHash, MD5_DIGEST_LENGTH, &FTI_Exec->dcpInfoPosix.LayerHash[dcpLayer*MD5_DIGEST_STRING_LENGTH] );

    // layer size is needed in order to create layer hash during recovery
    FTI_Exec->dcpInfoPosix.LayerSize[dcpLayer] = FTI_Exec->iCPInfo.layerSize;

    FTI_Exec->dcpInfoPosix.Counter++;
    
    if( (dcpLayer == 0) ) {
        char ofn[512];
        snprintf( ofn, FTI_BUFS, "%s/dcp-id%d-rank%d.fti", FTI_Ckpt[4].dcpDir, dcpFileId-1, FTI_Topo->myRank );
        if( (remove(ofn) < 0) && (errno != ENOENT) ) {
            snprintf(errstr, FTI_BUFS, "cannot delete file '%s'", ofn );
            FTI_Print( errstr, FTI_WARN ); 
        }
    }
    
    */
    return FTI_SCES;
}



/*-------------------------------------------------------------------------*/
/**
  @brief      Initializes iCP for FTI-FF I/O.
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
int FTI_InitFtiffICP(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt,
        FTIT_dataset* FTI_Data, FTIT_IO *ignore)
{
    char fn[FTI_BUFS], strerr[FTI_BUFS];
    WritePosixInfo_t *write_info = (WritePosixInfo_t*) malloc (sizeof(WritePosixInfo_t));
    FTI_Print("I/O mode: FTI File Format.", FTI_DBUG);
    // only for printout of dCP share in FTI_Checkpoint
    FTI_Exec->FTIFFMeta.dcpSize = 0;
    // important for reading and writing operations
    FTI_Exec->FTIFFMeta.dataSize = 0;
    FTI_Exec->FTIFFMeta.pureDataSize = 0;

    //If inline L4 save directly to global directory
    int level = FTI_Exec->ckptLvel;
    if (level == 4 && FTI_Ckpt[4].isInline) { 
        if( FTI_Conf->dcpFtiff&& FTI_Ckpt[4].isDcp ) {
            snprintf(fn, FTI_BUFS, "%s/%s", FTI_Ckpt[4].dcpDir, FTI_Ckpt[4].dcpName);
        } else {
            snprintf(fn, FTI_BUFS, "%s/%s", FTI_Conf->gTmpDir, FTI_Exec->meta[0].ckptFile);
        }
    } else if ( level == 4 && !FTI_Ckpt[4].isInline )
        if( FTI_Conf->dcpFtiff && FTI_Ckpt[4].isDcp ) {
            snprintf(fn, FTI_BUFS, "%s/%s", FTI_Ckpt[1].dcpDir, FTI_Ckpt[4].dcpName);
        } else {
            snprintf(fn, FTI_BUFS, "%s/%s", FTI_Conf->lTmpDir, FTI_Exec->meta[0].ckptFile);
        }
        else {
            snprintf(fn, FTI_BUFS, "%s/%s", FTI_Conf->lTmpDir, FTI_Exec->meta[0].ckptFile);
        }

    // for dCP: create if not exists, open if exists
    if ( FTI_Conf->dcpFtiff && FTI_Ckpt[4].isDcp ){ 
        if (access(fn,R_OK) != 0){ 
            write_info->flag = 'w'; 
        }
        else {
            write_info->flag = 'e'; //e means extend file 
        }
    }
    else {
        write_info->flag = 'w';
    }
    write_info->offset = 0;
    FTI_PosixOpen(fn,write_info);
    FTI_Exec -> iCPInfo.fd = write_info;
    return FTI_SCES;

}

/*-------------------------------------------------------------------------*/
/**
  @brief      Writes dataset into ckpt file using FTI-FF.
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Topo        Topology metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @param      FTI_Data        Dataset metadata.
  @return     integer         FTI_SCES if successful.
 **/
/*-------------------------------------------------------------------------*/
int FTI_WriteFtiffVar(int varID, FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt,
        FTIT_dataset* FTI_Data, FTIT_IO *ignore)
{
    char str[FTI_BUFS];

    FTIFF_db *db = FTI_Exec->firstdb;
    FTIFF_dbvar *dbvar = NULL;
    unsigned char *dptr;
    int dbvar_idx, dbcounter=0;
    int isnextdb;
    long dcpSize = 0;
    long dataSize = 0;
    long pureDataSize = 0;

    int pvar_idx = -1, pvar_idx_;
    for( pvar_idx_=0; pvar_idx_<FTI_Exec->nbVar; pvar_idx_++ ) {
        if( FTI_Data[pvar_idx_].id == varID ) {
            pvar_idx = pvar_idx_;
        }
    }
    if( pvar_idx == -1 ) {
        FTI_Print("FTI_WriteFtiffVar: Illegal ID", FTI_WARN);
        return FTI_NSCS;
    }

    FTIFF_UpdateDatastructVarFTIFF( FTI_Exec, FTI_Data, FTI_Conf, pvar_idx );

    // check if metadata exists
    if( FTI_Exec->firstdb == NULL ) {
        FTI_Print("No data structure found to write data to file. Discarding checkpoint.", FTI_WARN);
        return FTI_NSCS;
    }

    WritePosixInfo_t *fd = FTI_Exec->iCPInfo.fd;

    db = FTI_Exec->firstdb;

    do {    

        isnextdb = 0;

        for(dbvar_idx=0;dbvar_idx<db->numvars;dbvar_idx++) {

            dbvar = &(db->dbvars[dbvar_idx]);

            if( dbvar->id == varID ) {
                unsigned char hashchk[MD5_DIGEST_LENGTH];
                // important for dCP!
                // TODO check if we can use:
                // 'dataSize += dbvar->chunksize'
                // for dCP disabled
                dataSize += dbvar->containersize;
                if( dbvar->hascontent ) 
                    pureDataSize += dbvar->chunksize;

                FTI_ProcessDBVar(FTI_Exec, FTI_Conf, dbvar , FTI_Data, hashchk, fd, FTI_Exec->iCPInfo.fn , &dcpSize, &dptr);
                // create hash for datachunk and assign to member 'hash'
                if( dbvar->hascontent ) {
                    memcpy( dbvar->hash, hashchk, MD5_DIGEST_LENGTH );
                }

                // debug information
                snprintf(str, FTI_BUFS, "FTIFF: CKPT(id:%i) dataBlock:%i/dataBlockVar%i id: %i, idx: %i"
                        ", dptr: %ld, fptr: %ld, chunksize: %ld, "
                        "base_ptr: 0x%" PRIxPTR " ptr_pos: 0x%" PRIxPTR " ", 
                        FTI_Exec->ckptID, dbcounter, dbvar_idx,  
                        dbvar->id, dbvar->idx, dbvar->dptr,
                        dbvar->fptr, dbvar->chunksize,
                        (uintptr_t)FTI_Data[dbvar->idx].ptr, (uintptr_t)dptr);
                FTI_Print(str, FTI_DBUG);

            }

        }

        if (db->next) {
            db = db->next;
            isnextdb = 1;
        }

        dbcounter++;

    } while( isnextdb );

    // only for printout of dCP share in FTI_Checkpoint
    FTI_Exec->FTIFFMeta.dcpSize += dcpSize;
    FTI_Exec->FTIFFMeta.pureDataSize += pureDataSize;

    // important for reading and writing operations
    FTI_Exec->FTIFFMeta.dataSize += dataSize;

    FTI_Exec->iCPInfo.result = FTI_SCES;

    return FTI_SCES;

}

/*-------------------------------------------------------------------------*/
/**
  @brief      Finalizes iCP for FTI-FF I/O.
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Topo        Topology metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @param      FTI_Data        Dataset metadata.
  @return     integer         FTI_SCES if successful.

  This function takes care of the I/O specific actions needed to
  finalize iCP.
 **/
/*-------------------------------------------------------------------------*/
int FTI_FinalizeFtiffICP(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt,
        FTIT_dataset* FTI_Data, FTIT_IO *ignore)
{   
    if ( FTI_Exec->iCPInfo.status == FTI_ICP_FAIL ) {
        return FTI_NSCS;
    }


    if ( FTI_Try( FTIFF_CreateMetadata( FTI_Exec, FTI_Topo, FTI_Data, FTI_Conf ), "Create FTI-FF meta data" ) != FTI_SCES ) {
        return FTI_NSCS;
    }

    WritePosixInfo_t *write_info = FTI_Exec->iCPInfo.fd;
    FTIFF_writeMetaDataFTIFF( FTI_Exec, write_info);

    FTI_PosixSync(write_info);
    FTI_PosixClose(write_info);
    free(write_info);
    FTI_Exec->iCPInfo.fd = NULL;

    return FTI_SCES;

}


/* 
 * As long SIONlib does not support seek in a single file
 * FTI does not support SIONlib I/O for incremental
 * checkpointing
 */
#if 0
#ifdef ENABLE_SIONLIB // --> If SIONlib is installed
/*-------------------------------------------------------------------------*/
/**
  @brief      Initializes iCP for SIONlib I/O.
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
int FTI_InitSionlibICP(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt,
        FTIT_dataset* FTI_Data)
{
    int res;
    FTI_Print("I/O mode: SIONlib.", FTI_DBUG);

    int numFiles = 1;
    int nlocaltasks = 1;
    int* file_map = calloc(1, sizeof(int));
    int* ranks = talloc(int, 1);
    int* rank_map = talloc(int, 1);
    sion_int64* chunkSizes = talloc(sion_int64, 1);
    int fsblksize = -1;
    chunkSizes[0] = FTI_Exec->ckptSize;
    ranks[0] = FTI_Topo->splitRank;
    rank_map[0] = FTI_Topo->splitRank;

    // open parallel file
    char fn[FTI_BUFS], str[FTI_BUFS];
    snprintf(str, FTI_BUFS, "Ckpt%d-sionlib.fti", FTI_Exec->ckptID);
    snprintf(fn, FTI_BUFS, "%s/%s", FTI_Conf->gTmpDir, str);
    int sid = sion_paropen_mapped_mpi(fn, "wb,posix", &numFiles, FTI_COMM_WORLD, &nlocaltasks, &ranks, &chunkSizes, &file_map, &rank_map, &fsblksize, NULL);


    // check if successful
    if (sid == -1) {
        errno = 0;
        FTI_Print("SIONlib: File could no be opened", FTI_EROR);

        free(file_map);
        free(rank_map);
        free(ranks);
        free(chunkSizes);
        return FTI_NSCS;
    }

    memcpy(FTI_Exec->iCPInfo.fh, &sid, sizeof(int));

    free(file_map);
    free(rank_map);
    free(ranks);
    free(chunkSizes);

    return FTI_SCES;

}

/*-------------------------------------------------------------------------*/
/**
  @brief      Writes dataset into ckpt file using SIONlib.
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Topo        Topology metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @param      FTI_Data        Dataset metadata.
  @return     integer         FTI_SCES if successful.
 **/
/*-------------------------------------------------------------------------*/
int FTI_WriteSionlibVar(int varID, FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt,
        FTIT_dataset* FTI_Data)
{

    int sid;
    memcpy( &sid, FTI_Exec->iCPInfo.fh, sizeof(FTI_SL_FH) );

    unsigned long offset = 0;
    // write datasets into file
    int i;
    for (i = 0; i < FTI_Exec->nbVar; i++) {

        if( FTI_Data[i].id == varID ) {

            // set file pointer to corresponding block in sionlib file
            int res = sion_seek(sid, FTI_Topo->splitRank, SION_CURRENT_BLK, offset);

            // check if successful
            if (res != SION_SUCCESS) {
                errno = 0;
                FTI_Print("SIONlib: Could not set file pointer", FTI_EROR);
                sion_parclose_mapped_mpi(sid);
                return FTI_NSCS;
            }

            // SIONlib write call
            res = sion_fwrite(FTI_Data[i].ptr, FTI_Data[i].size, 1, sid);

            // check if successful
            if (res < 0) {
                errno = 0;
                FTI_Print("SIONlib: Data could not be written", FTI_EROR);
                res =  sion_parclose_mapped_mpi(sid);
                return FTI_NSCS;
            }

        }

        offset += FTI_Data[i].size;

    }

    FTI_Exec->iCPInfo.result = FTI_SCES;
    return FTI_SCES;

}

/*-------------------------------------------------------------------------*/
/**
  @brief      Finalizes iCP for SIONlib I/O.
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Topo        Topology metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @param      FTI_Data        Dataset metadata.
  @return     integer         FTI_SCES if successful.

  This function takes care of the I/O specific actions needed to
  finalize iCP.
 **/
/*-------------------------------------------------------------------------*/
int FTI_FinalizeSionlibICP(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt,
        FTIT_dataset* FTI_Data)
{

    int sid;
    memcpy( &sid, FTI_Exec->iCPInfo.fh, sizeof(FTI_SL_FH) );

    // close parallel file
    if (sion_parclose_mapped_mpi(sid) == -1) {
        FTI_Print("Cannot close sionlib file.", FTI_WARN);
        return FTI_NSCS;
    }

    return FTI_SCES;

}
#endif // SIONlib enabled
#endif
