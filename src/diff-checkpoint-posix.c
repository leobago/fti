#include "interface.h"

int FTI_WritePosixDcp(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt,
        FTIT_dataset* FTI_Data)
{
    char errstr[FTI_BUFS];
    
    // dcpFileId increments every dcpStackSize checkpoints.
    int dcpFileId = FTI_Exec->dcpInfoPosix.Counter / FTI_Conf->dcpInfoPosix.StackSize;

    // dcpLayer corresponds to the additional layers towards the base layer.
    int dcpLayer = FTI_Exec->dcpInfoPosix.Counter % FTI_Conf->dcpInfoPosix.StackSize;
    
    snprintf( FTI_Exec->meta[0].ckptFile, FTI_BUFS, "%s/dcp-id%d-rank%d.fti", FTI_Ckpt[4].dcpDir, dcpFileId, FTI_Topo->myRank );

    FILE *fd;
    if( dcpLayer == 0 ) {
        fd = fopen( FTI_Exec->meta[0].ckptFile, "wb" );
        if( fd == NULL ) {
            snprintf( errstr, FTI_BUFS, "Cannot create file '%s'!", FTI_Exec->meta[0].ckptFile ); 
            FTI_Print( errstr, FTI_EROR );
            return FTI_NSCS;
        }
    } else {
        fd = fopen( FTI_Exec->meta[0].ckptFile, "ab" );
        if( fd == NULL ) {
            snprintf( errstr, FTI_BUFS, "Cannot open file '%s' in append mode!", FTI_Exec->meta[0].ckptFile );
            FTI_Print( errstr, FTI_EROR );
            return FTI_NSCS;
        }
    }

    // for file hash create hash only from data block hashes
    MD5_CTX ctx;
    MD5_Init( &ctx );

    unsigned long layerSize = 0;

    unsigned char * block = (unsigned char*) malloc( FTI_Conf->dcpInfoPosix.BlockSize );
    int i = 0;
    if( dcpLayer == 0 ) FTI_Exec->dcpInfoPosix.FileSize = 0;
    // write constant meta data in the beginning of file
    // - number of variables in basis
    // - blocksize
    // - stacksize
    if( dcpLayer == 0 ) {
        while( !fwrite( &FTI_Conf->dcpInfoPosix.BlockSize, sizeof(unsigned long), 1, fd ) ) {
            if(ferror(fd)) {
                snprintf( errstr, FTI_BUFS, "unable to write in file %s", FTI_Exec->meta[0].ckptFile );
                FTI_Print( errstr, FTI_EROR );
                return FTI_NSCS;
            }
        }
        while( !fwrite( &FTI_Conf->dcpInfoPosix.StackSize, sizeof(unsigned int), 1, fd ) ) {
            if(ferror(fd)) {
                snprintf( errstr, FTI_BUFS, "unable to write in file %s", FTI_Exec->meta[0].ckptFile );
                FTI_Print( errstr, FTI_EROR );
                return FTI_NSCS;
            }
        }
        FTI_Exec->dcpInfoPosix.FileSize += sizeof(unsigned long) + sizeof(unsigned int);
        layerSize += sizeof(unsigned long) + sizeof(unsigned int);
    }
    
    // write actual amount of variables at the beginning of each layer
    while( !fwrite( &FTI_Exec->nbVar, sizeof(int), 1, fd ) ) {
        if(ferror(fd)) {
            snprintf( errstr, FTI_BUFS, "unable to write in file %s", FTI_Exec->meta[0].ckptFile );
            FTI_Print( errstr, FTI_EROR );
            return FTI_NSCS;
        }
    }
    FTI_Exec->dcpInfoPosix.FileSize += sizeof(int);
    layerSize += sizeof(int);
    
    for(; i<FTI_Exec->nbVar; i++) {
         
        unsigned int varId = FTI_Data[i].id;
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
                    snprintf( errstr, FTI_BUFS, "unable to write in file %s", FTI_Exec->meta[0].ckptFile );
                    FTI_Print( errstr, FTI_EROR );
                    return FTI_NSCS;
                }
            }
            while( !fwrite( &dataSize, sizeof(unsigned long), 1, fd ) ) {
                if(ferror(fd)) {
                    snprintf( errstr, FTI_BUFS, "unable to write in file %s", FTI_Exec->meta[0].ckptFile );
                    FTI_Print( errstr, FTI_EROR );
                }
            }
            FTI_Exec->dcpInfoPosix.FileSize += (sizeof(int) + sizeof(unsigned long));
            layerSize += sizeof(int) + sizeof(unsigned long);
        }
        unsigned long pos = 0;
        unsigned char * ptr = FTI_Data[i].ptr;
        
        while( pos < dataSize ) {
            
            // hash index
            unsigned int blockId = pos/FTI_Conf->dcpInfoPosix.BlockSize;
            unsigned int hashIdx = blockId*FTI_Conf->dcpInfoPosix.digestWidth;
            
            blockMeta.blockId = blockId;

            unsigned int chunkSize = ( (dataSize-pos) < FTI_Conf->dcpInfoPosix.BlockSize ) ? dataSize-pos : FTI_Conf->dcpInfoPosix.BlockSize;
           
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
                layerSize += success*fileUpdate;
                if(success) {
                    MD5_Update( &ctx, &FTI_Data[i].dcpInfoPosix.hashArrayTmp[hashIdx], MD5_DIGEST_LENGTH ); 
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

    free(block);

    fsync(fileno(fd));
    fclose( fd );
   
    // copy dcpFileSize for metadatacreation
    FTI_Exec->ckptSize = FTI_Exec->dcpInfoPosix.FileSize;

    // create final dcp layer hash
    unsigned char LayerHash[MD5_DIGEST_LENGTH];
    MD5_Final( LayerHash, &ctx );
    hashHex( LayerHash, MD5_DIGEST_LENGTH, &FTI_Exec->dcpInfoPosix.LayerHash[dcpLayer*MD5_DIGEST_STRING_LENGTH] );

    // layer size is needed in order to create layer hash during recovery
    FTI_Exec->dcpInfoPosix.LayerSize[dcpLayer] = layerSize;

    FTI_Exec->dcpInfoPosix.Counter++;
    if( (dcpLayer == (FTI_Conf->dcpInfoPosix.StackSize-1)) ) {
        int i = 0;
        for(; i<FTI_Exec->nbVar; i++) {
            //free(FTI_Data[i].dcpInfoPosix.hashArray);
            FTI_Data[i].dcpInfoPosix.hashDataSize = 0;
        }
    }
    if( (dcpLayer == 0) ) {
        char ofn[512];
        snprintf( ofn, FTI_BUFS, "%s/dcp-id%d-rank%d.fti", FTI_Ckpt[4].dcpDir, dcpFileId-1, FTI_Topo->splitRank );
        if( (remove(ofn) < 0) && (errno != ENOENT) ) {
            snprintf(errstr, FTI_BUFS, "cannot delete file '%s'", ofn );
            FTI_Print( errstr, FTI_WARN ); 
        }
    }

    // TEMPORARY TO CHECK FUNCTIONALITY
    char mfn[FTI_BUFS];
    snprintf( mfn, FTI_BUFS, "%s/dcp-rank%d.meta", FTI_Ckpt[4].dcpDir, FTI_Topo->splitRank );
    FILE* mfd = fopen( mfn, "wb" );
    fwrite( &FTI_Exec->dcpInfoPosix.FileSize, sizeof(unsigned long), 1, mfd );
    fwrite( &dcpFileId, sizeof(int), 1, mfd );
    fwrite( &FTI_Exec->nbVar, sizeof(int), 1, mfd);
    for(i=0; i<FTI_Exec->nbVar; i++) {
        unsigned long dataSize = FTI_Data[i].size;
        fwrite( &FTI_Data[i].id, sizeof(int), 1, mfd );
        fwrite( &dataSize, sizeof(unsigned long), 1, mfd );
    }
    fclose(mfd);

    return FTI_SCES;
   
}

/*-------------------------------------------------------------------------*/
/**
  @brief      It checks if a file exist and that its size is 'correct'.
  @param      fn              The ckpt. file name to check.
  @param      fs              The ckpt. file size to check.
  @param      checksum        The file checksum to check.
  @return     integer         0 if file exists, 1 if not or wrong size.

  This function checks whether a file exist or not and if its size is
  the expected one.

 **/
/*-------------------------------------------------------------------------*/
int FTI_CheckFileDcpPosix(char* fn, long fs, char* checksum)
{
    if (access(fn, F_OK) == 0) {
        struct stat fileStatus;
        if (stat(fn, &fileStatus) == 0) {
            if (fileStatus.st_size == fs) {
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
                return 1;
            }
        }
        else {
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

  This function calculates checksum of the checkpoint file based on
  MD5 algorithm. It compares calculated hash value with the one saved
  in the file.

 **/
/*-------------------------------------------------------------------------*/
int FTI_VerifyChecksumDcpPosix(char* fileName)
{

    FTIT_execution* exec = FTI_DcpPosixRecoverRuntimeInfo( DCP_POSIX_EXEC_TAG, NULL, NULL );
    FTIT_configuration* conf = FTI_DcpPosixRecoverRuntimeInfo( DCP_POSIX_CONF_TAG, NULL, NULL ); 

    char dummyBuffer[FTI_BUFS];
    char checksumToCmp[MD5_DIGEST_LENGTH];
    unsigned long blockSize;
    int nbVarBasis;

    FILE *fd = fopen(fileName, "rb");
    if (fd == NULL) {
        char str[FTI_BUFS];
        sprintf(str, "FTI failed to open file %s to calculate checksum.", fileName);
        FTI_Print(str, FTI_WARN);
        return FTI_NSCS;
    }

    unsigned char md5_tmp[MD5_DIGEST_LENGTH];
    unsigned char md5_final[MD5_DIGEST_LENGTH];

    MD5_CTX mdContext;

    size_t pos = 0;
    pos += fread( &blockSize, sizeof(unsigned long), 1, fd );
    pos += fread( &dummyBuffer, sizeof(unsigned int), 1, fd );

    int i;
    int layer = 0;
    int nbVarLayer;

    void* buffer = malloc( blockSize );
    while( pos < exec->meta[i].fs[0] ) {
        size_t pos_layer = 0;
        pos_layer += fread( &nbVarLayer, sizeof(int), 1, fd ); 
        while( pos_layer < exec->dcpInfoPosix.LayerSize[layer] ) { 
            if( layer == 0 ) {
                for( i=0; i<nbVarLayer; i++) {
                    // advance after id
                    pos_layer += fread( dummyBuffer, sizeof(int), 1, fd );
                    size_t dataSize;
                    pos_layer += fread( &dataSize, sizeof(unsigned long), 1, fd );
                    size_t pos_data = 0;
                    while( pos_data < dataSize ) {
                        pos_data += fread( buffer, blockSize, 1, fd );
                        MD5( buffer, blockSize, md5_tmp );
                        MD5_Update( &mdContext, md5_tmp, MD5_DIGEST_LENGTH );
                    }
                    pos_layer += pos_data;
                }
            }
            if( layer > 0 ) {
                
            }
        }
        layer++;
        pos += pos_layer;
    }

    fclose (fd);

    return FTI_SCES;
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
    
    uint32_t digest = crc32( 0L, Z_NULL, 0 );
    digest = crc32( digest, d, nBytes );

    memcpy( hash, &digest, CRC32_DIGEST_LENGTH );

    return hash;
}

