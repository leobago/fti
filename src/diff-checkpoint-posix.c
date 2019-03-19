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

    unsigned char * block = (unsigned char*) malloc( FTI_Conf->dcpInfoPosix.BlockSize );
    int i = 0;
    
    size_t dcpSize = 0;
    unsigned long glbDataSize = 0;
    if( dcpLayer == 0 ) FTI_Exec->dcpInfoPosix.FileSize = 0;
    
    for(; i<FTI_Exec->nbVar; i++) {
         
        unsigned int varId = FTI_Data[i].id;
        unsigned long dataSize = FTI_Data[i].size;
        unsigned long nbHashes = dataSize/FTI_Conf->dcpInfoPosix.BlockSize + (bool)(dataSize%FTI_Conf->dcpInfoPosix.BlockSize);
        
        glbDataSize += dataSize;

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
            FTI_Exec->dcpInfoPosix.FileSize += (sizeof(int) + sizeof(long));
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
                dcpSize += success*chunkSize;
                FTI_Exec->dcpInfoPosix.FileSize += success*fileUpdate;
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
    fwrite( &glbDataSize, sizeof(unsigned long), 1, mfd );
    fwrite( &dcpFileId, sizeof(int), 1, mfd );
    fwrite( &FTI_Conf->dcpInfoPosix.BlockSize, sizeof(unsigned long), 1, mfd );
    fwrite( &FTI_Exec->nbVar, sizeof(int), 1, mfd);
    for(i=0; i<FTI_Exec->nbVar; i++) {
        unsigned long dataSize = FTI_Data[i].size;
        fwrite( &FTI_Data[i].id, sizeof(int), 1, mfd );
        fwrite( &dataSize, sizeof(unsigned long), 1, mfd );
    }
    fclose(mfd);

    return FTI_SCES;
   
}

// HELPER FUNCTIONS

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

