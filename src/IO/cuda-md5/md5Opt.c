#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "md5Opt.h"
#include <pthread.h>
#include <fti.h>
#include <interface.h>
#define CPU 1
#define GPU 2
#define CFILE 3

int MD5GPU(FTIT_dataset *);
int MD5CPU(FTIT_dataset *);
int usesAsync = 0;

pthread_t thread;
pthread_mutex_t worker;
pthread_mutex_t application;
long totalWork= 0;
long worker_exit = 0;
int deviceId;
unsigned char* (*cpuHash)( const unsigned char *data, unsigned long nBytes, unsigned char *hash );





long tempBufferSize;
long md5ChunkSize;

int FTI_initMD5(long cSize, long tempSize, FTIT_configuration *FTI_Conf){
    if ( FTI_Conf->dcpInfoPosix.cachedCkpt)
        usesAsync = 1;
    else
        usesAsync = 0;

    cpuHash = FTI_Conf->dcpInfoPosix.hashFunc;
    tempBufferSize = tempSize;
    md5ChunkSize = cSize;
    return FTI_SCES;
}

int MD5CPU(FTIT_dataset *FTI_DataVar){
    unsigned long dataSize = FTI_DataVar->size;
    unsigned char block[md5ChunkSize];
    size_t i;
    unsigned char *ptr = (unsigned char *) FTI_DataVar->ptr;
    for ( i = 0 ; i < FTI_DataVar->size; i+=md5ChunkSize){
        unsigned int blockId = i/md5ChunkSize;
        unsigned int hashIdx = blockId*16;
        unsigned int chunkSize = ( (dataSize-i) < md5ChunkSize ) ? dataSize-i: md5ChunkSize;
        if( chunkSize < md5ChunkSize ) {
            memset( block, 0x0, md5ChunkSize );
            memcpy( block, &ptr[i], chunkSize );
            cpuHash( block, md5ChunkSize , &FTI_DataVar->dcpInfoPosix.currentHashArray[hashIdx] );
        } else {
            cpuHash( &ptr[i], md5ChunkSize , &FTI_DataVar->dcpInfoPosix.currentHashArray[hashIdx] );
        }
    }
    return FTI_SCES;
}


int FTI_MD5CPU(FTIT_dataset *FTI_DataVar){
    MD5CPU(FTI_DataVar);
    return 1;
}

int FTI_MD5GPU(FTIT_dataset *FTI_DataVar){
    return 1;
}

int FTI_CLOSE_ASYNC(FILE *f){
    return 1;
}

int FTI_SyncMD5(){
    return FTI_SCES;
}

int FTI_startMD5(){
    return FTI_SCES;
}

int FTI_destroyMD5(){
    return FTI_SCES;
}
