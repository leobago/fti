#ifndef _FAILURE_INJECTION_H
#define _FAILURE_INJECTION_H

#include <fti.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdint.h>
#include <limits.h>

static inline uint64_t get_ruint() {
    uint64_t buffer;
    int fd = open("/dev/urandom", O_RDWR);
    read(fd, &buffer, 8);
    close(fd);
    return buffer%INT_MAX;
}

void FTI_InitFIIO();
float PROBABILITY();
unsigned int FUNCTION( const char *testFunction );

#ifdef ENABLE_FTI_FI_IO
#define FTI_FI_WRITE( ERR, FD, BUF, COUNT, FN ) \
    do { \
        if( FUNCTION(__FUNCTION__) ) { \
            if( get_ruint() < ((uint64_t)((double)PROBABILITY()*INT_MAX)) ) { \
                close(FD); \
                FD = open(FN, O_RDONLY); \
            }  \
        } \
        ERR = write( FD, BUF, COUNT ); \
        (void)(ERR); \
    } while(0)
#define FTI_FI_FWRITE( ERR, BUF, SIZE, COUNT, FSTREAM, FN ) \
    do { \
        if( FUNCTION(__FUNCTION__) ) { \
            if( get_ruint() < ((uint64_t)((double)PROBABILITY()*INT_MAX)) ) { \
                fclose(FSTREAM); \
                FSTREAM = fopen(FN, "rb"); \
            } \
        } \
        ERR = fwrite( BUF, SIZE, COUNT, FSTREAM ); \
        (void)(ERR); \
    } while(0)
#else
#define FTI_FI_WRITE( ERR, FD, BUF, COUNT, FN ) ( ERR = write( FD, BUF, COUNT ) )
#define FTI_FI_FWRITE( ERR, BUF, SIZE, COUNT, FSTREAM, FN ) ( ERR = fwrite( BUF, SIZE, COUNT, FSTREAM ) )
#endif

#endif //_FAILURE_INJECTION_H
