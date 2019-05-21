#ifndef DEFS_H
#define DEFS_H

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

#ifndef _FTI_PUBLIC
#   define FTI_BUFS 256
#   define FTI_WORD 16
#   define FTI_DONE 1
#   define FTI_SCES 0
#   define FTI_NSCS -1
#   define FTI_NREC -2
#   define FTI_HEAD 2
#endif
  
#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

#define DBG_MSG(MSG,RANK,...) do { \
    int rank; \
    MPI_Comm_rank(MPI_COMM_WORLD,&rank); \
    if ( rank == RANK ) \
        printf( "%s:%d[DEBUG-%d] " MSG "\n", __FILENAME__,__LINE__,rank, ##__VA_ARGS__); \
    if ( RANK == -1 ) \
        printf( "%s:%d[DEBUG-%d] " MSG "\n", __FILENAME__,__LINE__,rank, ##__VA_ARGS__); \
} while (0)

/** Define RED color for FTI output.                                       */
#define FTI_COLOR_RED   "\x1B[31m"
/** Define ORANGE color for FTI output.                                    */
#define FTI_COLOR_ORG   "\x1B[38;5;202m"
/** Define GREEN color for FTI output.                                     */
#define FTI_COLOR_GRN   "\x1B[32m"
/** Define BLUE color for FTI output.                                       */
#define FTI_COLOR_BLU   "\x1B[34m"
/** Define color RESET for FTI output.                                     */
#define FTI_COLOR_RESET "\x1B[0m"

/** Verbosity level to print only errors.                                  */
#define FTI_EROR 4
/** Verbosity level to print only warning and errors.                      */
#define FTI_WARN 3
/** Verbosity level to print main information.                             */
#define FTI_IDCP 5
/** Verbosity level to print debug messages.                               */
#define FTI_INFO 2
/** Verbosity level to print debug messages.                               */
#define FTI_DBUG 1

/** Token for checkpoint Baseline.                                         */
#define FTI_BASE 990
/** Token for checkpoint Level 1.                                          */
#define FTI_CKTW 991
/** Token for checkpoint Level 2.                                          */
#define FTI_XORW 992
/** Token for checkpoint Level 3.                                          */
#define FTI_RSEW 993
/** Token for checkpoint Level 4.                                          */
#define FTI_PFSW 994
/** Token for end of the execution.                                        */
#define FTI_ENDW 995
/** Token to reject checkpoint.                                            */
#define FTI_REJW 996
/** Token for IO mode Posix.                                               */
#define FTI_IO_POSIX 1001
/** Token for IO mode MPI.                                                 */
#define FTI_IO_MPI 1002
/** Token for IO mode FTI-FF.                                              */
#define FTI_IO_FTIFF 1003
#ifdef ENABLE_SIONLIB // --> If SIONlib is installed
/** Token for IO mode SIONlib.                                             */
#define FTI_IO_SIONLIB 1004
#endif
/** Token for IO mode HDF5.                                         */
#define FTI_IO_HDF5 1005

/** MD5-hash: unsigned char digest length.                                 */
#ifndef MD5_DIGEST_LENGTH
#   define MD5_DIGEST_LENGTH 16
#endif
/** MD5-hash: hex converted char digest length.                            */
#define MD5_DIGEST_STRING_LENGTH 33

// need this parameter in one fti api function
#ifndef ENABLE_HDF5
typedef size_t 	hsize_t;
#endif

typedef uintptr_t           FTI_ADDRVAL;        /**< for ptr manipulation       */
typedef void*               FTI_ADDRPTR;        /**< void ptr type              */ 

#include <fti-int/stage-defs.h>
#include <fti-int/icp-defs.h>
#include <fti-int/ftiff-defs.h>

#endif // DEFS_H
