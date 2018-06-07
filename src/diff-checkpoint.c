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
 *  @file   diff-checkpoint.c
 *  @date   February, 2018
 *  @brief  Differential checkpointing routines.
 */

#define _DEFAULT_SOURCE
#define _BSD_SOURCE

#define FTI_ASSERT_ALIGNED( VAL, ALIGNMENT ) assert( ( VAL % ALIGNMENT ) == 0 )

#include "diff-checkpoint.h"

/** TEST VARIABLES FOR PAPER                                                            */


static const long counterStride = 4096L;
static const long counterElem = (4L*1024L*1024L*1024L)/counterStride;
static const long counterHBStride = 8L;
static const long counterHBElem = (32768L)/counterHBStride;
static long dataWritten;
static long dataStored;
static double timerReceiveDiffChunk_t;
static double timerHashCmp_t;
static double timerHashUpdate_t;
static double timerWriteData_t;     // records the time that needs a rank to write the datasets
static double timerWriteFtiff_t;    // records time to execute FTIFF_Write + FTI_UpdateHashChanges 
static double timerCreateMetaData_t;
static double timerFileChecksum_t;
static double timerBlockMeta_t;
static double timerPadding_t;
static double timerUpdateData_t;
static double timerRenameFile_t;
static long *counter_t;
static long *counterAll_t;
static long *counterHB_t;
static long *counterHBAll_t;

void initDiffStats() {
    dataWritten = 0;
    dataStored = 0;
    timerReceiveDiffChunk_t = 0;
    timerHashCmp_t = 0;
    timerHashUpdate_t = 0;
    timerWriteData_t = 0;
    timerWriteFtiff_t = 0;
    timerCreateMetaData_t = 0;
    timerFileChecksum_t = 0;
    timerBlockMeta_t = 0;
    timerPadding_t = 0;
    timerUpdateData_t = 0;
    timerRenameFile_t = 0;
    counterHB_t = (long*) calloc( counterHBElem, sizeof(long) );
    counterHBAll_t = (long*) calloc( counterHBElem, sizeof(long) );
    counter_t = (long*) calloc( counterElem, sizeof(long) );
    counterAll_t = (long*) calloc( counterElem, sizeof(long) );
}

void accumulateFileChecksumTime( struct timespec t1, struct timespec t2 ) {
    double dsec2 = t2.tv_sec;
    double dsec1 = t1.tv_sec;
    double dt = (dsec2-dsec1)*BILLION + (t2.tv_nsec-t1.tv_nsec);
    dt /= BILLION;
    timerFileChecksum_t += dt;
}

void accumulateUpdateDataTime( struct timespec t1, struct timespec t2 ) {
    double dsec2 = t2.tv_sec;
    double dsec1 = t1.tv_sec;
    double dt = (dsec2-dsec1)*BILLION + (t2.tv_nsec-t1.tv_nsec);
    dt /= BILLION;
    timerUpdateData_t += dt;
}

void accumulateRenameFileTime( struct timespec t1, struct timespec t2 ) {
    double dsec2 = t2.tv_sec;
    double dsec1 = t1.tv_sec;
    double dt = (dsec2-dsec1)*BILLION + (t2.tv_nsec-t1.tv_nsec);
    dt /= BILLION;
    timerRenameFile_t += dt;
}

void accumulateBlockMetaTime( struct timespec t1, struct timespec t2 ) {
    double dsec2 = t2.tv_sec;
    double dsec1 = t1.tv_sec;
    double dt = (dsec2-dsec1)*BILLION + (t2.tv_nsec-t1.tv_nsec);
    dt /= BILLION;
    timerBlockMeta_t += dt;
}

void accumulatePaddingTime( struct timespec t1, struct timespec t2 ) {
    double dsec2 = t2.tv_sec;
    double dsec1 = t1.tv_sec;
    double dt = (dsec2-dsec1)*BILLION + (t2.tv_nsec-t1.tv_nsec);
    dt /= BILLION;
    timerPadding_t += dt;
}

void accumulateHashCmpTime( struct timespec t1, struct timespec t2 ) {
    double dsec2 = t2.tv_sec;
    double dsec1 = t1.tv_sec;
    double dt = (dsec2-dsec1)*BILLION + (t2.tv_nsec-t1.tv_nsec);
    dt /= BILLION;
    timerHashCmp_t += dt;
}

void accumulateReceiveDiffChunkTime( struct timespec t1, struct timespec t2 ) {
    double dsec2 = t2.tv_sec;
    double dsec1 = t1.tv_sec;
    double dt = (dsec2-dsec1)*BILLION + (t2.tv_nsec-t1.tv_nsec);
    dt /= BILLION;
    timerReceiveDiffChunk_t += dt;
}

void accumulateHashUpdateTime( struct timespec t1, struct timespec t2 ) {
    double dsec2 = t2.tv_sec;
    double dsec1 = t1.tv_sec;
    double dt = (dsec2-dsec1)*BILLION + (t2.tv_nsec-t1.tv_nsec);
    dt /= BILLION;
    timerHashUpdate_t = dt;
}

void accumulateCreateMetaDataTime( struct timespec t1, struct timespec t2 ) {
    double dsec2 = t2.tv_sec;
    double dsec1 = t1.tv_sec;
    double dt = (dsec2-dsec1)*BILLION + (t2.tv_nsec-t1.tv_nsec);
    dt /= BILLION;
    timerCreateMetaData_t += dt;
}

void accumulateWriteFtiff( struct timespec t1, struct timespec t2 ) {
    double dsec2 = t2.tv_sec;
    double dsec1 = t1.tv_sec;
    double dt = (dsec2-dsec1)*BILLION + (t2.tv_nsec-t1.tv_nsec);
    dt /= BILLION;
    timerWriteFtiff_t += dt;
}

void accumulateWriteDataTime( struct timespec t1, struct timespec t2 ) {
    double dsec2 = t2.tv_sec;
    double dsec1 = t1.tv_sec;
    double dt = (dsec2-dsec1)*BILLION + (t2.tv_nsec-t1.tv_nsec);
    dt /= BILLION;
    timerWriteData_t += dt;
}

void accumulateDiffStats( long written, long stored ) {
    dataWritten += written;
    dataStored += stored;
}

void printDiffStats( FTIT_topology* FTI_Topo, double totalTimeCkpt ) {
    
    long totalDataWritten;
    long totalDataStored;
    double totalInvTimerHashCmp;
    double totalInvTimerReceiveDiffChunk;
    double totalInvTimerHashUpdate;
    double totalInvTimerWriteData;
    double totalTimeCkptFtiff;
    double totalTimeReceiveDiffChunk;
    double totalTimeHashUpdate;
    double totalTimeWriteData;
    double totalTimerCreateMetaData;
    double totalTimerFileChecksum;
    double totalTimerBlockMeta;
    double totalTimerPadding;
    double totalTimerUpdateData;
    double totalTimerRenameFile;
    int wsize;
    MPI_Comm_size(FTI_COMM_WORLD, &wsize);
    
    MPI_Allreduce( &timerWriteFtiff_t, &totalTimeCkptFtiff, 1, MPI_DOUBLE, MPI_SUM, FTI_COMM_WORLD );  
    
    double invTimerHashCmp = timerHashCmp_t/totalTimeCkptFtiff;
    double invTimerWriteData = timerWriteData_t/totalTimeCkptFtiff;
    double invTimerHashUpdate = timerHashUpdate_t/totalTimeCkptFtiff;
    double invTimerReceiveDiffChunk = timerReceiveDiffChunk_t/totalTimeCkptFtiff;

    MPI_Allreduce( &dataWritten, &totalDataWritten, 1, MPI_LONG, MPI_SUM, FTI_COMM_WORLD );  
    MPI_Allreduce( &dataStored, &totalDataStored, 1, MPI_LONG, MPI_SUM, FTI_COMM_WORLD );  
    MPI_Allreduce( &invTimerHashCmp, &totalInvTimerHashCmp, 1, MPI_DOUBLE, MPI_SUM, FTI_COMM_WORLD );  
    MPI_Allreduce( &invTimerReceiveDiffChunk, &totalInvTimerReceiveDiffChunk, 1, MPI_DOUBLE, MPI_SUM, FTI_COMM_WORLD );  
    MPI_Allreduce( &invTimerHashUpdate, &totalInvTimerHashUpdate, 1, MPI_DOUBLE, MPI_SUM, FTI_COMM_WORLD );  
    MPI_Allreduce( &invTimerWriteData, &totalInvTimerWriteData, 1, MPI_DOUBLE, MPI_SUM, FTI_COMM_WORLD );  
    MPI_Allreduce( &timerReceiveDiffChunk_t, &totalTimeReceiveDiffChunk, 1, MPI_DOUBLE, MPI_SUM, FTI_COMM_WORLD );  
    MPI_Allreduce( &timerHashUpdate_t, &totalTimeHashUpdate, 1, MPI_DOUBLE, MPI_SUM, FTI_COMM_WORLD );  
    MPI_Allreduce( &timerWriteData_t, &totalTimeWriteData, 1, MPI_DOUBLE, MPI_SUM, FTI_COMM_WORLD );  
    MPI_Allreduce( &timerCreateMetaData_t, &totalTimerCreateMetaData, 1, MPI_DOUBLE, MPI_SUM, FTI_COMM_WORLD );  
    MPI_Allreduce( &timerFileChecksum_t, &totalTimerFileChecksum, 1, MPI_DOUBLE, MPI_SUM, FTI_COMM_WORLD );  
    MPI_Allreduce( &timerBlockMeta_t, &totalTimerBlockMeta, 1, MPI_DOUBLE, MPI_SUM, FTI_COMM_WORLD );  
    MPI_Allreduce( &timerPadding_t, &totalTimerPadding, 1, MPI_DOUBLE, MPI_SUM, FTI_COMM_WORLD );  
    MPI_Allreduce( &timerUpdateData_t, &totalTimerUpdateData, 1, MPI_DOUBLE, MPI_SUM, FTI_COMM_WORLD );  
    MPI_Allreduce( &timerRenameFile_t, &totalTimerRenameFile, 1, MPI_DOUBLE, MPI_SUM, FTI_COMM_WORLD );  
    
    double ratio = 100*((double) totalDataWritten)/totalDataStored; // ratio dirty to all
    double ratioReceiveDiffChunk = 100*(totalTimeReceiveDiffChunk/totalTimeCkptFtiff); 
    double ratioWrite = 100*(totalTimeWriteData/totalTimeCkptFtiff);
    double ratioHashUpdate = 100*(totalTimeHashUpdate/totalTimeCkptFtiff);
    double ratioCreateMetaData = 100*(totalTimerCreateMetaData/totalTimeCkptFtiff);
    double ratioFileChecksum = 100*(totalTimerFileChecksum/totalTimeCkptFtiff);
    double ratioBlockMeta = 100*(totalTimerBlockMeta/totalTimeCkptFtiff);
    double ratioPadding = 100*(totalTimerPadding/totalTimeCkptFtiff);
    double ratioUpdateData = 100*(totalTimerUpdateData/totalTimeCkptFtiff);
    double ratioRenameFile = 100*(totalTimerRenameFile/totalTimeCkptFtiff);
    //double ratioReceiveDiffChunk = 100*(totalInvTimerReceiveDiffChunk/wsize);
    //double ratioWrite = 100*(totalInvTimerWriteData/wsize);
    //double ratioHashUpdate = 100*(totalInvTimerHashUpdate/wsize);
    double ratioTotal = 100*(totalTimeCkptFtiff/totalTimeCkpt);
    
    totalTimeCkptFtiff /= wsize;

    /* == KEY ==
     * G: Global Checkpoint time
     * L: Local average per Rank of FTIFF_Write call
     * Th: Time for hash part of diff mechanism (hash compare, c, and Hash update, u)
     * Tw: time to write datasets
     * Tmf: Time for meta data creation
     * Tmb: Time to write Block meta data
     * Tcs: Time for file checksum creation
     * Tud: Time to update FTI-FF datastructure
     * Trf: Time to rename CP files (will vanish in stable version)
     */

    if (FTI_Topo->splitRank == 0) {
        printf("[ " BLU "FTI  DIFFCKPT" RESET " ] : (DATA STATS) %ld of %ld Bytes written (%.2lf%%).\n", 
                totalDataWritten, totalDataStored,
                ratio);
        printf("[ " BLU "FTI  DIFFCKPT" RESET " ] : (TIME STATS) G: %.2lf s. L: %.2lf s. (%.2lf %% 'Th' [%.2lf %% c. | %.2lf %% u.], %.2lf %% 'Tw', %.2lf %% 'Tmf', %.2lf %% 'Tmb', %.2lf %% 'Tcs', %.2lf %% 'Tp', %.2lf %% 'Tud', %.2lf %% 'Trf').\n", 
                totalTimeCkpt, totalTimeCkptFtiff, ratioReceiveDiffChunk+ratioHashUpdate, ratioReceiveDiffChunk, ratioHashUpdate, ratioWrite, ratioCreateMetaData, ratioBlockMeta, ratioFileChecksum, ratioPadding, ratioUpdateData, ratioRenameFile);
    }

    long s = 0;
    long *counterOutX = NULL;   // bin
    long *counterOutY = NULL;   // count
    long counterOutElem = 0;
     
    if (FTI_Topo->splitRank == 0) {
        printf("[ " BLU "FTI  DIFFCKPT" RESET " ] : (COUNTER STAT BEGIN)\n"); 
    }

    MPI_Allreduce( counter_t, counterAll_t, counterElem, MPI_LONG, MPI_SUM, FTI_COMM_WORLD );

    for(; s<counterElem; ++s) {
        if( counterAll_t[s] != 0 ) {
            counterOutElem++;
            counterOutX = (long*) realloc( counterOutX, counterOutElem*sizeof(long) );
            counterOutY = (long*) realloc( counterOutY, counterOutElem*sizeof(long) );
            counterOutX[counterOutElem-1] = s*counterStride;
            counterOutY[counterOutElem-1] = counterAll_t[s];
            if (FTI_Topo->splitRank == 0) {
                printf("BIN : %ld - %ld -> %ld\n", counterOutX[counterOutElem-1], counterOutX[counterOutElem-1]+counterStride-1, counterOutY[counterOutElem-1]); 
            }
        }
    }

    if (FTI_Topo->splitRank == 0) {
        printf("[ " BLU "FTI  DIFFCKPT" RESET " ] : (COUNTER STAT END)\n"); 
    }

    s=0;
    long *counterHBOutX = NULL;   // bin
    long *counterHBOutY = NULL;   // count
    long counterHBOutElem = 0;
     
    if (FTI_Topo->splitRank == 0) {
        printf("[ " BLU "FTI  DIFFCKPT" RESET " ] : (HASH BLOCK COUNTER STAT BEGIN)\n"); 
    }

    MPI_Allreduce( counterHB_t, counterHBAll_t, counterHBElem, MPI_LONG, MPI_SUM, FTI_COMM_WORLD );

    for(; s<counterHBElem; ++s) {
        if( counterHBAll_t[s] != 0 ) {
            counterHBOutElem++;
            counterHBOutX = (long*) realloc( counterHBOutX, counterHBOutElem*sizeof(long) );
            counterHBOutY = (long*) realloc( counterHBOutY, counterHBOutElem*sizeof(long) );
            counterHBOutX[counterHBOutElem-1] = s*counterHBStride;
            counterHBOutY[counterHBOutElem-1] = counterHBAll_t[s];
            if (FTI_Topo->splitRank == 0) {
                printf("BIN : %ld - %ld -> %ld\n", counterHBOutX[counterHBOutElem-1], counterHBOutX[counterHBOutElem-1]+counterHBStride-1, counterHBOutY[counterHBOutElem-1]); 
            }
        }
    }

    if (FTI_Topo->splitRank == 0) {
        printf("[ " BLU "FTI  DIFFCKPT" RESET " ] : (HASH BLOCK COUNTER STAT END)\n"); 
    }

    dataWritten = 0;
    dataStored = 0;
    timerReceiveDiffChunk_t = 0;
    timerHashCmp_t = 0;
    timerHashUpdate_t = 0;
    timerWriteData_t = 0;
    timerWriteFtiff_t = 0;
    timerCreateMetaData_t = 0;
    timerFileChecksum_t = 0;
    timerBlockMeta_t = 0;
    timerPadding_t = 0;
    timerUpdateData_t = 0;
    timerRenameFile_t = 0;
    memset( counterHB_t, 0x0, counterHBElem*sizeof(long) );
    memset( counterHBAll_t, 0x0, counterHBElem*sizeof(long) );
    memset( counter_t, 0x0, counterElem*sizeof(long) );
    memset( counterAll_t, 0x0, counterElem*sizeof(long) );
}




/**                                                                                     */
/** Static Global Variables                                                             */

static int                  HASH_MODE;
static int                  DIFF_BLOCK_SIZE;

static FTI_ADDRVAL          FTI_PageSize;       /**< memory page size                   */
static FTI_ADDRVAL          FTI_PageMask;       /**< memory page mask                   */

static FTIT_DataDiffInfoSignal    FTI_SignalDiffInfo;   /**< container for diff of datasets     */
static FTIT_DataDiffInfoHash      FTI_HashDiffInfo;   /**< container for diff of datasets     */

/** File Local Variables                                                                */

static bool enableDiffCkpt;
static int diffMode;

static struct sigaction     FTI_SigAction;       /**< sigaction meta data               */
static struct sigaction     OLD_SigAction;       /**< previous sigaction meta data      */

static long countPages;
static FTI_ADDRVAL* pagesGlobal;

/** Function Definitions                                                                */
int compare( const void* a, const void* b)
{
     FTI_ADDRVAL A = * ( (FTI_ADDRVAL*) a );
     FTI_ADDRVAL B = * ( (FTI_ADDRVAL*) b );

     if ( A == B ) return 0;
     else if ( A < B ) return -1;
     else return 1;
}

long getCountPages() {
    return countPages;
}

void resetPageCounter() {
    countPages = 0;
}

int FTI_InitDiffCkpt( FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec, FTIT_dataset* FTI_Data )
{
    
    // for paper
    initDiffStats();
    //
    int rank;
    MPI_Comm_rank(FTI_COMM_WORLD, &rank);
    if( getenv("FTI_HASH_MODE") != 0 ) {
        HASH_MODE = atoi(getenv("FTI_HASH_MODE"));
    } else {
        HASH_MODE = 0;
    }
    if( getenv("FTI_DIFF_BLOCK_SIZE") != 0 ) {
        DIFF_BLOCK_SIZE = atoi(getenv("FTI_DIFF_BLOCK_SIZE"));
    } else {
        DIFF_BLOCK_SIZE = 2048;
    }

    if(rank == 0) {
        switch (HASH_MODE) {
            case -1:
                printf("[ " BLU "FTI  DIFFCKPT" RESET " ] : HASH MODE IS -> OFF\n");
                break;
            case 0:
                printf("[ " BLU "FTI  DIFFCKPT" RESET " ] : HASH MODE IS -> MD5\n");
                break;
            case 1:
                printf("[ " BLU "FTI  DIFFCKPT" RESET " ] : HASH MODE IS -> CRC32\n");
                break;
            case 2:
                printf("[ " BLU "FTI  DIFFCKPT" RESET " ] : HASH MODE IS -> ADLER32\n");
                break;
            case 3:
                printf("[ " BLU "FTI  DIFFCKPT" RESET " ] : HASH MODE IS -> FLETCHER32\n");
                break;
        }
        printf("[ " BLU "FTI  DIFFCKPT" RESET " ] : DIFF_BLOCK_SIZE IS -> %d\n", DIFF_BLOCK_SIZE);
    }

    enableDiffCkpt = FTI_Conf->enableDiffCkpt;
    
    diffMode = FTI_Conf->diffMode;
    if( enableDiffCkpt && FTI_Conf->diffMode == 0 ) {
        FTI_HashDiffInfo.dataDiff = NULL;
        FTI_HashDiffInfo.nbProtVar = 0;
        return FTI_SCES;
    }
    if( enableDiffCkpt && FTI_Conf->diffMode == 1 ) {
        // get page mask
        FTI_PageSize = (FTI_ADDRVAL) sysconf(_SC_PAGESIZE);
        FTI_PageMask = ~((FTI_ADDRVAL)0x0);
        FTI_ADDRVAL tail = (FTI_ADDRVAL)0x1;
        for(; tail!=FTI_PageSize; FTI_PageMask<<=1, tail<<=1); 
        // init data diff structure
        FTI_SignalDiffInfo.dataDiff = NULL;
        FTI_SignalDiffInfo.nbProtVar = 0;
        
        pagesGlobal = NULL;
        countPages = 0;
        
        
        // register signal handler
        return FTI_RegisterSigHandler();
    } else {
        return FTI_SCES;
    }
}

void printReport() {
    long num_unchanged = 0;
    long num_prot = 0; 
    int i,j;
    for(i=0; i<FTI_SignalDiffInfo.nbProtVar; ++i) {
        num_prot += FTI_SignalDiffInfo.dataDiff[i].totalSize;
        for(j=0; j<FTI_SignalDiffInfo.dataDiff[i].rangeCnt; ++j) {
            num_unchanged += FTI_SignalDiffInfo.dataDiff[i].ranges[j].size;
        }
    }
    num_prot /= FTI_PageSize;
    num_unchanged /= FTI_PageSize;

    printf(
            "Diff Ckpt Summary\n"
            "-------------------------------------\n"
            "number of pages protected:       %lu\n"
            "number of pages changed:         %lu\n",
            num_prot, num_unchanged);
    fflush(stdout);
    
}

int FTI_FinalizeDiffCkpt(){
    int res = 0;
    if( enableDiffCkpt ) {
        if( diffMode == 1 ) {
            //printReport();
            res += FTI_RemoveSigHandler();
            res += FTI_RemoveProtections();
            res += FTI_FreeDiffCkptStructs();
        }
    }
    return ( res == 0 ) ? FTI_SCES : FTI_NSCS;
}

int FTI_FreeDiffCkptStructs() {
    int i;
    
    for(i=0; i<FTI_SignalDiffInfo.nbProtVar; ++i) {
        if(FTI_SignalDiffInfo.dataDiff[i].ranges != NULL) {
            free(FTI_SignalDiffInfo.dataDiff[i].ranges);
        }
    }
    free(FTI_SignalDiffInfo.dataDiff);

    return FTI_SCES;
}

int FTI_RemoveProtections() {
    int i,j;
    for(i=0; i<FTI_SignalDiffInfo.nbProtVar; ++i){
        if ( FTI_SignalDiffInfo.dataDiff[i].rangeCnt == 0 ) { continue; }
        for(j=0; j<FTI_SignalDiffInfo.dataDiff[i].rangeCnt; ++j) {
            FTI_ADDRPTR ptr = (FTI_ADDRPTR) (FTI_SignalDiffInfo.dataDiff[i].basePtr + FTI_SignalDiffInfo.dataDiff[i].ranges[j].offset);
            long size = (long) FTI_SignalDiffInfo.dataDiff[i].ranges[j].size;
            if ( size == 0 ) { continue; }
            if ( mprotect( ptr, size, PROT_READ|PROT_WRITE ) == -1 ) {
                // ENOMEM is return e.g. if allocation was already freed, which will (hopefully) 
                // always be the case if FTI_Finalize() is called at the end and the buffer was allocated dynamically
                if ( errno != ENOMEM ) {
                    FTI_Print( "FTI was unable to restore the data access", FTI_EROR );
                    return FTI_NSCS;
                }
                errno = 0;
            }
        }
    }
    return FTI_SCES;
}

int FTI_RemoveSigHandler() 
{ 
    if( sigaction(SIGSEGV, &OLD_SigAction, NULL) == -1 ){
        FTI_Print( "FTI was unable to restore the default signal handler", FTI_EROR );
        errno = 0;
        return FTI_NSCS;
    }
    return FTI_SCES;
}

/*
    TODO: It might be good to remove all mprotect for all pages if a SIGSEGV gets raised
    which does not belong to the change-log mechanism.
*/

int FTI_RegisterSigHandler() 
{ 
    // SA_SIGINFO -> flag to allow detailed info about signal
    FTI_SigAction.sa_flags = SA_SIGINFO;
    sigemptyset(&FTI_SigAction.sa_mask);
    FTI_SigAction.sa_sigaction = FTI_SigHandler;
    if( sigaction(SIGSEGV, &FTI_SigAction, &OLD_SigAction) == -1 ){
        FTI_Print( "FTI was unable to register the signal handler", FTI_EROR );
        errno = 0;
        return FTI_NSCS;
    }
    return FTI_SCES;
}

void FTI_SigHandler( int signum, siginfo_t* info, void* ucontext ) 
{
    char strdbg[FTI_BUFS], strerr[FTI_BUFS];

    if ( signum == SIGSEGV ) {
        if( FTI_isValidRequest( (FTI_ADDRVAL)info->si_addr ) ){
            
            // debug information
            snprintf( strdbg, FTI_BUFS, "FTI_DIFFCKPT: 'FTI_SigHandler' - SIGSEGV signal was raised at address %p\n", info->si_addr );
            FTI_Print( strdbg, FTI_DBUG );
            
            // remove protection from page
            if ( mprotect( (FTI_ADDRPTR)(((FTI_ADDRVAL)info->si_addr) & FTI_PageMask), FTI_PageSize, PROT_READ|PROT_WRITE ) == -1) {
                snprintf(strerr, FTI_BUFS, "FTI was unable to protect address: %p", (FTI_ADDRPTR)(((FTI_ADDRVAL)info->si_addr) & FTI_PageMask));
                FTI_Print( strerr, FTI_EROR );
                errno = 0;
                if( sigaction(SIGSEGV, &OLD_SigAction, NULL) == -1 ){
                    FTI_Print( "FTI was unable to restore the parent signal handler", FTI_EROR );
                    errno = 0;
                }
            }

            pagesGlobal = (FTI_ADDRVAL*) realloc( pagesGlobal, (++countPages) * sizeof(FTI_ADDRVAL) );
            pagesGlobal[countPages-1] = ((FTI_ADDRVAL)info->si_addr) & FTI_PageMask;
            
            // register page as dirty
            FTI_ExcludePage( (FTI_ADDRVAL)info->si_addr );
        
        } else {
            /*
                NOTICE: tested that this works also if the application that leverages FTI uses signal() and NOT sigaction(). 
                I.e. the handler registered by signal from the application is called for the case that the SIGSEGV was raised from 
                an address outside of the FTI protected region.
                TODO: However, needs to be tested with applications that use signal handling.
            */

            // forward to default handler and raise signal again
            if( sigaction(SIGSEGV, &OLD_SigAction, NULL) == -1 ){
                FTI_Print( "FTI was unable to restore the parent handler", FTI_EROR );
                errno = 0;
            }
            printf("BEFORE - address: %p\n", info->si_addr);    
            raise(SIGSEGV);
            // if returns from old signal handler (which it shouldn't), register FTI handler again.
            // TODO since we are talking about seg faults, we might not attempt to register our handler again here.
            //if( sigaction(SIGSEGV, &FTI_SigAction, &OLD_SigAction) == -1 ){
            //    FTI_Print( "FTI was unable to register the signal handler", FTI_EROR );
            //    errno = 0;
            //}
        }
    }
}

int FTI_ExcludePage( FTI_ADDRVAL addr ) {
    bool found; 
    FTI_ADDRVAL page = addr & FTI_PageMask;
    int idx;
    long pos;
    if( FTI_GetRangeIndices( page, &idx, &pos) == FTI_NSCS ) {
        return FTI_NSCS;
    }
    // swap array elements i -> i+1 for i > pos and increase counter
    FTI_ADDRVAL base = FTI_SignalDiffInfo.dataDiff[idx].basePtr;
    FTI_ADDRVAL offset = FTI_SignalDiffInfo.dataDiff[idx].ranges[pos].offset;
    FTI_ADDRVAL end = base + offset + FTI_SignalDiffInfo.dataDiff[idx].ranges[pos].size;
    // update values
    FTI_SignalDiffInfo.dataDiff[idx].ranges[pos].size = page - (base + offset);
    if ( FTI_SignalDiffInfo.dataDiff[idx].ranges[pos].size == 0 ) {
        FTI_SignalDiffInfo.dataDiff[idx].ranges[pos].offset = page - base + FTI_PageSize;
        FTI_SignalDiffInfo.dataDiff[idx].ranges[pos].size = end - (page + FTI_PageSize);
        if ( FTI_SignalDiffInfo.dataDiff[idx].ranges[pos].size == 0 ) {
            FTI_ShiftPageItemsLeft( idx, pos );
        }
    } else {
        FTI_ShiftPageItemsRight( idx, pos );
        FTI_SignalDiffInfo.dataDiff[idx].ranges[pos+1].offset = page - base + FTI_PageSize;
        FTI_SignalDiffInfo.dataDiff[idx].ranges[pos+1].size = end - (page + FTI_PageSize);
    }
    // add dirty page to buffer
}

int FTI_GetRangeIndices( FTI_ADDRVAL page, int* idx, long* pos)
{
    // binary search for page
    int i;
    for(i=0; i<FTI_SignalDiffInfo.nbProtVar; ++i){
        if(FTI_SignalDiffInfo.dataDiff[i].rangeCnt == 0) { continue; }
        bool found = false;
        long LOW = 0;
        long UP = FTI_SignalDiffInfo.dataDiff[i].rangeCnt - 1;
        long MID = (LOW+UP)/2; 
        if ( FTI_RangeCmpPage(i, MID, page) == 0 ) {
            found = true;
        }
        while( LOW < UP ) {
            int cmp = FTI_RangeCmpPage(i, MID, page);
            // page is in first half
            if( cmp < 0 ) {
                UP = MID - 1;
            // page is in second half
            } else if ( cmp > 0 ) {
                LOW = MID + 1;
            }
            MID = (LOW+UP)/2;
            if ( FTI_RangeCmpPage(i, MID, page) == 0 ) {
                found = true;
                break;
            }
        }
        if ( found ) {
            *idx = i;
            *pos = MID;
            return FTI_SCES;
        }
    }
    return FTI_NSCS;
}

int FTI_RangeCmpPage(int idx, long idr, FTI_ADDRVAL page) {
    FTI_ADDRVAL base = FTI_SignalDiffInfo.dataDiff[idx].basePtr;
    FTI_ADDRVAL size = FTI_SignalDiffInfo.dataDiff[idx].ranges[idr].size;
    FTI_ADDRVAL first = FTI_SignalDiffInfo.dataDiff[idx].ranges[idr].offset + base;
    FTI_ADDRVAL last = FTI_SignalDiffInfo.dataDiff[idx].ranges[idr].offset + base + size - FTI_PageSize;
    if( page < first ) {
        return -1;
    } else if ( page > last ) {
        return 1;
    } else if ( (page >= first) && (page <= last) ) {
        return 0;
    }
}

int FTI_ShiftPageItemsLeft( int idx, long pos ) {
    // decrease array size by 1 and decrease counter and return if at the end of the array
    if ( pos == FTI_SignalDiffInfo.dataDiff[idx].rangeCnt - 1 ) {
        if ( pos == 0 ) {
            --FTI_SignalDiffInfo.dataDiff[idx].rangeCnt;
            return FTI_SCES;
        }
        FTI_SignalDiffInfo.dataDiff[idx].ranges = (FTIT_DataRange*) realloc(FTI_SignalDiffInfo.dataDiff[idx].ranges, (--FTI_SignalDiffInfo.dataDiff[idx].rangeCnt) * sizeof(FTIT_DataRange));
        assert( FTI_SignalDiffInfo.dataDiff[idx].ranges != NULL );
        return FTI_SCES;
    }
    long i;
    // shift elements of array starting at the end
    for(i=FTI_SignalDiffInfo.dataDiff[idx].rangeCnt - 1; i>pos; --i) {
        memcpy( &(FTI_SignalDiffInfo.dataDiff[idx].ranges[i-1]), &(FTI_SignalDiffInfo.dataDiff[idx].ranges[i]), sizeof(FTIT_DataRange));
    }
    FTI_SignalDiffInfo.dataDiff[idx].ranges = (FTIT_DataRange*) realloc(FTI_SignalDiffInfo.dataDiff[idx].ranges, (--FTI_SignalDiffInfo.dataDiff[idx].rangeCnt) * sizeof(FTIT_DataRange));
    assert( FTI_SignalDiffInfo.dataDiff[idx].ranges != NULL );
    return FTI_SCES;
}

int FTI_ShiftPageItemsRight( int idx, long pos ) {
    // increase array size by 1 and increase counter
    FTI_SignalDiffInfo.dataDiff[idx].ranges = (FTIT_DataRange*) realloc(FTI_SignalDiffInfo.dataDiff[idx].ranges, (++FTI_SignalDiffInfo.dataDiff[idx].rangeCnt) * sizeof(FTIT_DataRange));
    assert( FTI_SignalDiffInfo.dataDiff[idx].ranges != NULL );
    
    long i;
    // shift elements of array starting at the end
    assert(FTI_SignalDiffInfo.dataDiff[idx].rangeCnt > 0);
    for(i=FTI_SignalDiffInfo.dataDiff[idx].rangeCnt-1; i>(pos+1); --i) {
        memcpy( &(FTI_SignalDiffInfo.dataDiff[idx].ranges[i]), &(FTI_SignalDiffInfo.dataDiff[idx].ranges[i-1]), sizeof(FTIT_DataRange));
    }
}

int FTI_RegisterProtections(int idx, FTIT_dataset* FTI_Data, FTIT_execution* FTI_Exec) 
{   
    if( diffMode == 0 ) {
        return FTI_GenerateHashBlocks( idx, FTI_Data, FTI_Exec );
    } else if ( diffMode == 1 ) {
        return FTI_ProtectPages ( idx, FTI_Data );
    } else {
        return FTI_SCES;
    }

}

int FTI_UpdateProtections(int idx, FTIT_dataset* FTI_Data, FTIT_execution* FTI_Exec) 
{   
    if( diffMode == 0 ) {
        return FTI_UpdateHashBlocks( idx, FTI_Data, FTI_Exec );
    //} else if ( diffMode == 1 ) {
    //    return FTI_ProtectPages ( idx, FTI_Data, FTI_Exec );
    } else {
        return FTI_SCES;
    }

}


// TODO think about logic here: is called before a checkpoint in many cases. have to assure that data with new hashes are written.
int FTI_UpdateHashBlocks(int idx, FTIT_dataset* FTI_Data, FTIT_execution* FTI_Exec) 
{
    FTI_ADDRVAL data_ptr = (FTI_ADDRVAL) FTI_Data[idx].ptr;
    FTI_ADDRVAL data_end = (FTI_ADDRVAL) FTI_Data[idx].ptr + (FTI_ADDRVAL) FTI_Data[idx].size;
    FTI_ADDRVAL data_size = (FTI_ADDRVAL) FTI_Data[idx].size;

    FTI_HashDiffInfo.dataDiff[idx].basePtr = data_ptr; 
    long newNbBlocks = data_size/DIFF_BLOCK_SIZE;
    long oldNbBlocks;
    newNbBlocks += ((data_size%DIFF_BLOCK_SIZE) == 0) ? 0 : 1;
    oldNbBlocks = FTI_HashDiffInfo.dataDiff[idx].nbBlocks;
    
    assert(oldNbBlocks > 0);
    
    FTI_HashDiffInfo.dataDiff[idx].nbBlocks = newNbBlocks;
    FTI_HashDiffInfo.dataDiff[idx].totalSize = data_size;

    // if number of blocks decreased
    if ( newNbBlocks < oldNbBlocks ) {
        
        // reduce hash array
        FTI_HashDiffInfo.dataDiff[idx].hashBlocks = (FTIT_HashBlock*) realloc(FTI_HashDiffInfo.dataDiff[idx].hashBlocks, sizeof(FTIT_HashBlock)*(newNbBlocks));
        if ( HASH_MODE == 0 ) {
            FTI_HashDiffInfo.dataDiff[idx].hashBlocks[0].md5hash = (unsigned char*) realloc( FTI_HashDiffInfo.dataDiff[idx].hashBlocks[0].md5hash, (MD5_DIGEST_LENGTH)*(newNbBlocks) );
            int hashIdx;
            for(hashIdx = 0; hashIdx<newNbBlocks; ++hashIdx) {
                FTI_HashDiffInfo.dataDiff[idx].hashBlocks[hashIdx].md5hash = FTI_HashDiffInfo.dataDiff[idx].hashBlocks[0].md5hash + (hashIdx) * MD5_DIGEST_LENGTH;
            }
        }

    // if number of blocks increased
    } else if ( newNbBlocks > oldNbBlocks ) {
        
        // extend hash array
        FTI_HashDiffInfo.dataDiff[idx].hashBlocks = (FTIT_HashBlock*) realloc(FTI_HashDiffInfo.dataDiff[idx].hashBlocks, sizeof(FTIT_HashBlock)*(newNbBlocks));    
        int hashIdx;
        if ( HASH_MODE == 0 ) {
            FTI_HashDiffInfo.dataDiff[idx].hashBlocks[0].md5hash = (unsigned char*) realloc( FTI_HashDiffInfo.dataDiff[idx].hashBlocks[0].md5hash, (MD5_DIGEST_LENGTH) * newNbBlocks );
            for(hashIdx = 0; hashIdx<newNbBlocks; ++hashIdx) {
                FTI_HashDiffInfo.dataDiff[idx].hashBlocks[hashIdx].md5hash = FTI_HashDiffInfo.dataDiff[idx].hashBlocks[0].md5hash + (hashIdx) * MD5_DIGEST_LENGTH;
            }
        }
        data_ptr += oldNbBlocks * DIFF_BLOCK_SIZE;
        // set new hash values
        for(hashIdx = oldNbBlocks; hashIdx<newNbBlocks; ++hashIdx) {
            //int hashBlockSize = ( (data_end - data_ptr) > DIFF_BLOCK_SIZE ) ? DIFF_BLOCK_SIZE : data_end - data_ptr;
            //MD5_CTX ctx;
            //MD5_Init(&ctx);
            //MD5_Update(&ctx, (FTI_ADDRPTR) data_ptr, hashBlockSize);
            //MD5_Final(FTI_HashDiffInfo.dataDiff[idx].hashBlocks[hashIdx].hash, &ctx);
            FTI_HashDiffInfo.dataDiff[idx].hashBlocks[hashIdx].dirty = true; 
            FTI_HashDiffInfo.dataDiff[idx].hashBlocks[hashIdx].isValid = false; 
            //data_ptr += hashBlockSize;
        }
    
    }
    return 0;
}

int FTI_GenerateHashBlocks( int idx, FTIT_dataset* FTI_Data, FTIT_execution* FTI_Exec ) {
   
    FTI_HashDiffInfo.dataDiff = (FTIT_DataDiffHash*) realloc( FTI_HashDiffInfo.dataDiff, (FTI_HashDiffInfo.nbProtVar+1) * sizeof(FTIT_DataDiffHash));
    assert( FTI_HashDiffInfo.dataDiff != NULL );
    FTI_ADDRVAL basePtr = (FTI_ADDRVAL) FTI_Data[idx].ptr;
    FTI_ADDRVAL ptr = (FTI_ADDRVAL) FTI_Data[idx].ptr;
    FTI_ADDRVAL end = (FTI_ADDRVAL) FTI_Data[idx].ptr + (FTI_ADDRVAL) FTI_Data[idx].size;
    long nbHashBlocks = FTI_Data[idx].size/DIFF_BLOCK_SIZE;
    nbHashBlocks += ( (FTI_Data[idx].size%DIFF_BLOCK_SIZE) == 0 ) ? 0 : 1; 
    FTI_HashDiffInfo.dataDiff[FTI_HashDiffInfo.nbProtVar].nbBlocks = nbHashBlocks;
    FTIT_HashBlock* hashBlocks = (FTIT_HashBlock*) malloc( sizeof(FTIT_HashBlock) * nbHashBlocks );
    assert( hashBlocks != NULL );
    // keep hashblocks array dense
    if ( HASH_MODE == 0 ) {
        hashBlocks[0].md5hash = (unsigned char*) malloc( (MD5_DIGEST_LENGTH) * nbHashBlocks );
        assert( hashBlocks[0].md5hash != NULL );
    }
    long cnt = 0;
    while( ptr < end ) {
        int hashBlockSize = ( (end - ptr) > DIFF_BLOCK_SIZE ) ? DIFF_BLOCK_SIZE : end-ptr;
        if ( HASH_MODE == 0 ) {
            hashBlocks[cnt].md5hash = hashBlocks[0].md5hash + cnt*MD5_DIGEST_LENGTH;
        }
        //MD5_CTX ctx;
        //MD5_Init(&ctx);
        //MD5_Update(&ctx, (FTI_ADDRPTR)ptr, hashBlockSize);
        //MD5_Final(hashBlocks[cnt].hash, &ctx);
        hashBlocks[cnt].dirty = true;
        hashBlocks[cnt].isValid = false;
        cnt++;
        ptr+=hashBlockSize;
    }
    assert( nbHashBlocks == cnt );
    FTI_HashDiffInfo.dataDiff[FTI_HashDiffInfo.nbProtVar].hashBlocks    = hashBlocks;
    FTI_HashDiffInfo.dataDiff[FTI_HashDiffInfo.nbProtVar].basePtr       = basePtr;
    FTI_HashDiffInfo.dataDiff[FTI_HashDiffInfo.nbProtVar].totalSize     = end - basePtr;
    FTI_HashDiffInfo.dataDiff[FTI_HashDiffInfo.nbProtVar].id            = FTI_Data[idx].id;
    FTI_HashDiffInfo.nbProtVar++;
    return FTI_SCES;
}

int FTI_ProtectPages ( int idx, FTIT_dataset* FTI_Data ) {
    char strdbg[FTI_BUFS];
    FTI_ADDRVAL first_page = FTI_GetFirstInclPage((FTI_ADDRVAL)FTI_Data[idx].ptr);
    FTI_ADDRVAL last_page = FTI_GetLastInclPage((FTI_ADDRVAL)FTI_Data[idx].ptr+FTI_Data[idx].size);
    
    FTI_ASSERT_ALIGNED(first_page, FTI_PageSize); 
    FTI_ASSERT_ALIGNED(last_page, FTI_PageSize); 
    
    FTI_ADDRVAL psize = 0;

    // check if dataset includes at least one full page.
    if (first_page < last_page) {
        psize = last_page - first_page + FTI_PageSize;
        
        FTI_ASSERT_ALIGNED(psize, FTI_PageSize); 
        
        if ( mprotect((FTI_ADDRPTR) first_page, psize, PROT_READ) == -1 ) {
            FTI_Print( "FTI was unable to register the pages", FTI_EROR );
            errno = 0;
            return FTI_NSCS;
        }

        // TODO no support for datasets that change size yet
        FTI_SignalDiffInfo.dataDiff = (FTIT_DataDiffSignal*) realloc( FTI_SignalDiffInfo.dataDiff, (FTI_SignalDiffInfo.nbProtVar+1) * sizeof(FTIT_DataDiffSignal));
        assert( FTI_SignalDiffInfo.dataDiff != NULL );
        FTI_SignalDiffInfo.dataDiff[FTI_SignalDiffInfo.nbProtVar].ranges = (FTIT_DataRange*) malloc( sizeof(FTIT_DataRange) );
        assert( FTI_SignalDiffInfo.dataDiff[FTI_SignalDiffInfo.nbProtVar].ranges != NULL );
        FTI_SignalDiffInfo.dataDiff[FTI_SignalDiffInfo.nbProtVar].rangeCnt       = 1;
        FTI_SignalDiffInfo.dataDiff[FTI_SignalDiffInfo.nbProtVar].ranges->offset = (FTI_ADDRVAL) 0x0;
        FTI_SignalDiffInfo.dataDiff[FTI_SignalDiffInfo.nbProtVar].ranges->size   = psize;
        FTI_SignalDiffInfo.dataDiff[FTI_SignalDiffInfo.nbProtVar].basePtr        = first_page;
        FTI_SignalDiffInfo.dataDiff[FTI_SignalDiffInfo.nbProtVar].totalSize      = psize;
        FTI_SignalDiffInfo.dataDiff[FTI_SignalDiffInfo.nbProtVar].id             = FTI_Data[idx].id;
        FTI_SignalDiffInfo.nbProtVar++;

    // if not don't protect anything. NULL just for debug output.
    } else {
        FTI_SignalDiffInfo.dataDiff = (FTIT_DataDiffSignal*) realloc( FTI_SignalDiffInfo.dataDiff, (FTI_SignalDiffInfo.nbProtVar+1) * sizeof(FTIT_DataDiffSignal));
        assert( FTI_SignalDiffInfo.dataDiff != NULL );
        FTI_SignalDiffInfo.dataDiff[FTI_SignalDiffInfo.nbProtVar].ranges         = (FTIT_DataRange*) malloc( sizeof(FTIT_DataRange) );
        FTI_SignalDiffInfo.dataDiff[FTI_SignalDiffInfo.nbProtVar].rangeCnt       = 0;
        FTI_SignalDiffInfo.dataDiff[FTI_SignalDiffInfo.nbProtVar].ranges->offset = (FTI_ADDRVAL) 0x0;
        FTI_SignalDiffInfo.dataDiff[FTI_SignalDiffInfo.nbProtVar].ranges->size   = 0;
        FTI_SignalDiffInfo.dataDiff[FTI_SignalDiffInfo.nbProtVar].basePtr        = (FTI_ADDRVAL) 0x0;
        FTI_SignalDiffInfo.dataDiff[FTI_SignalDiffInfo.nbProtVar].totalSize      = 0;
        FTI_SignalDiffInfo.dataDiff[FTI_SignalDiffInfo.nbProtVar].id             = FTI_Data[idx].id;
        FTI_SignalDiffInfo.nbProtVar++;
        first_page = (FTI_ADDRVAL) NULL;
        last_page = (FTI_ADDRVAL) NULL;
    }

    // debug information
    snprintf(strdbg, FTI_BUFS, "FTI-DIFFCKPT: 'FTI_ProtectPages' - ID: %d, size: %lu, pages protect: %lu, addr: %p, first page: %p, last page: %p rangeCnt: %ld\n", 
            FTI_Data[idx].id, 
            FTI_Data[idx].size, 
            psize/FTI_PageSize,
            FTI_Data[idx].ptr,
            (FTI_ADDRPTR) first_page,
            (FTI_ADDRPTR) last_page,
            FTI_SignalDiffInfo.dataDiff[FTI_SignalDiffInfo.nbProtVar-1].rangeCnt );
    FTI_Print( strdbg, FTI_DBUG );
    return FTI_SCES;
}

FTI_ADDRVAL FTI_GetFirstInclPage(FTI_ADDRVAL addr) 
{
    FTI_ADDRVAL page; 
    page = (addr + FTI_PageSize - 1) & FTI_PageMask;
    return page;
}

FTI_ADDRVAL FTI_GetLastInclPage(FTI_ADDRVAL addr) 
{
    FTI_ADDRVAL page; 
    page = (addr - FTI_PageSize + 1) & FTI_PageMask;
    return page;
}

bool FTI_isValidRequest( FTI_ADDRVAL addr_val ) 
{
    
    if( addr_val == (FTI_ADDRVAL) NULL ) return false;

    FTI_ADDRVAL page = ((FTI_ADDRVAL) addr_val) & FTI_PageMask;

    if ( FTI_ProtectedPageIsValid( page ) && FTI_isProtectedPage( page ) ) {
        return true;
    }

    return false;

}

bool FTI_ProtectedPageIsValid( FTI_ADDRVAL page ) 
{
    // binary search for page
    bool isValid = false;
    int i;
    for(i=0; i<FTI_SignalDiffInfo.nbProtVar; ++i){
        if(FTI_SignalDiffInfo.dataDiff[i].rangeCnt == 0) { continue; }
        long LOW = 0;
        long UP = FTI_SignalDiffInfo.dataDiff[i].rangeCnt - 1;
        long MID = (LOW+UP)/2; 
        if ( FTI_RangeCmpPage(i, MID, page) == 0 ) {
            isValid = true;
        }
        while( LOW < UP ) {
            int cmp = FTI_RangeCmpPage(i, MID, page);
            // page is in first half
            if( cmp < 0 ) {
                UP = MID - 1;
            // page is in second half
            } else if ( cmp > 0 ) {
                LOW = MID + 1;
            }
            MID = (LOW+UP)/2;
            if ( FTI_RangeCmpPage(i, MID, page) == 0 ) {
                isValid = true;
                break;
            }
        }
        if ( isValid ) {
            break;
        }
    }
    return isValid;
}

bool FTI_isProtectedPage( FTI_ADDRVAL page ) 
{
    bool inRange = false;
    int i;
    for(i=0; i<FTI_SignalDiffInfo.nbProtVar; ++i) {
        if(FTI_SignalDiffInfo.dataDiff[i].rangeCnt == 0) { continue; }
        if( (page >= FTI_SignalDiffInfo.dataDiff[i].basePtr) && (page <= FTI_SignalDiffInfo.dataDiff[i].basePtr + FTI_SignalDiffInfo.dataDiff[i].totalSize - FTI_PageSize) ) {
            inRange = true;
            break;
        }
    }
    return inRange;
}

int FTI_HashCmp( int varIdx, long hashIdx, FTI_ADDRPTR ptr, int hashBlockSize ) {
    
    struct timespec t1;
    struct timespec t2;
    counterHB_t[hashBlockSize/counterHBStride]++; // INT_DIVIDE(size,stride) * stride -> bin
    clock_gettime( CLOCK_REALTIME, &t1 );

    // check if in range
    if ( hashIdx < FTI_HashDiffInfo.dataDiff[varIdx].nbBlocks ) {
        unsigned char md5hashNow[MD5_DIGEST_LENGTH];
        uint32_t bit32hashNow;
        bool clean;
        if ( HASH_MODE == 0 ) {
            MD5(ptr, hashBlockSize, md5hashNow);
            clean = memcmp(md5hashNow, FTI_HashDiffInfo.dataDiff[varIdx].hashBlocks[hashIdx].md5hash, MD5_DIGEST_LENGTH) == 0;
        } else {
            switch ( HASH_MODE ) {
                case 1:
                    bit32hashNow = crc_32( ptr, hashBlockSize );
                    break;
                case 2:
                    bit32hashNow = adler32( ptr, hashBlockSize );
                    break;
                case 3:
                    bit32hashNow = fletcher32( ptr, hashBlockSize );
                    break;
            }
            clean = bit32hashNow == FTI_HashDiffInfo.dataDiff[varIdx].hashBlocks[hashIdx].bit32hash;
        }
        // set clean if unchanged
        if ( clean ) {
            FTI_HashDiffInfo.dataDiff[varIdx].hashBlocks[hashIdx].dirty = false;
            clock_gettime( CLOCK_REALTIME, &t2 );
            accumulateHashCmpTime( t1, t2 );
            return 0;
        // set dirty if changed
        } else {
            FTI_HashDiffInfo.dataDiff[varIdx].hashBlocks[hashIdx].dirty = true;
            clock_gettime( CLOCK_REALTIME, &t2 );
            accumulateHashCmpTime( t1, t2 );
            return 1;
        }
    // return -1 if end
    } else {
        clock_gettime( CLOCK_REALTIME, &t2 );
        accumulateHashCmpTime( t1, t2 );
        return -1;
    }
}

int FTI_UpdateChanges(FTIT_dataset* FTI_Data) 
{
    if( diffMode == 0 ) {
        return FTI_UpdateHashChanges( FTI_Data );
    } else if ( diffMode == 1 ) {
        return FTI_UpdateSigChanges( FTI_Data );
    } else {
        return FTI_SCES;
    }
}

int FTI_UpdateSigChanges(FTIT_dataset* FTI_Data) 
{
    FTI_RemoveProtections();
    FTI_FreeDiffCkptStructs();
    int varIdx;
    int nbProtVar = FTI_SignalDiffInfo.nbProtVar; 
    FTI_SignalDiffInfo.dataDiff = NULL;
    FTI_SignalDiffInfo.nbProtVar = 0;
    for(varIdx=0; varIdx<nbProtVar; ++varIdx) {
        FTI_ProtectPages( varIdx, FTI_Data );     
    }
}

int FTI_UpdateHashChanges(FTIT_dataset* FTI_Data) 
{

    struct timespec t1;
    clock_gettime( CLOCK_REALTIME, &t1 );
    double start_t = MPI_Wtime();
    int varIdx;
    int nbProtVar = FTI_HashDiffInfo.nbProtVar;
    long memuse = 0;
    long totalmemprot = 0;
    for(varIdx=0; varIdx<nbProtVar; ++varIdx) {
        assert(FTI_Data[varIdx].size == FTI_HashDiffInfo.dataDiff[varIdx].totalSize);
        totalmemprot += FTI_Data[varIdx].size;
        FTI_ADDRPTR ptr = FTI_Data[varIdx].ptr;
        long pos = 0;
        int width = 0;
        int blockIdx;
        int nbBlocks = FTI_HashDiffInfo.dataDiff[varIdx].nbBlocks;
        for(blockIdx=0; blockIdx<nbBlocks; ++blockIdx) {
            if ( HASH_MODE == 0 ) {
                memuse += MD5_DIGEST_LENGTH;
            } else {
                memuse += sizeof(uint32_t); 
            }
            if ( !FTI_HashDiffInfo.dataDiff[varIdx].hashBlocks[blockIdx].isValid || FTI_HashDiffInfo.dataDiff[varIdx].hashBlocks[blockIdx].dirty ) {
                width = ( (FTI_HashDiffInfo.dataDiff[varIdx].totalSize - pos) > DIFF_BLOCK_SIZE ) ? DIFF_BLOCK_SIZE : (FTI_HashDiffInfo.dataDiff[varIdx].totalSize - pos);
                switch ( HASH_MODE ) {
                    case 0:
                        MD5(ptr, width, FTI_HashDiffInfo.dataDiff[varIdx].hashBlocks[blockIdx].md5hash);
                        break;
                    case 1:
                        FTI_HashDiffInfo.dataDiff[varIdx].hashBlocks[blockIdx].bit32hash = crc_32( ptr, width );
                        break;
                    case 2:
                        FTI_HashDiffInfo.dataDiff[varIdx].hashBlocks[blockIdx].bit32hash = adler32( ptr, width );
                        break;
                    case 3:
                        FTI_HashDiffInfo.dataDiff[varIdx].hashBlocks[blockIdx].bit32hash = fletcher32( ptr, width );
                        break;
                }
                FTI_HashDiffInfo.dataDiff[varIdx].hashBlocks[blockIdx].dirty = false;
                FTI_HashDiffInfo.dataDiff[varIdx].hashBlocks[blockIdx].isValid = true;
            }
            ptr += (FTI_ADDRVAL) width;
            pos += width;
        }
    }
    char strout[FTI_BUFS];
    int rank;
    
    struct timespec t2;
    clock_gettime( CLOCK_REALTIME, &t2 );
    accumulateHashUpdateTime( t1, t2 );
    accumulateWriteFtiff( t1, t2 );
    
    MPI_Comm_rank(FTI_COMM_WORLD, &rank);
    if (rank==0) {
        printf("[ " BLU "FTI  DIFFCKPT" RESET " ] : hash arrays in memory: %lf MB, total memory protected: %lf MB\n", ((double)memuse)/(1024*1024), ((double)totalmemprot)/(1024*1024));
        printf("[ " BLU "FTI  DIFFCKPT" RESET " ] : Update hashes took: %lf second's\n", MPI_Wtime()-start_t);
    }
}

int FTI_ReceiveDiffChunk(int id, FTI_ADDRVAL data_offset, FTI_ADDRVAL data_size, FTI_ADDRVAL* buffer_offset, FTI_ADDRVAL* buffer_size, FTIT_execution* FTI_Exec, FTIFF_dbvar* dbvar) {
    
    struct timespec t1;
    struct timespec t2;
    clock_gettime( CLOCK_REALTIME, &t1 );
   
    // FOR TESTING
    int rank;
    MPI_Comm_rank(FTI_COMM_WORLD, &rank);
    static bool init = true;
    static long pos;
    static FTI_ADDRVAL data_ptr;
    static FTI_ADDRVAL data_end;
    static long hash_ptr;
    char strdbg[FTI_BUFS];
    if ( init ) {
        hash_ptr = dbvar->dptr;
        pos = 0;
        data_ptr = data_offset;
        data_end = data_offset + data_size;
        init = false;
    }
    
    int idx;
    long i;
    bool flag;
    // reset function and return not found
    if ( pos == -1 ) {
        init = true;
        clock_gettime( CLOCK_REALTIME, &t2 );
        accumulateReceiveDiffChunkTime( t1, t2 );
        return 0;
    }
   
    // if differential ckpt is disabled, return whole chunk and finalize call
    if ( !enableDiffCkpt ) {
        pos = -1;
        *buffer_offset = data_ptr;
        *buffer_size = data_size;
        counter_t[(*buffer_size)/counterStride]++; // INT_DIVIDE(size,stride) * stride -> bin
        clock_gettime( CLOCK_REALTIME, &t2 );
        accumulateReceiveDiffChunkTime( t1, t2 );
        return 1;
    }

    if ( diffMode == 0 ) {
        for(idx=0; (flag = FTI_HashDiffInfo.dataDiff[idx].id != id) && (idx < FTI_HashDiffInfo.nbProtVar); ++idx);
        if ( !flag ) {
            //if ( !dbvar->hasCkpt ) {
            //    // set pos = -1 to ensure a return value of 0 and a function reset at next invokation
            //    pos = -1;
            //    *buffer_offset = data_ptr;
            //    *buffer_size = data_size;
            //    return 1;
            //}
            //if (pos == 0) {
            //    long checkamount = FTI_CheckDiffAmount(idx, (FTI_ADDRPTR) data_offset, data_size);
            //    if(rank == 0) {
            //        printf("[var-id:%d] expected amount: %ld\n",id,checkamount);
            //    }
            //}
            int hashBlockSize = ( (data_end - data_ptr) > DIFF_BLOCK_SIZE ) ? DIFF_BLOCK_SIZE : data_end - data_ptr;
            long hashIdx = hash_ptr/DIFF_BLOCK_SIZE;
            
            // advance *buffer_offset for clean regions
            bool clean = FTI_HashCmp( idx, hashIdx, (FTI_ADDRPTR) data_ptr, hashBlockSize ) == 0;
            clean &= FTI_HashDiffInfo.dataDiff[idx].hashBlocks[hashIdx].isValid;
            while( clean ) {
                data_ptr += hashBlockSize;
                hash_ptr += hashBlockSize;
                hashIdx = hash_ptr/DIFF_BLOCK_SIZE;
                hashBlockSize = ( (data_end - data_ptr) > DIFF_BLOCK_SIZE ) ? DIFF_BLOCK_SIZE : data_end - data_ptr;
                clean = FTI_HashCmp( idx, hashIdx, (FTI_ADDRPTR) data_ptr, hashBlockSize ) == 0;
                clean &= FTI_HashDiffInfo.dataDiff[idx].hashBlocks[hashIdx].isValid;
            }
            
            /* if at call pointer to dirty region then data_ptr unchanged */
            *buffer_offset = data_ptr;
            *buffer_size = 0;
            
            // advance *buffer_size for dirty regions
            bool dirty = FTI_HashCmp( idx, hashIdx, (FTI_ADDRPTR) data_ptr, hashBlockSize ) == 1;
            dirty |= !(FTI_HashDiffInfo.dataDiff[idx].hashBlocks[hashIdx].isValid);
            bool inRange = data_ptr < data_end;
            while( dirty && inRange ) {
                *buffer_size += hashBlockSize;
                data_ptr += hashBlockSize;
                hash_ptr += hashBlockSize;
                hashIdx = hash_ptr/DIFF_BLOCK_SIZE;
                hashBlockSize = ( (data_end - data_ptr) > DIFF_BLOCK_SIZE ) ? DIFF_BLOCK_SIZE : data_end - data_ptr;
                dirty = FTI_HashCmp( idx, hashIdx, (FTI_ADDRPTR) data_ptr, hashBlockSize ) == 1;
                dirty |= !(FTI_HashDiffInfo.dataDiff[idx].hashBlocks[hashIdx].isValid);
                inRange = data_ptr < data_end;
            }
            // check if we are at the end of the data region
            //if ( FTI_HashCmp( idx, hashIdx, (FTI_ADDRPTR) data_ptr, hashBlockSize ) == -1 ) {
            if ( data_ptr == data_end ) {
                //if ( data_ptr != data_end ) {
                //    FTI_Print("DIFF-CKPT: meta-data inconsistency: data size stored does not match runtime data size", FTI_WARN);
                //    init = true;
                //    return 0;
                //}
                if ( *buffer_size != 0 ) {
                    pos = -1;
                    counter_t[(*buffer_size)/counterStride]++;
                    clock_gettime( CLOCK_REALTIME, &t2 );
                    accumulateReceiveDiffChunkTime( t1, t2 );
                    return 1;
                } else {
                    init = true;
                    clock_gettime( CLOCK_REALTIME, &t2 );
                    accumulateReceiveDiffChunkTime( t1, t2 );
                    return 0;
                }
            }
            pos = hashIdx;
            counter_t[(*buffer_size)/counterStride]++;
            clock_gettime( CLOCK_REALTIME, &t2 );
            accumulateReceiveDiffChunkTime( t1, t2 );
            return 1;
        }
    }

    if ( diffMode == 1 ) {
        for(idx=0; (flag = FTI_SignalDiffInfo.dataDiff[idx].id != id) && (idx < FTI_SignalDiffInfo.nbProtVar); ++idx);
        if( !flag ) {
            FTI_ADDRVAL base = FTI_SignalDiffInfo.dataDiff[idx].basePtr;
            // all memory dirty or not protected or first proper checkpoint
            if ( FTI_SignalDiffInfo.dataDiff[idx].rangeCnt == 0 || !dbvar->hasCkpt ) {
                // set pos = -1 to ensure a return value of 0 and a function reset at next invokation
                pos = -1;
                *buffer_offset = data_ptr;
                *buffer_size = data_size;
                return 1;
            }
            for(i=pos; i<FTI_SignalDiffInfo.dataDiff[idx].rangeCnt; ++i) {
                FTI_ADDRVAL range_size = FTI_SignalDiffInfo.dataDiff[idx].ranges[i].size;
                FTI_ADDRVAL range_offset = FTI_SignalDiffInfo.dataDiff[idx].ranges[i].offset + base;
                FTI_ADDRVAL range_end = range_offset + range_size;
                FTI_ADDRVAL dirty_range_end; 
                // dirty pages at beginning of data buffer
                if ( data_ptr < range_offset ) {
                    snprintf(strdbg, FTI_BUFS, "FTI-DIFFCKPT: 'FTI_ReceiveDiffChunk:%d' - id: %d, idx-ranges: %lu, offset: %p, size: %lu\n",
                            __LINE__, 
                            FTI_SignalDiffInfo.dataDiff[idx].id, 
                            i,
                            range_offset,
                            range_size);
                    FTI_Print(strdbg, FTI_DBUG);
                    *buffer_offset = data_ptr;
                    *buffer_size = range_offset - data_ptr;
                    // at next call, data_ptr should be equal to range_offset of range[pos]
                    // and one of the next if clauses should be invoked
                    data_ptr = range_offset;
                    pos = i;
                    return 1;
                }
                // dirty pages after the beginning of data buffer
                if ( (data_ptr >= range_offset) && (range_end < data_end) ) {
                    if ( i < FTI_SignalDiffInfo.dataDiff[idx].rangeCnt-1 ) {
                        snprintf(strdbg, FTI_BUFS, "FTI-DIFFCKPT: 'FTI_ReceiveDiffChunk:%d' - id: %d, idx-ranges: %lu, offset: %p, size: %lu\n",
                                __LINE__, 
                                FTI_SignalDiffInfo.dataDiff[idx].id, 
                                i,
                                range_offset,
                                range_size);
                        FTI_Print(strdbg, FTI_DBUG);
                        data_ptr = FTI_SignalDiffInfo.dataDiff[idx].ranges[i+1].offset + base;
                        pos = i+1;
                        *buffer_offset = range_end;
                        *buffer_size = data_ptr - range_end;
                        return 1;
                        // this is the last clean range
                    } else if ( data_end != range_end ) {
                        snprintf(strdbg, FTI_BUFS, "FTI-DIFFCKPT: 'FTI_ReceiveDiffChunk:%d' - id: %d, idx-ranges: %lu, offset: %p, size: %lu\n",
                                __LINE__, 
                                FTI_SignalDiffInfo.dataDiff[idx].id, 
                                i,
                                range_offset,
                                range_size);
                        FTI_Print(strdbg, FTI_DBUG);
                        data_ptr = data_end;
                        pos = -1;
                        *buffer_offset = range_end;
                        *buffer_size = data_ptr - range_end;
                        return 1;
                    }
                }
                // data buffer ends inside clean range
                if ( (data_ptr >= range_offset) && (range_end >= data_end) ) {
                    break;
                }
            }
        }
    }
    // nothing to return -> function reset
    init = true;
    return 0;
}

long FTI_CheckDiffAmount(int idx, FTI_ADDRPTR ptr, FTI_ADDRVAL size) {
    long counter = 0;
    long dirty = 0;
    long nbBlocks = size/DIFF_BLOCK_SIZE;// + ((size%DIFF_BLOCK_SIZE)==0)?0:1;
    nbBlocks += ((size%DIFF_BLOCK_SIZE)==0)?0:1;
    int i;
    //printf("nbBlocks:%ld, size:%lu, diff_block_size:%d\n",nbBlocks, size, DIFF_BLOCK_SIZE);
    for(i=0; i<nbBlocks; ++i) {
        int hbs = ((size-counter)>DIFF_BLOCK_SIZE)?DIFF_BLOCK_SIZE:size-counter;
        int res = FTI_HashCmp( idx, i, ptr, hbs ); 
        assert(res != -1);
        if ( res == 1 ) {
            dirty += hbs;
        }
        counter += hbs;
        ptr += hbs;
    }
    return dirty;
}

bool verifyRanges() {
    FTI_ADDRVAL* pagesLocal = NULL;
    long localPageCount = 0;
    long memalloc = 0;
    int varIdx;
    int pageIdx;
    int rangeIdx;
    for(varIdx = 0; varIdx<FTI_SignalDiffInfo.nbProtVar; ++varIdx) {
        FTI_ADDRVAL numPages;
        FTI_ADDRVAL offset;
        FTI_ADDRVAL base;
        FTI_ADDRVAL endLastRange;
        FTI_ADDRVAL end;
        FTI_ADDRVAL first_page = FTI_SignalDiffInfo.dataDiff[varIdx].basePtr;
        assert(first_page%FTI_PageSize == 0);
        long rangeCnt = FTI_SignalDiffInfo.dataDiff[varIdx].rangeCnt;
        for(rangeIdx=0; rangeIdx<rangeCnt; ++rangeIdx) {
            assert(FTI_SignalDiffInfo.dataDiff[varIdx].ranges[rangeIdx].size%FTI_PageSize == 0);
            if ( rangeIdx > 0 ) {
                offset = first_page + FTI_SignalDiffInfo.dataDiff[varIdx].ranges[rangeIdx].offset;
                base = first_page + FTI_SignalDiffInfo.dataDiff[varIdx].ranges[rangeIdx-1].offset + FTI_SignalDiffInfo.dataDiff[varIdx].ranges[rangeIdx-1].size;
                assert(offset%FTI_PageSize == 0);
                assert(base%FTI_PageSize == 0);
            } else {
                offset = first_page + FTI_SignalDiffInfo.dataDiff[varIdx].ranges[rangeIdx].offset;
                base = first_page;
                assert(offset%FTI_PageSize == 0);
                assert(base%FTI_PageSize == 0);
            }
            numPages = offset - base;
            FTI_ASSERT_ALIGNED(numPages, FTI_PageSize);
            numPages /= FTI_PageSize;
            for(pageIdx = 0; pageIdx<numPages; ++pageIdx) {
                pagesLocal = (FTI_ADDRVAL*) realloc( pagesLocal, sizeof(FTI_ADDRVAL) * (++localPageCount) );
                pagesLocal[localPageCount-1] = base + pageIdx*FTI_PageSize;
            }
        }
        endLastRange = offset + FTI_SignalDiffInfo.dataDiff[varIdx].ranges[rangeCnt-1].size;
        end = FTI_SignalDiffInfo.dataDiff[varIdx].basePtr + FTI_SignalDiffInfo.dataDiff[varIdx].totalSize;
        assert(endLastRange%FTI_PageSize == 0);
        assert(end%FTI_PageSize == 0);
        if ( end != endLastRange ) {
            numPages = (end - endLastRange)/FTI_PageSize;
            for(pageIdx = 0; pageIdx<numPages; ++pageIdx) {
                pagesLocal = (FTI_ADDRVAL*) realloc( pagesLocal, sizeof(FTI_ADDRVAL) * (++localPageCount) );
                pagesLocal[localPageCount-1] = endLastRange + pageIdx*FTI_PageSize;
            }
        }
    }
    assert( localPageCount == countPages );
    qsort( pagesLocal, localPageCount, sizeof(FTI_ADDRVAL), compare );
    qsort( pagesGlobal, countPages, sizeof(FTI_ADDRVAL), compare );
    for(pageIdx=0; pageIdx<countPages; ++pageIdx) {
        assert( compare( &pagesGlobal[pageIdx], &pagesLocal[pageIdx] ) == 0 );
    }
}

