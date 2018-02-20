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


#include "diff-checkpoint.h"

/**                                                                                     */
/** Static Global Variables                                                             */

static FTI_ADDRVAL          FTI_PageSize;       /**< memory page size                   */
static FTI_ADDRVAL          FTI_PageMask;       /**< memory page mask                   */

static FTIT_DataDiffInfo    FTI_DataDiffInfo;   /**< container for diff of datasets     */

/** File Local Variables                                                                */

static struct sigaction     FTI_SigAction;       /**< sigaction meta data               */
static struct sigaction     OLD_SigAction;       /**< previous sigaction meta data      */

/** Function Definitions                                                                */

int FTI_InitDiffCkpt( FTIT_execution* FTI_Exec, FTIT_dataset* FTI_Data )
{
    // get page mask
    FTI_PageSize = (FTI_ADDRVAL) sysconf(_SC_PAGESIZE);
    FTI_PageMask = ~((FTI_ADDRVAL)0x0);
    FTI_ADDRVAL tail = (FTI_ADDRVAL)0x1;
    for(; tail!=FTI_PageSize; FTI_PageMask<<=1, tail<<=1); 
    // init data diff structure
    FTI_DataDiffInfo.dataDiff = NULL;
    FTI_DataDiffInfo.nbProtVar = 0;
    // register signal handler
    return FTI_RegisterSigHandler();
}

void printReport() {
    long num_changed = 0;
    long num_prot=0; 
    int i,j;
    for(i=0; i<FTI_DataDiffInfo.nbProtVar; ++i) {
        num_changed += FTI_DataDiffInfo.dataDiff[i].dirtyPagesCnt;
        for(j=0; j<FTI_DataDiffInfo.dataDiff[i].rangeCnt; ++j) {
            num_prot += FTI_DataDiffInfo.dataDiff[i].ranges[j].size;
        }
    }
    num_prot /= FTI_PageSize;

    printf(
            "Diff Ckpt Summary\n"
            "-------------------------------------\n"
            "number of pages protected:       %lu\n"
            "number of pages changed:         %lu\n",
            num_prot, num_changed);
    fflush(stdout);
    
}

int FTI_FinalizeDiffCkpt(){
    int res = 0;
    printReport();
    res += FTI_RemoveSigHandler();
    res += FTI_RemoveProtections();
    res += FTI_FreeDiffCkptStructs();
    return ( res == 0 ) ? FTI_SCES : FTI_NSCS;
}

int FTI_FreeDiffCkptStructs() {
    int i;
    
    for(i=0; i<FTI_DataDiffInfo.nbProtVar; ++i) {
        free(FTI_DataDiffInfo.dataDiff[i].ranges);
    }
    free(FTI_DataDiffInfo.dataDiff);

    return FTI_SCES;
}

int FTI_RemoveProtections() {
    int i,j;
    for(i=0; i<FTI_DataDiffInfo.nbProtVar; ++i){
        for(j=0; j<FTI_DataDiffInfo.dataDiff[i].rangeCnt; ++j) {
            FTI_ADDRPTR ptr = (FTI_ADDRPTR) (FTI_DataDiffInfo.dataDiff[i].basePtr + FTI_DataDiffInfo.dataDiff[i].ranges[j].offset);
            long size = (long) FTI_DataDiffInfo.dataDiff[i].ranges[j].size;
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
    // SA_NODEFER -> flag to allow the signal to be raised inside handler
    FTI_SigAction.sa_flags = SA_SIGINFO|SA_NODEFER;
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
    char strdbg[FTI_BUFS];

    if ( signum == SIGSEGV ) {
        if( FTI_isValidRequest( (FTI_ADDRVAL)info->si_addr ) ){
            
            // debug information
            snprintf( strdbg, FTI_BUFS, "FTI_DIFFCKPT: 'FTI_SigHandler' - SIGSEGV signal was raised at address %p\n", info->si_addr );
            FTI_Print( strdbg, FTI_DBUG );
            
            // remove protection from page
            if ( mprotect( (FTI_ADDRPTR)(((FTI_ADDRVAL)info->si_addr) & FTI_PageMask), FTI_PageSize, PROT_READ|PROT_WRITE ) == -1) {
                FTI_Print( "FTI was unable to register the signal handler", FTI_EROR );
                errno = 0;
                if( sigaction(SIGSEGV, &OLD_SigAction, NULL) == -1 ){
                    FTI_Print( "FTI was unable to restore the parent signal handler", FTI_EROR );
                    errno = 0;
                }
            }
            
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
            raise(SIGSEGV);
            
            // if returns from old signal handler (which it shouldn't), register FTI handler again.
            // TODO since we are talking about seg faults, we might not attempt to register our handler again here.
            if( sigaction(SIGSEGV, &FTI_SigAction, &OLD_SigAction) == -1 ){
                FTI_Print( "FTI was unable to register the signal handler", FTI_EROR );
                errno = 0;
            }
        }
    }
}

int FTI_ExcludePage( FTI_ADDRVAL addr ) {
    bool found; 
    FTI_ADDRVAL page = addr & FTI_PageMask;
    int idx = -1;
    long pos;
    // binary search for page
    int i;
    for(i=0; i<FTI_DataDiffInfo.nbProtVar; ++i){
        bool found = false;
        long LOW = 0;
        long UP = FTI_DataDiffInfo.dataDiff[i].rangeCnt - 1;
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
            idx = i;
            pos = MID;
            break;
        }
    }
    if( idx == -1 ) {
        return FTI_NSCS;
    }
    // swap array elements i -> i+1 for i > pos and increase counter
    FTI_ShiftPageItems( idx, pos );
    FTI_ADDRVAL base = FTI_DataDiffInfo.dataDiff[idx].basePtr;
    FTI_ADDRVAL offset = FTI_DataDiffInfo.dataDiff[idx].ranges[pos].offset;
    FTI_ADDRVAL end = base + offset + FTI_DataDiffInfo.dataDiff[idx].ranges[pos].size;
    // update values
    FTI_DataDiffInfo.dataDiff[idx].ranges[pos].size = (page + FTI_PageSize) - (base + offset);
    FTI_DataDiffInfo.dataDiff[idx].ranges[pos+1].offset = page - base + FTI_PageSize;
    FTI_DataDiffInfo.dataDiff[idx].ranges[pos+1].size = end - (page + FTI_PageSize);
    // add dirty page to buffer
    FTI_DataDiffInfo.dataDiff[idx].dirtyPages = (FTI_ADDRVAL*) realloc( FTI_DataDiffInfo.dataDiff[idx].dirtyPages, (++FTI_DataDiffInfo.dataDiff[idx].dirtyPagesCnt)*sizeof(FTI_ADDRVAL));
    assert(FTI_DataDiffInfo.dataDiff[idx].dirtyPages != NULL);
    FTI_DataDiffInfo.dataDiff[idx].dirtyPages[FTI_DataDiffInfo.dataDiff[idx].dirtyPagesCnt-1] = page;
}

int FTI_RangeCmpPage(int idx, long idr, FTI_ADDRVAL page) {
    FTI_ADDRVAL base = FTI_DataDiffInfo.dataDiff[idx].basePtr;
    FTI_ADDRVAL size = FTI_DataDiffInfo.dataDiff[idx].ranges[idr].size;
    FTI_ADDRVAL first = FTI_DataDiffInfo.dataDiff[idx].ranges[idr].offset + base;
    FTI_ADDRVAL last = FTI_DataDiffInfo.dataDiff[idx].ranges[idr].offset + base + size - FTI_PageSize;
    if( page < first ) {
        return -1;
    } else if ( page > last ) {
        return 1;
    } else if ( (page >= first) && (page <= last) ) {
        return 0;
    }
}

int FTI_ShiftPageItems( int idx, long pos ) {
    // increase array size by 1 and increase counter
    FTI_DataDiffInfo.dataDiff[idx].ranges = (FTIT_DataRange*) realloc(FTI_DataDiffInfo.dataDiff[idx].ranges, (++FTI_DataDiffInfo.dataDiff[idx].rangeCnt) * sizeof(FTIT_DataRange));
    assert( FTI_DataDiffInfo.dataDiff[idx].ranges != NULL );
    
    long i;
    // shift elements of array starting at the end
    assert(FTI_DataDiffInfo.dataDiff[idx].rangeCnt > 0);
    for(i=FTI_DataDiffInfo.dataDiff[idx].rangeCnt-1; i>(pos+1); --i) {
        memcpy( &(FTI_DataDiffInfo.dataDiff[idx].ranges[i]), &(FTI_DataDiffInfo.dataDiff[idx].ranges[i-1]), sizeof(FTIT_DataRange));
    }
}

int FTI_ProtectPages(int idx, FTIT_dataset* FTI_Data, FTIT_execution* FTI_Exec) 
{
    char strdbg[FTI_BUFS];
    FTI_ADDRVAL first_page = FTI_GetFirstInclPage((FTI_ADDRVAL)FTI_Data[idx].ptr);
    FTI_ADDRVAL last_page = FTI_GetLastInclPage((FTI_ADDRVAL)FTI_Data[idx].ptr+FTI_Data[idx].size);
    FTI_ADDRVAL psize = 0;

    // check if dataset includes at least one full page.
    if (first_page < last_page) {
        psize = last_page - first_page + FTI_PageSize;
        if ( mprotect((FTI_ADDRPTR) first_page, psize, PROT_READ) == -1 ) {
            FTI_Print( "FTI was unable to register the pages", FTI_EROR );
            errno = 0;
            return FTI_NSCS;
        }
        // TODO no support for datasets that change size yet
        FTI_DataDiffInfo.dataDiff = (FTIT_DataDiff*) realloc( FTI_DataDiffInfo.dataDiff, (FTI_DataDiffInfo.nbProtVar+1) * sizeof(FTIT_DataDiff));
        assert( FTI_DataDiffInfo.dataDiff != NULL );
        FTI_DataDiffInfo.dataDiff[FTI_DataDiffInfo.nbProtVar].dirtyPages     = NULL;
        FTI_DataDiffInfo.dataDiff[FTI_DataDiffInfo.nbProtVar].dirtyPagesCnt  = 0;
        FTI_DataDiffInfo.dataDiff[FTI_DataDiffInfo.nbProtVar].ranges = (FTIT_DataRange*) malloc( sizeof(FTIT_DataRange) );
        assert( FTI_DataDiffInfo.dataDiff[FTI_DataDiffInfo.nbProtVar].ranges != NULL );
        FTI_DataDiffInfo.dataDiff[FTI_DataDiffInfo.nbProtVar].rangeCnt       = 1;
        FTI_DataDiffInfo.dataDiff[FTI_DataDiffInfo.nbProtVar].ranges->offset = (FTI_ADDRVAL) 0x0;
        FTI_DataDiffInfo.dataDiff[FTI_DataDiffInfo.nbProtVar].ranges->size   = psize;
        FTI_DataDiffInfo.dataDiff[FTI_DataDiffInfo.nbProtVar].basePtr        = first_page;
        FTI_DataDiffInfo.dataDiff[FTI_DataDiffInfo.nbProtVar].totalSize      = psize;
        FTI_DataDiffInfo.dataDiff[FTI_DataDiffInfo.nbProtVar].id             = FTI_Data[idx].id;
        FTI_DataDiffInfo.nbProtVar++;

    // if not don't protect anything. NULL just for debug output.
    } else {
        first_page = (FTI_ADDRVAL) NULL;
        last_page = (FTI_ADDRVAL) NULL;
    }

    // debug information
    snprintf(strdbg, FTI_BUFS, "FTI_DIFFCKPT: 'FTI_ProtectPages' - ID: %d, size: %lu, pages protect: %lu, addr: %p, first page: %p, last page: %p\n", 
            FTI_Data[idx].id, 
            FTI_Data[idx].size, 
            psize/FTI_PageSize,
            FTI_Data[idx].ptr,
            (FTI_ADDRPTR) first_page,
            (FTI_ADDRPTR) last_page);
    FTI_Print( strdbg, FTI_INFO );
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

    if ( FTI_isProtectedPage( page ) && FTI_ProtectedPageIsValid( page ) ) {
        return true;
    }
    printf("### LINE %d ### \n",__LINE__);

    return false;

}

//TODO implement insertion sort for dirty pages.
bool FTI_ProtectedPageIsValid( FTI_ADDRVAL page ) 
{
    // page is valid if not already found to be dirty.
    int i=0;
    // binary search for page
    for(i=0; i<FTI_DataDiffInfo.nbProtVar; ++i){
        if( FTI_DataDiffInfo.dataDiff[i].dirtyPages == NULL ) {
            break;
        }
        long LOW = 0;
        long UP = FTI_DataDiffInfo.dataDiff[i].dirtyPagesCnt - 1;
        long MID = (LOW+UP)/2;
        while( LOW < UP ) {
            FTI_ADDRVAL page_mid = FTI_DataDiffInfo.dataDiff[i].dirtyPages[MID];
            // page is in first half
            if( page < page_mid ) {
                assert(UP>0);
                UP = MID - 1;
            // page is in second half
            } else if ( page > page_mid ) {
                LOW = MID + 1;
            }
            MID = (LOW+UP)/2;
            if ( page == page_mid ) {
                return false;
            }
        }
    }
    return true;
}

bool FTI_isProtectedPage( FTI_ADDRVAL page ) 
{
    bool inRange = false;
    int i;
    for(i=0; i<FTI_DataDiffInfo.nbProtVar; ++i) {
        if( (page >= FTI_DataDiffInfo.dataDiff[i].basePtr) && (page <= FTI_DataDiffInfo.dataDiff[i].basePtr + FTI_DataDiffInfo.dataDiff[i].totalSize - FTI_PageSize) ) {
            inRange = true;
            break;
        }
    }
    return inRange;
}
