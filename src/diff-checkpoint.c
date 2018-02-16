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
static FTIT_PageInfo        FTI_PageInfo;       /**< Container of dirty pages           */

/** File Local Variables                                                                */

struct sigaction            FTI_SigAction;       /**< sigaction meta data               */
struct sigaction            OLD_SigAction;       /**< previous sigaction meta data      */

/** Function Definitions                                                                */

int FTI_InitDiffCkpt(){
    // get page mask
    FTI_PageSize = (FTI_ADDRVAL) sysconf(_SC_PAGESIZE);
    FTI_PageMask = ~((FTI_ADDRVAL)0x0);
    FTI_ADDRVAL tail = (FTI_ADDRVAL)0x1;
    for(; tail!=FTI_PageSize; FTI_PageMask<<=1, tail<<=1); 
    // init data structures
    memset(&FTI_PageInfo, 0x0, sizeof(FTIT_PageInfo));
    if ( FTI_RegisterSigHandler() == FTI_NSCS ) {
        return FTI_NSCS;
    } else {
        return FTI_SCES;
    }
}

void printReport() {
    
    char str[512];
    char gstr[8*512];

    size_t num_changed = FTI_PageInfo.dirtyPagesCount;
    size_t num_prot=0; 
    for(int i=0; i<FTI_PageInfo.protPagesCount; ++i) {
        num_prot += FTI_PageInfo.protPageRanges[i].size;
    }
    num_prot /= FTI_PageSize;

    snprintf(str, FTI_BUFS, 
            "Diff Ckpt Summary\n"
            "-------------------------------------\n"
            "number of pages protected: %lu\n"
            "number of pages changed:   %lu\n",
            num_prot, num_changed);
    MPI_Allgather(str, 512, MPI_CHAR, gstr, 512, MPI_CHAR, MPI_COMM_WORLD);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(rank==0)
        printf("%s", gstr);
    
    
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
    free(FTI_PageInfo.dirtyPages);
    free(FTI_PageInfo.protPageRanges);
    return FTI_SCES;
}

int FTI_RemoveProtections() {
    for(int i=0; i<FTI_PageInfo.protPagesCount; ++i){
        if ( mprotect( (FTI_ADDRPTR) FTI_PageInfo.protPageRanges[i].basePtr, FTI_PageInfo.protPageRanges[i].size, PROT_READ|PROT_WRITE ) == -1 ) {
            if ( errno != ENOMEM ) {
                FTI_Print( "FTI was unable to restore the data access", FTI_EROR );
                return FTI_NSCS;
            }
            errno = 0;
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
            FTI_PageInfo.dirtyPages = (FTI_ADDRPTR) realloc(FTI_PageInfo.dirtyPages, (++(FTI_PageInfo.dirtyPagesCount))*sizeof(FTI_ADDRVAL));
            FTI_PageInfo.dirtyPages[FTI_PageInfo.dirtyPagesCount - 1] = ((FTI_ADDRVAL)info->si_addr) & FTI_PageMask;
        
        } else {
            /*
                NOTICE: tested that this works also if the application that leverages FTI uses signal() and NOT sigaction(). 
                I.e. the handler registered by signal is called for the case that the SIGSEGV was raised from an address 
                outside of the FTI protected region.
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

int FTI_ProtectPages(int idx, FTIT_dataset* FTI_Data) 
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

        FTI_PageInfo.protPagesCount++;
        FTI_PageInfo.protPageRanges = (FTIT_PageRange*) realloc(FTI_PageInfo.protPageRanges, sizeof(FTIT_PageRange));
        FTI_PageInfo.protPageRanges[FTI_PageInfo.protPagesCount-1].basePtr = first_page;
        FTI_PageInfo.protPageRanges[FTI_PageInfo.protPagesCount-1].size = psize;

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

    return false;

}

bool FTI_ProtectedPageIsValid( FTI_ADDRVAL page ) 
{
    // page is valid if not already marked dirty.
    bool isValid = true;
    
    int i=0;
    for(; i<FTI_PageInfo.dirtyPagesCount; ++i) {
        if(page == FTI_PageInfo.dirtyPages[i]) {
            isValid = false;
            break;
        }
    }

    return isValid;
}

bool FTI_isProtectedPage( FTI_ADDRVAL page ) 
{
    bool inRange = false;
    for(int i=0; i<FTI_PageInfo.protPagesCount; ++i) {
        if( (page >= FTI_PageInfo.protPageRanges[i].basePtr) && (page <= (FTI_PageInfo.protPageRanges[i].basePtr+FTI_PageInfo.protPageRanges[i].size)) ) {
            inRange = true;
            break;
        }
    }
    return inRange;
}
