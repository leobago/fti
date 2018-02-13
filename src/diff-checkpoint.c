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
static FTI_dPageVector      FTI_dirtyPages;     /**< Container of dirty pages           */

/** File Local Variables                                                                */

struct sigaction            FTI_SigAction;       /**< sigaction meta data               */
struct sigaction            OLD_SigAction;       /**< previous sigaction meta data      */

/** Function Definitions                                                                */

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
        if( FTI_IsValidRequest( (FTI_ADDRVAL)info->si_addr ) ){
            // debug information
            snprintf( strdbg, FTI_BUFS, "SIGSEGV signal was raised at address %p\n", info->si_addr );
            FTI_Print( strdbg, FTI_DBUG );
            // remove protection from page
            mprotect( (FTI_ADDRPTR)((FTI_ADDRVAL)info->si_addr & FTI_PageMask), FTI_PageSize, PROT_READ|PROT_WRITE );
            // register page as dirty
            FTI_dirtyPages.addr_vec = (FTI_ADDRPTR*) realloc(FTI_dirtyPages.addr_vec, (++(FTI_dirtyPages.count))*sizeof(FTI_ADDRPTR));
            FTI_dirtyPages.addr_vec[FTI_dirtyPages.count - 1] = (FTI_ADDRPTR)((FTI_ADDRVAL)info->si_addr & FTI_PageMask);
        } else {
            // forward to default handler and raise signal again
            sigaction(SIGSEGV, &OLD_SigAction, NULL);
            raise(SIGSEGV);
            // if returns from old signal handler (which it shouldn't), register FTI handler again
            sigaction(SIGSEGV, &FTI_SigAction, &OLD_SigAction);
        }
    }
}

int FTI_RegisterPages(FTIT_dataset data) {

}

bool FTI_IsValidRequest( FTI_ADDRVAL addr_val ) {
// TODO add request for the case that the page was set dirty already!
    if( addr_val == NULL ) return false;
}