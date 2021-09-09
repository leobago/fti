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
 *  @file   dcp.c
 *  @date   October, 2017
 *  @brief  Checkpointing functions for the FTI library.
 */

#include "dcp.h"
#include <math.h>

static size_t _mb = 1024L*1024L;
static size_t _gb = 1024L*1024L*1024L;
static size_t _tb = 1024L*1024L*1024L*1024L;

/**
 *  Calculates the most adequate metric in bytes for N among TB, GB and MB
 *  @param n A number N
 *  @param s A string buffer to print the result, must not be null
 *  @remark Sets s to be either TB, GB or MB
 *  @return The adequate metric size in bytes
 **/
static inline size_t get_metric(size_t n, char* s) {
    s[1] = 'B';
    if (n > _tb) {
        s[0] = 'T';
        return _tb;
    }
    if (n > _gb) {
        s[0] = 'G';
        return _gb;
    }
    s[0] = 'M';
    return _mb;
}

void FTI_PrintDcpStats(FTIT_configuration FTI_Conf, FTIT_execution FTI_Exec,
 FTIT_topology FTI_Topo) {
    char str[FTI_BUFS];
    char data_metric[3] = "_B";
    char dcp_metric[3] = "_B";
    char cp_print_mode[6];

    size_t norder_data, norder_dcp, pureDataSize, dcpSize;

    if (FTI_Conf.dcpFtiff) {
        dcpSize = FTI_Exec.FTIFFMeta.dcpSize;
        pureDataSize = FTI_Exec.FTIFFMeta.pureDataSize;
    } else {
        dcpSize = FTI_Exec.dcpInfoPosix.dcpSize;
        pureDataSize = FTI_Exec.dcpInfoPosix.dataSize;
    }
  
    double relErrAvg = -1;
    if( FTI_Exec.isPbdcp == 4 ) {
      double sum_error=FTI_Exec.dcpInfoPosix.errorSum;
      int64_t nVals=FTI_Exec.dcpInfoPosix.nbValues;
      double sum_error_tot;
      int64_t nVals_tot;
      MPI_Reduce(&sum_error, &sum_error_tot, 1, MPI_DOUBLE, MPI_SUM, 0, FTI_COMM_WORLD);
      MPI_Reduce(&nVals, &nVals_tot, 1, MPI_INT64_T, MPI_SUM, 0, FTI_COMM_WORLD);
      if( nVals_tot > 0 ) relErrAvg=sqrt(sum_error_tot/nVals_tot);
    }

    int64_t *data_Size = (FTI_Conf.dcpFtiff)?
    (int64_t*)&FTI_Exec.FTIFFMeta.pureDataSize:
    &FTI_Exec.dcpInfoPosix.dataSize;
    int64_t *dcp_Size = (FTI_Conf.dcpFtiff)?
      (int64_t*)&FTI_Exec.FTIFFMeta.dcpSize:
      &FTI_Exec.dcpInfoPosix.dcpSize;
    int64_t dcpStats[2];  // 0:totalDcpSize, 1:totalDataSize
    int64_t sendBuf[] = { *dcp_Size, *data_Size };
    MPI_Reduce(sendBuf, dcpStats, 2, MPI_INT64_T, MPI_SUM, 0, FTI_COMM_WORLD);
    if (FTI_Topo.splitRank ==  0) {
        *dcp_Size = dcpStats[0];
        *data_Size = dcpStats[1];
    }
    
    norder_data = get_metric(pureDataSize, data_metric);
    norder_dcp = get_metric(dcpSize, dcp_metric);

    // If not head
    if (FTI_Topo.splitRank)
        snprintf(cp_print_mode, sizeof(cp_print_mode), "Local");
    else
        snprintf(cp_print_mode, sizeof(cp_print_mode), "Total");

    snprintf(str, FTI_BUFS, "%s CP data: %.2lf %s, %s dCP update:"
            " %.2lf %s, dCP share: %.2lf%%",
            cp_print_mode,
            (double)pureDataSize/norder_data,
            data_metric,
            cp_print_mode,
            (double)dcpSize/norder_dcp,
            dcp_metric,
            ((double)dcpSize/pureDataSize)*100);
    if ( (FTI_Exec.isPbdcp == 4) && FTI_Conf.pbdcpEnabled && (relErrAvg != -1) ){
        char rmseStr[FTI_BUFS];
        snprintf(rmseStr,FTI_BUFS," [PBDCP active -> average relative error: %.5lf%%]",relErrAvg*100);
        strcat(str,rmseStr);
    } else {
        char rmseStr[FTI_BUFS] = " [PBDCP inactive]";
        strcat(str,rmseStr);
    }
    // If not head
    if (FTI_Topo.splitRank)
        FTI_Print(str, FTI_DBUG);
    FTI_Print(str, FTI_IDCP);
}
