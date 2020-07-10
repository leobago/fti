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

#include "./interface.h"

void FTI_PrintDcpStats(FTIT_configuration FTI_Conf, FTIT_execution FTI_Exec,
 FTIT_topology FTI_Topo) {
    char str[FTI_BUFS];
    uint32_t pureDataSize = (FTI_Conf.dcpFtiff) ?
     FTI_Exec.FTIFFMeta.pureDataSize : FTI_Exec.dcpInfoPosix.dataSize;
    uint32_t dcpSize = (FTI_Conf.dcpFtiff) ?
     FTI_Exec.FTIFFMeta.dcpSize : FTI_Exec.dcpInfoPosix.dcpSize;
    int32_t norder_data, norder_dcp;
    char corder_data[3], corder_dcp[3];
    int32_t DCP_TB = (1024L*1024L*1024L*1024L);
    int32_t DCP_GB = (1024L*1024L*1024L);
    int32_t DCP_MB = (1024L*1024L);
    if (pureDataSize > DCP_TB) {
        norder_data = DCP_TB;
        snprintf(corder_data, sizeof(corder_data), "TB");
    } else if (pureDataSize > DCP_GB) {
        norder_data = DCP_GB;
        snprintf(corder_data, sizeof(corder_data), "TB");
    } else {
        norder_data = DCP_MB;
        snprintf(corder_data, sizeof(corder_data), "TB");
    }
    if (dcpSize > DCP_TB) {
        norder_dcp = DCP_TB;
        snprintf(corder_dcp, sizeof(corder_dcp), "TB");
    } else if (dcpSize > DCP_GB) {
        norder_dcp = DCP_GB;
        snprintf(corder_dcp, sizeof(corder_dcp), "TB");
    } else {
        norder_dcp = DCP_MB;
        snprintf(corder_dcp, sizeof(corder_dcp), "TB");
    }

    if (FTI_Topo.splitRank != 0) {
        snprintf(str, FTI_BUFS, "Local CP data: %.2lf %s, Local dCP update:"
            " %.2lf %s, dCP share: %.2lf%%",
                (double)pureDataSize/norder_data, corder_data,
                (double)dcpSize/norder_dcp, corder_dcp,
                ((double)dcpSize/pureDataSize)*100);
        FTI_Print(str, FTI_DBUG);
    } else {
        snprintf(str, FTI_BUFS, "Total CP data: %.2lf %s, Total dCP update:"
            " %.2lf %s, dCP share: %.2lf%%",
                (double)pureDataSize/norder_data, corder_data,
                (double)dcpSize/norder_dcp, corder_dcp,
                ((double)dcpSize/pureDataSize)*100);
    }

    FTI_Print(str, FTI_IDCP);
}
