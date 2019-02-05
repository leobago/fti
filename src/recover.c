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
 *  @file   recover.c
 *  @date   October, 2017
 *  @brief  Recovery functions for the FTI library.
 */

#include "interface.h"

/*-------------------------------------------------------------------------*/
/**
  @brief      It checks if a file exist and that its size is 'correct'.
  @param      fn              The ckpt. file name to check.
  @param      fs              The ckpt. file size to check.
  @param      checksum        The file checksum to check.
  @return     integer         0 if file exists, 1 if not or wrong size.

  This function checks whether a file exist or not and if its size is
  the expected one.

 **/
/*-------------------------------------------------------------------------*/
int FTI_CheckFile(char* fn, long fs, char* checksum)
{
    if (access(fn, F_OK) == 0) {
        struct stat fileStatus;
        if (stat(fn, &fileStatus) == 0) {
            if (fileStatus.st_size == fs) {
                if (strlen(checksum)) {
                    int res = FTI_VerifyChecksum(fn, checksum);
                    if (res != FTI_SCES) {
                        return 1;
                    }
                    return 0;
                }
                return 0;
            }
            else {
                return 1;
            }
        }
        else {
            return 1;
        }
    }
    else {
        char str[FTI_BUFS];
        sprintf(str, "Missing file: \"%s\"", fn);
        FTI_Print(str, FTI_WARN);
        return 1;
    }
}

/*-------------------------------------------------------------------------*/
/**
  @brief      It detects all the erasures for a particular level.
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Topo        Topology metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @param      erased          The array of erasures to fill.
  @return     integer         FTI_SCES if successful.

  This function detects all the erasures for L1, L2 and L3. It return the
  results in the erased array. The search for erasures is done at the
  three levels independently on the current recovery level.

 **/
/*-------------------------------------------------------------------------*/
int FTI_CheckErasures(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt,
        int *erased)
{
    int level = FTI_Exec->ckptLvel;
    long fs = FTI_Exec->meta[level].fs[0];
    long pfs = FTI_Exec->meta[level].pfs[0];
    long maxFs = FTI_Exec->meta[level].maxFs[0];
    char ckptFile[FTI_BUFS];
    strncpy(ckptFile, FTI_Exec->meta[level].ckptFile, FTI_BUFS);

    char checksum[MD5_DIGEST_STRING_LENGTH], ptnerChecksum[MD5_DIGEST_STRING_LENGTH], rsChecksum[MD5_DIGEST_STRING_LENGTH];
    FTI_GetChecksums(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt, checksum, ptnerChecksum, rsChecksum);
    char str[FTI_BUFS];
    snprintf(str, FTI_BUFS, "Checking file %s and its erasures.", ckptFile);
    FTI_Print(str, FTI_DBUG);
    char fn[FTI_BUFS]; //Path to the checkpoint/partner file name
    int buf;
    int ckptID, rank; //Variables for proper partner file name
    switch (level) {
        case 1:
            snprintf(fn, FTI_BUFS, "%s/%s", FTI_Ckpt[1].dir, ckptFile);
            buf = FTI_CheckFile(fn, fs, checksum);
            MPI_Allgather(&buf, 1, MPI_INT, erased, 1, MPI_INT, FTI_Exec->groupComm);
            break;
        case 2:
            snprintf(fn, FTI_BUFS, "%s/%s", FTI_Ckpt[2].dir, ckptFile);
            buf = FTI_CheckFile(fn, fs, checksum);
            MPI_Allgather(&buf, 1, MPI_INT, erased, 1, MPI_INT, FTI_Exec->groupComm);

            sscanf(ckptFile, "Ckpt%d-Rank%d.fti", &ckptID, &rank);
            snprintf(fn, FTI_BUFS, "%s/Ckpt%d-Pcof%d.fti", FTI_Ckpt[2].dir, ckptID, rank);
            buf = FTI_CheckFile(fn, pfs, ptnerChecksum);
            MPI_Allgather(&buf, 1, MPI_INT, erased + FTI_Topo->groupSize, 1, MPI_INT, FTI_Exec->groupComm);
            break;
        case 3:
            snprintf(fn, FTI_BUFS, "%s/%s", FTI_Ckpt[3].dir, ckptFile);
            buf = FTI_CheckFile(fn, fs, checksum);
            MPI_Allgather(&buf, 1, MPI_INT, erased, 1, MPI_INT, FTI_Exec->groupComm);

            sscanf(ckptFile, "Ckpt%d-Rank%d.fti", &ckptID, &rank);
            snprintf(fn, FTI_BUFS, "%s/Ckpt%d-RSed%d.fti", FTI_Ckpt[3].dir, ckptID, rank);
            buf = FTI_CheckFile(fn, maxFs, rsChecksum);
            MPI_Allgather(&buf, 1, MPI_INT, erased + FTI_Topo->groupSize, 1, MPI_INT, FTI_Exec->groupComm);
            break;
        case 4:
            snprintf(fn, FTI_BUFS, "%s/%s", FTI_Ckpt[4].dir, ckptFile);
            buf = FTI_CheckFile(fn, fs, checksum);
            MPI_Allgather(&buf, 1, MPI_INT, erased, 1, MPI_INT, FTI_Exec->groupComm);
            break;
    }
    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      It decides wich action take depending on the restart level.
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Topo        Topology metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @return     integer         FTI_SCES if successful.

  This function launches the required action depending on the recovery
  level. The recovery level is detected from the checkpoint ID of the
  last checkpoint taken.

 **/
/*-------------------------------------------------------------------------*/
int FTI_RecoverFiles(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt)
{

    if( FTI_Conf->dcpEnabled ) {
        FTI_LoadCkptMetaData( FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt );
    }

    if (!FTI_Topo->amIaHead) {
        if( FTI_Exec->reco == 3 ) {
            int res = FTI_SCES, allRes;
            if( FTI_Topo->splitRank == 0 ) {
#ifdef ENABLE_HDF5
                res = FTI_H5CheckSingleFile( FTI_Conf );
#else       
                res = FTI_NSCS;
#endif
            }
            MPI_Allreduce(&res, &allRes, 1, MPI_INT, MPI_SUM, FTI_Exec->globalComm);
            if( allRes == FTI_SCES ) {
                char str[FTI_BUFS];
                snprintf(str, FTI_BUFS, "VPR recovery successfull from file '%s'", FTI_Conf->h5SingleFilePath );
                FTI_Print(str, FTI_INFO);
                FTI_Exec->h5SingleFile = true;
            } else {
                char str[FTI_BUFS];
                snprintf(str, FTI_BUFS, "VPR recovery from file '%s' failed!", FTI_Conf->h5SingleFilePath );
                FTI_Print(str, FTI_WARN);
                FTI_Exec->h5SingleFile = false;
            }
            return allRes;
        }
        //FTI_LoadMeta(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt);
        int level;
        for (level = 1; level < 5; level++) { //For every level (from 1 to 4, because of reliability)
            if (FTI_Exec->meta[level].exists[0] || FTI_Conf->ioMode == FTI_IO_FTIFF) {
                //Get ckptID from checkpoint file name

                int ckptID;
                if ( FTI_Conf->ioMode != FTI_IO_FTIFF ) {
                    sscanf(FTI_Exec->meta[level].ckptFile, "Ckpt%d", &ckptID);

                    //Temporary for Recover functions
                    FTI_Exec->ckptLvel = level;
                    FTI_Exec->ckptID = ckptID;
                } else {
                    ckptID = FTI_Exec->ckptID;
                    FTI_Exec->ckptLvel = level;
                }

                char str[FTI_BUFS];
                snprintf(str, FTI_BUFS, "Trying recovery with Ckpt. %d at level %d.", ckptID, level);
                FTI_Print(str, FTI_DBUG);
                

                int res;
                switch (level) {
                    case 4:
                        FTI_Clean(FTI_Conf, FTI_Topo, FTI_Ckpt, 1);
                        MPI_Barrier(FTI_COMM_WORLD);
                        if ( FTI_Ckpt[4].isDcp ) {
                            res = FTI_RecoverL4(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt);
                            FTI_Ckpt[4].isDcp = false;
                            
                            if (res == FTI_SCES ) {
                                break;
                            }
                            snprintf(str, FTI_BUFS, "Recover failed from level %d_dCP with Ckpt. %d.", level, ckptID);
                            FTI_Print(str, FTI_INFO);
                        }
                        res = FTI_RecoverL4(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt);
                        break;
                    case 3:
                        res = FTI_RecoverL3(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt);
                        break;
                    case 2:
                        res = FTI_RecoverL2(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt);
                        break;
                    case 1:
                        res = FTI_RecoverL1(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt);
                        break;
                }
                int allRes;

                MPI_Allreduce(&res, &allRes, 1, MPI_INT, MPI_SUM, FTI_COMM_WORLD);
                if (allRes == FTI_SCES) {
                    //Inform heads that recovered successfully
                    MPI_Allreduce(&res, &allRes, 1, MPI_INT, MPI_SUM, FTI_Exec->globalComm);
                     
                    // FTI-FF: ckptID is already set properly
                    if(FTI_Conf->ioMode == FTI_IO_FTIFF) {
                        ckptID = FTI_Exec->ckptID;
                    }

                    snprintf(str, FTI_BUFS, "Recovering successfully from level %d with Ckpt. %d.", level, ckptID);
                    FTI_Print(str, FTI_INFO);

                    //Update ckptID and ckptLevel and lastCkptLvel
                    FTI_Exec->ckptID = ckptID;
                    FTI_Exec->ckptLvel = level;
                    FTI_Exec->lastCkptLvel = level;
                    if ( FTI_Conf->keepL4Ckpt ) {
                        ckptID = FTI_LoadL4CkptMetaData( FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt );
                        int hasL4Ckpt = ( ckptID >= 0 ) ? 1 : 0;
                        if ( (FTI_Topo->nbHeads > 0 ) && (FTI_Topo->nodeRank == 1) ) {
                            // send level and ckpt ID to head process in node
                            int sendBuf[2] = { hasL4Ckpt, ckptID };
                            MPI_Send( sendBuf, 2, MPI_INT, FTI_Topo->headRank, FTI_Conf->generalTag, FTI_Exec->globalComm ); 
                        }
                        if( hasL4Ckpt ) {
                            snprintf(FTI_Exec->meta[0].currentL4CkptFile, 
                                FTI_BUFS, "Ckpt%d-Rank%d.fti", ckptID, FTI_Topo->myRank );
                            FTI_Ckpt[4].hasCkpt = true;
                        }
                    }
                    return FTI_SCES; //Recovered successfully
                }
                else {
                    snprintf(str, FTI_BUFS, "Recover failed from level %d with Ckpt. %d.", level, ckptID);
                    FTI_Print(str, FTI_INFO);
                }
            }
        }
        //Looped all levels with no success
        FTI_Print("Cannot recover from any checkpoint level.", FTI_INFO);

        //Inform heads that cannot recover
        int res = FTI_NSCS, allRes;
        MPI_Allreduce(&res, &allRes, 1, MPI_INT, MPI_SUM, FTI_Exec->globalComm);

        //Reset ckptID and ckptLevel
        FTI_Exec->ckptLvel = 0;
        FTI_Exec->ckptID = 0;
        return FTI_NSCS;
    }
    else { //Head processes
        int res = FTI_SCES, allRes;
        MPI_Allreduce(&res, &allRes, 1, MPI_INT, MPI_SUM, FTI_Exec->globalComm);
        if (allRes != FTI_SCES) {
            //Recover not successful
            return FTI_NSCS;
        }
        if ( FTI_Conf->keepL4Ckpt ) {
            // receive level and ckpt ID from first application process in node
            int recvBuf[2];
            MPI_Recv( recvBuf, 2, MPI_INT, FTI_Topo->body[0], FTI_Conf->generalTag, FTI_Exec->globalComm, MPI_STATUS_IGNORE ); 
            if ( recvBuf[0] == 4 ) {
                FTI_Ckpt[4].hasCkpt = true;
                FTI_Exec->ckptLvel = recvBuf[0];
                FTI_Exec->ckptID = recvBuf[1];
                int i; 
                for ( i=1; i<FTI_Topo->nodeSize; ++i ) {
                    snprintf(&FTI_Exec->meta[0].currentL4CkptFile[i * FTI_BUFS], 
                            FTI_BUFS, "Ckpt%d-Rank%d.fti", FTI_Exec->ckptID, FTI_Topo->body[i-1] ); 
                }
            }
        }
        return FTI_SCES;
    }
}
