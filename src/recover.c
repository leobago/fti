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
    char str[FTI_BUFS];
    if (access(fn, F_OK) == 0) {
        struct stat fileStatus;
        if (stat(fn, &fileStatus) == 0) {
            if (fileStatus.st_size == fs) {
                if (strlen(checksum)) {
                    int res = FTI_VerifyChecksum(fn, checksum);
                    if (res != FTI_SCES) {
                        sprintf(str, "Missing file: \"%s\"", fn);
                        FTI_Print(str, FTI_WARN);

                        return 1;
                    }
                    return 0;
                }
                return 0;
            }
            else {
                sprintf(str, "Missing file: \"%s\"", fn);
                FTI_Print(str, FTI_WARN);

                return 1;
            }
        }
        else {
            sprintf(str, "Missing file: \"%s\"", fn);
            FTI_Print(str, FTI_WARN);
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
    int level = FTI_Exec->ckptMeta.level;
    long fs = FTI_Exec->ckptMeta.fs;
    long pfs = FTI_Exec->ckptMeta.pfs;
    long maxFs = FTI_Exec->ckptMeta.maxFs;
    char ckptFile[FTI_BUFS];
    strncpy(ckptFile, FTI_Exec->ckptMeta.ckptFile, FTI_BUFS);

    char checksum[MD5_DIGEST_STRING_LENGTH], ptnerChecksum[MD5_DIGEST_STRING_LENGTH], rsChecksum[MD5_DIGEST_STRING_LENGTH];
    FTI_GetChecksums(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt, checksum, ptnerChecksum, rsChecksum);
    char str[FTI_BUFS];
    snprintf(str, FTI_BUFS, "Checking file %s and its erasures %d.", ckptFile,level);
    FTI_Print(str, FTI_DBUG);
    char fn[FTI_BUFS]; //Path to the checkpoint/partner file name
    int buf;
    int ckptId, rank; //Variables for proper partner file name
    int (*consistency)(char *, long , char*);
#ifdef ENABLE_HDF5
    if (FTI_Conf->ioMode == FTI_IO_HDF5)
        consistency = &FTI_CheckHDF5File;
    else if( FTI_Ckpt[FTI_Exec->ckptMeta.level].recoIsDcp && FTI_Conf->dcpPosix ) {
        FTI_DcpPosixRecoverRuntimeInfo( DCP_POSIX_INIT_TAG, FTI_Exec, FTI_Conf );
        consistency = &FTI_CheckFileDcpPosix;
    }
    else
        consistency = &FTI_CheckFile;
#else
    if( FTI_Ckpt[FTI_Exec->ckptMeta.level].recoIsDcp && FTI_Conf->dcpPosix ) {
        FTI_DcpPosixRecoverRuntimeInfo( DCP_POSIX_INIT_TAG, FTI_Exec, FTI_Conf );
        consistency = &FTI_CheckFileDcpPosix;
    } else {
        consistency = &FTI_CheckFile;
    }
#endif

    switch (level) {
        case 1:
            snprintf(fn, FTI_BUFS, "%s/%s", FTI_Ckpt[1].dir, ckptFile);
            buf = consistency(fn, fs, checksum);
            MPI_Allgather(&buf, 1, MPI_INT, erased, 1, MPI_INT, FTI_Exec->groupComm);
            break;
        case 2:
            snprintf(fn, FTI_BUFS, "%s/%s", FTI_Ckpt[2].dir, ckptFile);
            buf = consistency(fn, fs, checksum);
            MPI_Allgather(&buf, 1, MPI_INT, erased, 1, MPI_INT, FTI_Exec->groupComm);

            sscanf(ckptFile, "Ckpt%d-Rank%d.fti", &ckptId, &rank);
            snprintf(fn, FTI_BUFS, "%s/Ckpt%d-Pcof%d.fti", FTI_Ckpt[2].dir, ckptId, rank);
            buf = consistency(fn, pfs, ptnerChecksum);
            MPI_Allgather(&buf, 1, MPI_INT, erased + FTI_Topo->groupSize, 1, MPI_INT, FTI_Exec->groupComm);
            break;
        case 3:
            snprintf(fn, FTI_BUFS, "%s/%s", FTI_Ckpt[3].dir, ckptFile);
            buf = consistency(fn, fs, checksum);
            MPI_Allgather(&buf, 1, MPI_INT, erased, 1, MPI_INT, FTI_Exec->groupComm);

            sscanf(ckptFile, "Ckpt%d-Rank%d.fti", &ckptId, &rank);
            snprintf(fn, FTI_BUFS, "%s/Ckpt%d-RSed%d.fti", FTI_Ckpt[3].dir, ckptId, rank);
            buf = FTI_CheckFile(fn, maxFs, rsChecksum);
            MPI_Allgather(&buf, 1, MPI_INT, erased + FTI_Topo->groupSize, 1, MPI_INT, FTI_Exec->groupComm);
            break;
        case 4:
            if( FTI_Ckpt[FTI_Exec->ckptMeta.level].recoIsDcp && FTI_Conf->dcpPosix ) {
                snprintf(fn, FTI_BUFS, "%s/%s", FTI_Ckpt[4].dcpDir, ckptFile); 
            } else {
                snprintf(fn, FTI_BUFS, "%s/%s", FTI_Ckpt[4].dir, ckptFile);
            }
            buf = consistency(fn, fs, checksum);
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

    int res = FTI_Try(FTI_LoadMetaRecovery(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt), "load checkpoint metadata");
    if (res == FTI_NSCS) {
        return FTI_NSCS;
    }

    if (!FTI_Topo->amIaHead) {
        if( FTI_Exec->reco == 3 ) {
            if( FTI_Conf->h5SingleFileEnable ) {
                int allRes = FTI_NSCS;
#ifdef ENABLE_HDF5
                int ckptId, res = FTI_SCES;
                if( FTI_Topo->splitRank == 0 ) {
                    res = FTI_H5CheckSingleFile( FTI_Conf, &ckptId );
                }
                MPI_Allreduce(&res, &allRes, 1, MPI_INT, MPI_SUM, FTI_Exec->globalComm);
                if( allRes == FTI_SCES ) {
                    char str[FTI_BUFS];
                    snprintf(str, FTI_BUFS, "VPR recovery successfull from file '%s/%s-ID%08d.h5'", 
                            FTI_Conf->h5SingleFileDir, FTI_Conf->h5SingleFilePrefix, ckptId );
                    FTI_Print(str, FTI_INFO);
                    FTI_Exec->h5SingleFile = true;
                    MPI_Bcast( &ckptId, 1, MPI_INT, 0, FTI_COMM_WORLD );
                    FTI_Exec->ckptId = ckptId;
                    snprintf(FTI_Exec->h5SingleFileReco, FTI_BUFS, "%s/%s-ID%08d.h5", FTI_Conf->h5SingleFileDir, FTI_Conf->h5SingleFilePrefix, ckptId);
                } else {
                    FTI_Print("VPR recovery failed!", FTI_WARN);
                    FTI_Exec->h5SingleFile = false;
                }
#else       
                FTI_Print("FTI is not compiled with HDF5 support!", FTI_EROR);
#endif
                return allRes;
            } else {
                FTI_Print("VPR is disabled. Please enable with 'h5_single_file_enable=1'!", FTI_EROR);
                return FTI_NSCS;
            }
        }

        while( !FTI_Exec->mqueue.empty() ) 
        {

            FTI_Exec->mqueue.pop( &FTI_Exec->ckptMeta );

            int level = FTI_Exec->ckptMeta.level;

            int ckptId;
            if ( FTI_Conf->ioMode != FTI_IO_FTIFF ) {
                sscanf(FTI_Exec->ckptMeta.ckptFile, "Ckpt%d", &ckptId);

                //Temporary for Recover functions
                FTI_Exec->ckptMeta.level = level;
                FTI_Exec->ckptId = ckptId;
            } else {
                ckptId = FTI_Exec->ckptId;
                FTI_Exec->ckptMeta.level = level;
            }

            char str[FTI_BUFS];
            snprintf(str, FTI_BUFS, "Trying recovery with Ckpt. %d at level %d.", ckptId, level);
            FTI_Print(str, FTI_DBUG);

            FTI_Try(FTI_LoadMetaDcp(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt), "load dcp metadata");

            int res;
            switch (level) {
                case 0:
                    if ( FTI_Ckpt[4].recoIsDcp ) {
                        res = FTI_RecoverL4(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt);
                        if( FTI_Conf->dcpFtiff ) FTI_Ckpt[4].recoIsDcp = false;
                        if (res == FTI_SCES ) {
                            level = 4;
                        } else {
                            snprintf(str, FTI_BUFS, "Recover failed from level %d_dCP with Ckpt. %d.", level, ckptId);
                            FTI_Print(str, FTI_INFO);
                        }
                    } else {
                        res = FTI_NSCS;
                    }
                    break;
                case 4:
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

                // FTI-FF: ckptId is already set properly
                if((FTI_Conf->ioMode == FTI_IO_FTIFF) || (FTI_Ckpt[4].recoIsDcp && FTI_Conf->dcpPosix) ) {
                    ckptId = FTI_Exec->ckptId;
                }

                if( level == 4 && !FTI_Ckpt[4].recoIsDcp ) {
                    FTI_Clean(FTI_Conf, FTI_Topo, FTI_Ckpt, 1);
                    MPI_Barrier(FTI_COMM_WORLD);
                    if( !(FTI_Topo->nodeRank - FTI_Topo->nbHeads) ) {
                        RENAME(FTI_Conf->lTmpDir, FTI_Ckpt[1].dir);
                    }
                    MPI_Barrier(FTI_COMM_WORLD);
                }

                snprintf(str, FTI_BUFS, "Recovering successfully from level %d with Ckpt. %d.", level, ckptId);
                FTI_Print(str, FTI_INFO);

                if ( FTI_Conf->keepL4Ckpt ) {
                    ckptId = FTI_LoadL4CkptMetaData( FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt );
                    int hasL4Ckpt = ( ckptId >= 0 ) ? 1 : 0;
                    if ( (FTI_Topo->nbHeads > 0 ) && (FTI_Topo->nodeRank == 1) ) {
                        // send level and ckpt ID to head process in node
                        int sendBuf[2] = { hasL4Ckpt, ckptId };
                        MPI_Send( sendBuf, 2, MPI_INT, FTI_Topo->headRank, FTI_Conf->generalTag, FTI_Exec->globalComm ); 
                    }
                    if( hasL4Ckpt ) {
                        FTI_Exec->ckptMeta.ckptIdL4 = ckptId; 
                        FTI_Ckpt[4].hasCkpt = true;
                    }
                }
                FTI_Exec->mqueue.clear();

                //Update ckptId and ckptLevel and lastCkptLvel
                FTI_Exec->ckptId = ckptId;
                FTI_Exec->ckptLvel = level;
                FTI_Exec->lastCkptLvel = level;

                return FTI_SCES; //Recovered successfully
            }
            else {
                snprintf(str, FTI_BUFS, "Recover failed from level %d with Ckpt. %d.", level, ckptId);
                FTI_Print(str, FTI_INFO);
            }
        }
        //Looped all levels with no success
        FTI_Print("Cannot recover from any checkpoint level.", FTI_INFO);

        //Inform heads that cannot recover
        int res = FTI_NSCS, allRes;
        MPI_Allreduce(&res, &allRes, 1, MPI_INT, MPI_SUM, FTI_Exec->globalComm);

        //Reset ckptId and ckptLevel
        FTI_Exec->ckptLvel = 0;
        FTI_Exec->ckptId = 0;
        return FTI_NSCS;
    }

    else { //Head processes
        int res = FTI_SCES, allRes;
        MPI_Allreduce(&res, &allRes, 1, MPI_INT, MPI_SUM, FTI_Exec->globalComm);
        if (allRes != FTI_SCES) {
            //Recover not successful
            return FTI_NSCS;
        }
        if ( FTI_Conf->keepL4Ckpt && !(FTI_Exec->reco == 3) ) {
            // receive level and ckpt ID from first application process in node
            int recvBuf[2];
            MPI_Recv( recvBuf, 2, MPI_INT, FTI_Topo->body[0], FTI_Conf->generalTag, FTI_Exec->globalComm, MPI_STATUS_IGNORE ); 
            if ( recvBuf[0] == 4 ) {
                FTI_Ckpt[4].hasCkpt = true;
                FTI_Exec->ckptLvel = recvBuf[0];
                FTI_Exec->ckptId = recvBuf[1];
                FTI_Exec->ckptMeta.ckptIdL4 = recvBuf[1];
            }
        }
        return FTI_SCES;
    }
}
