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
 *  @file   checkpoint.c
 *  @date   October, 2017
 *  @brief  Checkpointing functions for the FTI library.
 */

#ifndef LUSTRE
#    define _POSIX_C_SOURCE 200809L
#endif

#include <string.h>

#include "interface.h"
#include "ftiff.h"
#include "api_cuda.h"
#include "utility.h"

/*-------------------------------------------------------------------------*/
/**
  @brief      It updates the local and global mean iteration time.
  @param      FTI_Exec        Execution metadata.
  @return     integer         FTI_SCES if successful.

  This function updates the local and global mean iteration time. It also
  recomputes the checkpoint interval in iterations and corrects the next
  checkpointing iteration based on the observed mean iteration duration.

 **/
/*-------------------------------------------------------------------------*/
int FTI_UpdateIterTime(FTIT_execution* FTI_Exec)
{
    int nbProcs, res;
    char str[FTI_BUFS];
    double last = FTI_Exec->iterTime;
    FTI_Exec->iterTime = MPI_Wtime();
    if (FTI_Exec->ckptIcnt > 0) {
        FTI_Exec->lastIterTime = FTI_Exec->iterTime - last;
        FTI_Exec->totalIterTime = FTI_Exec->totalIterTime + FTI_Exec->lastIterTime;
        if (FTI_Exec->ckptIcnt % FTI_Exec->syncIter == 0) {
            FTI_Exec->meanIterTime = FTI_Exec->totalIterTime / FTI_Exec->ckptIcnt;
            MPI_Allreduce(&FTI_Exec->meanIterTime, &FTI_Exec->globMeanIter, 1, MPI_DOUBLE, MPI_SUM, FTI_COMM_WORLD);
            MPI_Comm_size(FTI_COMM_WORLD, &nbProcs);
            FTI_Exec->globMeanIter = FTI_Exec->globMeanIter / nbProcs;
            if (FTI_Exec->globMeanIter > 60) {
                FTI_Exec->ckptIntv = 1;
            }
            else {
                FTI_Exec->ckptIntv = rint(60.0 / FTI_Exec->globMeanIter);
            }
            res = FTI_Exec->ckptLast + FTI_Exec->ckptIntv;
            if (FTI_Exec->ckptLast == 0) {
                res = res + 1;
            }
            if (res >= FTI_Exec->ckptIcnt) {
                FTI_Exec->ckptNext = res;
            }
            snprintf(str, FTI_BUFS, "Current iter : %d ckpt intv. : %d . Next ckpt. at iter. %d . Sync. intv. : %d",
                    FTI_Exec->ckptIcnt, FTI_Exec->ckptIntv, FTI_Exec->ckptNext, FTI_Exec->syncIter);
            FTI_Print(str, FTI_DBUG);
            if ((FTI_Exec->syncIter < (FTI_Exec->ckptIntv / 2)) && (FTI_Exec->syncIter < FTI_Exec->syncIterMax)) {
                FTI_Exec->syncIter = FTI_Exec->syncIter * 2;
                snprintf(str, FTI_BUFS, "Iteration frequency : %.2f sec/iter => %d iter/min. Resync every %d iter.",
                        FTI_Exec->globMeanIter, FTI_Exec->ckptIntv, FTI_Exec->syncIter);
                FTI_Print(str, FTI_DBUG);
                if (FTI_Exec->syncIter == FTI_Exec->syncIterMax) {
                    snprintf(str, FTI_BUFS, "Sync. intv. has reached max value => %i iterations", FTI_Exec->syncIterMax);
                    FTI_Print(str, FTI_DBUG);
                }

            }
        }
    }
    FTI_Exec->ckptIcnt++; // Increment checkpoint loop counter
    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      It writes the checkpoint data in the target file.
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Topo        Topology metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @param      FTI_Data        Dataset metadata.
  @return     integer         FTI_SCES if successful.

  This function checks whether the checkpoint needs to be local or remote,
  opens the target file and writes dataset per dataset, the checkpoint data,
  it finally flushes and closes the checkpoint file.

 **/
/*-------------------------------------------------------------------------*/
int FTI_WriteCkpt(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt,
        FTIT_dataset* FTI_Data)
{
    char str[FTI_BUFS]; //For console output
    snprintf(str, FTI_BUFS, "Starting writing checkpoint (ID: %d, Lvl: %d)",
            FTI_Exec->ckptID, FTI_Exec->ckptLvel);
    FTI_Print(str, FTI_DBUG);

    //update ckpt file name
    snprintf(FTI_Exec->meta[0].ckptFile, FTI_BUFS,
            "Ckpt%d-Rank%d.fti", FTI_Exec->ckptID, FTI_Topo->myRank);

#ifdef ENABLE_HDF5 //If HDF5 is installed overwrite the name
    if (FTI_Conf->ioMode == FTI_IO_HDF5) {
        snprintf(FTI_Exec->meta[0].ckptFile, FTI_BUFS,
                "Ckpt%d-Rank%d.h5", FTI_Exec->ckptID, FTI_Topo->myRank);
    }
#endif
    
    //If checkpoint is inlin and level 4 save directly to PFS
    int res; //response from writing funcitons
    if (FTI_Ckpt[4].isInline && FTI_Exec->ckptLvel == 4) {
        
        if ( !(FTI_Conf->dcpEnabled && FTI_Ckpt[4].isDcp) ) {
            FTI_Print("Saving to temporary global directory", FTI_DBUG);

            //Create global temp directory
            if (mkdir(FTI_Conf->gTmpDir, 0777) == -1) {
                if (errno != EEXIST) {
                    FTI_Print("Cannot create global directory", FTI_EROR);
                    return FTI_NSCS;
                }
            }
        } else {
            if ( !FTI_Ckpt[4].hasDcp ) {
                if (mkdir(FTI_Ckpt[4].dcpDir, 0777) == -1) {
                    if (errno != EEXIST) {
                        FTI_Print("Cannot create global dCP directory", FTI_EROR);
                        return FTI_NSCS;
                    }
                }
            }
        }

        switch (FTI_Conf->ioMode) {
            case FTI_IO_POSIX:
                res = FTI_Try(FTI_WritePosix(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt, FTI_Data), "write checkpoint to PFS (POSIX I/O).");
                break;
            case FTI_IO_MPI:
                res = FTI_Try(FTI_WriteMPI(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Data), "write checkpoint to PFS (MPI-IO).");
                break;
#ifdef ENABLE_SIONLIB //If SIONlib is installed
            case FTI_IO_SIONLIB:
                res = FTI_Try(FTI_WriteSionlib(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Data), "write checkpoint to PFS (Sionlib).");
                break;
#endif
            case FTI_IO_FTIFF:
                res = FTI_Try(FTIFF_WriteFTIFF(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt, FTI_Data), "write checkpoint to PFS (FTI-FF).");
                break;
#ifdef ENABLE_HDF5 //If HDF5 is installed
            case FTI_IO_HDF5:
                res = FTI_Try(FTI_WriteHDF5(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt, FTI_Data), "write checkpoint to PFS (HDF5).");
                break;
#endif
        }
    }
    else {
        if ( !(FTI_Conf->dcpEnabled && FTI_Ckpt[4].isDcp) ) {
            FTI_Print("Saving to temporary local directory", FTI_DBUG);
            //Create local temp directory
            if (mkdir(FTI_Conf->lTmpDir, 0777) == -1) {
                if (errno != EEXIST) {
                    FTI_Print("Cannot create local directory", FTI_EROR);
                }
            }
        } else {
            if ( !FTI_Ckpt[4].hasDcp ) {
                if (mkdir(FTI_Ckpt[1].dcpDir, 0777) == -1) {
                    if (errno != EEXIST) {
                        FTI_Print("Cannot create global dCP directory", FTI_EROR);
                        return FTI_NSCS;
                    }
                }
            }
        }
        switch (FTI_Conf->ioMode) {
            case FTI_IO_FTIFF:
        
                res = FTI_Try(FTIFF_WriteFTIFF(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt, FTI_Data), "write checkpoint using FTI-FF.");
                break;
#ifdef ENABLE_HDF5 //If HDF5 is installed
            case FTI_IO_HDF5:
                res = FTI_Try(FTI_WriteHDF5(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt, FTI_Data), "write checkpoint (HDF5).");
                break;
#endif
            default:
                res = FTI_Try(FTI_WritePosix(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt, FTI_Data),"write checkpoint.");
                break;
        }

    }
        

    //Check if all processes have written correctly (every process must succeed)
    int allRes;
    MPI_Allreduce(&res, &allRes, 1, MPI_INT, MPI_SUM, FTI_COMM_WORLD);
    if (allRes != FTI_SCES) {
        return FTI_NSCS;
    }
    if ( FTI_Conf->dcpEnabled && FTI_Ckpt[4].isDcp ) {
        // After dCP update store total data and dCP sizes in application rank 0
        long dcpStats[2]; // 0:totalDcpSize, 1:totalDataSize
        long sendBuf[] = { FTI_Exec->FTIFFMeta.dcpSize, FTI_Exec->FTIFFMeta.pureDataSize };
        MPI_Reduce( sendBuf, dcpStats, 2, MPI_LONG, MPI_SUM, 0, FTI_COMM_WORLD );
        if ( FTI_Topo->splitRank ==  0 ) {
            FTI_Exec->FTIFFMeta.dcpSize = dcpStats[0]; 
            FTI_Exec->FTIFFMeta.pureDataSize = dcpStats[1];
        }
    }

    res = FTI_Try(FTI_CreateMetadata(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt, FTI_Data), "create metadata.");
    
    if ( (FTI_Conf->dcpEnabled || FTI_Conf->keepL4Ckpt) && (FTI_Topo->splitRank == 0) ) {
        FTI_WriteCkptMetaData( FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt );
    }

    return res;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Decides wich action start depending on the ckpt. level.
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Topo        Topology metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @return     integer         FTI_SCES if successful.

  This function launches the required action dependeing on the ckpt. level.
  It does that for each group (application process in the node) if executed
  by the head, or only locally if executed by an application process. The
  parameter pr determines if the for loops have 1 or number of App. procs.
  iterations. The group parameter helps determine the groupID in both cases.

 **/
/*-------------------------------------------------------------------------*/
int FTI_PostCkpt(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt)
{
    char str[FTI_BUFS]; //For console output

    double t1 = MPI_Wtime(); //Start time

    int res; //Response from post-processing functions
    switch (FTI_Exec->ckptLvel) {
        case 4:
            res = FTI_Flush(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt, 0);
            break;
        case 3:
            res = FTI_RSenc(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt);
            break;
        case 2:
            res = FTI_Ptner(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt);
            break;
        case 1:
            res = FTI_Local(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt);
            break;
    }

    //Check if all processes done post-processing correctly
    int allRes;
    MPI_Allreduce(&res, &allRes, 1, MPI_INT, MPI_SUM, FTI_COMM_WORLD);
    if (allRes != FTI_SCES) {
        FTI_Print("Error postprocessing checkpoint. Discarding current checkpoint...", FTI_WARN);
        FTI_Clean(FTI_Conf, FTI_Topo, FTI_Ckpt, 0); //Remove temporary files
        return FTI_NSCS;
    }

    double t2 = MPI_Wtime(); //Post-processing time

    // rename l4 checkpoint file before deleting l4 folder if keepL4Ckpt enabled
    if ( FTI_Conf->keepL4Ckpt && FTI_Exec->ckptLvel == 4 ) {
        if ( FTI_Ckpt[4].hasCkpt ) {
            FTI_ArchiveL4Ckpt( FTI_Conf, FTI_Exec, FTI_Ckpt, FTI_Topo );
        }
        // store current ckpt file name in meta data.
        if ( !FTI_Topo->amIaHead ) {
            strncpy(FTI_Exec->meta[0].currentL4CkptFile, FTI_Exec->meta[0].ckptFile, FTI_BUFS);
        } else {
            int i;
            for( i=1; i<FTI_Topo->nodeSize; ++i ) {
                strncpy(&FTI_Exec->meta[0].currentL4CkptFile[i * FTI_BUFS], &FTI_Exec->meta[0].ckptFile[i * FTI_BUFS], FTI_BUFS);
            }
        }
    }

    FTI_Clean(FTI_Conf, FTI_Topo, FTI_Ckpt, FTI_Exec->ckptLvel); //delete previous files on this checkpoint level
    int nodeFlag = (((!FTI_Topo->amIaHead) && ((FTI_Topo->nodeRank - FTI_Topo->nbHeads) == 0)) || (FTI_Topo->amIaHead)) ? 1 : 0;
    nodeFlag = (!FTI_Ckpt[4].isDcp && (nodeFlag != 0));
    if (nodeFlag) { //True only for one process in the node.
        //Debug message needed to test nodeFlag (./tests/nodeFlag/nodeFlag.c)
        snprintf(str, FTI_BUFS, "Has nodeFlag = 1 and nodeID = %d. CkptLvel = %d.", FTI_Topo->nodeID, FTI_Exec->ckptLvel);
        FTI_Print(str, FTI_DBUG);
        if (!(FTI_Ckpt[4].isInline && FTI_Exec->ckptLvel == 4)) {
            //checkpoint was not saved in global temporary directory
            int level = (FTI_Exec->ckptLvel != 4) ? FTI_Exec->ckptLvel : 1; //if level 4: head moves local ckpt files to PFS
            if (rename(FTI_Conf->lTmpDir, FTI_Ckpt[level].dir) == -1) {
                char dbg_str[FTI_BUFS];
                snprintf(dbg_str, FTI_BUFS, "Cannot rename local directory (%s)", FTI_Conf->lTmpDir);
                FTI_Print(dbg_str, FTI_EROR);
            }
            else {
                FTI_Print("Local directory renamed", FTI_DBUG);
            }
        }
    }
    int globalFlag = !FTI_Topo->splitRank;
    globalFlag = (!FTI_Ckpt[4].isDcp && (globalFlag != 0));
    if (globalFlag) { //True only for one process in the FTI_COMM_WORLD.
        if (FTI_Exec->ckptLvel == 4) {
            if (rename(FTI_Conf->gTmpDir, FTI_Ckpt[4].dir) == -1) {
                FTI_Print("Cannot rename global directory", FTI_EROR);
            }
        }
        // there is no temp meta data folder for FTI-FF
        if ( FTI_Conf->ioMode != FTI_IO_FTIFF ) {
            if (rename(FTI_Conf->mTmpDir, FTI_Ckpt[FTI_Exec->ckptLvel].metaDir) == -1) {
                FTI_Print("Cannot rename meta directory", FTI_EROR);
            }
        }
    }
    MPI_Barrier(FTI_COMM_WORLD); //barrier needed to wait for process to rename directories (new temporary could be needed in next checkpoint)

    double t3 = MPI_Wtime(); //Renaming directories time

    snprintf(str, FTI_BUFS, "Post-checkpoint took %.2f sec. (Pt:%.2fs, Cl:%.2fs)",
            t3 - t1, t2 - t1, t3 - t2);
    FTI_Print(str, FTI_INFO);

    // expose to FTI that a checkpoint exists for level
    FTI_Ckpt[FTI_Exec->ckptLvel].hasCkpt = true;

    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      It listens for checkpoint notifications.
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Topo        Topology metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @return     integer         FTI_SCES if successful.
  This function listens for notifications from the application processes
  and takes the required actions after notification. This function is only
  executed by the head of the nodes and its complementary with the
  FTI_Checkpoint function in terms of communications.
 **/
/*-------------------------------------------------------------------------*/
int FTI_Listen(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt)
{

    MPI_Status ckpt_status;         int ckpt_flag = 0;
    MPI_Status stage_status;        int stage_flag = 0;
    MPI_Status finalize_status;     int finalize_flag = 0;

    FTI_Print("Head starts listening...", FTI_DBUG);
    while (1) { //heads can stop only by receiving FTI_ENDW

        FTI_Print("Head waits for message...", FTI_DBUG);

        MPI_Iprobe( MPI_ANY_SOURCE, FTI_Conf->finalTag, FTI_Exec->globalComm, &finalize_flag, &finalize_status );
        if ( FTI_Conf->stagingEnabled ) {
            MPI_Iprobe( MPI_ANY_SOURCE, FTI_Conf->stageTag, FTI_Exec->nodeComm, &stage_flag, &stage_status );
        }
        MPI_Iprobe( MPI_ANY_SOURCE, FTI_Conf->ckptTag, FTI_Exec->globalComm, &ckpt_flag, &ckpt_status );
        if( ckpt_flag ) {

            // head will process the whole checkpoint
            // (treated second due to priority)
            FTI_HandleCkptRequest( FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt ); 
            ckpt_flag = 0;
            continue;

        } 

        if ( stage_flag ) {
            
            // head will process each unstage request on its own
            // [A MAYBE: we could interrupt the unstageing process if 
            // we receive a checkpoint request.]
            FTI_HandleStageRequest( FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt, stage_status.MPI_SOURCE );
            stage_flag = 0;
            continue;

        } 

        // the 'continue' statement ensures that we first process all
        // checkpoint and staging request before we call finalize.
        if ( finalize_flag ) {
            
            char str[FTI_BUFS];
            FTI_Print("Head waits for message...", FTI_DBUG);
            
            int val = 0, i;
            for (i = 0; i < FTI_Topo->nbApprocs; i++) { // Iterate on the application processes in the node
                int buf;
                MPI_Recv(&buf, 1, MPI_INT, FTI_Topo->body[i], FTI_Conf->finalTag, FTI_Exec->globalComm, MPI_STATUS_IGNORE);
                snprintf(str, FTI_BUFS, "The head received a %d message", buf);
                FTI_Print(str, FTI_DBUG);
                val += buf;
            }

            val /= FTI_Topo->nbApprocs;

            if ( val != FTI_ENDW) { // If we were asked to finalize
                FTI_Print( "Inconsistency in Finalize request.", FTI_WARN );
            }
            
            FTI_Print("Head stopped listening.", FTI_DBUG);
            FTI_Finalize();

            if ( FTI_Conf->keepHeadsAlive ) {
                break;
            }

        }

    }

    // will be reached only if keepHeadsAlive is TRUE
    return FTI_SCES;

}

/*-------------------------------------------------------------------------*/
/**
  @brief      handles checkpoint requests from application ranks (if head).
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Topo        Topology metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @return     integer         FTI_SCES if successful.
 **/
/*-------------------------------------------------------------------------*/
int FTI_HandleCkptRequest(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt)
{   
    char str[FTI_BUFS]; //For console output
    int flags[7]; //Increment index if get corresponding value from application process
    //(index (1 - 4): checkpoint level; index 5: stops head; index 6: reject checkpoint)
    int i;
    for (i = 0; i < 7; i++) { // Initialize flags
        flags[i] = 0;
    }
    FTI_Print("Head waits for message...", FTI_DBUG);
    for (i = 0; i < FTI_Topo->nbApprocs; i++) { // Iterate on the application processes in the node
        int buf;
        MPI_Recv(&buf, 1, MPI_INT, FTI_Topo->body[i], FTI_Conf->ckptTag, FTI_Exec->globalComm, MPI_STATUS_IGNORE);
        snprintf(str, FTI_BUFS, "The head received a %d message", buf);
        FTI_Print(str, FTI_DBUG);
        flags[buf - FTI_BASE] = flags[buf - FTI_BASE] + 1;
    }
    for (i = 1; i < 7; i++) {
        if (flags[i] == FTI_Topo->nbApprocs) { // Determining checkpoint level
            FTI_Exec->ckptLvel = i;
        }
    }
    if (flags[6] > 0) {
        FTI_Exec->ckptLvel = 6;
    }

    int isDcpCnt = 0;
    // FTI-FF: receive meta data information from the application ranks.
    if ( FTI_Conf->ioMode == FTI_IO_FTIFF &&  FTI_Exec->ckptLvel != 6 &&  FTI_Exec->ckptLvel != 5 ) {

        // init headInfo
        FTIFF_headInfo *headInfo;
        headInfo = malloc(FTI_Topo->nbApprocs * sizeof(FTIFF_headInfo));

        int k;
        for (i = 0; i < FTI_Topo->nbApprocs; i++) { // Iterate on the application processes in the node
            k = i+1;
            MPI_Recv(&(headInfo[i]), 1, FTIFF_MpiTypes[FTIFF_HEAD_INFO], FTI_Topo->body[i], FTI_Conf->generalTag, FTI_Exec->globalComm, MPI_STATUS_IGNORE);
            FTI_Exec->meta[0].exists[k] = headInfo[i].exists;
            FTI_Exec->meta[0].nbVar[k] = headInfo[i].nbVar;
            FTI_Exec->meta[0].maxFs[k] = headInfo[i].maxFs;
            FTI_Exec->meta[0].fs[k] = headInfo[i].fs;
            FTI_Exec->meta[0].pfs[k] = headInfo[i].pfs;
            isDcpCnt += headInfo[i].isDcp;
            MPI_Recv(&(FTI_Exec->meta[0].varID[k * FTI_BUFS]), headInfo[i].nbVar, MPI_INT, FTI_Topo->body[i], FTI_Conf->generalTag, FTI_Exec->globalComm, MPI_STATUS_IGNORE);
            MPI_Recv(&(FTI_Exec->meta[0].varSize[k * FTI_BUFS]), headInfo[i].nbVar, MPI_LONG, FTI_Topo->body[i], FTI_Conf->generalTag, FTI_Exec->globalComm, MPI_STATUS_IGNORE);
            strncpy(&(FTI_Exec->meta[0].ckptFile[k * FTI_BUFS]), headInfo[i].ckptFile , FTI_BUFS);
            sscanf(&(FTI_Exec->meta[0].ckptFile[k * FTI_BUFS]), "Ckpt%d", &FTI_Exec->ckptID);
        }
        strcpy(FTI_Exec->meta[FTI_Exec->ckptLvel].ckptFile, FTI_Exec->meta[0].ckptFile);

        if ( FTI_Conf->dcpEnabled ) {
            if ( (isDcpCnt == FTI_Topo->nbApprocs) && FTI_Conf->dcpEnabled ) {
                FTI_Ckpt[4].isDcp = true;
            }
        } else {
            isDcpCnt = 0;
        }

        free(headInfo);

    }

    //Check if checkpoint was written correctly by all processes
    int res = (FTI_Exec->ckptLvel == 6) ? FTI_NSCS : FTI_SCES;

    // check for consistency of dCP request (isDcpCnt is 0 if dCP is disabled)
    if ( (isDcpCnt > 0) && (isDcpCnt < FTI_Topo->nbApprocs) ) {
        FTI_Print( "dCP was requested by some but not all ranks, discarding checkpoint request!", FTI_WARN );
        res = FTI_NSCS;
    }
    int allRes;
    MPI_Allreduce(&res, &allRes, 1, MPI_INT, MPI_SUM, FTI_COMM_WORLD);
    if (allRes == FTI_SCES) { //If checkpoint was written correctly do post-processing
        res = FTI_Try(FTI_PostCkpt(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt), "postprocess the checkpoint.");
        if (res == FTI_SCES) {
            res = FTI_Exec->ckptLvel; //return ckptLvel if post-processing succeeds
        }
    }
    else {  //If checkpoint wasn't written correctly
        FTI_Print("Checkpoint have not been witten correctly. Discarding current checkpoint...", FTI_WARN);
        FTI_Clean(FTI_Conf, FTI_Topo, FTI_Ckpt, 0); //Remove temporary files
        res = FTI_NSCS;
    }
    for (i = 0; i < FTI_Topo->nbApprocs; i++) { // Send msg. to avoid checkpoint collision
        MPI_Send(&res, 1, MPI_INT, FTI_Topo->body[i], FTI_Conf->generalTag, FTI_Exec->globalComm);
    }
    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Writes ckpt to PFS using POSIX.
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Topo        Topology metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @param      FTI_Data        Dataset metadata.
  @return     integer         FTI_SCES if successful.

 **/
/*-------------------------------------------------------------------------*/
int FTI_WritePosix(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt,
        FTIT_dataset* FTI_Data)
{
    int res;
    FTI_Print("I/O mode: Posix.", FTI_DBUG);
    char str[FTI_BUFS], fn[FTI_BUFS];
    int level = FTI_Exec->ckptLvel;
    if (level == 4 && FTI_Ckpt[4].isInline) { //If inline L4 save directly to global directory
        snprintf(fn, FTI_BUFS, "%s/%s", FTI_Conf->gTmpDir, FTI_Exec->meta[0].ckptFile);
    }
    else {
        snprintf(fn, FTI_BUFS, "%s/%s", FTI_Conf->lTmpDir, FTI_Exec->meta[0].ckptFile);
    }

    // open task local ckpt file
    FILE* fd = fopen(fn, "wb");
    if (fd == NULL) {
        snprintf(str, FTI_BUFS, "FTI checkpoint file (%s) could not be opened.", fn);
        FTI_Print(str, FTI_EROR);

        return FTI_NSCS;
    }

    // write data into ckpt file
    int i;

    for (i = 0; i < FTI_Exec->nbVar; i++) {
        clearerr(fd);
        if (!ferror(fd)) {
            errno = 0;
            // if data are stored to CPU just write them
            if ( !(FTI_Data[i].isDevicePtr) ){
                snprintf(str, FTI_BUFS, "ID:  %d Data are . %d %p %p", FTI_Data[i].id,FTI_Data[i].isDevicePtr,FTI_Data[i].ptr,FTI_Data[i].devicePtr);
                FTI_Print(str,FTI_DBUG);
                res = write_posix(FTI_Data[i].ptr, FTI_Data[i].size, fd);
            }
#ifdef GPUSUPPORT            
            // if data are stored to the GPU move them from device
            // memory to cpu memory and store them.
            else {
                FTI_Print(str,FTI_INFO);
                if ((res = FTI_Try(
                                FTI_pipeline_gpu_to_storage(&FTI_Data[i],  FTI_Exec, FTI_Conf, write_posix, fd),
                                "moving data from GPU to storage")) != FTI_SCES) {
                    snprintf(str, FTI_BUFS, "Dataset #%d could not be written.", FTI_Data[i].id);
                    FTI_Print(str, FTI_EROR);
                    fclose(fd);
                    return res;
                }
            }
#endif            
        }

        if (ferror(fd)) {
            char error_msg[FTI_BUFS];
            error_msg[0] = 0;
            strerror_r(errno, error_msg, FTI_BUFS);
            snprintf(str, FTI_BUFS, "%d Dataset #%d could not be written: %s.",__LINE__, FTI_Data[i].id, error_msg);
            FTI_Print(str, FTI_EROR);
            fclose(fd);
            return FTI_NSCS;
        }
    }

    // close file
    if (fclose(fd) != 0) {
        FTI_Print("FTI checkpoint file could not be closed.", FTI_EROR);

        return FTI_NSCS;
    }

    return FTI_SCES;

}

/*-------------------------------------------------------------------------*/
/**
  @brief      Writes ckpt to PFS using MPI I/O.
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Topo        Topology metadata.
  @param      FTI_Data        Dataset metadata.
  @return     integer         FTI_SCES if successful.

  In here it is taken into account, that in MPIIO the count parameter
  in both, MPI_Type_contiguous and MPI_File_write_at, are integer
  types. The ckpt data is split into chunks of maximal (MAX_INT-1)/2
  elements to form contiguous data types. It was experienced, that
  if the size is greater then that, it may lead to problems.

 **/
/*-------------------------------------------------------------------------*/
int FTI_WriteMPI(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_dataset* FTI_Data)
{
    WriteMPIInfo_t write_info;
    int res;
    FTI_Print("I/O mode: MPI-IO.", FTI_DBUG);
    char str[FTI_BUFS], mpi_err[FTI_BUFS];

    write_info.FTI_Conf = FTI_Conf;

    // enable collective buffer optimization
    MPI_Info info;
    MPI_Info_create(&info);
    MPI_Info_set(info, "romio_cb_write", "enable");

    // TODO enable to set stripping unit in the config file (Maybe also other hints)
    // set stripping unit to 4MB
    MPI_Info_set(info, "stripping_unit", "4194304");

    MPI_Offset chunkSize = FTI_Exec->ckptSize;

    // collect chunksizes of other ranks
    MPI_Offset* chunkSizes = talloc(MPI_Offset, FTI_Topo->nbApprocs * FTI_Topo->nbNodes);
    MPI_Allgather(&chunkSize, 1, MPI_OFFSET, chunkSizes, 1, MPI_OFFSET, FTI_COMM_WORLD);

    char gfn[FTI_BUFS], ckptFile[FTI_BUFS];
    snprintf(ckptFile, FTI_BUFS, "Ckpt%d-mpiio.fti", FTI_Exec->ckptID);
    snprintf(gfn, FTI_BUFS, "%s/%s", FTI_Conf->gTmpDir, ckptFile);
    // open parallel file (collective call)
    //    MPI_File pfh;

#ifdef LUSTRE
    if (FTI_Topo->splitRank == 0) {
        res = llapi_file_create(gfn, FTI_Conf->stripeUnit, FTI_Conf->stripeOffset, FTI_Conf->stripeFactor, 0);
        if (res) {
            char error_msg[FTI_BUFS];
            error_msg[0] = 0;
            strerror_r(-res, error_msg, FTI_BUFS);
            snprintf(str, FTI_BUFS, "[Lustre] %s.", error_msg);
            FTI_Print(str, FTI_WARN);
        } else {
            snprintf(str, FTI_BUFS, "[LUSTRE] file:%s striping_unit:%i striping_factor:%i striping_offset:%i",
                    ckptFile, FTI_Conf->stripeUnit, FTI_Conf->stripeFactor, FTI_Conf->stripeOffset);
            FTI_Print(str, FTI_DBUG);
        }
    }
#endif
    res = MPI_File_open(FTI_COMM_WORLD, gfn, MPI_MODE_WRONLY|MPI_MODE_CREATE, info, &(write_info.pfh));

    // check if successful
    if (res != 0) {
        errno = 0;
        int reslen;
        MPI_Error_string(res, mpi_err, &reslen);
        snprintf(str, FTI_BUFS, "unable to create file [MPI ERROR - %i] %s", res, mpi_err);
        FTI_Print(str, FTI_EROR);
        free(chunkSizes);
        return FTI_NSCS;
    }

    // set file offset
    write_info.offset = 0;
    int i;
    for (i = 0; i < FTI_Topo->splitRank; i++) {
        write_info.offset += chunkSizes[i];
    }
    free(chunkSizes);

    for (i = 0; i < FTI_Exec->nbVar; i++) {
        // determine the type of data pointer
        // Data are stored in the CPU side. 
        if ( !(FTI_Data[i].isDevicePtr) ){
            FTI_Print(str,FTI_INFO);
            res = write_mpi(FTI_Data[i].ptr, FTI_Data[i].size, &write_info);
        }
#ifdef GPUSUPPORT
        // dowload data from the GPU if necessary
        // Data are stored in the GPU side.
        else {
            snprintf(str, FTI_BUFS, "Dataset #%d Writing GPU Data.", FTI_Data[i].id);
            FTI_Print(str,FTI_INFO);

            if ((res = FTI_Try(
                            FTI_pipeline_gpu_to_storage(&FTI_Data[i],  FTI_Exec, FTI_Conf, write_mpi, &write_info),
                            "moving data from GPU to storage")) != FTI_SCES) {
                snprintf(str, FTI_BUFS, "Dataset #%d could not be written.", FTI_Data[i].id);
                FTI_Print(str, FTI_EROR);
                MPI_File_close(&write_info.pfh);
                return res;
            }
        }
#endif


        // check if successful
        if (res != 0) {
            errno = 0;
            int reslen;
            MPI_Error_string(write_info.err, mpi_err, &reslen);
            snprintf(str, FTI_BUFS, "Failed to write protected_var[%i] to PFS  [MPI ERROR - %i] %s", i, write_info.err, mpi_err);
            FTI_Print(str, FTI_EROR);
            MPI_File_close(&write_info.pfh);
            return FTI_NSCS;
        }
    }
    MPI_File_close(&write_info.pfh);
    MPI_Info_free(&info);
    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Writes ckpt to PFS using SIONlib.
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Topo        Topology metadata.
  @param      FTI_Data        Dataset metadata.
  @return     integer         FTI_SCES if successful.

 **/
/*-------------------------------------------------------------------------*/
#ifdef ENABLE_SIONLIB // --> If SIONlib is installed
int FTI_WriteSionlib(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo,FTIT_dataset* FTI_Data)
{
    int numFiles = 1;
    int nlocaltasks = 1;
    int* file_map = calloc(1, sizeof(int));
    int* ranks = talloc(int, 1);
    int* rank_map = talloc(int, 1);
    sion_int64* chunkSizes = talloc(sion_int64, 1);
    int fsblksize = -1;
    chunkSizes[0] = FTI_Exec->ckptSize;
    ranks[0] = FTI_Topo->splitRank;
    rank_map[0] = FTI_Topo->splitRank;

    // open parallel file
    char fn[FTI_BUFS], str[FTI_BUFS];
    snprintf(str, FTI_BUFS, "Ckpt%d-sionlib.fti", FTI_Exec->ckptID);
    snprintf(fn, FTI_BUFS, "%s/%s", FTI_Conf->gTmpDir, str);
    int sid = sion_paropen_mapped_mpi(fn, "wb,posix", &numFiles, FTI_COMM_WORLD, &nlocaltasks, &ranks, &chunkSizes, &file_map, &rank_map, &fsblksize, NULL);

    // check if successful
    if (sid == -1) {
        errno = 0;
        FTI_Print("SIONlib: File could no be opened", FTI_EROR);

        free(file_map);
        free(rank_map);
        free(ranks);
        free(chunkSizes);
        return FTI_NSCS;
    }

    // set file pointer to corresponding block in sionlib file
    int res = sion_seek(sid, FTI_Topo->splitRank, SION_CURRENT_BLK, SION_CURRENT_POS);

    // check if successful
    if (res != SION_SUCCESS) {
        errno = 0;
        FTI_Print("SIONlib: Could not set file pointer", FTI_EROR);
        sion_parclose_mapped_mpi(sid);
        free(file_map);
        free(rank_map);
        free(ranks);
        free(chunkSizes);
        return FTI_NSCS;
    }

    // write datasets into file
    int i;
    for (i = 0; i < FTI_Exec->nbVar; i++) {
        // SIONlib write call

        if ( !(FTI_Data[i].isDevicePtr) ){
            FTI_Print(str,FTI_INFO);
            res = write_sion(FTI_Data[i].ptr, FTI_Data[i].size, &sid);
        }
#ifdef GPUSUPPORT            
        // if data are stored to the GPU move them from device
        // memory to cpu memory and store them.
        else {
            FTI_Print(str,FTI_INFO);
            if ((res = FTI_Try(
                            FTI_pipeline_gpu_to_storage(&FTI_Data[i],  FTI_Exec, FTI_Conf, write_sion, &sid),
                            "moving data from GPU to storage")) != FTI_SCES) {
                snprintf(str, FTI_BUFS, "Dataset #%d could not be written.", FTI_Data[i].id);
                FTI_Print(str, FTI_EROR);
                errno = 0;
                FTI_Print("SIONlib: Data could not be written", FTI_EROR);
                res =  sion_parclose_mapped_mpi(sid);
                free(file_map);
                free(rank_map);
                free(ranks);
                free(chunkSizes);
                return res;
            }
        }
#endif            
    }

    // close parallel file
    if (sion_parclose_mapped_mpi(sid) == -1) {
        FTI_Print("Cannot close sionlib file.", FTI_WARN);
        free(file_map);
        free(rank_map);
        free(ranks);
        free(chunkSizes);
        return FTI_NSCS;
    }
    free(file_map);
    free(rank_map);
    free(ranks);
    free(chunkSizes);

    return FTI_SCES;
}
#endif
