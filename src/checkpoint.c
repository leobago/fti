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

#include "interface.h"
#include <math.h>

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
        FTIT_keymap* FTI_Data)
{
    char str[FTI_BUFS]; //For console output
    snprintf(str, FTI_BUFS, "Starting writing checkpoint (ID: %d, Lvl: %d)", FTI_Exec->ckptId, FTI_Exec->ckptMeta.level);
    FTI_Print(str, FTI_DBUG);

    if ( FTI_Conf->keepL4Ckpt && FTI_Exec->ckptMeta.level == 4 ) {
        int ckptId = FTI_LoadL4CkptMetaData( FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt );
        if( ckptId > 0 ) {
            FTI_Exec->ckptMeta.ckptIdL4 = ckptId;
            FTI_ArchiveL4Ckpt( FTI_Conf, FTI_Exec, FTI_Ckpt, FTI_Topo );
            MPI_Barrier( FTI_COMM_WORLD );
        }
    }
    //If checkpoint is inlin and level 4 save directly to PFS
    int res; //response from writing funcitons
    int offset = 2*(FTI_Conf->dcpPosix || FTI_Conf->dcpFtiff);
    if (FTI_Ckpt[4].isInline && FTI_Exec->ckptMeta.level == 4) {

        if ( !((FTI_Conf->dcpFtiff || FTI_Conf->dcpPosix) && FTI_Ckpt[4].isDcp) && !FTI_Exec->h5SingleFile ) {
            MKDIR(FTI_Conf->gTmpDir, 0777);
        } else if ( !FTI_Ckpt[4].hasDcp && !FTI_Exec->h5SingleFile ) {
            MKDIR(FTI_Ckpt[4].dcpDir, 0777);
        }
        //Actually call the respecitve function to store the checkpoint 
        res = FTI_Exec->ckptFunc[GLOBAL](FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt, FTI_Data, &ftiIO[offset + GLOBAL]);
    }
    else {
        if ( !((FTI_Conf->dcpFtiff || FTI_Conf->dcpPosix) && FTI_Ckpt[4].isDcp) && !FTI_Exec->h5SingleFile ) {
            MKDIR(FTI_Conf->lTmpDir,0777);
        } else if ( !FTI_Ckpt[4].hasDcp && !FTI_Exec->h5SingleFile ){
            MKDIR(FTI_Ckpt[1].dcpDir, 0777);
        }
        //Actually call the respecitve function to store the checkpoint 
        res = FTI_Exec->ckptFunc[LOCAL](FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt, FTI_Data, &ftiIO[offset + LOCAL] );
    }

    //Check if all processes have written correctly (every process must succeed)
    int allRes;
    MPI_Allreduce(&res, &allRes, 1, MPI_INT, MPI_SUM, FTI_COMM_WORLD);
    if (allRes != FTI_SCES) {
        return FTI_NSCS;
    } else if( FTI_Exec->h5SingleFile ) {
        return FTI_SCES;
    }
    if ( (FTI_Conf->dcpFtiff||FTI_Conf->dcpPosix) && FTI_Ckpt[4].isDcp ) {
        // After dCP update store total data and dCP sizes in application rank 0
        unsigned long *dataSize = (FTI_Conf->dcpFtiff)?(unsigned long*)&FTI_Exec->FTIFFMeta.pureDataSize:&FTI_Exec->dcpInfoPosix.dataSize;
        unsigned long *dcpSize = (FTI_Conf->dcpFtiff)?(unsigned long*)&FTI_Exec->FTIFFMeta.dcpSize:&FTI_Exec->dcpInfoPosix.dcpSize;
        unsigned long dcpStats[2]; // 0:totalDcpSize, 1:totalDataSize
        unsigned long sendBuf[] = { *dcpSize, *dataSize };
        MPI_Reduce( sendBuf, dcpStats, 2, MPI_UNSIGNED_LONG, MPI_SUM, 0, FTI_COMM_WORLD );
        if ( FTI_Topo->splitRank ==  0 ) {
            *dcpSize = dcpStats[0]; 
            *dataSize = dcpStats[1];
        }
    }

    res = FTI_Try(FTI_CreateMetadata(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt, FTI_Data), "create metadata.");

    if ( (FTI_Conf->dcpFtiff || FTI_Conf->keepL4Ckpt) && (FTI_Topo->splitRank == 0) ) {
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
    switch (FTI_Exec->ckptMeta.level) {
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

    FTI_Clean(FTI_Conf, FTI_Topo, FTI_Ckpt, FTI_Exec->ckptMeta.level); //delete previous files on this checkpoint level
    int nodeFlag = (((!FTI_Topo->amIaHead) && ((FTI_Topo->nodeRank - FTI_Topo->nbHeads) == 0)) || (FTI_Topo->amIaHead)) ? 1 : 0;
    nodeFlag = (!FTI_Ckpt[4].isDcp && (nodeFlag != 0));
    if (nodeFlag) { //True only for one process in the node.
        //Debug message needed to test nodeFlag (./tests/nodeFlag/nodeFlag.c)
        snprintf(str, FTI_BUFS, "Has nodeFlag = 1 and nodeID = %d. CkptLvel = %d.", FTI_Topo->nodeID, FTI_Exec->ckptMeta.level);
        FTI_Print(str, FTI_DBUG);
        if (!(FTI_Ckpt[4].isInline && FTI_Exec->ckptMeta.level == 4)) {
            //checkpoint was not saved in global temporary directory
            int level = (FTI_Exec->ckptMeta.level != 4) ? FTI_Exec->ckptMeta.level : 1; //if level 4: head moves local ckpt files to PFS
            RENAME(FTI_Conf->lTmpDir, FTI_Ckpt[level].dir);
        }
    }
    int globalFlag = !FTI_Topo->splitRank;
    globalFlag = (!(FTI_Ckpt[4].isDcp && FTI_Conf->dcpFtiff) && (globalFlag != 0));
    if (globalFlag) { //True only for one process in the FTI_COMM_WORLD.
        if ((FTI_Exec->ckptMeta.level == 4) && !(FTI_Ckpt[4].isDcp)) {
            RENAME(FTI_Conf->gTmpDir, FTI_Ckpt[4].dir);
        }
        // there is no temp meta data folder for FTI-FF
        if ( FTI_Conf->ioMode != FTI_IO_FTIFF ) {
            RENAME(FTI_Conf->mTmpDir, FTI_Ckpt[FTI_Exec->ckptMeta.level].metaDir);
        }
    }
    MPI_Barrier(FTI_COMM_WORLD); //barrier needed to wait for process to rename directories (new temporary could be needed in next checkpoint)

    double t3 = MPI_Wtime(); //Renaming directories time

    snprintf(str, FTI_BUFS, "Post-checkpoint took %.2f sec. (Pt:%.2fs, Cl:%.2fs)",
            t3 - t1, t2 - t1, t3 - t2);
    FTI_Print(str, FTI_INFO);

    // expose to FTI that a checkpoint exists for level
    FTI_Ckpt[FTI_Exec->ckptMeta.level].hasCkpt = true;

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
        int isDCP;
        MPI_Recv(&isDCP, 1, MPI_INT, FTI_Topo->body[i], FTI_Conf->ckptTag, FTI_Exec->globalComm, MPI_STATUS_IGNORE);
        FTI_Ckpt[4].isDcp = isDCP;
        MPI_Recv(&FTI_Exec->ckptId, 1, MPI_INT, FTI_Topo->body[i], FTI_Conf->ckptTag, FTI_Exec->globalComm, MPI_STATUS_IGNORE);
        snprintf(str, FTI_BUFS, "The head received a %d message", buf);
        FTI_Print(str, FTI_DBUG);
        flags[buf - FTI_BASE] = flags[buf - FTI_BASE] + 1;
    }
    for (i = 1; i < 7; i++) {
        if (flags[i] == FTI_Topo->nbApprocs) { // Determining checkpoint level
            FTI_Exec->ckptMeta.level = i;
        }
    }
    if (flags[6] > 0) {
        FTI_Exec->ckptMeta.level = 6;
    }

    if ( FTI_Conf->ioMode == FTI_IO_FTIFF &&  FTI_Exec->ckptMeta.level != 6 &&  FTI_Exec->ckptMeta.level != 5 ) {

        FTI_Exec->mqueue.clear();

    }

    //Check if checkpoint was written correctly by all processes
    int res = (FTI_Exec->ckptMeta.level == 6) ? FTI_NSCS : FTI_SCES;

    int allRes;
    MPI_Allreduce(&res, &allRes, 1, MPI_INT, MPI_SUM, FTI_COMM_WORLD);
    if (allRes == FTI_SCES) { //If checkpoint was written correctly do post-processing
        res = FTI_Try(FTI_PostCkpt(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt), "postprocess the checkpoint.");
        if (res == FTI_SCES) {
            res = FTI_Exec->ckptMeta.level; // send checkpoint level if post-processing succeeds
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
  @brief      Writes ckpt.
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Topo        Topology metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @param      FTI_Data        Dataset metadata.
  @param      io              IO function pointers
  @return     integer         FTI_SCES if successful.

  This function performs a normal checkpoint by calling the respective file format procedures,
  initalize ckpt, write data, compute integrity and finalize files.
 **/
/*-------------------------------------------------------------------------*/
int FTI_Write(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt,
        FTIT_keymap* FTI_Data, FTIT_IO *io)
{

    int i;
    void *write_info = io->initCKPT(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt, FTI_Data);
    if( !write_info ) {
        FTI_Print("unable to initialize checkpoint!", FTI_EROR);
        return FTI_NSCS;
    }

    FTIT_dataset* data;
    if( FTI_Data->data( &data, FTI_Exec->nbVar ) != FTI_SCES ) return FTI_NSCS;

    for (i = 0; i < FTI_Exec->nbVar; i++) {
        data[i].filePos = io->getPos(write_info);
        int ret = io->WriteData(&data[i], write_info);
        if (ret != FTI_SCES)
            return ret;
    }

    io->finIntegrity(FTI_Exec->integrity, write_info);
    io->finCKPT(write_info);
    free (write_info);
    return FTI_SCES;

}
