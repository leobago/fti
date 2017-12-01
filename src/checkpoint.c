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
            sprintf(str, "Current iter : %d ckpt intv. : %d . Next ckpt. at iter. %d . Sync. intv. : %d",
                    FTI_Exec->ckptIcnt, FTI_Exec->ckptIntv, FTI_Exec->ckptNext, FTI_Exec->syncIter);
            FTI_Print(str, FTI_DBUG);
            if ((FTI_Exec->syncIter < (FTI_Exec->ckptIntv / 2)) && (FTI_Exec->syncIter < FTI_Exec->syncIterMax)) {
                FTI_Exec->syncIter = FTI_Exec->syncIter * 2;
                sprintf(str, "Iteration frequency : %.2f sec/iter => %d iter/min. Resync every %d iter.",
                        FTI_Exec->globMeanIter, FTI_Exec->ckptIntv, FTI_Exec->syncIter);
                FTI_Print(str, FTI_DBUG);
                if (FTI_Exec->syncIter == FTI_Exec->syncIterMax) {
                    sprintf(str, "Sync. intv. has reached max value => %i iterations", FTI_Exec->syncIterMax);
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
    sprintf(str, "Starting writing checkpoint (ID: %d, Lvl: %d)",
            FTI_Exec->ckptID, FTI_Exec->ckptLvel);
    FTI_Print(str, FTI_DBUG);

    double tt = MPI_Wtime(); //Start time

    //update ckpt file name
    snprintf(FTI_Exec->meta[0].ckptFile, FTI_BUFS,
            "Ckpt%d-Rank%d.fti", FTI_Exec->ckptID, FTI_Topo->myRank);

    //If checkpoint is inlin and level 4 save directly to PFS
    int res; //response from writing funcitons
    if (FTI_Ckpt[4].isInline && FTI_Exec->ckptLvel == 4) {
        FTI_Print("Saving to temporary global directory", FTI_DBUG);

        //Create global temp directory
        if (mkdir(FTI_Conf->gTmpDir, 0777) == -1) {
            if (errno != EEXIST) {
                FTI_Print("Cannot create global directory", FTI_EROR);
                return FTI_NSCS;
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
        }
    }
    else {
        FTI_Print("Saving to temporary local directory", FTI_DBUG);
        //Create local temp directory
        if (mkdir(FTI_Conf->lTmpDir, 0777) == -1) {
            if (errno != EEXIST) {
                FTI_Print("Cannot create local directory", FTI_EROR);
            }
        }
        res = FTI_Try(FTI_WritePosix(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt, FTI_Data),"write checkpoint.");
    }

    //Check if all processes have written correctly (every process must succeed)
    int allRes;
    MPI_Allreduce(&res, &allRes, 1, MPI_INT, MPI_SUM, FTI_COMM_WORLD);
    if (allRes != FTI_SCES) {
        return FTI_NSCS;
    }

    sprintf(str, "Time writing checkpoint file : %f seconds.", MPI_Wtime() - tt);
    FTI_Print(str, FTI_DBUG);

    res = FTI_Try(FTI_CreateMetadata(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt, FTI_Data), "create metadata.");
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

    FTI_Clean(FTI_Conf, FTI_Topo, FTI_Ckpt, FTI_Exec->ckptLvel); //delete previous files on this checkpoint level
    int nodeFlag = (((!FTI_Topo->amIaHead) && ((FTI_Topo->nodeRank - FTI_Topo->nbHeads) == 0)) || (FTI_Topo->amIaHead)) ? 1 : 0;
    if (nodeFlag) { //True only for one process in the node.
        //Debug message needed to test nodeFlag (./tests/nodeFlag/nodeFlag.c)
        sprintf(str, "Has nodeFlag = 1 and nodeID = %d. CkptLvel = %d.", FTI_Topo->nodeID, FTI_Exec->ckptLvel);
        FTI_Print(str, FTI_DBUG);
        if (!(FTI_Ckpt[4].isInline && FTI_Exec->ckptLvel == 4)) {
            //checkpoint was not saved in global temporary directory
            int level = (FTI_Exec->ckptLvel != 4) ? FTI_Exec->ckptLvel : 1; //if level 4: head moves local ckpt files to PFS
            if (rename(FTI_Conf->lTmpDir, FTI_Ckpt[level].dir) == -1) {
                FTI_Print("Cannot rename local directory", FTI_EROR);
            }
            else {
                FTI_Print("Local directory renamed", FTI_DBUG);
            }
        }
    }
    int globalFlag = !FTI_Topo->splitRank;
    if (globalFlag) { //True only for one process in the FTI_COMM_WORLD.
        if (FTI_Exec->ckptLvel == 4) {
            if (rename(FTI_Conf->gTmpDir, FTI_Ckpt[4].dir) == -1) {
                FTI_Print("Cannot rename global directory", FTI_EROR);
            }
        }
        if (rename(FTI_Conf->mTmpDir, FTI_Ckpt[FTI_Exec->ckptLvel].metaDir) == -1) {
            FTI_Print("Cannot rename meta directory", FTI_EROR);
        }
    }
    MPI_Barrier(FTI_COMM_WORLD); //barrier needed to wait for process to rename directories (new temporary could be needed in next checkpoint)

    double t3 = MPI_Wtime(); //Renaming directories time

    sprintf(str, "Post-checkpoint took %.2f sec. (Pt:%.2fs, Cl:%.2fs)",
            t3 - t1, t2 - t1, t3 - t2);
    FTI_Print(str, FTI_INFO);
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
    FTI_Print("Head starts listening...", FTI_DBUG);
    while (1) { //heads can stop only by receiving FTI_ENDW
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
            MPI_Recv(&buf, 1, MPI_INT, FTI_Topo->body[i], FTI_Conf->tag, FTI_Exec->globalComm, MPI_STATUS_IGNORE);
            sprintf(str, "The head received a %d message", buf);
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
        if (FTI_Exec->ckptLvel == 5) { // If we were asked to finalize
            FTI_Print("Head stopped listening.", FTI_DBUG);
            FTI_Finalize();
        }

        //Check if checkpoint was written correctly by all processes
        int res = (FTI_Exec->ckptLvel == 6) ? FTI_NSCS : FTI_SCES;
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
            MPI_Send(&res, 1, MPI_INT, FTI_Topo->body[i], FTI_Conf->tag, FTI_Exec->globalComm);
        }
    }
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
    FTI_Print("I/O mode: Posix.", FTI_DBUG);
    char str[FTI_BUFS], fn[FTI_BUFS];
    int level = FTI_Exec->ckptLvel;
    if (level == 4 && FTI_Ckpt[4].isInline) { //If inline L4 save directly to global directory
        sprintf(fn, "%s/%s", FTI_Conf->gTmpDir, FTI_Exec->meta[0].ckptFile);
    }
    else {
        sprintf(fn, "%s/%s", FTI_Conf->lTmpDir, FTI_Exec->meta[0].ckptFile);
    }

    // open task local ckpt file
    FILE* fd = fopen(fn, "wb");
    if (fd == NULL) {
        sprintf(str, "FTI checkpoint file (%s) could not be opened.", fn);
        FTI_Print(str, FTI_EROR);

        return FTI_NSCS;
    }

    // write data into ckpt file
    int i;
    for (i = 0; i < FTI_Exec->nbVar; i++) {
        clearerr(fd);
        size_t written = 0;
        int fwrite_errno;
        while (written < FTI_Data[i].count && !ferror(fd)) {
            errno = 0;
            written += fwrite(((char*)FTI_Data[i].ptr) + (FTI_Data[i].eleSize*written), FTI_Data[i].eleSize, FTI_Data[i].count - written, fd);
            fwrite_errno = errno;
        }
        if (ferror(fd)) {
            char error_msg[FTI_BUFS];
            error_msg[0] = 0;
            strerror_r(fwrite_errno, error_msg, FTI_BUFS);
            sprintf(str, "Dataset #%d could not be written: %s.", FTI_Data[i].id, error_msg);
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
    int res;
    FTI_Print("I/O mode: MPI-IO.", FTI_DBUG);
    char str[FTI_BUFS], mpi_err[FTI_BUFS];

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
    sprintf(gfn, "%s/%s", FTI_Conf->gTmpDir, ckptFile);
    // open parallel file (collective call)
    MPI_File pfh;

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
    res = MPI_File_open(FTI_COMM_WORLD, gfn, MPI_MODE_WRONLY|MPI_MODE_CREATE, info, &pfh);

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
    MPI_Offset offset = 0;
    int i;
    for (i = 0; i < FTI_Topo->splitRank; i++) {
        offset += chunkSizes[i];
    }
    free(chunkSizes);

    for (i = 0; i < FTI_Exec->nbVar; i++) {
        long pos = 0;
        long varSize = FTI_Data[i].size;
        long bSize = FTI_Conf->transferSize;
        char *data_ptr = FTI_Data[i].ptr;
        while (pos < varSize) {
            if ((varSize - pos) < FTI_Conf->transferSize) {
                bSize = varSize - pos;
            }

            MPI_Datatype dType;
            MPI_Type_contiguous(bSize, MPI_BYTE, &dType);
            MPI_Type_commit(&dType);

            res = MPI_File_write_at(pfh, offset, data_ptr, 1, dType, MPI_STATUS_IGNORE);
            // check if successful
            if (res != 0) {
                errno = 0;
                int reslen;
                MPI_Error_string(res, mpi_err, &reslen);
                snprintf(str, FTI_BUFS, "Failed to write protected_var[%i] to PFS  [MPI ERROR - %i] %s", i, res, mpi_err);
                FTI_Print(str, FTI_EROR);
                MPI_File_close(&pfh);
                return FTI_NSCS;
            }
            MPI_Type_free(&dType);
            data_ptr += bSize;
            offset += bSize;
            pos = pos + bSize;
        }
    }
    MPI_File_close(&pfh);
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
    sprintf(fn, "%s/%s", FTI_Conf->gTmpDir, str);
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
        res = sion_fwrite(FTI_Data[i].ptr, FTI_Data[i].size, 1, sid);

        // check if successful
        if (res < 0) {
            errno = 0;
            FTI_Print("SIONlib: Data could not be written", FTI_EROR);
            res =  sion_parclose_mapped_mpi(sid);
            free(file_map);
            free(rank_map);
            free(ranks);
            free(chunkSizes);
            return FTI_NSCS;
        }

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
