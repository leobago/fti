/**
 *  @file   checkpoint.c
 *  @author Leonardo A. Bautista Gomez (leobago@gmail.com)
 *  @date   December, 2013
 *  @brief  Checkpointing functions for the FTI library.
 */

#include "fti.h"

/*-------------------------------------------------------------------------*/
/**
    @brief      It updates the local and global mean iteration time.
    @return     integer         FTI_SCES if successful.

    This function updates the local and global mean iteration time. It also
    recomputes the checkpoint interval in iterations and correct the next
    checkpointing iteration based on the observed mean iteration duration.

 **/
/*-------------------------------------------------------------------------*/
int FTI_UpdateIterTime()
{
    int nbProcs, res;
    char str[FTI_BUFS];
    double last = FTI_Exec.iterTime;
    FTI_Exec.iterTime = MPI_Wtime();
    if (FTI_Exec.ckptIcnt > 0) {
        FTI_Exec.lastIterTime = FTI_Exec.iterTime - last;
        FTI_Exec.totalIterTime = FTI_Exec.totalIterTime + FTI_Exec.lastIterTime;
        if (FTI_Exec.ckptIcnt % FTI_Exec.syncIter == 0) {
            FTI_Exec.meanIterTime = FTI_Exec.totalIterTime / FTI_Exec.ckptIcnt;
            MPI_Allreduce(&FTI_Exec.meanIterTime, &FTI_Exec.globMeanIter, 1, MPI_DOUBLE, MPI_SUM, FTI_COMM_WORLD);
            MPI_Comm_size(FTI_COMM_WORLD, &nbProcs);
            FTI_Exec.globMeanIter = FTI_Exec.globMeanIter / nbProcs;
            if (FTI_Exec.globMeanIter > 60) {
                FTI_Exec.ckptIntv = 1;
            }
            else {
                FTI_Exec.ckptIntv = (1 * 60) / FTI_Exec.globMeanIter;
            }
            res = FTI_Exec.ckptLast + FTI_Exec.ckptIntv;
            if (res >= FTI_Exec.ckptIcnt) {
                FTI_Exec.ckptNext = res;
            }
            if (FTI_Exec.syncIter < (FTI_Exec.ckptIntv / 2)) {
                FTI_Exec.syncIter = FTI_Exec.syncIter * 2;
                sprintf(str, "Iteration frequency : %.2f sec/iter => %d iter/min. Resync every %d iter.",
                    FTI_Exec.globMeanIter, FTI_Exec.ckptIntv, FTI_Exec.syncIter);
                FTI_Print(str, FTI_DBUG);
            }
        }
    }
    FTI_Exec.ckptIcnt++; // Increment checkpoint loop counter
    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
    @brief      It writes the checkpoint data in the target file.
    @param      FTI_Data        Dataset array.
    @return     integer         FTI_SCES if successful.

    This function checks whether the checkpoint needs to be local or remote,
    opens the target file and write dataset per dataset, the checkpoint data,
    it finally flushes and closes the checkpoint file.

 **/
/*-------------------------------------------------------------------------*/
int FTI_WriteCkpt(FTIT_dataset* FTI_Data)
{
    int i, res;
    FILE* fd;
    double tt = MPI_Wtime();
    char fn[FTI_BUFS], str[FTI_BUFS];
    snprintf(FTI_Exec.ckptFile, FTI_BUFS, "Ckpt%d-Rank%d.fti", FTI_Exec.ckptID, FTI_Topo.myRank);
    if (FTI_Ckpt[4].isInline && FTI_Exec.ckptLvel == 4) {
        sprintf(fn, "%s/%s", FTI_Conf.gTmpDir, FTI_Exec.ckptFile);
        mkdir(FTI_Conf.gTmpDir, 0777);
    }
    else {
        sprintf(fn, "%s/%s", FTI_Conf.lTmpDir, FTI_Exec.ckptFile);
        mkdir(FTI_Conf.lTmpDir, 0777);
    }
    fd = fopen(fn, "wb");
    if (fd == NULL) {
        FTI_Print("FTI checkpoint file could not be opened.", FTI_EROR);
        return FTI_NSCS;
    }
    for (i = 0; i < FTI_Exec.nbVar; i++) {
        if (fwrite(FTI_Data[i].ptr, FTI_Data[i].eleSize, FTI_Data[i].count, fd) != FTI_Data[i].count) {
            sprintf(str, "Dataset #%d could not be written.", FTI_Data[i].id);
            FTI_Print(str, FTI_EROR);
            return FTI_NSCS;
        }
    }
    if (fflush(fd) != 0) {
        FTI_Print("FTI checkpoint file could not be flushed.", FTI_EROR);
        return FTI_NSCS;
    }
    if (fclose(fd) != 0) {
        FTI_Print("FTI checkpoint file could not be flushed.", FTI_EROR);
        return FTI_NSCS;
    }
    sprintf(str, "Time writing checkpoint file : %f seconds.", MPI_Wtime() - tt);
    FTI_Print(str, FTI_DBUG);
    int globalTmp = (FTI_Ckpt[4].isInline && FTI_Exec.ckptLvel == 4) ? 1 : 0;
    res = FTI_Try(FTI_CreateMetadata(globalTmp), "create metadata.");
    return res;
}

/*-------------------------------------------------------------------------*/
/**
    @brief      Decides wich action start depending on the ckpt. level.
    @param      level           Cleaning checkpoint level.
    @param      group           Must be groupID if App-proc. or 1 if Head.
    @param      pr              Must be 1 if App-proc. or nbApprocs if Head.
    @return     integer         FTI_SCES if successful.

    This function cleans the checkpoints of a group or a single process.
    It does that for each group (application process in the node) if executed
    by the head, or only locally if executed by an application process. The
    parameters pr determine if the for loops have 1 or number of App. procs.
    iterations. The group parameter help determine the groupID in both cases.

 **/
/*-------------------------------------------------------------------------*/
int FTI_GroupClean(int level, int group, int pr)
{
    int i, rank;
    if (level == 0) {
        FTI_Print("Error postprocessing checkpoint. Discarding checkpoint...", FTI_WARN);
    }
    rank = FTI_Topo.myRank;
    for (i = 0; i < pr; i++) {
        if (FTI_Topo.amIaHead)
            rank = FTI_Topo.body[i];
        FTI_Clean(level, i + group, rank);
    }
    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
    @brief      Decides wich action start depending on the ckpt. level.
    @param      group           Must be groupID if App-proc. or 1 if Head.
    @param      fo              Must be -1 if App-proc. or 0 if Head.
    @param      pr              Must be 1 if App-proc. or nbApprocs if Head.
    @return     integer         FTI_SCES if successful.

    This function launchs the required action dependeing on the ckpt. level.
    It does that for each group (application process in the node) if executed
    by the head, or only locally if executed by an application process. The
    parameters pr determine if the for loops have 1 or number of App. procs.
    iterations. The group parameter help determine the groupID in both cases.

 **/
/*-------------------------------------------------------------------------*/
int FTI_PostCkpt(int group, int fo, int pr)
{
    int i, tres, res, level, nodeFlag, globalFlag = FTI_Topo.splitRank;
    double t0, t1, t2, t3;
    char str[FTI_BUFS];
    t0 = MPI_Wtime();
    res = (FTI_Exec.ckptLvel == (FTI_REJW - FTI_BASE)) ? FTI_NSCS : FTI_SCES;
    MPI_Allreduce(&res, &tres, 1, MPI_INT, MPI_SUM, FTI_COMM_WORLD);
    if (tres != FTI_SCES) {
        FTI_GroupClean(0, group, pr);
        return FTI_NSCS;
    }
    t1 = MPI_Wtime();
    for (i = 0; i < pr; i++) {
        switch (FTI_Exec.ckptLvel) {
        case 4:
            res += FTI_Flush(i + group, fo);
            break;
        case 3:
            res += FTI_RSenc(i + group);
            break;
        case 2:
            res += FTI_Ptner(i + group);
            break;
        case 1:
            res += FTI_Local(i + group);
            break;
        }
    }
    MPI_Allreduce(&res, &tres, 1, MPI_INT, MPI_SUM, FTI_COMM_WORLD);
    if (tres != FTI_SCES) {
        FTI_GroupClean(0, group, pr);
        return FTI_NSCS;
    }
    t2 = MPI_Wtime();
    FTI_GroupClean(FTI_Exec.ckptLvel, group, pr);
    MPI_Barrier(FTI_COMM_WORLD);
    nodeFlag = (((!FTI_Topo.amIaHead) && (FTI_Topo.nodeRank == 0)) || (FTI_Topo.amIaHead)) ? 1 : 0;
    if (nodeFlag) {
        level = (FTI_Exec.ckptLvel != 4) ? FTI_Exec.ckptLvel : 1;
        rename(FTI_Conf.lTmpDir, FTI_Ckpt[level].dir);
        FTI_Print("Local directory renamed", FTI_DBUG);
    }
    if (!globalFlag) {
        if (FTI_Exec.ckptLvel == 4) {
            rename(FTI_Conf.gTmpDir, FTI_Ckpt[FTI_Exec.ckptLvel].dir);
        }
        rename(FTI_Conf.mTmpDir, FTI_Ckpt[FTI_Exec.ckptLvel].metaDir);
    }
    t3 = MPI_Wtime();
    sprintf(str, "Post-checkpoint took %.2f sec.", t3 - t0);
    sprintf(str, "%s (Ag:%.2fs, Pt:%.2fs, Cl:%.2fs)", str, t1 - t0, t2 - t1, t3 - t2);
    FTI_Print(str, FTI_INFO);

    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
    @brief      It listens for checkpoint notifications.
    @return     integer         FTI_SCES if successful.

    This function listens for notifications from the application processes
    and take the required actions after notification. This function is only
    executed by the head of the nodes and its complementary with the
    FTI_Checkpoint function in terms of communications.

 **/
/*-------------------------------------------------------------------------*/
int FTI_Listen()
{
    MPI_Status status;
    char str[FTI_BUFS];
    int i, buf, res, flags[7];
    for (i = 0; i < 7; i++) { // Initialize flags
        flags[i] = 0;
    }
    FTI_Print("Head listening...", FTI_DBUG);
    for (i = 0; i < FTI_Topo.nbApprocs; i++) { // Iterate on the application processes in the node
        MPI_Recv(&buf, 1, MPI_INT, FTI_Topo.body[i], FTI_Conf.tag, FTI_Exec.globalComm, &status);
        sprintf(str, "The head received a %d message", buf);
        FTI_Print(str, FTI_DBUG);
        fflush(stdout);
        flags[buf - FTI_BASE] = flags[buf - FTI_BASE] + 1;
    }
    for (i = 1; i < 7; i++) {
        if (flags[i] == FTI_Topo.nbApprocs) { // Determining checkpoint level
            FTI_Exec.ckptLvel = i;
        }
    }
    if (flags[6] > 0) {
        FTI_Exec.ckptLvel = 6;
    }
    if (FTI_Exec.ckptLvel == 5) { // If we were asked to finalize
        return FTI_ENDW;
    }
    res = FTI_Try(FTI_PostCkpt(1, 0, FTI_Topo.nbApprocs), "postprocess the checkpoint.");
    if (res == FTI_SCES) {
        FTI_Exec.wasLastOffline = 1;
        FTI_Exec.lastCkptLvel = FTI_Exec.ckptLvel;
        res = FTI_Exec.ckptLvel;
    }
    for (i = 0; i < FTI_Topo.nbApprocs; i++) { // Send msg. to avoid checkpoint collision
        MPI_Send(&res, 1, MPI_INT, FTI_Topo.body[i], FTI_Conf.tag, FTI_Exec.globalComm);
    }
    return FTI_SCES;
}
