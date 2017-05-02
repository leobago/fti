/**
 *  @file   checkpoint.c
 *  @author Leonardo A. Bautista Gomez (leobago@gmail.com)
 *  @date   December, 2013
 *  @brief  Checkpointing functions for the FTI library.
 */

#define _POSIX_C_SOURCE 200809L
#include <string.h>

#include "interface.h"

/*-------------------------------------------------------------------------*/
/**
    @brief      It updates the local and global mean iteration time.
    @return     integer         FTI_SCES if successful.

    This function updates the local and global mean iteration time. It also
    recomputes the checkpoint interval in iterations and correct the next
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
            sprintf(str, "Current iter : %d ckpt iter. : %d . Next ckpt. at iter. %d",
                    FTI_Exec->ckptIcnt, FTI_Exec->ckptIntv, FTI_Exec->ckptNext);
            FTI_Print(str, FTI_DBUG);
            if (FTI_Exec->syncIter < (FTI_Exec->ckptIntv / 2)) {
                FTI_Exec->syncIter = FTI_Exec->syncIter * 2;
                sprintf(str, "Iteration frequency : %.2f sec/iter => %d iter/min. Resync every %d iter.",
                    FTI_Exec->globMeanIter, FTI_Exec->ckptIntv, FTI_Exec->syncIter);
                FTI_Print(str, FTI_DBUG);
            }
        }
    }
    FTI_Exec->ckptIcnt++; // Increment checkpoint loop counter
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
int FTI_WriteCkpt(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
                  FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt,
                  FTIT_dataset* FTI_Data)
{
    int i, res;
    char str[FTI_BUFS];

    double tt = MPI_Wtime();

    int globalTmp = (FTI_Ckpt[4].isInline && FTI_Exec->ckptLvel == 4) ? 1 : 0;
    
    if (globalTmp) {
        res = FTI_Try(FTI_WritePar(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Data),"Write checkpoint to PFS");
    }
    else {
        res = FTI_Try(FTI_WriteSer(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Data),"Write checkpoint to PFS");
    }
    
    sprintf(str, "Time writing checkpoint file : %f seconds.", MPI_Wtime() - tt);
    FTI_Print(str, FTI_DBUG);
    
    res = FTI_Try(FTI_CreateMetadata(FTI_Conf, FTI_Exec, FTI_Topo, globalTmp), "create metadata.");
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
int FTI_GroupClean(FTIT_configuration* FTI_Conf, FTIT_topology* FTI_Topo,
                   FTIT_checkpoint* FTI_Ckpt, int level, int group, int pr)
{
    int i, rank;
    if (level == 0) {
        FTI_Print("Error postprocessing checkpoint. Discarding checkpoint...", FTI_WARN);
    }
    rank = FTI_Topo->myRank;
    for (i = 0; i < pr; i++) {
        if (FTI_Topo->amIaHead) {
            rank = FTI_Topo->body[i];
        }
        FTI_Clean(FTI_Conf, FTI_Topo, FTI_Ckpt, level, i + group, rank);
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
int FTI_PostCkpt(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
                 FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt,
                 int group, int fo, int pr)
{
    int i, j, tres, res, level, nodeFlag, globalFlag = !FTI_Topo->splitRank, sid;
    double t0, t1, t2, t3;
    int save_sectorID, save_groupID, nbSectors;
    unsigned long fs, maxFs;
    sion_int64 chunksize;
    sion_int32 fsblksize = -1;
    char str[FTI_BUFS];
    char catstr[FTI_BUFS];

    t0 = MPI_Wtime();

    res = (FTI_Exec->ckptLvel == (FTI_REJW - FTI_BASE)) ? FTI_NSCS : FTI_SCES;
    MPI_Allreduce(&res, &tres, 1, MPI_INT, MPI_SUM, FTI_COMM_WORLD);
    if (tres != FTI_SCES) {
        FTI_GroupClean(FTI_Conf, FTI_Topo, FTI_Ckpt, 0, group, pr);
        return FTI_NSCS;
    }

    t1 = MPI_Wtime();
    
    if (FTI_Exec->ckptLvel == 4 && FTI_Topo->amIaHead && !FTI_Ckpt[4].isInline) {
        res = FTI_FlushInit(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt, fo);
    }
    
    MPI_Allreduce(&res, &tres, 1, MPI_INT, MPI_SUM, FTI_COMM_WORLD);
    if (tres != FTI_SCES) {
        FTI_GroupClean(FTI_Conf, FTI_Topo, FTI_Ckpt, 0, group, pr);
        return FTI_NSCS;
    }

    for (i = 0; i < pr; i++) {
        switch (FTI_Exec->ckptLvel) {
            case 4:
                res += FTI_Flush(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt, i + group, fo);
                break;
            case 3:
                res += FTI_RSenc(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt, i + group);
                break;
            case 2:
                res += FTI_Ptner(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt, i + group);
                break;
            case 1:
                res += FTI_Local(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt, i + group);
                break;
        }
    }

    if (FTI_Exec->ckptLvel == 4 && FTI_Topo->amIaHead && !FTI_Ckpt[4].isInline) {
        res += FTI_FlushFinalize(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt);
    }

    MPI_Allreduce(&res, &tres, 1, MPI_INT, MPI_SUM, FTI_COMM_WORLD);
    if (tres != FTI_SCES) {
        FTI_GroupClean(FTI_Conf, FTI_Topo, FTI_Ckpt, 0, group, pr);
        return FTI_NSCS;
    }

    t2 = MPI_Wtime();

    FTI_GroupClean(FTI_Conf, FTI_Topo, FTI_Ckpt, FTI_Exec->ckptLvel, group, pr);
    nodeFlag = (((!FTI_Topo->amIaHead) && ((FTI_Topo->nodeRank - FTI_Topo->nbHeads) == 0)) || (FTI_Topo->amIaHead)) ? 1 : 0;
    if (nodeFlag) {
        //Debug message needed to test nodeFlag (./tests/nodeFlag/nodeFlag.c)
        sprintf(str, "Has nodeFlag = 1 and nodeID = %d. CkptLvel = %d.", FTI_Topo->nodeID, FTI_Exec->ckptLvel);
        FTI_Print(str, FTI_DBUG);
        if (!(FTI_Ckpt[4].isInline && FTI_Exec->ckptLvel == 4)) {
            level = (FTI_Exec->ckptLvel != 4) ? FTI_Exec->ckptLvel : 1;
            if (rename(FTI_Conf->lTmpDir, FTI_Ckpt[level].dir) == -1) {
                FTI_Print("Cannot rename local directory", FTI_EROR);
            }
            else {
                FTI_Print("Local directory renamed", FTI_DBUG);
            }
        }
    }
    if (globalFlag) {
        if (FTI_Exec->ckptLvel == 4) {
            if (rename(FTI_Conf->gTmpDir, FTI_Ckpt[FTI_Exec->ckptLvel].dir) == -1) {
                FTI_Print("Cannot rename global directory", FTI_EROR);
            }
        }
        if (rename(FTI_Conf->mTmpDir, FTI_Ckpt[FTI_Exec->ckptLvel].metaDir) == -1) {
            FTI_Print("Cannot rename meta directory", FTI_EROR);
        }
    }
    MPI_Barrier(FTI_COMM_WORLD);

    t3 = MPI_Wtime();

    sprintf(catstr, "Post-checkpoint took %.2f sec.", t3 - t0);
    sprintf(str, "%s (Ag:%.2fs, Pt:%.2fs, Cl:%.2fs)", catstr, t1 - t0, t2 - t1, t3 - t2);
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
int FTI_Listen(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
               FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt)
{
    MPI_Status status;
    char str[FTI_BUFS];
    int i, buf, res, flags[7];
    for (i = 0; i < 7; i++) { // Initialize flags
        flags[i] = 0;
    }
    FTI_Print("Head listening...", FTI_DBUG);
    for (i = 0; i < FTI_Topo->nbApprocs; i++) { // Iterate on the application processes in the node
        MPI_Recv(&buf, 1, MPI_INT, FTI_Topo->body[i], FTI_Conf->tag, FTI_Exec->globalComm, &status);
        sprintf(str, "The head received a %d message", buf);
        FTI_Print(str, FTI_DBUG);
        fflush(stdout);
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
        return FTI_ENDW;
    }
    //MPI_Barrier(FTI_COMM_WORLD);
    //exit(0);
    res = FTI_Try(FTI_PostCkpt(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt, 1, 0, FTI_Topo->nbApprocs), "postprocess the checkpoint.");
    if (res == FTI_SCES) {
        FTI_Exec->wasLastOffline = 1;
        FTI_Exec->lastCkptLvel = FTI_Exec->ckptLvel;
        res = FTI_Exec->ckptLvel;
    }
    for (i = 0; i < FTI_Topo->nbApprocs; i++) { // Send msg. to avoid checkpoint collision
        MPI_Send(&res, 1, MPI_INT, FTI_Topo->body[i], FTI_Conf->tag, FTI_Exec->globalComm);
    }
    return FTI_SCES;
}

int FTI_WritePosix(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,FTIT_topology* FTI_Topo,FTIT_dataset* FTI_Data)
{
    return FTI_SCES;
}

int FTI_WriteMpi(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,FTIT_topology* FTI_Topo,FTIT_dataset* FTI_Data)
{
    int i, dCount, res, reslen;
    char str[FTI_BUFS], mpi_err[FTI_BUFS];
    void *dataOffset;
    MPI_Datatype maxInt, dType, rType;
    MPI_Offset chunkSize = 0, *chunkSizes;
    MPI_Offset pfSector = 0, pfSize = 0, dSize, rSize;
    MPI_Info info;
    MPI_Status status;
    MPI_File pfh;

    // enable collective buffer optimization
    MPI_Info_create(&info);
    MPI_Info_set(info, "romio_cb_write", "enable");
        
    // set stripping unit to 4MB
    MPI_Info_set(info, "stripping_unit", "4194304");
    
        
    // determine own chunksize
    for (i=0; i < FTI_Exec->nbVar; i++) {
        chunkSize += FTI_Data[i].size;
    }

    // collect chunksizes of other ranks
    chunkSizes = talloc(MPI_Offset, FTI_Topo->nbApprocs*FTI_Topo->nbNodes);
    MPI_Allgather(&chunkSize, 1, MPI_OFFSET, chunkSizes, 1, MPI_OFFSET, FTI_COMM_WORLD);
   
    // determine parallel file size
    for(i=0; i<FTI_Topo->nbApprocs*FTI_Topo->nbNodes; i++) {
        pfSize += chunkSizes[i];
    }

    // open parallel file (collective call)
    res = MPI_File_open(FTI_COMM_WORLD, FTI_Exec->fn, MPI_MODE_WRONLY|MPI_MODE_CREATE, info, &pfh); 
     
    // check if successfull
    if (res != 0) {
        MPI_Error_string(res, mpi_err, &reslen);
        snprintf(str, FTI_BUFS, "unable to create file [MPI ERROR - %i] %s", res, mpi_err);
        FTI_Print(str, FTI_EROR);
        MPI_File_close(&pfh);
        return FTI_NSCS;
    }
    
    //set parallel file size (collective)
    res = MPI_File_set_size(pfh, pfSize);

    // set file offset
    for (i=0; i<FTI_Topo->splitRank; i++) {
        pfSector += chunkSizes[i];
    }
    
    for (i=0; i < FTI_Exec->nbVar; i++) {
        
        // determine data types
        dSize = FTI_Data[i].size;                       // check if > 1 MPI stypes needed
        dCount = ((int)dSize)/INT_MAX;                  // determine # chunks of datatype
        dataOffset = FTI_Data[i].ptr;                   // create new reference ptr
        rSize = dSize%INT_MAX; 
        
        // if size of data > INT_MAX
        if ( dCount > 1 ) {
            
            // create MPI data type
            MPI_Type_contiguous(INT_MAX, MPI_BYTE, &dType); 
            MPI_Type_commit(&dType);

            for (i = 0; i < dCount; i++) {
                
                // write ckpt data to file
                res = MPI_File_write_at(pfh, pfSector, dataOffset, dCount, dType, &status);
                pfSector += INT_MAX;                    // increment file pointer
                dataOffset += INT_MAX;                  // increment data pointer

                // check if successfull
                if (res != 0) {
                    MPI_Error_string(res, mpi_err, &reslen);
                    snprintf(str, FTI_BUFS, "unable to write ckpt data [MPI ERROR - %i] %s", res, mpi_err);
                    FTI_Print(str, FTI_EROR);
                    MPI_File_close(&pfh);
                    return FTI_NSCS;
                }

            }
            
            // writing modulo of data size and INT_MAX
            if ( rSize > 0 ) {

                // create MPI data type
                MPI_Type_contiguous(rSize, MPI_BYTE, &rType);
                
                // write ckpt data to file
                res = MPI_File_write_at(pfh, pfSector, dataOffset, dCount, dType, &status);
                pfSector += rSize;                      // increment file pointer

                // check if successfull
                if (res != 0) {
                    MPI_Error_string(res, mpi_err, &reslen);
                    snprintf(str, FTI_BUFS, "unable to write ckpt data [MPI ERROR - %i] %s", res, mpi_err);
                    FTI_Print(str, FTI_EROR);
                    MPI_File_close(&pfh);
                    return FTI_NSCS;
                }

            }
        } else {
            // create MPI data type
            MPI_Type_contiguous(dSize, MPI_BYTE, &dType); 
            MPI_Type_commit(&dType);
                
            // write ckpt data to file
            res = MPI_File_write_at(pfh, pfSector, FTI_Data[i].ptr, 1, dType, &status);
            pfSector += dSize;                         // increment file pointer

            // check if successfull
            if (res != 0) {
                MPI_Error_string(res, mpi_err, &reslen);
                snprintf(str, FTI_BUFS, "unable to write ckpt data [MPI ERROR - %i] %s", res, mpi_err);
                FTI_Print(str, FTI_EROR);
                MPI_File_close(&pfh);
                return FTI_NSCS;
            }

        }

        MPI_Type_free(&dType);
        if (dCount > 0 && rSize > 0) { 
            MPI_Type_free(&rType); 
        }

    }

    // This barrier is actually not supposed to be here since close call collective...
    MPI_Barrier(FTI_COMM_WORLD);
    MPI_File_close(&pfh);
    //MPI_Info_free(&info);
    return FTI_SCES;
}

int FTI_WriteSionlib(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,FTIT_topology* FTI_Topo,FTIT_dataset* FTI_Data)
{

    int i, res, numFiles = 1, nlocaltasks, *file_map, *rank_map, *gRankList;
    sion_int64 *chunkSizes;
    sion_int32 fsblksize=-1;
    size_t eleSize;
    size_t ckptSize = 0;
    
    // determine chunksize
    for (i=0; i < FTI_Exec->nbVar; i++) {
        ckptSize += FTI_Data[i].size;
    }

    // set parameter for paropen mapped call
    if (FTI_Topo->nbHeads == 1 && FTI_Topo->groupID == 1) {

        // for the case that we have a head
        nlocaltasks = 2;
        gRankList = talloc(int, 2);
        chunkSizes = talloc(sion_int64, 2);
        file_map = talloc(int, 2);
        rank_map = talloc(int, 2);

        chunkSizes[0] = 0;
        chunkSizes[1] = ckptSize;
        gRankList[0] = FTI_Topo->headRank;
        gRankList[1] = FTI_Topo->myRank;
        file_map[0] = 0;
        file_map[1] = 0;
        rank_map[0] = gRankList[0];
        rank_map[1] = gRankList[1];
    
    } else {
    
        nlocaltasks = 1;
        gRankList = talloc(int, 1);
        chunkSizes = talloc(sion_int64, 1);
        file_map = talloc(int, 1);
        rank_map = talloc(int, 1);

        *chunkSizes = ckptSize;
        *gRankList = FTI_Topo->myRank;
        *file_map = 0;
        *rank_map = *gRankList;
    
    }

    // set chunksize for the metadata creation
    FTI_Exec->meta[0].fs = (size_t) ckptSize;
    
    // open parallel file
    FTI_Exec->sid = sion_paropen_mapped_mpi(FTI_Exec->fn, "wb,posix", &numFiles, FTI_COMM_WORLD, &nlocaltasks, &gRankList, &chunkSizes, &file_map, &rank_map, &fsblksize, NULL); 
    
    // check if successful
    if (FTI_Exec->sid==-1) {
        errno = 0;
        FTI_Print("SIONlib: File could no be opened", FTI_EROR);
        return FTI_NSCS;
    }
        
    // set file pointer to corresponding block in sionlib file
    res = sion_seek(FTI_Exec->sid, FTI_Topo->myRank, SION_CURRENT_BLK, SION_CURRENT_POS);
		
    // check if successful
    if (res != SION_SUCCESS) {
        errno = 0;
        FTI_Print("SIONlib: Could not set file pointer", FTI_EROR);
        return FTI_NSCS;
    }
    
    // write datasets into file
    for (i=0; i < FTI_Exec->nbVar; i++) {
        eleSize = FTI_Data[i].size;
    
        // SIONlib write call
        res = sion_fwrite(FTI_Data[i].ptr, eleSize, 1, FTI_Exec->sid);
		
        // check if successful
        if (res < 0) {
            errno = 0;
            FTI_Print("SIONlib: Data could not be written", FTI_EROR);
            return FTI_NSCS;
        }
    }

    // close parallel file
    res =  sion_parclose_mapped_mpi(FTI_Exec->sid);
    
    // check if successful
    if (res<0) {
        errno = 0;
        FTI_Print("SIONlib: Parclose mapped error", FTI_EROR);
        return FTI_NSCS;
     }

    return FTI_SCES;

}

int FTI_WritePar(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,FTIT_topology* FTI_Topo,FTIT_dataset* FTI_Data)
{
    
    int res;
    
    // create global temp directory
    if (mkdir(FTI_Conf->gTmpDir, 0777) == -1) {
        if (errno != EEXIST) {
            FTI_Print("Cannot create global directory", FTI_EROR);
            return FTI_NSCS;
        }
    }
    
    // select IO
    switch(FTI_Conf->ioMode) {

        case FTI_IO_POSIX:

            res = FTI_WritePosix(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Data);
            break;
        
        case FTI_IO_MPI:
            
            // set parallel file name
            snprintf(FTI_Exec->ckptFile, FTI_BUFS, "Ckpt%d-mpiio.fti", FTI_Exec->ckptID);
            sprintf(FTI_Exec->fn, "%s/%s", FTI_Conf->gTmpDir, FTI_Exec->ckptFile);

            res = FTI_WriteMpi(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Data);
            break;

        case FTI_IO_SIONLIB:

            // set parallel file name
            snprintf(FTI_Exec->ckptFile, FTI_BUFS, "Ckpt%d-sionlib.fti", FTI_Exec->ckptID);
            sprintf(FTI_Exec->fn, "%s/%s", FTI_Conf->gTmpDir, FTI_Exec->ckptFile);

            // write checkpoint
            res = FTI_WriteSionlib(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Data);
            break;

    }

    return res;

}

int FTI_WriteSer(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,FTIT_topology* FTI_Topo,FTIT_dataset* FTI_Data)
{

    int i;
    char fn[FTI_BUFS], str[FTI_BUFS];
    size_t glbWritten = 0; // checkpointsize
    FILE *fd;
    
    // create local temp directory
    snprintf(FTI_Exec->ckptFile, FTI_BUFS, "Ckpt%d-Rank%d.fti", FTI_Exec->ckptID, FTI_Topo->myRank);
    sprintf(fn, "%s/%s", FTI_Conf->lTmpDir, FTI_Exec->ckptFile);
    if (mkdir(FTI_Conf->lTmpDir, 0777) == -1) {
        if (errno != EEXIST) {
            FTI_Print("Cannot create local directory", FTI_EROR);
            return FTI_NSCS;
        }
    }

    // open task local ckpt file
    fd = fopen(fn, "wb");
    if (fd == NULL) {
        FTI_Print("FTI checkpoint file could not be opened.", FTI_EROR);

        return FTI_NSCS;
    }

    // write data into ckpt file
    for (i = 0; i < FTI_Exec->nbVar; i++) {
        clearerr(fd);
        size_t written = 0;
        int fwrite_errno;
        while ( written < FTI_Data[i].count && !ferror(fd) ) {
            errno = 0;
            written += fwrite(((char*)FTI_Data[i].ptr)+(FTI_Data[i].eleSize*written), FTI_Data[i].eleSize, FTI_Data[i].count-written, fd);
            fwrite_errno = errno;
        }
        if ( ferror(fd) ) {
            char error_msg[FTI_BUFS];
            error_msg[0] = 0;
            strerror_r(fwrite_errno, error_msg, FTI_BUFS);
            sprintf(str, "Dataset #%d could not be written: %s.", FTI_Data[i].id, error_msg);
            FTI_Print(str, FTI_EROR);
            fclose(fd);
            return FTI_NSCS;
        }
        glbWritten += written*FTI_Data[i].eleSize;
    }
    
    // store file size in runtime meta data
    FTI_Exec->meta[0].fs = glbWritten;
    
    // flush buffer
    if (fflush(fd) != 0) {
        FTI_Print("FTI checkpoint file could not be flushed.", FTI_EROR);

        fclose(fd);

        return FTI_NSCS;
    }
    
    // close file
    if (fclose(fd) != 0) {
        FTI_Print("FTI checkpoint file could not be flushed.", FTI_EROR);

        return FTI_NSCS;
    }
    
    return FTI_SCES;

}

