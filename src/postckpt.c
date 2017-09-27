/**
 *  @file   postckpt.c
 *  @author Leonardo A. Bautista Gomez (leobago@gmail.com)
 *  @date   December, 2013
 *  @brief  Post-checkpointing functions for the FTI library.
 */

#include "interface.h"

/*-------------------------------------------------------------------------*/
/**
  @brief      It returns FTI_SCES.
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Topo        Topology metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @return     integer         FTI_SCES.

  This function just returns FTI_SCES to have homogeneous code.

 **/
/*-------------------------------------------------------------------------*/
int FTI_Local(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
              FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt)
{
    FTI_Print("Starting checkpoint post-processing L1", FTI_DBUG);
    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      It sends Ckpt file.
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @param      destination     destination group rank
  @param      postFlag        0 if postckpt done by approc, > 0 if by head
  @return     integer         FTI_SCES if successful.

  This function sends ckpt file to partner process. Partner should call
  FTI_RecvPtner to receive this file.

 **/
/*-------------------------------------------------------------------------*/
int FTI_SendCkpt(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec, FTIT_checkpoint* FTI_Ckpt,
                 int destination, int postFlag)
{
    char lfn[FTI_BUFS], str[FTI_BUFS];
    sprintf(lfn, "%s/%s", FTI_Conf->lTmpDir, &FTI_Exec->meta[0].ckptFile[postFlag * FTI_BUFS]);

    //PostFlag is set to 0 if Post-processing is inline and set to processes nodeID if Post-processing done by head
    if (postFlag) {
        sprintf(str, "L2 trying to access process's %d ckpt. file (%s).", postFlag, lfn);
    }
    else {
        sprintf(str, "L2 trying to access local ckpt. file (%s).", lfn);
    }
    FTI_Print(str, FTI_DBUG);

    FILE* lfd = fopen(lfn, "rb");
    if (lfd == NULL) {
        FTI_Print("FTI failed to open L2 Ckpt. file.", FTI_DBUG);
        return FTI_NSCS;
    }

    char* buffer = talloc(char, FTI_Conf->blockSize);
    long toSend = FTI_Exec->meta[0].fs[postFlag]; //remaining data to send
    while (toSend > 0) {
        int sendSize = (toSend > FTI_Conf->blockSize) ? FTI_Conf->blockSize : toSend;
        int bytes = fread(buffer, sizeof(char), sendSize, lfd);

        if (ferror(lfd)) {
            FTI_Print("Error reading data from L2 ckpt file", FTI_DBUG);

            free(buffer);
            fclose(lfd);

            return FTI_NSCS;
        }

        MPI_Send(buffer, bytes, MPI_CHAR, destination, FTI_Conf->tag, FTI_Exec->groupComm);
        toSend -= bytes;
    }

    free(buffer);
    fclose(lfd);

    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      It receives Ptner file.
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @param      source          souce group rank
  @param      postFlag        0 if postckpt done by approc, > 0 if by head
  @return     integer         FTI_SCES if successful.

  This function receives ckpt file from partner process and saves it as
  Ptner file. Partner should call FTI_SendCkpt to send file.

 **/
/*-------------------------------------------------------------------------*/
int FTI_RecvPtner(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec, FTIT_checkpoint* FTI_Ckpt,
                  int source, int postFlag)
{
    //heads need to use ckptFile to get ckptID and rank
    int ckptID, rank;
    sscanf(&FTI_Exec->meta[0].ckptFile[postFlag * FTI_BUFS], "Ckpt%d-Rank%d.fti", &ckptID, &rank);

    char pfn[FTI_BUFS], str[FTI_BUFS];
    sprintf(pfn, "%s/Ckpt%d-Pcof%d.fti", FTI_Conf->lTmpDir, ckptID, rank);
    sprintf(str, "L2 trying to access Ptner file (%s).", pfn);
    FTI_Print(str, FTI_DBUG);

    FILE* pfd = fopen(pfn, "wb");
    if (pfd == NULL) {
        FTI_Print("FTI failed to open L2 ptner file.", FTI_DBUG);
        return FTI_NSCS;
    }

    char* buffer = talloc(char, FTI_Conf->blockSize);
    unsigned long toRecv = FTI_Exec->meta[0].pfs[postFlag]; //remaining data to receive
    while (toRecv > 0) {
        int recvSize = (toRecv > FTI_Conf->blockSize) ? FTI_Conf->blockSize : toRecv;
        MPI_Recv(buffer, recvSize, MPI_CHAR, source, FTI_Conf->tag, FTI_Exec->groupComm, MPI_STATUS_IGNORE);
        fwrite(buffer, sizeof(char), recvSize, pfd);

        if (ferror(pfd)) {
            FTI_Print("Error writing data to L2 ptner file", FTI_DBUG);

            free(buffer);
            fclose(pfd);

            return FTI_NSCS;
        }
        toRecv -= recvSize;
    }

    free(buffer);
    fclose(pfd);

    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      It copies ckpt. files in to the partner node.
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Topo        Topology metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @return     integer         FTI_SCES if successful.

  This function copies the checkpoint files into the partner node. It
  follows a ring, where the ring size is the group size given in the FTI
  configuration file.

 **/
/*-------------------------------------------------------------------------*/
int FTI_Ptner(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
              FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt)
{
    FTI_Print("Starting checkpoint post-processing L2", FTI_DBUG);
    if (FTI_Topo->amIaHead) {
        int res = FTI_Try(FTI_LoadTmpMeta(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt), "load temporary metadata.");
        if (res != FTI_SCES) {
            return FTI_NSCS;
        }
    }
    int startProc, endProc;
    if (FTI_Topo->amIaHead) { //post-processing for every process in the node
        startProc = 1;
        endProc = FTI_Topo->nodeSize;
    }
    else { //post-processing only for itself
        startProc = 0;
        endProc = 1;
    }

    int source = FTI_Topo->left; //receive Ckpt file from this process
    int destination = FTI_Topo->right; //send Ckpt file to this process
    int i;
    for (i = startProc; i < endProc; i++) {
        if (FTI_Topo->groupRank % 2) { //first send, then receive
            int res = FTI_SendCkpt(FTI_Conf, FTI_Exec, FTI_Ckpt, destination, i);
            if (res != FTI_SCES) {
                return FTI_NSCS;
            }
            res = FTI_RecvPtner(FTI_Conf, FTI_Exec, FTI_Ckpt, source, i);
            if (res != FTI_SCES) {
                return FTI_NSCS;
            }
        } else { //first receive, then send
            int res = FTI_RecvPtner(FTI_Conf, FTI_Exec, FTI_Ckpt, source, i);
            if (res != FTI_SCES) {
                return FTI_NSCS;
            }
            res = FTI_SendCkpt(FTI_Conf, FTI_Exec, FTI_Ckpt, destination, i);
            if (res != FTI_SCES) {
                return FTI_NSCS;
            }
        }
    }
    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      It performs RS encoding with the ckpt. files in to the group.
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Topo        Topology metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @return     integer         FTI_SCES if successful.

  This function performs the Reed-Solomon encoding for a given group. The
  checkpoint files are padded to the maximum size of the largest checkpoint
  file in the group +- the extra space to be a multiple of block size.

 **/
/*-------------------------------------------------------------------------*/
int FTI_RSenc(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
              FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt)
{
    FTI_Print("Starting checkpoint post-processing L3", FTI_DBUG);
    if (FTI_Topo->amIaHead) {
        int res = FTI_Try(FTI_LoadTmpMeta(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt), "load temporary metadata.");
        if (res != FTI_SCES) {
            return FTI_NSCS;
        }
    }

    int startProc, endProc;
    if (FTI_Topo->amIaHead) {
        startProc = 1;
        endProc = FTI_Topo->nodeSize;
    }
    else {
        startProc = 0;
        endProc = 1;
    }

    int proc;
    for (proc = startProc; proc < endProc; proc++) {
        int ckptID, rank;
        sscanf(&FTI_Exec->meta[0].ckptFile[proc * FTI_BUFS], "Ckpt%d-Rank%d.fti", &ckptID, &rank);
        char lfn[FTI_BUFS], efn[FTI_BUFS];

        sprintf(lfn, "%s/%s", FTI_Conf->lTmpDir, &FTI_Exec->meta[0].ckptFile[proc * FTI_BUFS]);
        sprintf(efn, "%s/Ckpt%d-RSed%d.fti", FTI_Conf->lTmpDir, ckptID, rank);

        char str[FTI_BUFS];
        sprintf(str, "L3 trying to access local ckpt. file (%s).", lfn);
        FTI_Print(str, FTI_DBUG);

        //all files in group must have the same size
        long maxFs = FTI_Exec->meta[0].maxFs[proc]; //max file size in group
        if (truncate(lfn, maxFs) == -1) {
            FTI_Print("Error with truncate on checkpoint file", FTI_WARN);
            return FTI_NSCS;
        }

        FILE* lfd = fopen(lfn, "rb");
        if (lfd == NULL) {
            FTI_Print("FTI failed to open L3 checkpoint file.", FTI_EROR);
            return FTI_NSCS;
        }

        FILE* efd = fopen(efn, "wb");
        if (efd == NULL) {
            FTI_Print("FTI failed to open encoded ckpt. file.", FTI_EROR);

            fclose(lfd);

            return FTI_NSCS;
        }

        int bs = FTI_Conf->blockSize;
        char* myData = talloc(char, bs);
        char* coding = talloc(char, bs);
        char* data = talloc(char, 2 * bs);
        int* matrix = talloc(int, FTI_Topo->groupSize* FTI_Topo->groupSize);

        int i;
        for (i = 0; i < FTI_Topo->groupSize; i++) {
            int j;
            for (j = 0; j < FTI_Topo->groupSize; j++) {
                matrix[i * FTI_Topo->groupSize + j] = galois_single_divide(1, i ^ (FTI_Topo->groupSize + j), FTI_Conf->l3WordSize);
            }
        }



        int remBsize = bs;
        long ps = ((maxFs / bs)) * bs;
        if (ps < maxFs) {
            ps = ps + bs;
        }

        // For each block
        long pos = 0;
        while (pos < ps) {
            if ((maxFs - pos) < bs) {
                remBsize = maxFs - pos;
            }

            // Reading checkpoint files
            size_t bytes = fread(myData, sizeof(char), remBsize, lfd);
            if (ferror(lfd)) {
                FTI_Print("FTI failed to read from L3 ckpt. file.", FTI_EROR);

                free(data);
                free(matrix);
                free(coding);
                free(myData);
                fclose(lfd);
                fclose(efd);

                return FTI_NSCS;
            }

            int dest = FTI_Topo->groupRank;
            i = FTI_Topo->groupRank;
            int offset = 0;
            int init = 0;
            int cnt = 0;

            // For each encoding
            MPI_Request reqSend, reqRecv; //used between iterations in while loop
            while (cnt < FTI_Topo->groupSize) {
                if (cnt == 0) {
                    memcpy(&(data[offset * bs]), myData, sizeof(char) * bytes);
                }
                else {
                    MPI_Wait(&reqSend, MPI_STATUS_IGNORE);
                    MPI_Wait(&reqRecv, MPI_STATUS_IGNORE);
                }

                // At every loop *but* the last one we send the data
                if (cnt != FTI_Topo->groupSize - 1) {
                    dest = (dest + FTI_Topo->groupSize - 1) % FTI_Topo->groupSize;
                    int src = (i + 1) % FTI_Topo->groupSize;
                    MPI_Isend(myData, bytes, MPI_CHAR, dest, FTI_Conf->tag, FTI_Exec->groupComm, &reqSend);
                    MPI_Irecv(&(data[(1 - offset) * bs]), bs, MPI_CHAR, src, FTI_Conf->tag, FTI_Exec->groupComm, &reqRecv);
                }

                int matVal = matrix[FTI_Topo->groupRank * FTI_Topo->groupSize + i];
                // First copy or xor any data that does not need to be multiplied by a factor
                if (matVal == 1) {
                    if (init == 0) {
                        memcpy(coding, &(data[offset * bs]), bs);
                        init = 1;
                    }
                    else {
                        galois_region_xor(&(data[offset * bs]), coding, bs);
                    }
                }

                // Then the data that needs to be multiplied by a factor
                if (matVal != 0 && matVal != 1) {
                    galois_w16_region_multiply(&(data[offset * bs]), matVal, bs, coding, init);
                    init = 1;
                }

                i = (i + 1) % FTI_Topo->groupSize;
                offset = 1 - offset;
                cnt++;
            }

            // Writting encoded checkpoints
            fwrite(coding, sizeof(char), remBsize, efd);

            // Next block
            pos = pos + bs;
        }

        free(data);
        free(matrix);
        free(coding);
        free(myData);
        fclose(lfd);
        fclose(efd);

        long fs = FTI_Exec->meta[0].fs[proc]; //ckpt file size
        if (truncate(lfn, fs) == -1) {
            FTI_Print("Error with re-truncate on checkpoint file", FTI_WARN);
            return FTI_NSCS;
        }

        //write checksum in metadata
        char checksum[MD5_DIGEST_LENGTH];
        int res = FTI_Checksum(efn, checksum);
        if (res != FTI_SCES) {
            return FTI_NSCS;
        }
        res = FTI_WriteRSedChecksum(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt, rank, checksum);
        if (res != FTI_SCES) {
            return FTI_NSCS;
        }
    }

    return FTI_SCES;
}


/*-------------------------------------------------------------------------*/
/**
  @brief      It flushes the local ckpt. files in to the PFS.
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Topo        Topology metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @param      level           The level from which ckpt. files are flushed.
  @return     integer         FTI_SCES if successful.

  This function flushes the local checkpoint files in to the PFS.

 **/
 /*-------------------------------------------------------------------------*/
int FTI_Flush(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
                FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt, int level)
{
    if (!FTI_Topo->amIaHead && level == 0) {
        return FTI_SCES; //inline L4 saves directly to PFS (nothing to flush)
    }
    char str[FTI_BUFS];
    sprintf(str, "Starting checkpoint post-processing L4 for level %d", level);
    FTI_Print(str, FTI_DBUG);
    // create global temp directory
    if (mkdir(FTI_Conf->gTmpDir, 0777) == -1) {
       if (errno != EEXIST) {
          FTI_Print("Cannot create global directory", FTI_EROR);
          return FTI_NSCS;
       }
    }
    int res = FTI_Try(FTI_LoadMeta(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt), "load metadata.");
    if (res != FTI_SCES) {
        return FTI_NSCS;
    }
    if (!FTI_Ckpt[4].isInline || FTI_Conf->ioMode == FTI_IO_POSIX) {
        //Just copy checkpoint files to PFS if L4 post-processing done by heads
        res = FTI_FlushPosix(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt, level);
    }
    else {
        switch(FTI_Conf->ioMode) {
            case FTI_IO_MPI:
                FTI_FlushMPI(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt, level);
                break;
#ifdef ENABLE_SIONLIB // --> If SIONlib is installed
            case FTI_IO_SIONLIB:
                FTI_FlushSionlib(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt, level);
                break;
#endif
        }
    }
    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
    @brief      It flushes the local ckpt. files in to the PFS using POSIX.
    @param      FTI_Conf        Configuration metadata.
    @param      FTI_Exec        Execution metadata.
    @param      FTI_Topo        Topology metadata.
    @param      FTI_Ckpt        Checkpoint metadata.
    @param      level           The level from which ckpt. files are flushed.
    @return     integer         FTI_SCES if successful.

    This function flushes the local checkpoint files in to the PFS.

 **/
/*-------------------------------------------------------------------------*/
int FTI_FlushPosix(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
                    FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt, int level)
{
    FTI_Print("Starting checkpoint post-processing L4 using Posix IO.", FTI_DBUG);
    int startProc, endProc, proc;
    if (FTI_Topo->amIaHead) {
        startProc = 1;
        endProc = FTI_Topo->nodeSize;
    }
    else {
        startProc = 0;
        endProc = 1;
    }

    for (proc = startProc; proc < endProc; proc++) {
        char str[FTI_BUFS];
        sprintf(str, "Post-processing for proc %d started.", proc);
        FTI_Print(str, FTI_DBUG);
        char lfn[FTI_BUFS], gfn[FTI_BUFS];
        sprintf(gfn, "%s/%s", FTI_Conf->gTmpDir, &FTI_Exec->meta[level].ckptFile[proc * FTI_BUFS]);
        sprintf(str, "Global temporary file name for proc %d: %s", proc, gfn);
        FTI_Print(str, FTI_DBUG);
        FILE* gfd = fopen(gfn, "wb");

        if (gfd == NULL) {
           FTI_Print("L4 cannot open ckpt. file in the PFS.", FTI_EROR);
           return FTI_NSCS;
        }

        if (level == 0) {
            sprintf(lfn, "%s/%s", FTI_Conf->lTmpDir, &FTI_Exec->meta[0].ckptFile[proc * FTI_BUFS]);
        }
        else {
            sprintf(lfn, "%s/%s", FTI_Ckpt[level].dir, &FTI_Exec->meta[level].ckptFile[proc * FTI_BUFS]);
        }
        sprintf(str, "Local file name for proc %d: %s", proc, lfn);
        FTI_Print(str, FTI_DBUG);
        // Open local file
        FILE* lfd = fopen(lfn, "rb");
        if (lfd == NULL) {
            FTI_Print("L4 cannot open the checkpoint file.", FTI_EROR);
            fclose(gfd);
            return FTI_NSCS;
        }

        char *readData = talloc(char, FTI_Conf->transferSize);
        long bSize = FTI_Conf->transferSize;
        long fs = FTI_Exec->meta[level].fs[proc];
        sprintf(str, "Local file size for proc %d: %ld", proc, fs);
        FTI_Print(str, FTI_DBUG);
        long pos = 0;
        // Checkpoint files exchange
        while (pos < fs) {
            if ((fs - pos) < FTI_Conf->transferSize)
              bSize = fs - pos;

            size_t bytes = fread(readData, sizeof(char), bSize, lfd);
            if (ferror(lfd)) {
                FTI_Print("L4 cannot read from the ckpt. file.", FTI_EROR);
                free(readData);
                fclose(lfd);
                fclose(gfd);
                return FTI_NSCS;
           }

            fwrite(readData, sizeof(char), bytes, gfd);
            if (ferror(gfd)) {
                FTI_Print("L4 cannot write to the ckpt. file in the PFS.", FTI_EROR);
                free(readData);
                fclose(lfd);
                fclose(gfd);
                return FTI_NSCS;
            }
            pos = pos + bytes;
        }
        free(readData);
        fclose(lfd);
        fclose(gfd);
    }
    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
    @brief      It flushes the local ckpt. files in to the PFS using MPI-I/O.
    @param      FTI_Conf        Configuration metadata.
    @param      FTI_Exec        Execution metadata.
    @param      FTI_Topo        Topology metadata.
    @param      FTI_Ckpt        Checkpoint metadata.
    @param      level           The level from which ckpt. files are flushed.
    @return     integer         FTI_SCES if successful.

    This function flushes the local checkpoint files in to the PFS.

 **/
/*-------------------------------------------------------------------------*/
int FTI_FlushMPI(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
                    FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt, int level)
{
    int res;
    FTI_Print("Starting checkpoint post-processing L4 using MPI-IO.", FTI_DBUG);
    // enable collective buffer optimization
    MPI_Info info;
    MPI_Info_create(&info);
    MPI_Info_set(info, "romio_cb_write", "enable");
    // TODO enable to set stripping unit in the config file (Maybe also other hints)
    // set stripping unit to 4MB
    MPI_Info_set(info, "stripping_unit", "4194304");

    // open parallel file (collective call)
    MPI_File pfh; // MPI-IO file handle
    char gfn[FTI_BUFS], lfn[FTI_BUFS], str[FTI_BUFS], ckptFile[FTI_BUFS];
    snprintf(ckptFile, FTI_BUFS, "Ckpt%d-mpiio.fti", FTI_Exec->ckptID);
    sprintf(gfn, "%s/%s", FTI_Conf->gTmpDir, ckptFile);
#ifdef LUSTRE
    if (FTI_Topo->splitRank == 0) {
        res = llapi_file_create(gfn, FTI_Conf->stripeUnit, FTI_Conf->stripeOffset, FTI_Conf->stripeFactor, 0);
        if (res) {
            char error_msg[FTI_BUFS];
            error_msg[0] = 0;
            strerror_r(-res, error_msg, FTI_BUFS);
            sprintf(str, "[Lustre] %s.", error_msg);
            FTI_Print(str, FTI_WARN);
        } else {
            snprintf(str, FTI_BUFS, "[LUSTRE] file:%s striping_unit:%i striping_factor:%i striping_offset:%i",
                    ckptFile, FTI_Conf->stripeUnit, FTI_Conf->stripeFactor, FTI_Conf->stripeOffset);
            FTI_Print(str, FTI_DBUG);
        }
    }
#endif
    res = MPI_File_open(FTI_COMM_WORLD, gfn, MPI_MODE_WRONLY|MPI_MODE_CREATE, info, &pfh);
    if (res != 0) {
       errno = 0;
       char mpi_err[FTI_BUFS];
       MPI_Error_string(res, mpi_err, NULL);
       snprintf(str, FTI_BUFS, "Unable to create file during MPI-IO flush [MPI ERROR - %i] %s", res, mpi_err);
       FTI_Print(str, FTI_EROR);
       MPI_Info_free(&info);
       return FTI_NSCS;
    }
    MPI_Info_free(&info);

    int proc, startProc, endProc;
    if (FTI_Topo->amIaHead) {
        startProc = 1;
        endProc = FTI_Topo->nodeSize;
    }
    else {
        startProc = 0;
        endProc = 1;
    }
    int nbProc = endProc - startProc;
    MPI_Offset* localFileSizes = talloc(MPI_Offset, nbProc);
    char* localFileNames = talloc(char, FTI_BUFS * endProc);
    int* splitRanks = talloc(int, endProc); //rank of process in FTI_COMM_WORLD
    for (proc = startProc; proc < endProc; proc++) {
        if (level == 0) {
            sprintf(&localFileNames[proc * FTI_BUFS], "%s/%s", FTI_Conf->lTmpDir, &FTI_Exec->meta[0].ckptFile[proc * FTI_BUFS]);
        }
        else {
            sprintf(&localFileNames[proc * FTI_BUFS], "%s/%s", FTI_Ckpt[level].dir, &FTI_Exec->meta[level].ckptFile[proc * FTI_BUFS]);
        }
        if (FTI_Topo->amIaHead) {
            splitRanks[proc] = (FTI_Topo->nodeSize - 1) * FTI_Topo->nodeID + proc - 1; //determine process splitRank if head
        }
        else {
            splitRanks[proc] = FTI_Topo->splitRank;
        }
        localFileSizes[proc - startProc] = FTI_Exec->meta[level].fs[proc]; //[proc - startProc] to get index from 0
    }

    MPI_Offset* allFileSizes = talloc(MPI_Offset, FTI_Topo->nbApprocs * FTI_Topo->nbNodes);
    MPI_Allgather(localFileSizes, nbProc, MPI_OFFSET, allFileSizes, nbProc, MPI_OFFSET, FTI_COMM_WORLD);
    free(localFileSizes);

    for (proc = startProc; proc < endProc; proc++) {
        MPI_Offset offset = 0;
        int i;
        for (i = 0; i < splitRanks[proc]; i++) {
           offset += allFileSizes[i];
        }

        FILE* lfd = fopen(&localFileNames[FTI_BUFS * proc], "rb");
        if (lfd == NULL) {
           FTI_Print("L4 cannot open the checkpoint file.", FTI_EROR);
           free(localFileNames);
           free(allFileSizes);
           free(splitRanks);
           return FTI_NSCS;
        }

        char* readData = talloc(char, FTI_Conf->transferSize);
        long bSize = FTI_Conf->transferSize;
        long fs = FTI_Exec->meta[level].fs[proc];

        long pos = 0;
        // Checkpoint files exchange
        while (pos < fs) {
            if ((fs - pos) < FTI_Conf->transferSize) {
                bSize = fs - pos;
            }

            size_t bytes = fread(readData, sizeof(char), bSize, lfd);
            if (ferror(lfd)) {
              FTI_Print("L4 cannot read from the ckpt. file.", FTI_EROR);
              free(localFileNames);
              free(allFileSizes);
              free(splitRanks);
              free(readData);
              fclose(lfd);
              MPI_File_close(&pfh);
              return FTI_NSCS;
            }
            MPI_Datatype dType;
            MPI_Type_contiguous(bytes, MPI_BYTE, &dType);
            MPI_Type_commit(&dType);

            res = MPI_File_write_at(pfh, offset, readData, 1, dType, MPI_STATUS_IGNORE);
            // check if successful
            if (res != 0) {
                errno = 0;
                char mpi_err[FTI_BUFS];
                MPI_Error_string(res, mpi_err, NULL);
                snprintf(str, FTI_BUFS, "Failed to write data to PFS during MPIIO Flush [MPI ERROR - %i] %s", res, mpi_err);
                FTI_Print(str, FTI_EROR);
                free(localFileNames);
                free(splitRanks);
                free(allFileSizes);
                fclose(lfd);
                MPI_File_close(&pfh);
                return FTI_NSCS;
            }
            MPI_Type_free(&dType);
            offset += bytes;
            pos = pos + bytes;
        }
        free(readData);
        fclose(lfd);
    }
    free(localFileNames);
    free(allFileSizes);
    free(splitRanks);
    MPI_File_close(&pfh);
    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
    @brief      It flushes the local ckpt. files in to the PFS using SIONlib.
    @param      FTI_Conf        Configuration metadata.
    @param      FTI_Exec        Execution metadata.
    @param      FTI_Topo        Topology metadata.
    @param      FTI_Ckpt        Checkpoint metadata.
    @param      level           The level from which ckpt. files are flushed.
    @return     integer         FTI_SCES if successful.

    This function flushes the local checkpoint files in to the PFS.

 **/
/*-------------------------------------------------------------------------*/
#ifdef ENABLE_SIONLIB // --> If SIONlib is installed
int FTI_FlushSionlib(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
      FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt, int level)
{
    int proc, startProc, endProc;
    if (FTI_Topo->amIaHead) {
        startProc = 1;
        endProc = FTI_Topo->nodeSize;
    }
    else {
        startProc = 0;
        endProc = 1;
    }
    int nbProc = endProc - startProc;

    long* localFileSizes = talloc(long, nbProc);
    char* localFileNames = talloc(char, FTI_BUFS * nbProc);
    int* splitRanks = talloc(int, nbProc); //rank of process in FTI_COMM_WORLD
    for (proc = startProc; proc < endProc; proc++) {
        // Open local file case 0:
        if (level == 0) {
            sprintf(&localFileNames[proc * FTI_BUFS], "%s/%s", FTI_Conf->lTmpDir, &FTI_Exec->meta[0].ckptFile[proc * FTI_BUFS]);
        }
        else {
            sprintf(&localFileNames[proc * FTI_BUFS], "%s/%s", FTI_Ckpt[level].dir, &FTI_Exec->meta[level].ckptFile[proc * FTI_BUFS]);
        }
        if (FTI_Topo->amIaHead) {
            splitRanks[proc - startProc] = (FTI_Topo->nodeSize - 1) * FTI_Topo->nodeID + proc - 1; //[proc - startProc] to get index from 0
        }
        else {
            splitRanks[proc - startProc] = FTI_Topo->splitRank; //[proc - startProc] to get index from 0
        }
        localFileSizes[proc - startProc] = FTI_Exec->meta[level].fs[proc]; //[proc - startProc] to get index from 0
    }

    int rank, ckptID;
    char fn[FTI_BUFS], str[FTI_BUFS];
    sscanf(&FTI_Exec->meta[level].ckptFile[0], "Ckpt%d-Rank%d.fti", &ckptID, &rank);
    snprintf(str, FTI_BUFS, "Ckpt%d-sionlib.fti", ckptID);
    sprintf(fn, "%s/%s", FTI_Conf->gTmpDir, str);

    int numFiles = 1;
    int nlocaltasks = nbProc;
    int* file_map = calloc(nbProc, sizeof(int));
    int* ranks = talloc(int, nbProc);
    int* rank_map = talloc(int, nbProc);
    sion_int64* chunkSizes = talloc(sion_int64, nbProc);
    int fsblksize = -1;
    int i;
    for (i = 0; i < nbProc; i++) {
        chunkSizes[i] = localFileSizes[i];
        ranks[i] = splitRanks[i];
        rank_map[i] = splitRanks[i];
    }
    int sid = sion_paropen_mapped_mpi(fn, "wb,posix", &numFiles, FTI_COMM_WORLD, &nlocaltasks, &ranks, &chunkSizes, &file_map, &rank_map, &fsblksize, NULL);
    if (sid == -1) {
       FTI_Print("Cannot open with sion_paropen_mapped_mpi.", FTI_EROR);

       free(file_map);
       free(ranks);
       free(rank_map);
       free(chunkSizes);

       return FTI_NSCS;
    }

    for (proc = startProc; proc < endProc; proc++) {
        FILE* lfd = fopen(&localFileNames[FTI_BUFS * proc], "rb");
        if (lfd == NULL) {
           FTI_Print("L4 cannot open the checkpoint file.", FTI_EROR);
           free(localFileNames);
           free(splitRanks);
           sion_parclose_mapped_mpi(sid);
           free(file_map);
           free(ranks);
           free(rank_map);
           free(chunkSizes);
           return FTI_NSCS;
        }


        int res = sion_seek(sid, splitRanks[proc - startProc], SION_CURRENT_BLK, SION_CURRENT_POS);
        if (res != SION_SUCCESS) {
            errno = 0;
            sprintf(str, "SIONlib: unable to set file pointer");
            FTI_Print(str, FTI_EROR);
            free(localFileNames);
            free(splitRanks);
            fclose(lfd);
            sion_parclose_mapped_mpi(sid);
            free(file_map);
            free(ranks);
            free(rank_map);
            free(chunkSizes);
            return FTI_NSCS;
        }

        char *readData = talloc(char, FTI_Conf->transferSize);
        long bSize = FTI_Conf->transferSize;
        long fs = FTI_Exec->meta[level].fs[proc];

        long pos = 0;
        // Checkpoint files exchange
        while (pos < fs) {
            if ((fs - pos) < FTI_Conf->transferSize)
                bSize = fs - pos;

            size_t bytes = fread(readData, sizeof(char), bSize, lfd);
            if (ferror(lfd)) {
                FTI_Print("L4 cannot read from the ckpt. file.", FTI_EROR);
                free(localFileNames);
                free(splitRanks);
                free(readData);
                fclose(lfd);
                sion_parclose_mapped_mpi(sid);
                free(file_map);
                free(ranks);
                free(rank_map);
                free(chunkSizes);
                return FTI_NSCS;
            }

            long data_written = sion_fwrite(readData, sizeof(char), bytes, sid);

            if (data_written < 0) {
                FTI_Print("Sionlib: could not write data", FTI_EROR);
                free(localFileNames);
                free(splitRanks);
                free(readData);
                fclose(lfd);
                sion_parclose_mapped_mpi(sid);
                free(file_map);
                free(ranks);
                free(rank_map);
                free(chunkSizes);
                return FTI_NSCS;
            }

            pos = pos + bytes;
        }
    }
    free(localFileNames);
    free(splitRanks);
    sion_parclose_mapped_mpi(sid);
    free(file_map);
    free(ranks);
    free(rank_map);
    free(chunkSizes);
}
#endif
