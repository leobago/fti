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
  @param      group           The group ID.
  @return     integer         FTI_SCES.

  This function just returns FTI_SCES to have homogeneous code.

 **/
/*-------------------------------------------------------------------------*/
int FTI_Local(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
              FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt, int group)
{
    unsigned long maxFs, fs;
    FTI_Print("Starting checkpoint post-processing L1", FTI_DBUG);
    int res = FTI_Try(FTI_GetMeta(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt, &fs, &maxFs, group, 0), "obtain metadata.");
    if (res == FTI_NSCS) {
        return FTI_NSCS;
    }
    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      It copies ckpt. files in to the partner node.
  @param      group           The group ID.
  @return     integer         FTI_SCES if successful.

  This function copies the checkpoint files into the partner node. It
  follows a ring, where the ring size is the group size given in the FTI
  configuration file.

 **/
/*-------------------------------------------------------------------------*/
int FTI_Ptner(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
              FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt, int group)
{
    char *blBuf1, *blBuf2, lfn[FTI_BUFS], pfn[FTI_BUFS], str[FTI_BUFS];
    unsigned long maxFs, fs, ps, pos = 0;
    MPI_Request reqSend, reqRecv;
    FILE *lfd, *pfd;
    int res, dest, src, bSize = FTI_Conf->blockSize;
    MPI_Status status;

    FTI_Print("Starting checkpoint post-processing L2", FTI_DBUG);
    res = FTI_Try(FTI_GetMeta(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt, &fs, &maxFs, group, 0), "obtain metadata.");
    if (res == FTI_NSCS) {
        return FTI_NSCS;
    }
    ps = (maxFs / FTI_Conf->blockSize) * FTI_Conf->blockSize;
    if (ps < maxFs) {
        ps = ps + FTI_Conf->blockSize;
    }
    sprintf(str, "Max. file size %ld and padding size %ld.", maxFs, ps);
    FTI_Print(str, FTI_DBUG);

    sscanf(FTI_Exec->ckptFile, "Ckpt%d-Rank%d.fti", &FTI_Exec->ckptID, &src);
    sprintf(lfn, "%s/%s", FTI_Conf->lTmpDir, FTI_Exec->ckptFile);
    sprintf(pfn, "%s/Ckpt%d-Pcof%d.fti", FTI_Conf->lTmpDir, FTI_Exec->ckptID, src);

    sprintf(str, "L2 trying to access local ckpt. file (%s).", lfn);
    FTI_Print(str, FTI_DBUG);

    dest = FTI_Topo->right;
    src = FTI_Topo->left;

    lfd = fopen(lfn, "rb");
    if (lfd == NULL) {
        FTI_Print("FTI failed to open L2 chckpt. file.", FTI_DBUG);
        return FTI_NSCS;
    }

    pfd = fopen(pfn, "wb");
    if (pfd == NULL) {
        FTI_Print("FTI failed to open L2 partner file.", FTI_DBUG);
        fclose(lfd);
        return FTI_NSCS;
    }

    blBuf1 = talloc(char, FTI_Conf->blockSize);
    blBuf2 = talloc(char, FTI_Conf->blockSize);
    // Checkpoint files partner copy
    while (pos < ps) {
        if ((fs - pos) < FTI_Conf->blockSize) {
            bSize = fs - pos;
        }

        size_t bytes = fread(blBuf1, sizeof(char), bSize, lfd);
        if (ferror(lfd)) {
            FTI_Print("Error reading data from the L2 ckpt. file", FTI_DBUG);

            free(blBuf1);
            free(blBuf2);
            fclose(lfd);
            fclose(pfd);

            return FTI_NSCS;
        }
        sprintf(str, "groupID = %d, groupSize = %d, groupRank = %d, src = %d, dest = %d", FTI_Topo->groupID, FTI_Topo->groupSize, FTI_Topo->groupRank, src, dest);
        FTI_Print(str, FTI_DBUG);
        if (FTI_Topo->groupRank%2 == 0) {
            sprintf(str, "%d: sending to %d;", FTI_Topo->groupRank, dest);
            FTI_Print(str, FTI_DBUG);
            MPI_Send(blBuf1, bytes, MPI_CHAR, dest, FTI_Conf->tag, FTI_Exec->groupComm);

            sprintf(str, "%d: sent, w8ing for %d", FTI_Topo->groupRank, src);
            FTI_Print(str, FTI_DBUG);
            //MPI_Recv(blBuf2, FTI_Conf->blockSize, MPI_CHAR, src, FTI_Conf->tag, FTI_Exec->groupComm, MPI_STATUS_IGNORE);

                int number_amount;
                MPI_Status status;
                // Probe for an incoming message from process zero
                MPI_Probe(src, FTI_Conf->tag, FTI_Exec->groupComm, &status);

                // When probe returns, the status object has the size and other
                // attributes of the incoming message. Get the message size
                MPI_Get_count(&status, MPI_CHAR, &number_amount);

                // Now receive the message with the allocated buffer
                MPI_Recv(blBuf2, number_amount, MPI_CHAR, src, FTI_Conf->tag, FTI_Exec->groupComm, MPI_STATUS_IGNORE);
                sprintf(str, "%d: dynamically received %d numbers from %d (block = %d).\n", FTI_Topo->groupRank, number_amount, src, FTI_Conf->blockSize);
                FTI_Print(str, FTI_DBUG);

            sprintf(str, "%d: received from %d;", FTI_Topo->groupRank, src);
            FTI_Print(str, FTI_DBUG);
        }
        else {
            sprintf(str, "%d: w8ing for %d;", FTI_Topo->groupRank, src);
            FTI_Print(str, FTI_DBUG);
            //MPI_Recv(blBuf2, FTI_Conf->blockSize, MPI_CHAR, src, FTI_Conf->tag, FTI_Exec->groupComm, MPI_STATUS_IGNORE);

                int number_amount;
                MPI_Status status;
                // Probe for an incoming message from process zero
                MPI_Probe(src, FTI_Conf->tag, FTI_Exec->groupComm, &status);

                // When probe returns, the status object has the size and other
                // attributes of the incoming message. Get the message size
                MPI_Get_count(&status, MPI_CHAR, &number_amount);

                // Now receive the message with the allocated buffer
                MPI_Recv(blBuf2, number_amount, MPI_CHAR, src, FTI_Conf->tag, FTI_Exec->groupComm, MPI_STATUS_IGNORE);
                sprintf(str, "%d: dynamically received %d numbers from %d (block = %d).\n", FTI_Topo->groupRank, number_amount, src, FTI_Conf->blockSize);
                FTI_Print(str, FTI_DBUG);

            sprintf(str, "%d: received, sending to %d;", FTI_Topo->groupRank, dest);
            FTI_Print(str, FTI_DBUG);
            MPI_Send(blBuf1, bytes, MPI_CHAR, dest, FTI_Conf->tag, FTI_Exec->groupComm);

            sprintf(str, "%d: sent to %d;", FTI_Topo->groupRank, dest);
            FTI_Print(str, FTI_DBUG);
        }





        fwrite(blBuf2, sizeof(char), bSize, pfd);
        if (ferror(pfd)) {
            FTI_Print("Error writing data to the L2 partner file", FTI_DBUG);

            free(blBuf1);
            free(blBuf2);
            fclose(lfd);
            fclose(pfd);

            return FTI_NSCS;
        }

        pos = pos + FTI_Conf->blockSize;
    }

    free(blBuf1);
    free(blBuf2);
    fclose(lfd);
    fclose(pfd);

    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      It performs RS encoding with the ckpt. files in to the group.
  @param      group           The group ID.
  @return     integer         FTI_SCES if successful.

  This function performs the Reed-Solomon encoding for a given group. The
  checkpoint files are padded to the maximum size of the largest checkpoint
  file in the group +- the extra space to be a multiple of block size.

 **/
/*-------------------------------------------------------------------------*/
int FTI_RSenc(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
              FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt, int group)
{
    char *myData, *data, *coding, lfn[FTI_BUFS], efn[FTI_BUFS], str[FTI_BUFS];
    int *matrix, cnt, i, j, init, src, offset, dest, matVal, res, bs = FTI_Conf->blockSize;
    unsigned long maxFs, fs, ps, pos = 0;
    MPI_Request reqSend, reqRecv;
    MPI_Status status;
    int remBsize = bs;
    FILE *lfd, *efd;

    FTI_Print("Starting checkpoint post-processing L3", FTI_DBUG);
    res = FTI_Try(FTI_GetMeta(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt, &fs, &maxFs, group, 0), "obtain metadata.");
    if (res != FTI_SCES) {
        return FTI_NSCS;
    }
    ps = ((maxFs / bs)) * bs;
    if (ps < maxFs) {
        ps = ps + bs;
    }

    sscanf(FTI_Exec->ckptFile, "Ckpt%d-Rank%d.fti", &FTI_Exec->ckptID, &i);
    sprintf(lfn, "%s/%s", FTI_Conf->lTmpDir, FTI_Exec->ckptFile);
    sprintf(efn, "%s/Ckpt%d-RSed%d.fti", FTI_Conf->lTmpDir, FTI_Exec->ckptID, i);

    sprintf(str, "L3 trying to access local ckpt. file (%s).", lfn);
    FTI_Print(str, FTI_DBUG);

    lfd = fopen(lfn, "rb");
    if (lfd == NULL) {
        FTI_Print("FTI failed to open L3 checkpoint file.", FTI_EROR);
        return FTI_NSCS;
    }

    efd = fopen(efn, "wb");
    if (efd == NULL) {
        FTI_Print("FTI failed to open encoded ckpt. file.", FTI_EROR);

        fclose(lfd);

        return FTI_NSCS;
    }

    myData = talloc(char, bs);
    coding = talloc(char, bs);
    data = talloc(char, 2 * bs);
    matrix = talloc(int, FTI_Topo->groupSize* FTI_Topo->groupSize);

    for (i = 0; i < FTI_Topo->groupSize; i++) {
        for (j = 0; j < FTI_Topo->groupSize; j++) {
            matrix[i * FTI_Topo->groupSize + j] = galois_single_divide(1, i ^ (FTI_Topo->groupSize + j), FTI_Conf->l3WordSize);
        }
    }

    // For each block
    while (pos < ps) {
        if ((fs - pos) < bs) {
            remBsize = fs - pos;
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

        dest = FTI_Topo->groupRank;
        i = FTI_Topo->groupRank;
        offset = 0;
        init = 0;
        cnt = 0;

        // For each encoding
        while (cnt < FTI_Topo->groupSize) {
            if (cnt == 0) {
                memcpy(&(data[offset * bs]), myData, sizeof(char) * bytes);
            }
            else {
                MPI_Wait(&reqSend, &status);
                MPI_Wait(&reqRecv, &status);
            }

            // At every loop *but* the last one we send the data
            if (cnt != FTI_Topo->groupSize - 1) {
                dest = (dest + FTI_Topo->groupSize - 1) % FTI_Topo->groupSize;
                src = (i + 1) % FTI_Topo->groupSize;
                MPI_Isend(myData, bytes, MPI_CHAR, dest, FTI_Conf->tag, FTI_Exec->groupComm, &reqSend);
                MPI_Irecv(&(data[(1 - offset) * bs]), bs, MPI_CHAR, src, FTI_Conf->tag, FTI_Exec->groupComm, &reqRecv);
            }

            matVal = matrix[FTI_Topo->groupRank * FTI_Topo->groupSize + i];
            // First copy or xor any data that does not need to be multiplied by a factor
            if (matVal == 1) {
                if (init == 0) {
                    memcpy(coding, &(data[offset * bs]), bs);
                    init = 1;
                }
                else {
                    galois_region_xor(&(data[offset * bs]), coding, coding, bs);
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

    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      It flushes the local ckpt. files in to the PFS.
  @param      group           The group ID.
  @param      level           The level from which ckpt. files are flushed.
  @return     integer         FTI_SCES if successful.

  This function flushes the local checkpoint files in to the PFS.

 **/
/*-------------------------------------------------------------------------*/
int FTI_Flush(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
              FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt, int group, int level)
{
    char lfn[FTI_BUFS], gfn[FTI_BUFS], str[FTI_BUFS];
    unsigned long maxFs, fs, ps, pos = 0;
    FILE *lfd, *gfd;
    if (level == -1) {
        return FTI_SCES; // Fake call for inline PFS checkpoint
    }

    FTI_Print("Starting checkpoint post-processing L4", FTI_DBUG);
    int res = FTI_Try(FTI_GetMeta(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt, &fs, &maxFs, group, level), "obtain metadata.");
    if (res != FTI_SCES) {
        return FTI_NSCS;
    }

    if (mkdir(FTI_Conf->gTmpDir, 0777) == -1) {
        if (errno != EEXIST) {
            FTI_Print("Cannot create directory", FTI_EROR);
        }
    }

    ps = (maxFs / FTI_Conf->blockSize) * FTI_Conf->blockSize;
    if (ps < maxFs) {
        ps = ps + FTI_Conf->blockSize;
    }
    switch (level) {
        case 0:
            sprintf(lfn, "%s/%s", FTI_Conf->lTmpDir, FTI_Exec->ckptFile);
            break;
        case 1:
            sprintf(lfn, "%s/%s", FTI_Ckpt[1].dir, FTI_Exec->ckptFile);
            break;
        case 2:
            sprintf(lfn, "%s/%s", FTI_Ckpt[2].dir, FTI_Exec->ckptFile);
            break;
        case 3:
            sprintf(lfn, "%s/%s", FTI_Ckpt[3].dir, FTI_Exec->ckptFile);
            break;
    }

    // Open and resize files
    sprintf(gfn, "%s/%s", FTI_Conf->gTmpDir, FTI_Exec->ckptFile);
    sprintf(str, "L4 trying to access local ckpt. file (%s).", lfn);
    FTI_Print(str, FTI_DBUG);

    lfd = fopen(lfn, "rb");
    if (lfd == NULL) {
        FTI_Print("L4 cannot open the checkpoint file.", FTI_EROR);
        return FTI_NSCS;
    }

    gfd = fopen(gfn, "wb");
    if (gfd == NULL) {
        FTI_Print("L4 cannot open ckpt. file in the PFS.", FTI_EROR);

        fclose(lfd);

        return FTI_NSCS;
    }

    char *blBuf1 = talloc(char, FTI_Conf->blockSize);
    unsigned long bSize = FTI_Conf->blockSize;

    // Checkpoint files exchange
    while (pos < ps) {
        if ((fs - pos) < FTI_Conf->blockSize) {
            bSize = fs - pos;
        }

        size_t bytes = fread(blBuf1, sizeof(char), bSize, lfd);
        if (ferror(lfd)) {
            FTI_Print("L4 cannot read from the ckpt. file.", FTI_EROR);

            free(blBuf1);
            fclose(lfd);
            fclose(gfd);

            return FTI_NSCS;
        }

        fwrite(blBuf1, sizeof(char), bytes, gfd);
        if (ferror(gfd)) {
            FTI_Print("L4 cannot write to the ckpt. file in the PFS.", FTI_EROR);

            free(blBuf1);
            fclose(lfd);
            fclose(gfd);

            return FTI_NSCS;
        }

        pos = pos + FTI_Conf->blockSize;
    }

    free(blBuf1);
    fclose(lfd);
    fclose(gfd);

    return FTI_SCES;
}
