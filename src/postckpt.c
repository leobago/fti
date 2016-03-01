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
int FTI_Local(int group)
{
    unsigned long maxFs, fs;
    FTI_Print("Starting checkpoint post-processing L1", FTI_DBUG);
    int res = FTI_Try(FTI_GetMeta(&fs, &maxFs, group, 0), "obtain metadata.");
    if (res == FTI_NSCS)
        return FTI_NSCS;
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
int FTI_Ptner(int group)
{
    char *blBuf1, *blBuf2, lfn[FTI_BUFS], pfn[FTI_BUFS], str[FTI_BUFS];
    unsigned long maxFs, fs, ps, pos = 0;
    MPI_Request reqSend, reqRecv;
    FILE *lfd, *pfd;
    int res, dest, src, bSize = FTI_Conf.blockSize;
    MPI_Status status;

    FTI_Print("Starting checkpoint post-processing L2", FTI_DBUG);
    res = FTI_Try(FTI_GetMeta(&fs, &maxFs, group, 0), "obtain metadata.");
    if (res == FTI_NSCS)
        return FTI_NSCS;
    ps = (maxFs / FTI_Conf.blockSize) * FTI_Conf.blockSize;
    if (ps < maxFs)
        ps = ps + FTI_Conf.blockSize;
    sprintf(str, "Max. file size %ld and padding size %ld.", maxFs, ps);
    FTI_Print(str, FTI_DBUG);

    sscanf(FTI_Exec.ckptFile, "Ckpt%d-Rank%d.fti", &FTI_Exec.ckptID, &src);
    sprintf(lfn, "%s/%s", FTI_Conf.lTmpDir, FTI_Exec.ckptFile);
    sprintf(pfn, "%s/Ckpt%d-Pcof%d.fti", FTI_Conf.lTmpDir, FTI_Exec.ckptID, src);

    sprintf(str, "L2 trying to access local ckpt. file (%s).", lfn);
    FTI_Print(str, FTI_DBUG);
    res = FTI_Try(access(lfn, R_OK), " access the L2 checkpoint file.");
    if (res == FTI_NSCS)
        return FTI_NSCS;
    dest = FTI_Topo.right;
    src = FTI_Topo.left;

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

    blBuf1 = talloc(char, FTI_Conf.blockSize);
    blBuf2 = talloc(char, FTI_Conf.blockSize);
    // Checkpoint files partner copy
    while (pos < ps) {
        if ((fs - pos) < FTI_Conf.blockSize)
            bSize = fs - pos;

        (void)fread(blBuf1, sizeof(char), bSize, lfd);
        if (ferror(lfd)) {
            FTI_Print("Error reading data from the L2 ckpt. file", FTI_DBUG);

            free(blBuf1);
            free(blBuf2);
            fclose(lfd);
            fclose(pfd);

            return FTI_NSCS;
        }

        MPI_Isend(blBuf1, FTI_Conf.blockSize, MPI_CHAR, dest, FTI_Conf.tag, FTI_Exec.groupComm, &reqSend);
        MPI_Irecv(blBuf2, FTI_Conf.blockSize, MPI_CHAR, src, FTI_Conf.tag, FTI_Exec.groupComm, &reqRecv);
        MPI_Wait(&reqSend, &status);
        MPI_Wait(&reqRecv, &status);

        fwrite(blBuf2, sizeof(char), bSize, pfd);
        if (ferror(pfd)) {
            FTI_Print("Error writing data to the L2 partner file", FTI_DBUG);

            free(blBuf1);
            free(blBuf2);
            fclose(lfd);
            fclose(pfd);

            return FTI_NSCS;
        }

        pos = pos + FTI_Conf.blockSize;
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
int FTI_RSenc(int group)
{
    char *myData, *data, *coding, lfn[FTI_BUFS], efn[FTI_BUFS], str[FTI_BUFS];
    int *matrix, cnt, i, j, init, src, offset, dest, matVal, res, bs = FTI_Conf.blockSize;
    unsigned long maxFs, fs, ps, pos = 0;
    MPI_Request reqSend, reqRecv;
    MPI_Status status;
    int remBsize = bs;
    FILE *lfd, *efd;

    FTI_Print("Starting checkpoint post-processing L3", FTI_DBUG);
    res = FTI_Try(FTI_GetMeta(&fs, &maxFs, group, 0), "obtain metadata.");
    if (res != FTI_SCES)
        return FTI_NSCS;
    ps = ((maxFs / bs)) * bs;
    if (ps < maxFs)
        ps = ps + bs;

    sscanf(FTI_Exec.ckptFile, "Ckpt%d-Rank%d.fti", &FTI_Exec.ckptID, &i);
    sprintf(lfn, "%s/%s", FTI_Conf.lTmpDir, FTI_Exec.ckptFile);
    sprintf(efn, "%s/Ckpt%d-RSed%d.fti", FTI_Conf.lTmpDir, FTI_Exec.ckptID, i);
    sprintf(str, "L3 trying to access local ckpt. file (%s).", lfn);
    FTI_Print(str, FTI_DBUG);
    res = FTI_Try(access(lfn, R_OK), "access the L3 checkpoint file.");
    if (res != FTI_SCES)
        return FTI_NSCS;

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
    matrix = talloc(int, FTI_Topo.groupSize* FTI_Topo.groupSize);

    for (i = 0; i < FTI_Topo.groupSize; i++) {
        for (j = 0; j < FTI_Topo.groupSize; j++) {
            matrix[i * FTI_Topo.groupSize + j] = galois_single_divide(1, i ^ (FTI_Topo.groupSize + j), FTI_Conf.l3WordSize);
        }
    }

    while (pos < ps) { // For each block
        if ((fs - pos) < bs)
            remBsize = fs - pos;
        fread(myData, sizeof(char), remBsize, lfd); // Reading checkpoint files
        dest = FTI_Topo.groupRank;
        i = FTI_Topo.groupRank;
        offset = 0;
        init = 0;
        cnt = 0;
        while (cnt < FTI_Topo.groupSize) { // For each encoding
            if (cnt == 0) {
                memcpy(&(data[offset * bs]), myData, sizeof(char) * bs);
            }
            else {
                MPI_Wait(&reqSend, &status);
                MPI_Wait(&reqRecv, &status);
            }
            if (cnt != FTI_Topo.groupSize - 1) { // At every loop *but* the last one we send the data
                dest = (dest + FTI_Topo.groupSize - 1) % FTI_Topo.groupSize;
                src = (i + 1) % FTI_Topo.groupSize;
                MPI_Isend(myData, bs, MPI_CHAR, dest, FTI_Conf.tag, FTI_Exec.groupComm, &reqSend);
                MPI_Irecv(&(data[(1 - offset) * bs]), bs, MPI_CHAR, src, FTI_Conf.tag, FTI_Exec.groupComm, &reqRecv);
            }
            matVal = matrix[FTI_Topo.groupRank * FTI_Topo.groupSize + i];
            if (matVal == 1) { // First copy or xor any data that does not need to be multiplied by a factor
                if (init == 0) {
                    memcpy(coding, &(data[offset * bs]), bs);
                    init = 1;
                }
                else {
                    galois_region_xor(&(data[offset * bs]), coding, coding, bs);
                }
            }
            if (matVal != 0 && matVal != 1) { // Then the data that needs to be multiplied by a factor
                galois_w16_region_multiply(&(data[offset * bs]), matVal, bs, coding, init);
                init = 1;
            }
            i = (i + 1) % FTI_Topo.groupSize;
            offset = 1 - offset;
            cnt++;
        }
        fwrite(coding, sizeof(char), remBsize, efd); // Writting encoded checkpoints
        pos = pos + bs; // Next block
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
int FTI_Flush(int group, int level)
{
    char lfn[FTI_BUFS], gfn[FTI_BUFS], str[FTI_BUFS];
    unsigned long maxFs, fs, ps, pos = 0;
    FILE *lfd, *gfd;
    if (level == -1)
        return FTI_SCES; // Fake call for inline PFS checkpoint

    FTI_Print("Starting checkpoint post-processing L4", FTI_DBUG);
    int res = FTI_Try(FTI_GetMeta(&fs, &maxFs, group, level), "obtain metadata.");
    if (res != FTI_SCES)
        return FTI_NSCS;

    if (access(FTI_Conf.gTmpDir, F_OK) != 0) {
        if (mkdir(FTI_Conf.gTmpDir, 0777) == -1)
            FTI_Print("Cannot create directory", FTI_EROR);
    }

    ps = (maxFs / FTI_Conf.blockSize) * FTI_Conf.blockSize;
    if (ps < maxFs)
        ps = ps + FTI_Conf.blockSize;
    switch (level) {
    case 0:
        sprintf(lfn, "%s/%s", FTI_Conf.lTmpDir, FTI_Exec.ckptFile);
        break;
    case 1:
        sprintf(lfn, "%s/%s", FTI_Ckpt[1].dir, FTI_Exec.ckptFile);
        break;
    case 2:
        sprintf(lfn, "%s/%s", FTI_Ckpt[2].dir, FTI_Exec.ckptFile);
        break;
    case 3:
        sprintf(lfn, "%s/%s", FTI_Ckpt[3].dir, FTI_Exec.ckptFile);
        break;
    }

    // Open and resize files
    sprintf(gfn, "%s/%s", FTI_Conf.gTmpDir, FTI_Exec.ckptFile);
    sprintf(str, "L4 trying to access local ckpt. file (%s).", lfn);
    FTI_Print(str, FTI_DBUG);
    if (access(lfn, R_OK) != 0) {
        FTI_Print("L4 cannot access the checkpoint file.", FTI_EROR);

        return FTI_NSCS;
    }

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

    char *blBuf1 = talloc(char, FTI_Conf.blockSize);
    unsigned long bSize = FTI_Conf.blockSize;

    // Checkpoint files exchange
    while (pos < ps) {
        if ((fs - pos) < FTI_Conf.blockSize)
            bSize = fs - pos;

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

        pos = pos + FTI_Conf.blockSize;
    }

    free(blBuf1);

    fclose(lfd);
    fclose(gfd);

    return FTI_SCES;
}
