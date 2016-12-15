/**
 *  @file   postreco.c
 *  @author Leonardo A. Bautista Gomez (leobago@gmail.com)
 *  @date   January, 2014
 *  @brief  Post recovery functions for the FTI library.
 */

#include "interface.h"
#ifdef HAVE_LIBCPPR
#include "cppr.h"
#endif

/*-------------------------------------------------------------------------*/
/**
    @brief      Recover a set of ckpt. files using RS decoding.
    @return     integer         FTI_SCES if successful.

    This function tries to recover the L3 ckpt. files missing using the
    RS decoding.

 **/
/*-------------------------------------------------------------------------*/
int FTI_Decode(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
               FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt,
               int fs, int maxFs, int* erased)
{
    int *matrix, *decMatrix, *dm_ids, *tmpmat, i, j, k, m, ps, bs, pos = 0;
    char **coding, **data, *dataTmp, fn[FTI_BUFS], efn[FTI_BUFS], str[FTI_BUFS];
    FILE *fd, *efd;

    bs = FTI_Conf->blockSize;
    k = FTI_Topo->groupSize;
    m = k;
    ps = ((maxFs / FTI_Conf->blockSize)) * FTI_Conf->blockSize;
    if (ps < maxFs)
        ps = ps + FTI_Conf->blockSize; // Calculating padding size

    if (mkdir(FTI_Ckpt[3].dir, 0777) == -1)
        if (errno != EEXIST)
            FTI_Print("Cannot create directory", FTI_EROR);

    sscanf(FTI_Exec->ckptFile, "Ckpt%d-Rank%d.fti", &FTI_Exec->ckptID, &i);
    sprintf(fn, "%s/%s", FTI_Ckpt[3].dir, FTI_Exec->ckptFile);
    sprintf(efn, "%s/Ckpt%d-RSed%d.fti", FTI_Ckpt[3].dir, FTI_Exec->ckptID, i);

    data = talloc(char*, k);
    coding = talloc(char*, m);
    dataTmp = talloc(char, FTI_Conf->blockSize* k);
    dm_ids = talloc(int, k);
    decMatrix = talloc(int, k* k);
    tmpmat = talloc(int, k* k);
    matrix = talloc(int, k* k);
    for (i = 0; i < FTI_Topo->groupSize; i++) {
        for (j = 0; j < FTI_Topo->groupSize; j++) {
            matrix[i * FTI_Topo->groupSize + j] = galois_single_divide(1, i ^ (FTI_Topo->groupSize + j), FTI_Conf->l3WordSize);
        }
    }
    for (i = 0; i < m; i++) {
        coding[i] = talloc(char, FTI_Conf->blockSize);
        data[i] = talloc(char, FTI_Conf->blockSize);
    }
    j = 0;
    for (i = 0; j < k; i++) {
        if (erased[i] == 0) {
            dm_ids[j] = i;
            j++;
        }
    }
    // Building the matrix
    for (i = 0; i < k; i++) {
        if (dm_ids[i] < k) {
            for (j = 0; j < k; j++)
                tmpmat[i * k + j] = 0;
            tmpmat[i * k + dm_ids[i]] = 1;
        }
        else
            for (j = 0; j < k; j++) {
                tmpmat[i * k + j] = matrix[(dm_ids[i] - k) * k + j];
            }
    }
    // Inversing the matrix
    if (jerasure_invert_matrix(tmpmat, decMatrix, k, FTI_Conf->l3WordSize) < 0) {
        FTI_Print("Error inversing matrix", FTI_DBUG);

        for (i = 0; i < m; i++) {
            free(coding[i]);
            free(data[i]);
        }
        free(tmpmat);
        free(dm_ids);
        free(decMatrix);
        free(matrix);
        free(data);
        free(dataTmp);
        free(coding);

        return FTI_NSCS;
    }
    if (erased[FTI_Topo->groupRank] == 0) { // Resize and open files
        if (truncate(fn, ps) == -1) {
            FTI_Print("Error with truncate on checkpoint file", FTI_DBUG);

            for (i = 0; i < m; i++) {
                free(coding[i]);
                free(data[i]);
            }
            free(tmpmat);
            free(dm_ids);
            free(decMatrix);
            free(matrix);
            free(data);
            free(dataTmp);
            free(coding);

            return FTI_NSCS;
        }
        fd = fopen(fn, "rb");
        efd = fopen(efn, "rb");
    }
    else {
        fd = fopen(fn, "wb");
        efd = fopen(efn, "wb");
    }
    if (fd == NULL) {
        FTI_Print("R3 cannot open checkpoint file.", FTI_DBUG);
        if (efd)
            fclose(efd);

        for (i = 0; i < m; i++) {
            free(coding[i]);
            free(data[i]);
        }
        free(tmpmat);
        free(dm_ids);
        free(decMatrix);
        free(matrix);
        free(data);
        free(dataTmp);
        free(coding);

        return FTI_NSCS;
    }
    if (efd == NULL) {
        FTI_Print("R3 cannot open encoded ckpt. file.", FTI_DBUG);

        fclose(fd);

        for (i = 0; i < m; i++) {
            free(coding[i]);
            free(data[i]);
        }
        free(tmpmat);
        free(dm_ids);
        free(decMatrix);
        free(matrix);
        free(data);
        free(dataTmp);
        free(coding);

        return FTI_NSCS;
    }

    // Main loop, block by block
    while (pos < ps) {
        // Reading the data
        if (erased[FTI_Topo->groupRank] == 0) {
            size_t data_size = fread(data[FTI_Topo->groupRank] + 0, sizeof(char), bs, fd);
            size_t coding_size = fread(coding[FTI_Topo->groupRank] + 0, sizeof(char), bs, efd);

            if (ferror(fd) || ferror(efd)) {
                FTI_Print("R3 cannot from the ckpt. file or the encoded ckpt. file.", FTI_DBUG);

                fclose(fd);
                fclose(efd);

                for (i = 0; i < m; i++) {
                    free(coding[i]);
                    free(data[i]);
                }
                free(tmpmat);
                free(dm_ids);
                free(decMatrix);
                free(matrix);
                free(data);
                free(dataTmp);
                free(coding);

                return FTI_NSCS;
            }
        }
        else {
            bzero(data[FTI_Topo->groupRank], bs);
            bzero(coding[FTI_Topo->groupRank], bs);
        } // Erasure found

        MPI_Allgather(data[FTI_Topo->groupRank] + 0, bs, MPI_CHAR, dataTmp, bs, MPI_CHAR, FTI_Exec->groupComm);
        for (i = 0; i < k; i++)
            memcpy(data[i] + 0, &(dataTmp[i * bs]), sizeof(char) * bs);

        MPI_Allgather(coding[FTI_Topo->groupRank] + 0, bs, MPI_CHAR, dataTmp, bs, MPI_CHAR, FTI_Exec->groupComm);
        for (i = 0; i < k; i++)
            memcpy(coding[i] + 0, &(dataTmp[i * bs]), sizeof(char) * bs);

        // Decoding the lost data work
        if (erased[FTI_Topo->groupRank])
            jerasure_matrix_dotprod(k, FTI_Conf->l3WordSize, decMatrix + (FTI_Topo->groupRank * k), dm_ids, FTI_Topo->groupRank, data, coding, bs);

        MPI_Allgather(data[FTI_Topo->groupRank] + 0, bs, MPI_CHAR, dataTmp, bs, MPI_CHAR, FTI_Exec->groupComm);
        for (i = 0; i < k; i++)
            memcpy(data[i] + 0, &(dataTmp[i * bs]), sizeof(char) * bs);

        // Finally, re-encode any erased encoded checkpoint file
        if (erased[FTI_Topo->groupRank + k])
            jerasure_matrix_dotprod(k, FTI_Conf->l3WordSize, matrix + (FTI_Topo->groupRank * k), NULL, FTI_Topo->groupRank + k, data, coding, bs);
        if (erased[FTI_Topo->groupRank])
            fwrite(data[FTI_Topo->groupRank] + 0, sizeof(char), bs, fd);
        if (erased[FTI_Topo->groupRank + k])
            fwrite(coding[FTI_Topo->groupRank] + 0, sizeof(char), bs, efd);

        pos = pos + bs;
    }

    // Closing files
    fclose(fd);
    fclose(efd);

    if (truncate(fn, fs) == -1) {
        FTI_Print("R3 cannot re-truncate checkpoint file.", FTI_DBUG);

        for (i = 0; i < m; i++) {
            free(coding[i]);
            free(data[i]);
        }
        free(tmpmat);
        free(dm_ids);
        free(decMatrix);
        free(matrix);
        free(data);
        free(dataTmp);
        free(coding);

        return FTI_NSCS;
    }
    if (truncate(efn, fs) == -1) {
        FTI_Print("R3 cannot re-truncate encoded ckpt. file.", FTI_DBUG);

        for (i = 0; i < m; i++) {
            free(coding[i]);
            free(data[i]);
        }
        free(tmpmat);
        free(dm_ids);
        free(decMatrix);
        free(matrix);
        free(data);
        free(dataTmp);
        free(coding);

        return FTI_NSCS;
    }

    for (i = 0; i < m; i++) {
        free(coding[i]);
        free(data[i]);
    }
    free(tmpmat);
    free(dm_ids);
    free(decMatrix);
    free(matrix);
    free(data);
    free(dataTmp);
    free(coding);

    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
    @brief      Checks that all L1 ckpt. files are present.
    @param      group           The group ID.
    @return     integer         FTI_SCES if successful.

    This function detects all the erasures for L1. If there is at least one,
    L1 is not considered as recoverable.

 **/
/*-------------------------------------------------------------------------*/
int FTI_RecoverL1(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
                  FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt, int group)
{
    int erased[FTI_BUFS], buf, i, j; // FTI_BUFS > 32*3
    unsigned long fs, maxFs;
    if (FTI_CheckErasures(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt, &fs, &maxFs, group, erased, 1) != FTI_SCES) {
        FTI_Print("Error checking erasures.", FTI_DBUG);
        return FTI_NSCS;
    }
    buf = 0;
    for (j = 0; j < FTI_Topo->groupSize; j++)
        if (erased[j])
            buf++; // Counting erasures
    if (buf > 0) {
        FTI_Print("Checkpoint files missing at L1.", FTI_DBUG);
        return FTI_NSCS;
    }
    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
    @brief      Recover L2 ckpt. files using the partner copy.
    @param      group           The group ID.
    @return     integer         FTI_SCES if successful.

    This function tries to recover the L2 ckpt. files missing using the
    partner copy. If a ckpt. file and its copy are both missing, then we
    consider this checkpoint unavailable.

 **/
/*-------------------------------------------------------------------------*/
int FTI_RecoverL2(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
                  FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt, int group)
{
    int erased[FTI_BUFS], gs, buf, j, src, dest;
    char str[FTI_BUFS], lfn[FTI_BUFS], pfn[FTI_BUFS], jfn[FTI_BUFS], qfn[FTI_BUFS];
    char *blBuf1, *blBuf2, *blBuf3, *blBuf4;
    unsigned long ps, fs, maxFs, pos = 0;

    FILE *lfd = NULL, *pfd = NULL, *jfd = NULL, *qfd = NULL;

    MPI_Request reqSend1, reqRecv1, reqSend2, reqRecv2;
    MPI_Status status;

    blBuf1 = talloc(char, FTI_Conf->blockSize);
    blBuf2 = talloc(char, FTI_Conf->blockSize);
    blBuf3 = talloc(char, FTI_Conf->blockSize);
    blBuf4 = talloc(char, FTI_Conf->blockSize);

    gs = FTI_Topo->groupSize;
    src = FTI_Topo->left;
    dest = FTI_Topo->right;

    if (mkdir(FTI_Ckpt[2].dir, 0777) == -1)
        if (errno != EEXIST)
            FTI_Print("Cannot create directory", FTI_EROR);

    // Checking erasures
    if (FTI_CheckErasures(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt, &fs, &maxFs, group, erased, 2) != FTI_SCES) {
        FTI_Print("Error checking erasures.", FTI_DBUG);

        free(blBuf1);
        free(blBuf2);
        free(blBuf3);
        free(blBuf4);

        return FTI_NSCS;
    }

    buf = -1;
    for (j = 0; j < gs; j++)
        if (erased[j] && erased[((j + 1) % gs) + gs])
            buf = j; // Counting erasures
    sprintf(str, "A checkpoint file and its partner copy (ID in group : %d) have been lost", buf);
    if (buf > -1) {
        FTI_Print(str, FTI_DBUG);

        free(blBuf1);
        free(blBuf2);
        free(blBuf3);
        free(blBuf4);

        return FTI_NSCS;
    }

    buf = 0;
    for (j = 0; j < gs * 2; j++)
        if (erased[j])
            buf++; // Counting erasures
    if (buf > 0) {
        ps = (maxFs / FTI_Conf->blockSize) * FTI_Conf->blockSize;
        pos = 0; // For the logic
        if (ps < maxFs)
            ps = ps + FTI_Conf->blockSize; // Calculating padding size
        sprintf(str, "File size: %ld, max. file size : %ld and padding size : %ld.", fs, maxFs, ps);
        FTI_Print(str, FTI_DBUG);

        // Open checkpoint file to recover
        if (erased[FTI_Topo->groupRank]) {
            sprintf(lfn, "%s/%s", FTI_Ckpt[2].dir, FTI_Exec->ckptFile);
            sscanf(FTI_Exec->ckptFile, "Ckpt%d-Rank%d.fti", &FTI_Exec->ckptID, &buf);
            sprintf(jfn, "%s/Ckpt%d-Pcof%d.fti", FTI_Ckpt[2].dir, FTI_Exec->ckptID, buf);
            sprintf(str, "Opening checkpoint file (%s) to recover (L2).", lfn);
            FTI_Print(str, FTI_DBUG);
            sprintf(str, "Opening partner ckpt. file (%s) to recover (L2).", jfn);
            FTI_Print(str, FTI_DBUG);

            lfd = fopen(lfn, "wb");
            if (lfd == NULL) {
                FTI_Print("R2 cannot open the checkpoint file.", FTI_DBUG);

                free(blBuf1);
                free(blBuf2);
                free(blBuf3);
                free(blBuf4);

                return FTI_NSCS;
            }

            jfd = fopen(jfn, "wb");
            if (jfd == NULL) {
                FTI_Print("R2 cannot open the partner ckpt. file.", FTI_DBUG);

                fclose(lfd);

                free(blBuf1);
                free(blBuf2);
                free(blBuf3);
                free(blBuf4);

                return FTI_NSCS;
            }
        }

        // Truncate and open partner file to transfer
        if (erased[src] && !erased[gs + FTI_Topo->groupRank]) {
            sscanf(FTI_Exec->ckptFile, "Ckpt%d-Rank%d.fti", &FTI_Exec->ckptID, &buf);
            sprintf(pfn, "%s/Ckpt%d-Pcof%d.fti", FTI_Ckpt[2].dir, FTI_Exec->ckptID, buf);
            sprintf(str, "Opening partner ckpt. file (%s) to transfer (L2).", pfn);
            FTI_Print(str, FTI_DBUG);

            if (truncate(pfn, ps) == -1) {
                FTI_Print("R2 cannot truncate the partner ckpt. file.", FTI_DBUG);

                if (jfd)
                    fclose(jfd);
                if (lfd)
                    fclose(lfd);

                free(blBuf1);
                free(blBuf2);
                free(blBuf3);
                free(blBuf4);

                return FTI_NSCS;
            }

            pfd = fopen(pfn, "rb");
            if (pfd == NULL) {
                FTI_Print("R2 cannot open partner ckpt. file.", FTI_DBUG);

                if (jfd)
                    fclose(jfd);
                if (lfd)
                    fclose(lfd);

                free(blBuf1);
                free(blBuf2);
                free(blBuf3);
                free(blBuf4);

                return FTI_NSCS;
            }
        }

        // Truncate and open partner file to transfer
        if (erased[dest] && !erased[gs + FTI_Topo->groupRank]) {
            sprintf(qfn, "%s/%s", FTI_Ckpt[2].dir, FTI_Exec->ckptFile);
            sprintf(str, "Opening ckpt. file (%s) to transfer (L2).", qfn);
            FTI_Print(str, FTI_DBUG);

            if (truncate(qfn, ps) == -1) {
                FTI_Print("R2 cannot truncate the ckpt. file.", FTI_DBUG);

                if (jfd)
                    fclose(jfd);
                if (lfd)
                    fclose(lfd);
                if (pfd)
                    fclose(pfd);

                free(blBuf1);
                free(blBuf2);
                free(blBuf3);
                free(blBuf4);

                return FTI_NSCS;
            }

            qfd = fopen(qfn, "rb");
            if (qfd == NULL) {
                FTI_Print("R2 cannot open ckpt. file.", FTI_DBUG);

                if (jfd)
                    fclose(jfd);
                if (lfd)
                    fclose(lfd);
                if (pfd)
                    fclose(pfd);

                free(blBuf1);
                free(blBuf2);
                free(blBuf3);
                free(blBuf4);

                return FTI_NSCS;
            }
        }

        // Checkpoint files exchange
        while (pos < ps) {
            if (erased[src] && !erased[gs + FTI_Topo->groupRank]) {
                size_t bytes = fread(blBuf1, sizeof(char), FTI_Conf->blockSize, pfd);

                if (ferror(pfd)) {
                    FTI_Print("Error reading the data from the partner ckpt. file.", FTI_DBUG);

                    fclose(pfd);

                    if (jfd)
                        fclose(jfd);
                    if (lfd)
                        fclose(lfd);
                    if (qfd)
                        fclose(qfd);

                    free(blBuf1);
                    free(blBuf2);
                    free(blBuf3);
                    free(blBuf4);

                    return FTI_NSCS;
                }

                MPI_Isend(blBuf1, bytes, MPI_CHAR, src, FTI_Conf->tag, FTI_Exec->groupComm, &reqSend1);
            }
            if (erased[dest] && !erased[gs + FTI_Topo->groupRank]) {
                size_t bytes = fread(blBuf3, sizeof(char), FTI_Conf->blockSize, qfd);

                if (ferror(qfd)) {
                    FTI_Print("Error reading the data from the ckpt. file.", FTI_DBUG);

                    fclose(qfd);

                    if (jfd)
                        fclose(jfd);
                    if (lfd)
                        fclose(lfd);
                    if (pfd)
                        fclose(pfd);

                    free(blBuf1);
                    free(blBuf2);
                    free(blBuf3);
                    free(blBuf4);

                    return FTI_NSCS;
                }

                MPI_Isend(blBuf3, bytes, MPI_CHAR, dest, FTI_Conf->tag, FTI_Exec->groupComm, &reqSend2);
            }
            if (erased[FTI_Topo->groupRank]) {
                MPI_Irecv(blBuf2, FTI_Conf->blockSize, MPI_CHAR, dest, FTI_Conf->tag, FTI_Exec->groupComm, &reqRecv1);
                MPI_Irecv(blBuf4, FTI_Conf->blockSize, MPI_CHAR, src, FTI_Conf->tag, FTI_Exec->groupComm, &reqRecv2);
            }
            if (erased[src] && !erased[gs + FTI_Topo->groupRank])
                MPI_Wait(&reqSend1, &status);
            if (erased[dest] && !erased[gs + FTI_Topo->groupRank])
                MPI_Wait(&reqSend2, &status);
            if (erased[FTI_Topo->groupRank]) {
                MPI_Wait(&reqRecv1, &status);
                MPI_Wait(&reqRecv2, &status);

                fwrite(blBuf2, sizeof(char), FTI_Conf->blockSize, lfd);
                if (ferror(lfd)) {
                    FTI_Print("Errors writting the data in the R2 checkpoint file.", FTI_DBUG);

                    fclose(lfd);

                    if (jfd)
                        fclose(jfd);
                    if (pfd)
                        fclose(pfd);
                    if (qfd)
                        fclose(qfd);

                    free(blBuf1);
                    free(blBuf2);
                    free(blBuf3);
                    free(blBuf4);

                    return FTI_NSCS;
                }

                fwrite(blBuf4, sizeof(char), FTI_Conf->blockSize, jfd);
                if (ferror(jfd)) {
                    FTI_Print("Errors writting the data in the R2 partner ckpt. file.", FTI_DBUG);

                    fclose(jfd);

                    fclose(lfd);
                    if (pfd)
                        fclose(pfd);
                    if (qfd)
                        fclose(qfd);

                    free(blBuf1);
                    free(blBuf2);
                    free(blBuf3);
                    free(blBuf4);

                    return FTI_NSCS;
                }
            }
            pos = pos + FTI_Conf->blockSize;
        }

        // Close files
        if (erased[FTI_Topo->groupRank]) {
            if (fclose(lfd) != 0) {
                FTI_Print("R2 cannot close the checkpoint file.", FTI_DBUG);

                if (jfd)
                    fclose(jfd);
                if (pfd)
                    fclose(pfd);
                if (qfd)
                    fclose(qfd);

                free(blBuf1);
                free(blBuf2);
                free(blBuf3);
                free(blBuf4);

                return FTI_NSCS;
            }
            if (truncate(lfn, fs) == -1) {
                FTI_Print("R2 cannot re-truncate the checkpoint file.", FTI_DBUG);

                if (jfd)
                    fclose(jfd);
                if (pfd)
                    fclose(pfd);
                if (qfd)
                    fclose(qfd);

                free(blBuf1);
                free(blBuf2);
                free(blBuf3);
                free(blBuf4);

                return FTI_NSCS;
            }

            if (fclose(jfd) != 0) {
                FTI_Print("R2 cannot close the partner ckpt. file.", FTI_DBUG);

                if (pfd)
                    fclose(pfd);
                if (qfd)
                    fclose(qfd);

                free(blBuf1);
                free(blBuf2);
                free(blBuf3);
                free(blBuf4);

                return FTI_NSCS;
            }
            if (truncate(jfn, fs) == -1) {
                FTI_Print("R2 cannot re-truncate the partner ckpt. file.", FTI_DBUG);

                if (pfd)
                    fclose(pfd);
                if (qfd)
                    fclose(qfd);

                free(blBuf1);
                free(blBuf2);
                free(blBuf3);
                free(blBuf4);

                return FTI_NSCS;
            }
        }

        if (erased[src] && !erased[gs + FTI_Topo->groupRank]) {
            if (fclose(pfd) != 0) {
                FTI_Print("R2 cannot close the partner ckpt. file", FTI_DBUG);

                if (qfd)
                    fclose(qfd);

                free(blBuf1);
                free(blBuf2);
                free(blBuf3);
                free(blBuf4);

                return FTI_NSCS;
            }
            if (truncate(pfn, fs) == -1) {
                FTI_Print("R2 cannot re-truncate the partner ckpt. file.", FTI_DBUG);

                if (qfd)
                    fclose(qfd);

                free(blBuf1);
                free(blBuf2);
                free(blBuf3);
                free(blBuf4);

                return FTI_NSCS;
            }
        }

        if (erased[dest] && !erased[gs + FTI_Topo->groupRank]) {
            if (fclose(qfd) != 0) {
                FTI_Print("R2 cannot close the ckpt. file", FTI_DBUG);

                free(blBuf1);
                free(blBuf2);
                free(blBuf3);
                free(blBuf4);

                return FTI_NSCS;
            }
            if (truncate(qfn, fs) == -1) {
                FTI_Print("R2 cannot re-truncate the ckpt. file.", FTI_DBUG);

                free(blBuf1);
                free(blBuf2);
                free(blBuf3);
                free(blBuf4);

                return FTI_NSCS;
            }
        }
    }

    free(blBuf1);
    free(blBuf2);
    free(blBuf3);
    free(blBuf4);

    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
    @brief      Recover L3 ckpt. files ordering the RS decoding algorithm.
    @param      group           The group ID.
    @return     integer         FTI_SCES if successful.

    This function tries to recover the L3 ckpt. files missing using the
    RS decoding. If to many files are missing in the group, then we
    consider this checkpoint unavailable.

 **/
/*-------------------------------------------------------------------------*/
int FTI_RecoverL3(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
                  FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt, int group)
{
    int erased[FTI_BUFS], gs, j, l = 0;
    unsigned long fs, maxFs;
    char str[FTI_BUFS];
    gs = FTI_Topo->groupSize;

    if (mkdir(FTI_Ckpt[3].dir, 0777) == -1)
        if (errno != EEXIST)
            FTI_Print("Cannot create directory", FTI_EROR);

    // Checking erasures
    if (FTI_CheckErasures(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt, &fs, &maxFs, group, erased, 3) != FTI_SCES) {
        FTI_Print("Error checking erasures.", FTI_DBUG);
        return FTI_NSCS;
    }

    // Counting erasures
    l = 0;
    for (j = 0; j < gs; j++) {
        if (erased[j])
            l++;
        if (erased[j + gs])
            l++;
    }
    if (l > gs) {
        FTI_Print("Too many erasures at L3.", FTI_DBUG);
        return FTI_NSCS;
    }

    // Reed-Solomon decoding
    if (l > 0) {
        sprintf(str, "There are %d encoded/checkpoint files missing in this group.", l);
        FTI_Print(str, FTI_DBUG);
        if (FTI_Decode(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt, fs, maxFs, erased) == FTI_NSCS) {
            FTI_Print("RS-decoding could not regenerate the missing data.", FTI_DBUG);
            return FTI_NSCS;
        }
    }

    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
    @brief      Recover L4 ckpt. files from the PFS.
    @param      group           The group ID.
    @return     integer         FTI_SCES if successful.

    This function tries to recover the ckpt. files using the L4 ckpt. files
    stored in the PFS. If at least one ckpt. file is missing in the PFS, we
    consider this checkpoint unavailable.

 **/
/*-------------------------------------------------------------------------*/
int FTI_RecoverL4(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
                  FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt, int group)
{
    unsigned long maxFs, fs, ps, pos = 0;
    int j, l, gs, erased[FTI_BUFS];
    char gfn[FTI_BUFS], lfn[FTI_BUFS];
    FILE *gfd, *lfd;

    gs = FTI_Topo->groupSize;
    if (FTI_Topo->nodeRank == 0 || FTI_Topo->nodeRank == 1) {
        if (mkdir(FTI_Ckpt[1].dir, 0777) == -1) {
            if (errno != EEXIST)
                FTI_Print("Directory L1 could NOT be created.", FTI_WARN);
        }
    }
    MPI_Barrier(FTI_COMM_WORLD);
    // Checking erasures
    if (FTI_CheckErasures(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt, &fs, &maxFs, group, erased, 4) != FTI_SCES) {
        FTI_Print("Error checking erasures.", FTI_DBUG);
        return FTI_NSCS;
    }

    l = 0;
    // Counting erasures
    for (j = 0; j < gs; j++) {
        if (erased[j])
            l++;
    }
    if (l > 0) {
        FTI_Print("Checkpoint file missing at L4.", FTI_DBUG);
        return FTI_NSCS;
    }

    ps = (fs / FTI_Conf->blockSize) * FTI_Conf->blockSize;
    pos = 0; // For the logic
    // Calculating padding size
    if (ps < fs)
        ps = ps + FTI_Conf->blockSize;
    // Open and resize files
    sprintf(gfn, "%s/%s", FTI_Ckpt[4].dir, FTI_Exec->ckptFile);
    sprintf(lfn, "%s/%s", FTI_Ckpt[1].dir, FTI_Exec->ckptFile);

    if (truncate(gfn, ps) == -1) {
        FTI_Print("R4 cannot truncate the ckpt. file in the PFS.", FTI_DBUG);
        return FTI_NSCS;
    }

#ifdef HAVE_LIBCPPR
    FTI_Print("using CPPR for file transfer in FTI_RecoverL4", FTI_WARN);
    char cppr_str[FTI_BUFS];
    int cppr_retval = cppr_mv_wait(0,
                                   0,
                                   NULL,
                                   FTI_Ckpt[1].dir,
                                   FTI_Ckpt[4].dir,
                                   FTI_Exec->ckptFile);

    if(cppr_retval != CPPR_SUCCESS){
            sprintf(cppr_str, "Cannot recover and transfer file with cppr:src %s dest: %s, file %s: %s %s ",
                    FTI_Ckpt[1].dir, FTI_Ckpt[4].dir, FTI_Exec->ckptFile,
                    cppr_err_to_str(cppr_retval),
                    cppr_err_to_desc(cppr_retval));
            FTI_Print(cppr_str, FTI_WARN);
            return FTI_NSCS;
    }

    sprintf(cppr_str, "cppr successfully (recovered) dest: %s src: %s, file %s",
            FTI_Ckpt[1].dir, FTI_Ckpt[4].dir, FTI_Exec->ckptFile);
    FTI_Print(cppr_str, FTI_WARN);
    goto end;
#endif

    gfd = fopen(gfn, "rb");
    if (gfd == NULL) {
        FTI_Print("R4 cannot open the ckpt. file in the PFS.", FTI_DBUG);
        return FTI_NSCS;
    }

    lfd = fopen(lfn, "wb");
    if (lfd == NULL) {
        FTI_Print("R4 cannot open the local ckpt. file.", FTI_DBUG);
        fclose(gfd);
        return FTI_NSCS;
    }

    char *blBuf1 = talloc(char, FTI_Conf->blockSize);
    // Checkpoint files transfer from PFS
    while (pos < ps) {
        size_t bytes = fread(blBuf1, sizeof(char), FTI_Conf->blockSize, gfd);
        if (ferror(gfd)) {
            FTI_Print("R4 cannot read from the ckpt. file in the PFS.", FTI_DBUG);

            free(blBuf1);

            fclose(gfd);
            fclose(lfd);

            return  FTI_NSCS;
        }

        fwrite(blBuf1, sizeof(char), bytes, lfd);
        if (ferror(lfd)) {
            FTI_Print("R4 cannot write to the local ckpt. file.", FTI_DBUG);

            free(blBuf1);

            fclose(gfd);
            fclose(lfd);

            return  FTI_NSCS;
        }

        pos = pos + FTI_Conf->blockSize;
    }

    free(blBuf1);

    fclose(gfd);
    fclose(lfd);
end:
    if (truncate(gfn, fs) == -1) {
        FTI_Print("R4 cannot re-truncate the checkpoint file in the PFS.", FTI_DBUG);
        return FTI_NSCS;
    }
    if (truncate(lfn, fs) == -1) {
        FTI_Print("R4 cannot re-truncate the local checkpoint file.", FTI_DBUG);
        return FTI_NSCS;
    }

    return FTI_SCES;
}
