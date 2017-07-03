/**
 *  @file   postreco.c
 *  @author Leonardo A. Bautista Gomez (leobago@gmail.com)
 *  @date   January, 2014
 *  @brief  Post recovery functions for the FTI library.
 */

#include "interface.h"

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
    if (ps < maxFs) {
        ps = ps + FTI_Conf->blockSize; // Calculating padding size
    }

    if (mkdir(FTI_Ckpt[3].dir, 0777) == -1) {
        if (errno != EEXIST) {
            FTI_Print("Cannot create directory", FTI_EROR);
        }
    }

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
            for (j = 0; j < k; j++) {
                tmpmat[i * k + j] = 0;
            }
            tmpmat[i * k + dm_ids[i]] = 1;
        }
        else {
            for (j = 0; j < k; j++) {
                tmpmat[i * k + j] = matrix[(dm_ids[i] - k) * k + j];
            }
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
    }
    else {
        fd = fopen(fn, "wb");
    }

    if (erased[FTI_Topo->groupRank + FTI_Topo->groupSize] == 0) {
        efd = fopen(efn, "rb");
    }
    else {
        efd = fopen(efn, "wb");
    }
    if (fd == NULL) {
        FTI_Print("R3 cannot open checkpoint file.", FTI_DBUG);

        if (efd) {
            fclose(efd);
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

            if (ferror(fd)) {
                FTI_Print("R3 cannot from the ckpt. file.", FTI_DBUG);

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
        } // Erasure found

        if (erased[FTI_Topo->groupRank + FTI_Topo->groupSize] == 0) {
            size_t coding_size = fread(coding[FTI_Topo->groupRank] + 0, sizeof(char), bs, efd);
            if (ferror(efd)) {
                FTI_Print("R3 cannot from the encoded ckpt. file.", FTI_DBUG);

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
            bzero(coding[FTI_Topo->groupRank], bs);
        }

        MPI_Allgather(data[FTI_Topo->groupRank] + 0, bs, MPI_CHAR, dataTmp, bs, MPI_CHAR, FTI_Exec->groupComm);
        for (i = 0; i < k; i++) {
            memcpy(data[i] + 0, &(dataTmp[i * bs]), sizeof(char) * bs);
        }

        MPI_Allgather(coding[FTI_Topo->groupRank] + 0, bs, MPI_CHAR, dataTmp, bs, MPI_CHAR, FTI_Exec->groupComm);
        for (i = 0; i < k; i++) {
            memcpy(coding[i] + 0, &(dataTmp[i * bs]), sizeof(char) * bs);
        }

        // Decoding the lost data work
        if (erased[FTI_Topo->groupRank]) {
            jerasure_matrix_dotprod(k, FTI_Conf->l3WordSize, decMatrix + (FTI_Topo->groupRank * k), dm_ids, FTI_Topo->groupRank, data, coding, bs);
        }

        MPI_Allgather(data[FTI_Topo->groupRank] + 0, bs, MPI_CHAR, dataTmp, bs, MPI_CHAR, FTI_Exec->groupComm);
        for (i = 0; i < k; i++) {
            memcpy(data[i] + 0, &(dataTmp[i * bs]), sizeof(char) * bs);
        }

        // Finally, re-encode any erased encoded checkpoint file
        if (erased[FTI_Topo->groupRank + k]) {
            jerasure_matrix_dotprod(k, FTI_Conf->l3WordSize, matrix + (FTI_Topo->groupRank * k), NULL, FTI_Topo->groupRank + k, data, coding, bs);
        }
        if (erased[FTI_Topo->groupRank]) {
            fwrite(data[FTI_Topo->groupRank] + 0, sizeof(char), bs, fd);
        }
        if (erased[FTI_Topo->groupRank + k]) {
            fwrite(coding[FTI_Topo->groupRank] + 0, sizeof(char), bs, efd);
        }

        pos = pos + bs;
    }

    // Closing files
    fclose(fd);
    fclose(efd);

    if (truncate(fn, fs) == -1) {
        FTI_Print("R3 cannot re-truncate checkpoint file.", FTI_WARN);

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
    for (j = 0; j < FTI_Topo->groupSize; j++) {
        if (erased[j]) {
            buf++; // Counting erasures
        }
    }
    if (buf > 0) {
        FTI_Print("Checkpoint files missing at L1.", FTI_DBUG);
        return FTI_NSCS;
    }
    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
    @brief      Sends checkpint file.
    @param      destination     destination group rank
    @param      fs              filesize
    @param      ptner           0 if sending Ckpt, 1 if PtnerCkpt

    @return     integer         FTI_SCES if successful.

    This function sends Ckpt or PtnerCkpt file from partner proccess.

 **/
/*-------------------------------------------------------------------------*/
int FTI_SendCkptFileL2(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
                        FTIT_checkpoint* FTI_Ckpt, int destination,
                         unsigned long fs, int ptner) {
    int i, j; //iterators
    int bytes; //bytes read by fread
    char filename[FTI_BUFS], str[FTI_BUFS];
    FILE *fileDesc;

    if (ptner) {    //if want to send Ptner file
        int rank;
        sscanf(FTI_Exec->ckptFile, "Ckpt%d-Rank%d.fti", &FTI_Exec->ckptID, &rank); //do we need this from filename?
        sprintf(filename, "%s/Ckpt%d-Pcof%d.fti", FTI_Ckpt[2].dir, FTI_Exec->ckptID, rank);
    } else {    //if want to send Ckpt file
        sprintf(filename, "%s/%s", FTI_Ckpt[2].dir, FTI_Exec->ckptFile);
    }

    sprintf(str, "Opening file (rb) (%s) (L2).", filename);
    FTI_Print(str, FTI_DBUG);
    fileDesc = fopen(filename, "rb");
    if (fileDesc == NULL) {
        FTI_Print("R2 cannot open the partner ckpt. file.", FTI_WARN);
        return FTI_NSCS;
    }
    char* buffer = talloc(char, FTI_Conf->blockSize);
    unsigned long toSend = fs; // remaining data to send
    while (toSend > 0) {
        int sendSize = (toSend > FTI_Conf->blockSize) ? FTI_Conf->blockSize : toSend;
        size_t bytes = fread(buffer, sizeof(char), sendSize, fileDesc);

        if (ferror(fileDesc)) {
            FTI_Print("Error reading the data from the ckpt. file.", FTI_WARN);

            fclose(fileDesc);
            free(buffer);

            return FTI_NSCS;
        }

        MPI_Send(buffer, bytes, MPI_CHAR, destination, FTI_Conf->tag, FTI_Exec->groupComm);
        toSend -= bytes;
    }

    fclose(fileDesc);
    free(buffer);

    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
    @brief      Receives checkpint file.
    @param      source          source group rank
    @param      fs              filesize
    @param      ptner           0 if receiving Ckpt, 1 if PtnerCkpt
    @return     integer         FTI_SCES if successful.

    This function receives Ckpt or PtnerCkpt file from partner proccess.

 **/
/*-------------------------------------------------------------------------*/
int FTI_RecvCkptFileL2(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
                        FTIT_checkpoint* FTI_Ckpt, int source, unsigned long fs,
                         int ptner) {
    int i, j; //iterators
    char filename[FTI_BUFS], str[FTI_BUFS];
    FILE *fileDesc;

    if (ptner) { //if want to receive Ptner file
        int rank;
        sscanf(FTI_Exec->ckptFile, "Ckpt%d-Rank%d.fti", &FTI_Exec->ckptID, &rank); //do we need this from filename?
        sprintf(filename, "%s/Ckpt%d-Pcof%d.fti", FTI_Ckpt[2].dir, FTI_Exec->ckptID, rank);
    } else { //if want to receive Ckpt file
        sprintf(filename, "%s/%s", FTI_Ckpt[2].dir, FTI_Exec->ckptFile);
    }

    sprintf(str, "Opening file (wb) (%s) (L2).", filename);
    FTI_Print(str, FTI_DBUG);
    fileDesc = fopen(filename, "wb");
    if (fileDesc == NULL) {
        FTI_Print("R2 cannot open the file.", FTI_WARN);
        return FTI_NSCS;
    }
    char* buffer = talloc(char, FTI_Conf->blockSize);
    int toRecv = fs;    //remaining data to receive
    while (toRecv > 0) {
        int recvSize = (toRecv > FTI_Conf->blockSize) ? FTI_Conf->blockSize : toRecv;
        MPI_Recv(buffer, recvSize, MPI_CHAR, source, FTI_Conf->tag, FTI_Exec->groupComm, MPI_STATUS_IGNORE);
        fwrite(buffer, sizeof(char), recvSize, fileDesc);

        if (ferror(fileDesc)) {
            FTI_Print("Error writing the data to the file.", FTI_WARN);

            fclose(fileDesc);
            free(buffer);

            return FTI_NSCS;
        }

        toRecv -= recvSize;
    }

    fclose(fileDesc);
    free(buffer);

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
    int erased[FTI_BUFS];
    unsigned long fs, maxFs; //fileSize, maxFileSize
    int i, j; //iterators
    int source = FTI_Topo->right; //to receive Ptner file from this process (to recover)
    int destination = FTI_Topo->left; //to send Ptner file (to let him recover)
    int res, tres;
    char ptnerFilename[FTI_BUFS], str[FTI_BUFS];
    FILE *ptnerFile;

    if (mkdir(FTI_Ckpt[2].dir, 0777) == -1) {
        if (errno != EEXIST) {
            FTI_Print("Cannot create directory", FTI_EROR);
        }
    }

    // Checking erasures
    if (FTI_CheckErasures(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt, &fs, &maxFs, group, erased, 2) != FTI_SCES) {
        FTI_Print("Error checking erasures.", FTI_WARN);
        return FTI_NSCS;
    }

    i = 0;
    for (j = 0; j < FTI_Topo->groupSize * 2; j++) {
        if (erased[j]) {
            i++; // Counting erasures
        }
    }

    if (i == 0) {
        FTI_Print("Have all checkpoint files.", FTI_DBUG);
        return FTI_SCES;
    }

    res = FTI_SCES;
    if (erased[FTI_Topo->groupRank] && erased[source + FTI_Topo->groupSize]) {
        FTI_Print("My checkpoint file and partner copy have been lost", FTI_WARN);
        res = FTI_NSCS;
    }

    if (erased[FTI_Topo->groupRank + FTI_Topo->groupSize] && erased[destination]) {
        FTI_Print("My Ptner checkpoint file and his checkpoint file have been lost", FTI_WARN);
        res = FTI_NSCS;
    }

    MPI_Allreduce(&res, &tres, 1, MPI_INT, MPI_SUM, FTI_Exec->groupComm);
    if (tres != FTI_SCES) {
        return FTI_NSCS; //needed FTI_Abort?
    }

    //recover checkpoint files
    if (FTI_Topo->groupRank % 2) {
        if (erased[destination]) { //first send file
            unsigned long pfs; //Ptner file size
            res = FTI_GetPtnerSize(FTI_Conf, FTI_Topo, FTI_Ckpt, &pfs, group, 2);
            if (res == FTI_NSCS) {
                return FTI_NSCS;
            }
            res = FTI_SendCkptFileL2(FTI_Conf, FTI_Exec, FTI_Ckpt, destination, pfs, 1);
            if (res != FTI_SCES) {
                return FTI_NSCS;
            }
        }
        if (erased[FTI_Topo->groupRank]) { //then receive file
            res = FTI_RecvCkptFileL2(FTI_Conf, FTI_Exec, FTI_Ckpt, source, fs, 0);
            if (res != FTI_SCES) {
                return FTI_NSCS;
            }
        }
    } else {
        if (erased[FTI_Topo->groupRank]) { //first receive file
            res = FTI_RecvCkptFileL2(FTI_Conf, FTI_Exec, FTI_Ckpt, source, fs, 0);
            if (res != FTI_SCES) {
                return FTI_NSCS;
            }
        }

        if (erased[destination]) { //then send file
            unsigned long pfs; //Ptner file size
            res = FTI_GetPtnerSize(FTI_Conf, FTI_Topo, FTI_Ckpt, &pfs, group, 2);
            if (res == FTI_NSCS) {
                return FTI_NSCS;
            }
            res = FTI_SendCkptFileL2(FTI_Conf, FTI_Exec, FTI_Ckpt, destination, pfs, 1);
            if (res != FTI_SCES) {
                return FTI_NSCS;
            }
        }
    }

    //recover partner files
    if (FTI_Topo->groupRank % 2) {
        if (erased[source + FTI_Topo->groupSize]) { //fisrst send file
            res = FTI_SendCkptFileL2(FTI_Conf, FTI_Exec, FTI_Ckpt, source, fs, 0);
            if (res != FTI_SCES) {
                return FTI_NSCS;
            }
        }
        if (erased[FTI_Topo->groupRank + FTI_Topo->groupSize]) { //receive file
            unsigned long pfs; //Ptner file size
            res = FTI_GetPtnerSize(FTI_Conf, FTI_Topo, FTI_Ckpt, &pfs, group, 2);
            if (res == FTI_NSCS) {
                return FTI_NSCS;
            }
            res = FTI_RecvCkptFileL2(FTI_Conf, FTI_Exec, FTI_Ckpt, destination, pfs, 1);
            if (res != FTI_SCES) {
                return FTI_NSCS;
            }
        }
    } else {
        if (erased[FTI_Topo->groupRank + FTI_Topo->groupSize]) { //first receive file
            unsigned long pfs; //Ptner file size
            res = FTI_GetPtnerSize(FTI_Conf, FTI_Topo, FTI_Ckpt, &pfs, group, 2);
            if (res == FTI_NSCS) {
                return FTI_NSCS;
            }
            res = FTI_RecvCkptFileL2(FTI_Conf, FTI_Exec, FTI_Ckpt, destination, pfs, 1);
            if (res != FTI_SCES) {
                return FTI_NSCS;
            }
        }

        if (erased[source + FTI_Topo->groupSize]) { //send file
            res = FTI_SendCkptFileL2(FTI_Conf, FTI_Exec, FTI_Ckpt, source, fs, 0);
            if (res != FTI_SCES) {
                return FTI_NSCS;
            }
        }
    }

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

    if (mkdir(FTI_Ckpt[3].dir, 0777) == -1) {
        if (errno != EEXIST) {
            FTI_Print("Cannot create directory", FTI_EROR);
        }
    }

    // Checking erasures
    if (FTI_CheckErasures(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt, &fs, &maxFs, group, erased, 3) != FTI_SCES) {
        FTI_Print("Error checking erasures.", FTI_DBUG);
        return FTI_NSCS;
    }

    // Counting erasures
    l = 0;
    for (j = 0; j < gs; j++) {
        if (erased[j]) {
            l++;
        }
        if (erased[j + gs]) {
            l++;
        }
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
   int res;

   // select IO
   switch(FTI_Conf->ioMode) {

      case FTI_IO_POSIX:

         res = FTI_RecoverL4Posix(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt, group);
         break;

      case FTI_IO_MPI:

         res = FTI_RecoverL4Mpi(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt);
         break;

#ifdef ENABLE_SIONLIB // --> If SIONlib is installed
      case FTI_IO_SIONLIB:

         res = FTI_RecoverL4Sionlib(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt);
         break;
#endif
   }

   return res;

}

/*-------------------------------------------------------------------------*/
/**
    @brief      Recover L4 ckpt. files from the PFS using POSIX.
    @param      group           The group ID.
    @return     integer         FTI_SCES if successful.

    This function tries to recover the ckpt. files using the L4 ckpt. files
    stored in the PFS. If at least one ckpt. file is missing in the PFS, we
    consider this checkpoint unavailable.

 **/
/*-------------------------------------------------------------------------*/
int FTI_RecoverL4Posix(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
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

/*-------------------------------------------------------------------------*/
/**
    @brief      Recover L4 ckpt. files from the PFS using MPI-I/O.
    @return     integer         FTI_SCES if successful.

    This function tries to recover the ckpt. files using the L4 ckpt. files
    stored in the PFS. If at least one ckpt. file is missing in the PFS, we
    consider this checkpoint unavailable.

 **/
/*-------------------------------------------------------------------------*/
int FTI_RecoverL4Mpi(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
      FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt)
{
   int i, j, l, reslen, buf;
   char gfn[FTI_BUFS], lfn[FTI_BUFS], mpi_err[FTI_BUFS], str[FTI_BUFS];
   FILE *lfd;
   MPI_File pfh;
   MPI_Offset offset = 0, tfs, sfs, nbBlocks, block = 0, lastBlockBytes, *chunkSizes;
   MPI_Status status;
   MPI_Info info;

   // TODO enable to set stripping unit in the config file (Maybe also other hints)
   // enable collective buffer optimization
   MPI_Info_create(&info);
   MPI_Info_set(info, "romio_cb_read", "enable");

   // set stripping unit to 4MB
   MPI_Info_set(info, "stripping_unit", "4194304");

   sprintf(gfn, "%s/%s", FTI_Ckpt[4].dir, FTI_Exec->ckptFile);

   // rename checkpoint file.
   sprintf(FTI_Exec->ckptFile, "Ckpt%d-Rank%d.fti", FTI_Exec->ckptID, FTI_Topo->myRank);
   sprintf(lfn, "%s/%s", FTI_Ckpt[1].dir, FTI_Exec->ckptFile);

   // open parallel file
   buf = MPI_File_open(FTI_COMM_WORLD, gfn, MPI_MODE_RDWR, info, &pfh);
   // check if successfull
   if (buf != 0) {
      errno = 0;
      MPI_Error_string(buf, mpi_err, &reslen);
      if (buf != MPI_ERR_NO_SUCH_FILE) {
         snprintf(str, FTI_BUFS, "unable to access file [MPI ERROR - %i] %s", buf, mpi_err);
         FTI_Print(str, FTI_EROR);
      }
      return FTI_NSCS;
   }

   // create local directories
   if (FTI_Topo->nodeRank == 0 || FTI_Topo->nodeRank == 1) {
      if (mkdir(FTI_Ckpt[1].dir, 0777) == -1) {
         if (errno != EEXIST)
            FTI_Print("Directory L1 could NOT be created.", FTI_WARN);
      }
   }
   MPI_Barrier(FTI_COMM_WORLD);

   // collect chunksizes of other ranks
   chunkSizes = talloc(MPI_Offset, FTI_Topo->nbApprocs*FTI_Topo->nbNodes);
   MPI_Allgather(&(FTI_Exec->meta[0].fs), 1, MPI_OFFSET, chunkSizes, 1, MPI_OFFSET, FTI_COMM_WORLD);

   // set file offset
   for (i=0; i<FTI_Topo->splitRank; i++) {
      offset += chunkSizes[i];
   }

   // number of blocks
   nbBlocks = (FTI_Exec->meta[0].fs) / FTI_Conf->transferSize;
   block = 0; // For the logic
   lastBlockBytes = (FTI_Exec->meta[0].fs) % FTI_Conf->transferSize; // modulo for size of last block in bytes

   lfd = fopen(lfn, "wb");
   if (lfd == NULL) {
      FTI_Print("R4 cannot open the local ckpt. file.", FTI_DBUG);
      MPI_File_close(&pfh);
      return FTI_NSCS;
   }

   char *blBuf1 = talloc(char, FTI_Conf->transferSize);

   // Checkpoint files transfer from PFS
   while (block < nbBlocks) {

      // read block in parallel file
      buf = MPI_File_read_at(pfh, offset, blBuf1, FTI_Conf->transferSize, MPI_BYTE, &status);
      // check if successfull
      if (buf != 0) {
         errno = 0;
         MPI_Error_string(buf, mpi_err, &reslen);
         snprintf(str, FTI_BUFS, "R4 cannot read from the ckpt. file in the PFS. [MPI ERROR - %i] %s", buf, mpi_err);
         FTI_Print(str, FTI_EROR);
         MPI_File_close(&pfh);
         fclose(lfd);
         return FTI_NSCS;
      }

      fwrite(blBuf1, sizeof(char), FTI_Conf->transferSize, lfd);
      if (ferror(lfd)) {
         FTI_Print("R4 cannot write to the local ckpt. file.", FTI_DBUG);
         free(blBuf1);
         fclose(lfd);
         MPI_File_close(&pfh);
         return  FTI_NSCS;
      }

      offset += FTI_Conf->transferSize;
      block++;
   }

   // read block in parallel file
   buf = MPI_File_read_at(pfh, offset, blBuf1, lastBlockBytes, MPI_BYTE, &status);
   // check if successfull
   if (buf != 0) {
      errno = 0;
      MPI_Error_string(buf, mpi_err, &reslen);
      snprintf(str, FTI_BUFS, "R4 cannot read from the ckpt. file in the PFS. [MPI ERROR - %i] %s", buf, mpi_err);
      FTI_Print(str, FTI_EROR);
      MPI_File_close(&pfh);
      fclose(lfd);
      return FTI_NSCS;
   }

   fwrite(blBuf1, sizeof(char), lastBlockBytes, lfd);
   if (ferror(lfd)) {
      FTI_Print("R4 cannot write to the local ckpt. file.", FTI_DBUG);
      free(blBuf1);
      fclose(lfd);
      MPI_File_close(&pfh);
      return  FTI_NSCS;
   }


   free(blBuf1);

   buf = MPI_File_close(&pfh);
   fclose(lfd);

   return FTI_SCES;

}

/*-------------------------------------------------------------------------*/
/**
    @brief      Recover L4 ckpt. files from the PFS using SIONlib.
    @return     integer         FTI_SCES if successful.

    This function tries to recover the ckpt. files using the L4 ckpt. files
    stored in the PFS. If at least one ckpt. file is missing in the PFS, we
    consider this checkpoint unavailable.

 **/
/*-------------------------------------------------------------------------*/
#ifdef ENABLE_SIONLIB // --> If SIONlib is installed
int FTI_RecoverL4Sionlib(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
      FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt)
{
   size_t nbBlocks, block = 0, lastBlockBytes;
   int j, l, sid, nlocaltasks = 1, numFiles=1, nlocalfiles = 1, res, noFile;
   int *gRankList, *file_map, *rank_map;
   char str[FTI_BUFS], gfn[FTI_BUFS], lfn[FTI_BUFS], *newfname = NULL;
   FILE *gfd, *lfd, *dfp;
   MPI_Comm lComm;
   sion_int64 *chunkSizes, chunksize;
   sion_int32 fsblksize=-1;

   if (FTI_Topo->nodeRank == 0 || FTI_Topo->nodeRank == 1) {
      if (mkdir(FTI_Ckpt[1].dir, 0777) == -1) {
         if (errno != EEXIST) {
            FTI_Print("Directory L1 could NOT be created.", FTI_WARN);
         }
      }
   }

   MPI_Barrier(FTI_COMM_WORLD);

   sprintf(gfn, "%s/%s", FTI_Ckpt[4].dir, FTI_Exec->ckptFile);

   sprintf(FTI_Exec->ckptFile, "Ckpt%d-Rank%d.fti", FTI_Exec->ckptID, FTI_Topo->myRank);
   sprintf(lfn, "%s/%s", FTI_Ckpt[1].dir, FTI_Exec->ckptFile);

   // this is done, since sionlib aborts if the file is not readable.
   if (access(gfn, F_OK) != 0) {
      return FTI_NSCS;
   }

   // set parameter for paropen mapped call
   if (FTI_Topo->nbHeads == 1 && FTI_Topo->groupID == 1) {

      // for the case that we have a head
      nlocaltasks = 2;
      gRankList = talloc(int, 2);
      chunkSizes = talloc(sion_int64, 2);
      file_map = talloc(int, 2);
      rank_map = talloc(int, 2);

      //        chunkSizes[0] = 0;
      //        chunkSizes[1] = ckptSize;
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

      //        *chunkSizes = ckptSize;
      *gRankList = FTI_Topo->myRank;
      *file_map = 0;
      *rank_map = *gRankList;

   }

   sid = sion_paropen_mapped_mpi(gfn, "rb,posix", &numFiles, FTI_COMM_WORLD, &nlocaltasks, &gRankList, &chunkSizes, &file_map, &rank_map, &fsblksize, &dfp); 

   lfd = fopen(lfn, "wb");
   if (lfd == NULL) {
      FTI_Print("R4 cannot open the local ckpt. file.", FTI_DBUG);
      return FTI_NSCS;
   }

   char *blBuf1 = talloc(char, FTI_Conf->blockSize);

   res = sion_seek(sid, FTI_Topo->myRank, SION_CURRENT_BLK, SION_CURRENT_POS);
   // check if successful
   if (res != SION_SUCCESS) {
      FTI_Print("SIONlib: Could not set file pointer", FTI_EROR);
      sion_parclose_mapped_mpi(FTI_Exec->sid);
      fclose(lfd);
      return FTI_NSCS;
   }


   // Checkpoint files transfer from PFS
   while (!sion_feof(sid)) {

      chunksize = FTI_Exec->meta[0].fs;

      nbBlocks = chunksize / FTI_Conf->blockSize;
      block = 0;
      lastBlockBytes = chunksize % FTI_Conf->blockSize;


      while (block < nbBlocks) {

         res = sion_fread(blBuf1, sizeof(char), FTI_Conf->blockSize, sid);
         if (res != FTI_Conf->blockSize) {
            sprintf(str, "SIONlib: Unable to read %i Bytes from file", FTI_Conf->blockSize);
            FTI_Print(str, FTI_EROR);
            sion_parclose_mapped_mpi(FTI_Exec->sid);
            fclose(lfd);
            return FTI_NSCS;
         }

         fwrite(blBuf1, sizeof(char), FTI_Conf->blockSize, lfd);
         if (ferror(lfd)) {
            FTI_Print("R4 cannot write to the local ckpt. file.", FTI_DBUG);
            free(blBuf1);
            fclose(lfd);
            sion_parclose_mapped_mpi(FTI_Exec->sid);
            return  FTI_NSCS;
         }

         block++;
      }

      res = sion_fread(blBuf1, sizeof(char), lastBlockBytes, sid);
      if (res != lastBlockBytes) {
         sprintf(str, "SIONlib: Unable to read data %i Bytes from file", FTI_Conf->blockSize);
         FTI_Print(str, FTI_EROR);
         sion_parclose_mapped_mpi(FTI_Exec->sid);
         fclose(lfd);
         return FTI_NSCS;
      }

      fwrite(blBuf1, sizeof(char), lastBlockBytes, lfd);
      if (ferror(lfd)) {
         FTI_Print("R4 cannot write to the local ckpt. file.", FTI_DBUG);
         free(blBuf1);
         fclose(lfd);
         sion_parclose_mapped_mpi(FTI_Exec->sid);
         return  FTI_NSCS;
      }


   }
   free(blBuf1);

   fclose(lfd);

   sion_parclose_mapped_mpi(sid);

   return FTI_SCES;

}
#endif
