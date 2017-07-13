/**
 *  @file   postreco.c
 *  @author Leonardo A. Bautista Gomez (leobago@gmail.com)
 *  @date   January, 2014
 *  @brief  Post recovery functions for the FTI library.
 */

#include "interface.h"

/*-------------------------------------------------------------------------*/
/**
    @brief      It recovers a set of ckpt. files using RS decoding.
    @return     integer         FTI_SCES if successful.

    This function tries to recover the L3 ckpt. files missing using the
    RS decoding.

 **/
/*-------------------------------------------------------------------------*/
int FTI_Decode(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
               FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt, int* erased)
{
    int *matrix, *decMatrix, *dm_ids, *tmpmat, i, j, k, m, ps, bs, pos = 0;
    char **coding, **data, *dataTmp, fn[FTI_BUFS], efn[FTI_BUFS];
    FILE *fd, *efd;
    long fs = FTI_Exec->meta[3].fs[0];
    long maxFs = FTI_Exec->meta[3].maxFs[0];

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
            fread(data[FTI_Topo->groupRank] + 0, sizeof(char), bs, fd);

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
            fread(coding[FTI_Topo->groupRank] + 0, sizeof(char), bs, efd);
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
    @brief      It checks that all L1 ckpt. files are present.
    @param      group           The group ID.
    @return     integer         FTI_SCES if successful.

    This function detects all the erasures for L1. If there is at least one,
    L1 is not considered as recoverable.

 **/
/*-------------------------------------------------------------------------*/
int FTI_RecoverL1(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
                  FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt)
{
    int erased[FTI_BUFS], buf, j; // FTI_BUFS > 32*3
    unsigned long fs, maxFs;
    if (FTI_CheckErasures(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt, erased) != FTI_SCES) {
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
    @brief      It sends checkpint file.
    @param      destination     destination group rank
    @param      fs              filesize
    @param      ptner           0 if sending Ckpt, 1 if PtnerCkpt

    @return     integer         FTI_SCES if successful.

    This function sends Ckpt or PtnerCkpt file from partner proccess.

 **/
/*-------------------------------------------------------------------------*/
int FTI_SendCkptFileL2(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
                        FTIT_checkpoint* FTI_Ckpt, int destination, int ptner) {
    char filename[FTI_BUFS], str[FTI_BUFS];
    FILE *fileDesc;

    long toSend ; // remaining data to send
    if (ptner) {    //if want to send Ptner file
        int rank;
        sscanf(FTI_Exec->meta[2].ckptFile, "Ckpt%d-Rank%d.fti", &FTI_Exec->ckptID, &rank); //do we need this from filename?
        sprintf(filename, "%s/Ckpt%d-Pcof%d.fti", FTI_Ckpt[2].dir, FTI_Exec->ckptID, rank);
        toSend = FTI_Exec->meta[2].pfs[0];
    } else {    //if want to send Ckpt file
        sprintf(filename, "%s/%s", FTI_Ckpt[2].dir, FTI_Exec->meta[2].ckptFile);
        toSend = FTI_Exec->meta[2].fs[0];
    }

    sprintf(str, "Opening file (rb) (%s) (L2).", filename);
    FTI_Print(str, FTI_DBUG);
    fileDesc = fopen(filename, "rb");
    if (fileDesc == NULL) {
        FTI_Print("R2 cannot open the partner ckpt. file.", FTI_WARN);
        return FTI_NSCS;
    }
    char* buffer = talloc(char, FTI_Conf->blockSize);

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
    @brief      It receives checkpint file.
    @param      source          source group rank
    @param      fs              filesize
    @param      ptner           0 if receiving Ckpt, 1 if PtnerCkpt
    @return     integer         FTI_SCES if successful.

    This function receives Ckpt or PtnerCkpt file from partner proccess.

 **/
/*-------------------------------------------------------------------------*/
int FTI_RecvCkptFileL2(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
                        FTIT_checkpoint* FTI_Ckpt, int source, int ptner) {
    char filename[FTI_BUFS], str[FTI_BUFS];
    FILE *fileDesc;

    long toRecv;    //remaining data to receive
    if (ptner) { //if want to receive Ptner file
        int rank;
        sscanf(FTI_Exec->meta[2].ckptFile, "Ckpt%d-Rank%d.fti", &FTI_Exec->ckptID, &rank); //do we need this from filename?
        sprintf(filename, "%s/Ckpt%d-Pcof%d.fti", FTI_Ckpt[2].dir, FTI_Exec->ckptID, rank);
        toRecv = FTI_Exec->meta[2].pfs[0];
    } else { //if want to receive Ckpt file
        sprintf(filename, "%s/%s", FTI_Ckpt[2].dir, FTI_Exec->meta[2].ckptFile);
        toRecv = FTI_Exec->meta[2].fs[0];
    }

    sprintf(str, "Opening file (wb) (%s) (L2).", filename);
    FTI_Print(str, FTI_DBUG);
    fileDesc = fopen(filename, "wb");
    if (fileDesc == NULL) {
        FTI_Print("R2 cannot open the file.", FTI_WARN);
        return FTI_NSCS;
    }
    char* buffer = talloc(char, FTI_Conf->blockSize);

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
    @brief      It recovers L2 ckpt. files using the partner copy.
    @param      group           The group ID.
    @return     integer         FTI_SCES if successful.

    This function tries to recover the L2 ckpt. files missing using the
    partner copy. If a ckpt. file and its copy are both missing, then we
    consider this checkpoint unavailable.

 **/
/*-------------------------------------------------------------------------*/
int FTI_RecoverL2(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
                  FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt)
{
    int erased[FTI_BUFS];
    int i, j; //iterators
    int source = FTI_Topo->right; //to receive Ptner file from this process (to recover)
    int destination = FTI_Topo->left; //to send Ptner file (to let him recover)
    int res, tres;

    if (mkdir(FTI_Ckpt[2].dir, 0777) == -1) {
        if (errno != EEXIST) {
            FTI_Print("Cannot create directory", FTI_EROR);
        }
    }

    // Checking erasures
    if (FTI_CheckErasures(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt, erased) != FTI_SCES) {
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
        return FTI_NSCS;
    }

    //recover checkpoint files
    if (FTI_Topo->groupRank % 2) {
        if (erased[destination]) { //first send file
            res = FTI_SendCkptFileL2(FTI_Conf, FTI_Exec, FTI_Ckpt, destination, 1);
            if (res != FTI_SCES) {
                return FTI_NSCS;
            }
        }
        if (erased[FTI_Topo->groupRank]) { //then receive file
            res = FTI_RecvCkptFileL2(FTI_Conf, FTI_Exec, FTI_Ckpt, source, 0);
            if (res != FTI_SCES) {
                return FTI_NSCS;
            }
        }
    } else {
        if (erased[FTI_Topo->groupRank]) { //first receive file
            res = FTI_RecvCkptFileL2(FTI_Conf, FTI_Exec, FTI_Ckpt, source, 0);
            if (res != FTI_SCES) {
                return FTI_NSCS;
            }
        }

        if (erased[destination]) { //then send file
            res = FTI_SendCkptFileL2(FTI_Conf, FTI_Exec, FTI_Ckpt, destination, 1);
            if (res != FTI_SCES) {
                return FTI_NSCS;
            }
        }
    }

    //recover partner files
    if (FTI_Topo->groupRank % 2) {
        if (erased[source + FTI_Topo->groupSize]) { //fisrst send file
            res = FTI_SendCkptFileL2(FTI_Conf, FTI_Exec, FTI_Ckpt, source, 0);
            if (res != FTI_SCES) {
                return FTI_NSCS;
            }
        }
        if (erased[FTI_Topo->groupRank + FTI_Topo->groupSize]) { //receive file
            res = FTI_RecvCkptFileL2(FTI_Conf, FTI_Exec, FTI_Ckpt, destination, 1);
            if (res != FTI_SCES) {
                return FTI_NSCS;
            }
        }
    } else {
        if (erased[FTI_Topo->groupRank + FTI_Topo->groupSize]) { //first receive file
            res = FTI_RecvCkptFileL2(FTI_Conf, FTI_Exec, FTI_Ckpt, destination, 1);
            if (res != FTI_SCES) {
                return FTI_NSCS;
            }
        }

        if (erased[source + FTI_Topo->groupSize]) { //send file
            res = FTI_SendCkptFileL2(FTI_Conf, FTI_Exec, FTI_Ckpt, source, 0);
            if (res != FTI_SCES) {
                return FTI_NSCS;
            }
        }
    }

    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
    @brief      It recovers L3 ckpt. files ordering the RS decoding algorithm.
    @param      group           The group ID.
    @return     integer         FTI_SCES if successful.

    This function tries to recover the L3 ckpt. files missing using the
    RS decoding. If to many files are missing in the group, then we
    consider this checkpoint unavailable.

 **/
/*-------------------------------------------------------------------------*/
int FTI_RecoverL3(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
                  FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt)
{
    int erased[FTI_BUFS], j;
    char str[FTI_BUFS];


    if (mkdir(FTI_Ckpt[3].dir, 0777) == -1) {
        if (errno != EEXIST) {
            FTI_Print("Cannot create directory", FTI_EROR);
        }
    }

    // Checking erasures
    if (FTI_CheckErasures(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt, erased) != FTI_SCES) {
        FTI_Print("Error checking erasures.", FTI_DBUG);
        return FTI_NSCS;
    }

    // Counting erasures
    int l = 0;
    int gs = FTI_Topo->groupSize;
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
        if (FTI_Decode(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt, erased) == FTI_NSCS) {
            FTI_Print("RS-decoding could not regenerate the missing data.", FTI_DBUG);
            return FTI_NSCS;
        }
    }

    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
    @brief      It recovers L4 ckpt. files from the PFS.
    @param      group           The group ID.
    @return     integer         FTI_SCES if successful.

    This function tries to recover the ckpt. files using the L4 ckpt. files
    stored in the PFS. If at least one ckpt. file is missing in the PFS, we
    consider this checkpoint unavailable.

 **/
/*-------------------------------------------------------------------------*/
int FTI_RecoverL4(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
                  FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt)
{
   int res;
   if (!FTI_Ckpt[4].isInline) {
       res = FTI_RecoverL4Posix(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt);
   }
   else {
       switch(FTI_Conf->ioMode) {
          case FTI_IO_POSIX:
             res = FTI_RecoverL4Posix(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt);
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
   }

   return res;
}

/*-------------------------------------------------------------------------*/
/**
    @brief      It recovers L4 ckpt. files from the PFS using POSIX.
    @param      group           The group ID.
    @return     integer         FTI_SCES if successful.

    This function tries to recover the ckpt. files using the L4 ckpt. files
    stored in the PFS. If at least one ckpt. file is missing in the PFS, we
    consider this checkpoint unavailable.

 **/
/*-------------------------------------------------------------------------*/
int FTI_RecoverL4Posix(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
                  FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt)
{
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
   if (FTI_CheckErasures(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt, erased) != FTI_SCES) {
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

   // Open and resize files
   sprintf(FTI_Exec->meta[1].ckptFile, "Ckpt%d-Rank%d.fti", FTI_Exec->ckptID, FTI_Topo->myRank);
   sprintf(lfn, "%s/%s", FTI_Ckpt[1].dir, FTI_Exec->meta[1].ckptFile);
   sprintf(gfn, "%s/%s", FTI_Ckpt[4].dir, FTI_Exec->meta[4].ckptFile);

   gfd = fopen(gfn, "rb");
   if (gfd == NULL) {
      FTI_Print("R4 cannot open the ckpt. file in the PFS.", FTI_WARN);
      return FTI_NSCS;
   }

   lfd = fopen(lfn, "wb");
   if (lfd == NULL) {
      FTI_Print("R4 cannot open the local ckpt. file.", FTI_WARN);
      fclose(gfd);
      return FTI_NSCS;
   }

   char *readData = talloc(char, FTI_Conf->transferSize);
   long bSize = FTI_Conf->transferSize;
   long fs = FTI_Exec->meta[4].fs[0];
   // Checkpoint files transfer from PFS
   long pos = 0;
   while (pos < fs) {
      if ((fs - pos) < FTI_Conf->transferSize) {
         bSize = fs - pos;
      }

      size_t bytes = fread(readData, sizeof(char), bSize, gfd);
      if (ferror(gfd)) {
         FTI_Print("R4 cannot read from the ckpt. file in the PFS.", FTI_DBUG);

         free(readData);

         fclose(gfd);
         fclose(lfd);

         return  FTI_NSCS;
      }

      fwrite(readData, sizeof(char), bytes, lfd);
      if (ferror(lfd)) {
         FTI_Print("R4 cannot write to the local ckpt. file.", FTI_DBUG);

         free(readData);

         fclose(gfd);
         fclose(lfd);

         return  FTI_NSCS;
      }

      pos = pos + bytes;
   }

   free(readData);

   fclose(gfd);
   fclose(lfd);

   return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
    @brief      It recovers L4 ckpt. files from the PFS using MPI-I/O.
    @return     integer         FTI_SCES if successful.

    This function tries to recover the ckpt. files using the L4 ckpt. files
    stored in the PFS. If at least one ckpt. file is missing in the PFS, we
    consider this checkpoint unavailable.

 **/
/*-------------------------------------------------------------------------*/
int FTI_RecoverL4Mpi(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
      FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt)
{
   int i, reslen, buf;
   char gfn[FTI_BUFS], lfn[FTI_BUFS], mpi_err[FTI_BUFS], str[FTI_BUFS];

   // TODO enable to set stripping unit in the config file (Maybe also other hints)
   // enable collective buffer optimization
   MPI_Info info;
   MPI_Info_create(&info);
   MPI_Info_set(info, "romio_cb_read", "enable");

   // set stripping unit to 4MB
   MPI_Info_set(info, "stripping_unit", "4194304");

   sprintf(FTI_Exec->meta[1].ckptFile, "Ckpt%d-Rank%d.fti", FTI_Exec->ckptID, FTI_Topo->myRank);
   sprintf(lfn, "%s/%s", FTI_Ckpt[1].dir, FTI_Exec->meta[1].ckptFile);
   sprintf(gfn, "%s/%s", FTI_Ckpt[4].dir, FTI_Exec->meta[4].ckptFile);

   // open parallel file
   MPI_File pfh;
   buf = MPI_File_open(FTI_COMM_WORLD, gfn, MPI_MODE_RDWR, info, &pfh);
   // check if successful
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
   if (mkdir(FTI_Ckpt[1].dir, 0777) == -1) {
      if (errno != EEXIST) {
          FTI_Print("Directory L1 could NOT be created.", FTI_WARN);
      }
   }

   // collect chunksizes of other ranks
   MPI_Offset* chunkSizes = talloc(MPI_Offset, FTI_Topo->nbApprocs*FTI_Topo->nbNodes);
   MPI_Allgather(FTI_Exec->meta[4].fs, 1, MPI_OFFSET, chunkSizes, 1, MPI_OFFSET, FTI_COMM_WORLD);

   MPI_Offset offset = 0;
   // set file offset
   for (i = 0; i < FTI_Topo->splitRank; i++) {
      offset += chunkSizes[i];
   }

   FILE *lfd = fopen(lfn, "wb");
   if (lfd == NULL) {
      FTI_Print("R4 cannot open the local ckpt. file.", FTI_DBUG);
      MPI_File_close(&pfh);
      return FTI_NSCS;
   }

   long fs = FTI_Exec->meta[4].fs[0];
   char *readData = talloc(char, FTI_Conf->transferSize);
   long bSize = FTI_Conf->transferSize;
   long pos = 0;
   // Checkpoint files transfer from PFS
   while (pos < fs) {
       if ((fs - pos) < FTI_Conf->transferSize) {
           bSize = fs - pos;
       }
      // read block in parallel file
      buf = MPI_File_read_at(pfh, offset, readData, bSize, MPI_BYTE, MPI_STATUS_IGNORE);
      // check if successful
      if (buf != 0) {
         errno = 0;
         MPI_Error_string(buf, mpi_err, &reslen);
         snprintf(str, FTI_BUFS, "R4 cannot read from the ckpt. file in the PFS. [MPI ERROR - %i] %s", buf, mpi_err);
         FTI_Print(str, FTI_EROR);
         MPI_File_close(&pfh);
         fclose(lfd);
         return FTI_NSCS;
      }

      fwrite(readData, sizeof(char), bSize, lfd);
      if (ferror(lfd)) {
         FTI_Print("R4 cannot write to the local ckpt. file.", FTI_DBUG);
         free(readData);
         fclose(lfd);
         MPI_File_close(&pfh);
         return  FTI_NSCS;
      }

      offset += bSize;
      pos = pos + bSize;
   }

   free(readData);
   fclose(lfd);

   if (MPI_File_close(&pfh) != 0) {
      FTI_Print("Cannot close MPI file.", FTI_WARN);
      return FTI_NSCS;
   }

   return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
    @brief      It recovers L4 ckpt. files from the PFS using SIONlib.
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
   int res;
   char str[FTI_BUFS], gfn[FTI_BUFS], lfn[FTI_BUFS];

   if (mkdir(FTI_Ckpt[1].dir, 0777) == -1) {
       if (errno != EEXIST) {
           FTI_Print("Directory L1 could NOT be created.", FTI_WARN);
       }
   }

   sprintf(lfn, "%s/%s", FTI_Ckpt[1].dir, FTI_Exec->meta[1].ckptFile);
   sprintf(gfn, "%s/%s", FTI_Ckpt[4].dir, FTI_Exec->meta[4].ckptFile);

   // this is done, since sionlib aborts if the file is not readable.
   if (access(gfn, F_OK) != 0) {
      return FTI_NSCS;
   }

   int numFiles = 1;
   int nlocaltasks = 1;
   int* file_map = calloc(1, sizeof(int));
   int* ranks = talloc(int, 1);
   int* rank_map = talloc(int, 1);
   sion_int64* chunkSizes = talloc(sion_int64, 1);
   int fsblksize = -1;
   chunkSizes[0] = FTI_Exec->meta[4].fs[0];
   ranks[0] = FTI_Topo->splitRank;
   rank_map[0] = FTI_Topo->splitRank;
   int sid = sion_paropen_mapped_mpi(gfn, "rb,posix", &numFiles, FTI_COMM_WORLD, &nlocaltasks, &ranks, &chunkSizes, &file_map, &rank_map, &fsblksize, NULL);

   FILE* lfd = fopen(lfn, "wb");
   if (lfd == NULL) {
      FTI_Print("R4 cannot open the local ckpt. file.", FTI_DBUG);
      return FTI_NSCS;
   }

   char *readData = talloc(char, FTI_Conf->transferSize);
   MPI_Barrier(FTI_COMM_WORLD);
   res = sion_seek(sid, FTI_Topo->splitRank, SION_CURRENT_BLK, SION_CURRENT_POS);
   // check if successful
   if (res != SION_SUCCESS) {
      FTI_Print("SIONlib: Could not set file pointer", FTI_EROR);
      sion_parclose_mapped_mpi(sid);
      free(readData);
      fclose(lfd);
      return FTI_NSCS;
   }

   // Checkpoint files transfer from PFS
   while (!sion_feof(sid)) {
      long fs = FTI_Exec->meta[4].fs[0];
      char *readData = talloc(char, FTI_Conf->transferSize);
      long bSize = FTI_Conf->transferSize;
      long pos = 0;
      // Checkpoint files transfer from PFS
      while (pos < fs) {
         if ((fs - pos) < FTI_Conf->transferSize) {
             bSize = fs - pos;
         }
         res = sion_fread(readData, sizeof(char), bSize, sid);
         if (res != bSize) {
            sprintf(str, "SIONlib: Unable to read %lu Bytes from file", bSize);
            FTI_Print(str, FTI_EROR);
            sion_parclose_mapped_mpi(sid);
            free(readData);
            fclose(lfd);
            return FTI_NSCS;
         }

         fwrite(readData, sizeof(char), bSize, lfd);
         if (ferror(lfd)) {
            FTI_Print("R4 cannot write to the local ckpt. file.", FTI_DBUG);
            free(readData);
            fclose(lfd);
            sion_parclose_mapped_mpi(sid);
            return  FTI_NSCS;
         }

         pos = pos + bSize;
      }
      if (FTI_Topo->splitRank == 3) {
          sprintf(str, "Read data %ld. S%dG%d", pos, FTI_Topo->sectorID, FTI_Topo->groupID);
          FTI_Print(str, FTI_WARN);
      }
   }
   free(readData);

   fclose(lfd);

   sion_parclose_mapped_mpi(sid);

   return FTI_SCES;
}
#endif
