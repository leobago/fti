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
 *  @file   postreco.c
 *  @date   October, 2017
 *  @brief  Post recovery functions for the FTI library.
 */
#include <time.h>

#include "postreco.h"
/*-------------------------------------------------------------------------*/
/**
  @brief      It recovers a set of ckpt. files using RS decoding.
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Topo        Topology metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @param      erased          The array of erasures.
  @return     integer         FTI_SCES if successful.

  This function tries to recover the L3 ckpt. files missing using the
  RS decoding.

 **/
/*-------------------------------------------------------------------------*/
int FTI_Decode(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt, int* erased) {
    int ckptId, rank;
    sscanf(FTI_Exec->ckptMeta.ckptFile, "Ckpt%d-Rank%d.%s", &ckptId, &rank,
     FTI_Conf->suffix);
    char fn[FTI_BUFS], efn[FTI_BUFS];
    snprintf(efn, FTI_BUFS, "%s/Ckpt%d-RSed%d.%s", FTI_Ckpt[3].dir, ckptId,
     rank, FTI_Conf->suffix);
    snprintf(fn, FTI_BUFS, "%s/%s", FTI_Ckpt[3].dir,
     FTI_Exec->ckptMeta.ckptFile);

    int bs = FTI_Conf->blockSize;
    int k = FTI_Topo->groupSize;
    int m = k;

    int64_t fs = FTI_Exec->ckptMeta.fs;

    char** data = talloc(char*, k);
    char** coding = talloc(char*, m);
    char* dataTmp = talloc(char, FTI_Conf->blockSize* k);
    int* dm_ids = talloc(int, k);
    int* decMatrix = talloc(int, k* k);
    int* tmpmat = talloc(int, k* k);
    int* matrix = talloc(int, k* k);
    int i, j;
    for (i = 0; i < FTI_Topo->groupSize; i++) {
        for (j = 0; j < FTI_Topo->groupSize; j++) {
            matrix[i * FTI_Topo->groupSize + j] = galois_single_divide(1,
             i ^ (FTI_Topo->groupSize + j), FTI_Conf->l3WordSize);
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
        } else {
            for (j = 0; j < k; j++) {
                tmpmat[i * k + j] = matrix[(dm_ids[i] - k) * k + j];
            }
        }
    }

    // Inversing the matrix
    if (jerasure_invert_matrix(tmpmat, decMatrix, k,
     FTI_Conf->l3WordSize) < 0) {
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

    FILE *fd, *efd;
    int64_t maxFs = FTI_Exec->ckptMeta.maxFs;
    int64_t ps = ((maxFs / FTI_Conf->blockSize)) * FTI_Conf->blockSize;
    if (ps < maxFs) {
        ps = ps + FTI_Conf->blockSize;  // Calculating padding size
    }
    if (erased[FTI_Topo->groupRank] == 0) {  // Resize and open files
        // determine file size in order to write at the end of the
        // elongated and padded file (i.e. write at the end of file
        // after 'truncate(.., maxFs)'
        struct stat st_;
        if (FTI_Conf->ioMode == FTI_IO_FTIFF) {
            stat(fn, &st_);
        }

        if (truncate(fn, maxFs) == -1) {
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

        // after truncation we need to write the filesize into the file
        // in order to have the same file as at the state we performed
        // the encoding. In order to do so, we need to determine the
        // file size with stat, before the truncation!
        if (FTI_Conf->ioMode == FTI_IO_FTIFF) {
            int lftmp_ = open(fn, O_RDWR);
            if (lftmp_ == -1) {
                FTI_Print("FTI_RSenc: (FTIFF) Unable to open file!", FTI_EROR);
                return FTI_NSCS;
            }
            if (lseek(lftmp_, -sizeof(off_t), SEEK_END) == -1) {
                FTI_Print("FTI_RSenc: (FTIFF) Unable to seek in file!",
                 FTI_EROR);
                return FTI_NSCS;
            }
            if (write(lftmp_, &st_.st_size, sizeof(off_t)) == -1) {
                FTI_Print("FTI_RSenc: (FTIFF) Unable to write meta data"
                " in file!", FTI_EROR);
                return FTI_NSCS;
            }
            close(lftmp_);
        }
        fd = fopen(fn, "rb");
    } else {
        fd = fopen(fn, "wb");
    }

    if (erased[FTI_Topo->groupRank + FTI_Topo->groupSize] == 0) {
        efd = fopen(efn, "rb");
    } else {
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
    int64_t pos = 0;
    int remBsize = bs;

    MD5_CTX md5ctxRS;
    MD5_Init(&md5ctxRS);
    while (pos < ps) {
        if ((maxFs - pos) < bs) {
            remBsize = maxFs - pos;
        }

        // Reading the data
        if (erased[FTI_Topo->groupRank] == 0) {
            bzero(data[FTI_Topo->groupRank], bs);
            fread(data[FTI_Topo->groupRank] + 0, sizeof(char), remBsize, fd);

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

        if (erased[FTI_Topo->groupRank + FTI_Topo->groupSize] == 0) {
            bzero(coding[FTI_Topo->groupRank], bs);
            fread(coding[FTI_Topo->groupRank] + 0, sizeof(char), remBsize, efd);

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

        MPI_Allgather(data[FTI_Topo->groupRank] + 0, bs, MPI_CHAR, dataTmp, bs,
         MPI_CHAR, FTI_Exec->groupComm);
        for (i = 0; i < k; i++) {
            memcpy(data[i] + 0, &(dataTmp[i * bs]), sizeof(char) * bs);
        }

        MPI_Allgather(coding[FTI_Topo->groupRank] + 0, bs, MPI_CHAR, dataTmp,
         bs, MPI_CHAR, FTI_Exec->groupComm);
        for (i = 0; i < k; i++) {
            memcpy(coding[i] + 0, &(dataTmp[i * bs]), sizeof(char) * bs);
        }

        // Decoding the lost data work
        if (erased[FTI_Topo->groupRank]) {
            jerasure_matrix_dotprod(k, FTI_Conf->l3WordSize,
             decMatrix + (FTI_Topo->groupRank * k), dm_ids,
             FTI_Topo->groupRank, data, coding, bs);
        }

        MPI_Allgather(data[FTI_Topo->groupRank] + 0, bs, MPI_CHAR,
         dataTmp, bs, MPI_CHAR, FTI_Exec->groupComm);
        for (i = 0; i < k; i++) {
            memcpy(data[i] + 0, &(dataTmp[i * bs]), sizeof(char) * bs);
        }

        // Finally, re-encode any erased encoded checkpoint file
        if (erased[FTI_Topo->groupRank + k]) {
            jerasure_matrix_dotprod(k, FTI_Conf->l3WordSize,
             matrix + (FTI_Topo->groupRank * k), NULL, FTI_Topo->groupRank + k,
              data, coding, bs);
        }
        if (erased[FTI_Topo->groupRank]) {
            fwrite(data[FTI_Topo->groupRank] + 0, sizeof(char), remBsize, fd);
        }
        if (erased[FTI_Topo->groupRank + k]) {
            MD5_Update(&md5ctxRS, coding[FTI_Topo->groupRank], remBsize);
            fwrite(coding[FTI_Topo->groupRank] + 0, sizeof(char),
             remBsize, efd);
        }

        pos = pos + bs;
    }
    unsigned char hashRS[MD5_DIGEST_LENGTH];
    MD5_Final(hashRS, &md5ctxRS);


    // Closing files
    fclose(fd);
    fclose(efd);

    // FTI-FF: if file ckpt file deleted, determine fs from recovered file
    if (FTI_Conf->ioMode == FTI_IO_FTIFF && erased[FTI_Topo->groupRank]) {
        char str[FTI_BUFS];

        int ifd = open(fn, O_RDONLY);
        if (ifd == -1) {
            snprintf(str, FTI_BUFS,
             "failed to read FTI-FF file meta data from file '%s'", fn);
            FTI_Print(str, FTI_EROR);
            errno = 0;
            return FTI_NSCS;
        }

        if (lseek(ifd, -sizeof(off_t), SEEK_END) == -1) {
            snprintf(str, FTI_BUFS,
             "failed to read FTI-FF file meta data from file '%s'", fn);
            FTI_Print(str, FTI_EROR);
            errno = 0;
            close(ifd);
            return FTI_NSCS;
        }

        off_t fs_;
        if (read(ifd, &fs_, sizeof(off_t)) == -1) {
            snprintf(str, FTI_BUFS,
             "failed to read FTI-FF file meta data from file '%s'", fn);
            FTI_Print(str, FTI_EROR);
            errno = 0;
            close(ifd);
            return FTI_NSCS;
        }
        
        fs = (int64_t) fs_;
        FTI_Exec->ckptMeta.fs = fs;

        close(ifd);
    }

    // FTI-FF: if encoded file deleted, append meta data to encoded file
    if (FTI_Conf->ioMode == FTI_IO_FTIFF && erased[FTI_Topo->groupRank + k]) {
        char str[FTI_BUFS];
        FTIFF_metaInfo *FTIFFMeta = malloc(sizeof(FTIFF_metaInfo));

        // get timestamp
        struct timespec ntime;
        clock_gettime(CLOCK_REALTIME, &ntime);

        FTIFFMeta->timestamp = ntime.tv_sec*1000000000 + ntime.tv_nsec;
        FTIFFMeta->fs = maxFs;
        FTIFFMeta->ptFs = -1;
        FTIFFMeta->maxFs = maxFs;
        FTIFFMeta->ckptSize = fs;

        char checksum[MD5_DIGEST_STRING_LENGTH];
        int ii = 0;
        for (i = 0; i < MD5_DIGEST_LENGTH; i++) {
            snprintf(&checksum[ii], sizeof(char[3]), "%02x", hashRS[i]);
            ii+=2;
        }
        strncpy(FTIFFMeta->checksum, checksum, MD5_DIGEST_STRING_LENGTH);

        // add hash of meta info to meta info structure
        FTIFF_GetHashMetaInfo(FTIFFMeta->myHash, FTIFFMeta);

        // append meta info to RS file
        int ifd = open(efn, O_WRONLY|O_APPEND);
        char* buffer_ser = talloc(char, FTI_filemetastructsize);
        if (buffer_ser == NULL) {
            FTI_Print("failed to allocate memory for FTI-FF file meta data.",
             FTI_EROR);
            errno = 0;
            close(ifd);
            return FTI_NSCS;
        }
        if (FTIFF_SerializeFileMeta(FTIFFMeta, buffer_ser) != FTI_SCES) {
            FTI_Print("failed to serialize FTI-FF file meta data.", FTI_EROR);
            errno = 0;
            close(ifd);
            return FTI_NSCS;
        }
        if (write(ifd, buffer_ser, FTI_filemetastructsize) == -1) {
            snprintf(str, FTI_BUFS,
             "failed to write FTI-FF file meta data to file '%s'", efn);
            FTI_Print(str, FTI_EROR);
            errno = 0;
            close(ifd);
            return FTI_NSCS;
        }
        close(ifd);
        free(buffer_ser);
    }
    
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
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Topo        Topology metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @return     integer         FTI_SCES if successful.

  This function detects all the erasures for L1. If there is at least one,
  L1 is not considered as recoverable.

 **/
/*-------------------------------------------------------------------------*/
int FTI_RecoverL1(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt) {
    if (FTI_Conf->ioMode == FTI_IO_FTIFF) {
        if ( FTIFF_CheckL1RecoverInit(FTI_Exec, FTI_Topo, FTI_Ckpt,
         FTI_Conf) != FTI_SCES) {
            FTI_Print("No restart possible from L1. Ckpt files missing.",
             FTI_DBUG);
            return FTI_NSCS;
        }
    } else {
        int erased[FTI_BUFS];  // FTI_BUFS > 32*3
        if (FTI_CheckErasures(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt,
         erased) != FTI_SCES) {
            FTI_Print("Error checking erasures.", FTI_DBUG);
            return FTI_NSCS;
        }
        int buf = 0;
        int i;
        for (i = 0; i < FTI_Topo->groupSize; i++) {
            if (erased[i]) {
                buf++;  // Counting erasures
            }
        }
        if (buf > 0) {
            FTI_Print("Checkpoint files missing at L1.", FTI_WARN);
            return FTI_NSCS;
        }
    }
    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      It sends checkpint file.
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @param      destination     destination group rank.
  @param      ptner           0 if sending Ckpt, 1 if PtnerCkpt.
  @return     integer         FTI_SCES if successful.

  This function sends Ckpt or PtnerCkpt file from partner proccess.

 **/
/*-------------------------------------------------------------------------*/
int FTI_SendCkptFileL2(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_checkpoint* FTI_Ckpt, int destination, int ptner) {
    int64_t toSend;  // remaining data to send
    char filename[FTI_BUFS], str[FTI_BUFS];
    if (ptner) {  // if want to send Ptner file
        int ckptId, rank;
        sscanf(FTI_Exec->ckptMeta.ckptFile, "Ckpt%d-Rank%d.%s", &ckptId, &rank,
         FTI_Conf->suffix);  // do we need this from filename?
        snprintf(filename, FTI_BUFS, "%s/Ckpt%d-Pcof%d.%s", FTI_Ckpt[2].dir,
         ckptId, rank, FTI_Conf->suffix);
        toSend = FTI_Exec->ckptMeta.pfs;
    } else {  // if want to send Ckpt file
        snprintf(filename, FTI_BUFS, "%s/%s", FTI_Ckpt[2].dir,
         FTI_Exec->ckptMeta.ckptFile);
        toSend = FTI_Exec->ckptMeta.fs;
    }

    snprintf(str, FTI_BUFS, "Opening file (rb) (%s) (L2).", filename);
    FTI_Print(str, FTI_DBUG);
    FILE* fileDesc = fopen(filename, "rb");
    if (fileDesc == NULL) {
        FTI_Print("R2 cannot open the partner ckpt. file.", FTI_WARN);
        return FTI_NSCS;
    }
    char* buffer = talloc(char, FTI_Conf->blockSize);

    while (toSend > 0) {
        int sendSize = (toSend > FTI_Conf->blockSize) ?
         FTI_Conf->blockSize : toSend;
        size_t bytes = fread(buffer, sizeof(char), sendSize, fileDesc);

        if (ferror(fileDesc)) {
            FTI_Print("Error reading the data from the ckpt. file.", FTI_WARN);

            fclose(fileDesc);
            free(buffer);

            return FTI_NSCS;
        }
        MPI_Send(buffer, bytes, MPI_CHAR, destination, FTI_Conf->generalTag,
         FTI_Exec->groupComm);
        toSend -= bytes;
    }

    fclose(fileDesc);
    free(buffer);

    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      It receives checkpint file.
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @param      source          Source group rank.
  @param      ptner           0 if receiving Ckpt, 1 if PtnerCkpt.
  @return     integer         FTI_SCES if successful.

  This function receives Ckpt or PtnerCkpt file from partner proccess.

 **/
/*-------------------------------------------------------------------------*/
int FTI_RecvCkptFileL2(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_checkpoint* FTI_Ckpt, int source, int ptner) {
    int64_t toRecv;  // remaining data to receive
    char filename[FTI_BUFS], str[FTI_BUFS];
    if (ptner) {  // if want to receive Ptner file
        int ckptId, rank;
        sscanf(FTI_Exec->ckptMeta.ckptFile, "Ckpt%d-Rank%d.%s", &ckptId, &rank,
         FTI_Conf->suffix);
        snprintf(filename, FTI_BUFS, "%s/Ckpt%d-Pcof%d.%s", FTI_Ckpt[2].dir,
         FTI_Exec->ckptId, rank, FTI_Conf->suffix);
        toRecv = FTI_Exec->ckptMeta.pfs;
    } else {  // if want to receive Ckpt file
        snprintf(filename, FTI_BUFS, "%s/%s", FTI_Ckpt[2].dir,
         FTI_Exec->ckptMeta.ckptFile);
        toRecv = FTI_Exec->ckptMeta.fs;
    }

    snprintf(str, FTI_BUFS, "Opening file (wb) (%s) (L2).", filename);
    FTI_Print(str, FTI_DBUG);
    FILE* fileDesc = fopen(filename, "wb");
    if (fileDesc == NULL) {
        FTI_Print("R2 cannot open the file.", FTI_WARN);
        return FTI_NSCS;
    }
    char* buffer = talloc(char, FTI_Conf->blockSize);

    while (toRecv > 0) {
        int recvSize = (toRecv > FTI_Conf->blockSize) ?
         FTI_Conf->blockSize : toRecv;
        MPI_Recv(buffer, recvSize, MPI_CHAR, source, FTI_Conf->generalTag,
         FTI_Exec->groupComm, MPI_STATUS_IGNORE);
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
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Topo        Topology metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @return     integer         FTI_SCES if successful.

  This function tries to recover the L2 ckpt. files missing using the
  partner copy. If a ckpt. file and its copy are both missing, then we
  consider this checkpoint unavailable.

 **/
/*-------------------------------------------------------------------------*/
int FTI_RecoverL2(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt) {
    if (mkdir(FTI_Ckpt[2].dir, 0777) == -1) {
        if (errno != EEXIST) {
            FTI_Print("Cannot create directory", FTI_EROR);
        }
    }

    int erased[FTI_BUFS];
    // to receive Ptner file from this process (to recover)
    int source = FTI_Topo->right;
    // to send Ptner file (to let him recover)
    int destination = FTI_Topo->left;
    int res;

    if (FTI_Conf->ioMode == FTI_IO_FTIFF) {
        enum {
            LEFT_FILE,  // ckpt file of left partner (on left node)
            MY_FILE,    // my ckpt file (on my node)
            MY_COPY,    // copy of my ckpt file (on right node)
            LEFT_COPY   // copy of ckpt file of my left partner (on my node)
        };

        int exists[4];

        if (FTIFF_CheckL2RecoverInit(FTI_Exec, FTI_Topo, FTI_Ckpt, FTI_Conf,
         exists) != FTI_SCES) {
            FTI_Print("No restart possible from L2. Ckpt files missing.",
             FTI_DBUG);
            return FTI_NSCS;
        }

        memset(erased, 0x0, FTI_BUFS*sizeof(int));

        erased[destination] = !exists[LEFT_FILE];
        erased[FTI_Topo->groupRank] = !exists[MY_FILE];
        erased[source + FTI_Topo->groupSize] = !exists[MY_COPY];
        erased[FTI_Topo->groupRank + FTI_Topo->groupSize] = !exists[LEFT_COPY];
    } else {
        // Checking erasures
        if (FTI_CheckErasures(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt,
         erased) != FTI_SCES) {
            FTI_Print("Error checking erasures.", FTI_WARN);
            return FTI_NSCS;
        }

        int i = 0;
        int j;
        for (j = 0; j < FTI_Topo->groupSize * 2; j++) {
            if (erased[j]) {
                i++;  // Counting erasures
            }
        }

        if (i == 0) {
            FTI_Print("Have all checkpoint files.", FTI_DBUG);
            return FTI_SCES;
        }

        res = FTI_SCES;
        if (erased[FTI_Topo->groupRank] &&
         erased[source + FTI_Topo->groupSize]) {
            FTI_Print("My checkpoint file and partner copy have been lost",
             FTI_WARN);
            res = FTI_NSCS;
        }

        if (erased[FTI_Topo->groupRank + FTI_Topo->groupSize] &&
         erased[destination]) {
            FTI_Print("My Ptner checkpoint file and his checkpoint"
            " file have been lost", FTI_WARN);
            res = FTI_NSCS;
        }

        int allRes;
        MPI_Allreduce(&res, &allRes, 1, MPI_INT, MPI_SUM, FTI_Exec->groupComm);
        if (allRes != FTI_SCES) {
            return FTI_NSCS;
        }
    }
    // recover checkpoint files
    if (FTI_Topo->groupRank % 2) {
        if (erased[destination]) {  // first send file
            res = FTI_SendCkptFileL2(FTI_Conf, FTI_Exec, FTI_Ckpt,
             destination, 1);
            if (res != FTI_SCES) {
                return FTI_NSCS;
            }
        }
        if (erased[FTI_Topo->groupRank]) {  // then receive file
            res = FTI_RecvCkptFileL2(FTI_Conf, FTI_Exec, FTI_Ckpt, source, 0);
            if (res != FTI_SCES) {
                return FTI_NSCS;
            }
        }
    } else {
        if (erased[FTI_Topo->groupRank]) {  // first receive file
            res = FTI_RecvCkptFileL2(FTI_Conf, FTI_Exec, FTI_Ckpt, source, 0);
            if (res != FTI_SCES) {
                return FTI_NSCS;
            }
        }

        if (erased[destination]) {  // then send file
            res = FTI_SendCkptFileL2(FTI_Conf, FTI_Exec, FTI_Ckpt,
             destination, 1);
            if (res != FTI_SCES) {
                return FTI_NSCS;
            }
        }
    }

    // recover partner files
    if (FTI_Topo->groupRank % 2) {
        if (erased[source + FTI_Topo->groupSize]) {  // first send file
            res = FTI_SendCkptFileL2(FTI_Conf, FTI_Exec, FTI_Ckpt, source, 0);
            if (res != FTI_SCES) {
                return FTI_NSCS;
            }
        }
        if (erased[FTI_Topo->groupRank + FTI_Topo->groupSize]) {
            // receive file
            res = FTI_RecvCkptFileL2(FTI_Conf, FTI_Exec, FTI_Ckpt,
             destination, 1);
            if (res != FTI_SCES) {
                return FTI_NSCS;
            }
        }
    } else {
        if (erased[FTI_Topo->groupRank + FTI_Topo->groupSize]) {
            // first receive file
            res = FTI_RecvCkptFileL2(FTI_Conf, FTI_Exec, FTI_Ckpt,
             destination, 1);
            if (res != FTI_SCES) {
                return FTI_NSCS;
            }
        }

        if (erased[source + FTI_Topo->groupSize]) {  // send file
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
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Topo        Topology metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @return     integer         FTI_SCES if successful.

  This function tries to recover the L3 ckpt. files missing using the
  RS decoding. If to many files are missing in the group, then we
  consider this checkpoint unavailable.

 **/
/*-------------------------------------------------------------------------*/
int FTI_RecoverL3(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt) {
    if (mkdir(FTI_Ckpt[3].dir, 0777) == -1) {
        if (errno != EEXIST) {
            FTI_Print("Cannot create directory", FTI_EROR);
        }
    }

    int erased[FTI_BUFS];

    if (FTI_Conf->ioMode == FTI_IO_FTIFF) {
        if (FTIFF_CheckL3RecoverInit(FTI_Exec, FTI_Topo, FTI_Ckpt,
         erased) != FTI_SCES) {
            FTI_Print("No restart possible from L3. Ckpt files missing.",
             FTI_DBUG);
            return FTI_NSCS;
        }
    } else {
        // Checking erasures
        if (FTI_CheckErasures(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt,
         erased) != FTI_SCES) {
            FTI_Print("Error checking erasures.", FTI_DBUG);
            return FTI_NSCS;
        }
    }

    // Counting erasures
    int l = 0;
    int gs = FTI_Topo->groupSize;
    int i;
    for (i = 0; i < gs; i++) {
        if (erased[i]) {
            l++;
        }
        if (erased[i + gs]) {
            l++;
        }
    }
    if (l > gs) {
        FTI_Print("Too many erasures at L3.", FTI_DBUG);
        return FTI_NSCS;
    }

    // Reed-Solomon decoding
    if (l > 0) {
        char str[FTI_BUFS];
        snprintf(str, FTI_BUFS, "There are %d encoded/checkpoint files"
        " missing in this group.", l);
        FTI_Print(str, FTI_DBUG);
        int res = FTI_Try(FTI_Decode(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt,
         erased), "use RS-decoding to regenerate the missing data.");
        if (res == FTI_NSCS) {
            return FTI_NSCS;
        }
    }

    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      It recovers L4 ckpt. files from the PFS.
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Topo        Topology metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @return     integer         FTI_SCES if successful.

  This function tries to recover the ckpt. files using the L4 ckpt. files
  stored in the PFS. If at least one ckpt. file is missing in the PFS, we
  consider this checkpoint unavailable.

 **/
/*-------------------------------------------------------------------------*/
int FTI_RecoverL4(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt) {
    switch (FTI_Conf->ioMode) {
        case FTI_IO_FTIFF:
        case FTI_IO_HDF5:
        case FTI_IO_IME:
        case FTI_IO_POSIX:
            return FTI_RecoverL4Posix(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt);
        case FTI_IO_MPI:
            return FTI_RecoverL4Mpi(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt);
#ifdef ENABLE_SIONLIB
        case FTI_IO_SIONLIB:
            return FTI_RecoverL4Sionlib(FTI_Conf, FTI_Exec, FTI_Topo,
             FTI_Ckpt);
#endif
        default:
            FTI_Print("unknown I/O mode", FTI_WARN);
            return FTI_NSCS;
    }
}

/*-------------------------------------------------------------------------*/
/**
  @brief      It recovers L4 ckpt. files from the PFS using POSIX.
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Topo        Topology metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @return     integer         FTI_SCES if successful.

  This function tries to recover the ckpt. files using the L4 ckpt. files
  stored in the PFS. If at least one ckpt. file is missing in the PFS, we
  consider this checkpoint unavailable.

 **/
/*-------------------------------------------------------------------------*/
int FTI_RecoverL4Posix(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt) {
    FTI_Print("Starting recovery L4 using Posix I/O.", FTI_DBUG);
    if (mkdir(FTI_Ckpt[1].dir, 0777) == -1) {
        if (errno != EEXIST) {
            FTI_Print("Directory L1 could NOT be created.", FTI_WARN);
        }
    }

    // Checking erasures
    if (FTI_Conf->ioMode == FTI_IO_FTIFF) {
        if (FTIFF_CheckL4RecoverInit(FTI_Exec, FTI_Topo, FTI_Ckpt) !=
            FTI_SCES) {
            FTI_Print("No restart possible from L4. Ckpt files missing.",
             FTI_DBUG);
            return FTI_NSCS;
        }
    } else {
        int erased[FTI_BUFS];
        if (FTI_CheckErasures(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt,
         erased) != FTI_SCES) {
            FTI_Print("Error checking erasures.", FTI_DBUG);
            return FTI_NSCS;
        }
        int l = 0;
        int i;
        // Counting erasures
        for (i = 0; i < FTI_Topo->groupSize; i++) {
            if (erased[i]) {
                l++;
            }
        }
        if (l > 0) {
            FTI_Print("Checkpoint file missing at L4.", FTI_WARN);
            return FTI_NSCS;
        }

        if (FTI_Ckpt[FTI_Exec->ckptMeta.level].recoIsDcp &&
         FTI_Conf->dcpPosix) {
            return FTI_SCES;
        }

        snprintf(FTI_Exec->ckptMeta.ckptFile, FTI_BUFS, "Ckpt%d-Rank%d.%s",
         FTI_Exec->ckptId, FTI_Topo->myRank, FTI_Conf->suffix);
    }

    char gfn[FTI_BUFS], lfn[FTI_BUFS];

    snprintf(lfn, FTI_BUFS, "%s/%s", FTI_Conf->lTmpDir,
     FTI_Exec->ckptMeta.ckptFile);

    if (FTI_Ckpt[4].recoIsDcp) {
        snprintf(gfn, FTI_BUFS, "%s/%s", FTI_Ckpt[4].dcpDir,
         FTI_Exec->ckptMeta.ckptFile);
    } else if (FTI_Ckpt[4].localReplica == 1) {
        snprintf(gfn, FTI_BUFS, "%s/%s", FTI_Ckpt[4].L4Replica,
         FTI_Exec->ckptMeta.ckptFile);
    } else {
        snprintf(gfn, FTI_BUFS, "%s/%s", FTI_Ckpt[4].dir,
         FTI_Exec->ckptMeta.ckptFile);
    }

    FILE* gfd = fopen(gfn, "rb");
    if (gfd == NULL) {
        FTI_Print("R4 cannot open the ckpt. file in the PFS.", FTI_WARN);
        return FTI_NSCS;
    }

    MKDIR(FTI_Conf->lTmpDir, 0777);
    FILE* lfd = fopen(lfn, "wb");
    if (lfd == NULL) {
        FTI_Print("R4 cannot open the local ckpt. file.", FTI_WARN);
        fclose(gfd);
        return FTI_NSCS;
    }

    char *readData = talloc(char, FTI_Conf->transferSize);
    int64_t bSize = FTI_Conf->transferSize;
    int64_t fs = FTI_Exec->ckptMeta.fs;

    // Checkpoint files transfer from PFS
    int64_t pos = 0;
    while (pos < fs) {
        if ((fs - pos) < FTI_Conf->transferSize) {
            bSize = fs - pos;
        }

        size_t bytes = fread(readData, sizeof(char), bSize, gfd);

        if (ferror(gfd)) {
            FTI_Print("R4 cannot read from the ckpt. file in the PFS.",
             FTI_DBUG);

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
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Topo        Topology metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @return     integer         FTI_SCES if successful.

  This function tries to recover the ckpt. files using the L4 ckpt. files
  stored in the PFS. If at least one ckpt. file is missing in the PFS, we
  consider this checkpoint unavailable.

 **/
/*-------------------------------------------------------------------------*/
int FTI_RecoverL4Mpi(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt) {
    FTI_Print("Starting recovery L4 using MPI-IO.", FTI_DBUG);
    // create local directories
    if (mkdir(FTI_Ckpt[1].dir, 0777) == -1) {
        if (errno != EEXIST) {
            FTI_Print("Directory L1 could NOT be created.", FTI_WARN);
        }
    }

    // enable collective buffer optimization
    MPI_Info info;
    MPI_Info_create(&info);
    MPI_Info_set(info, "romio_cb_read", "enable");

    // set stripping unit to 4MB
    MPI_Info_set(info, "stripping_unit", "4194304");

    char gfn[FTI_BUFS], lfn[FTI_BUFS], gfp[FTI_BUFS];
    snprintf(FTI_Exec->ckptMeta.ckptFile, FTI_BUFS, "Ckpt%d-Rank%d.%s",
     FTI_Exec->ckptId, FTI_Topo->myRank, FTI_Conf->suffix);
    snprintf(gfn, FTI_BUFS, "Ckpt%d-mpiio.fti", FTI_Exec->ckptId);
    snprintf(lfn, FTI_BUFS, "%s/%s", FTI_Conf->lTmpDir,
     FTI_Exec->ckptMeta.ckptFile);
    snprintf(gfp, FTI_BUFS, "%s/%s", FTI_Ckpt[4].dir, gfn);

    // open parallel file
    MPI_File pfh;
    int buf = MPI_File_open(FTI_COMM_WORLD, gfp, MPI_MODE_RDWR, info, &pfh);
    // check if successful
    if (buf != 0) {
        errno = 0;
        char mpi_err[FTI_BUFS];
        int reslen;
        MPI_Error_string(buf, mpi_err, &reslen);
        if (buf != MPI_ERR_NO_SUCH_FILE) {
            char str[FTI_BUFS];
            snprintf(str, FTI_BUFS,
             "Unable to access file [MPI ERROR - %i] %s", buf, mpi_err);
            FTI_Print(str, FTI_EROR);
        }
        return FTI_NSCS;
    }

    // collect chunksizes of other ranks
    int64_t* chunkSizes = talloc(int64_t,
     FTI_Topo->nbApprocs*FTI_Topo->nbNodes);
    MPI_Allgather(&FTI_Exec->ckptMeta.fs, 1, MPI_INT64_T, chunkSizes, 1,
     MPI_INT64_T, FTI_COMM_WORLD);

    MPI_Offset offset = 0;
    // set file offset
    int i;
    for (i = 0; i < FTI_Topo->splitRank; i++) {
        offset += chunkSizes[i];
    }
    free(chunkSizes);

    MKDIR(FTI_Conf->lTmpDir, 0777);
    FILE *lfd = fopen(lfn, "wb");
    if (lfd == NULL) {
        FTI_Print("R4 cannot open the local ckpt. file.", FTI_DBUG);
        MPI_File_close(&pfh);
        return FTI_NSCS;
    }

    int64_t fs = FTI_Exec->ckptMeta.fs;
    char *readData = talloc(char, FTI_Conf->transferSize);
    int32_t bSize = FTI_Conf->transferSize;
    int64_t pos = 0;
    // Checkpoint files transfer from PFS
    while (pos < fs) {
        if ((fs - pos) < FTI_Conf->transferSize) {
            bSize = fs - pos;
        }
        // read block in parallel file
        buf = MPI_File_read_at(pfh, offset, readData, (int)bSize, MPI_BYTE,
         MPI_STATUS_IGNORE);
        // check if successful
        if (buf != 0) {
            errno = 0;
            char mpi_err[FTI_BUFS];
            char str[FTI_BUFS];
            int reslen;
            MPI_Error_string(buf, mpi_err, &reslen);
            snprintf(str, FTI_BUFS, "R4 cannot read from the ckpt. file in"
            " the PFS. [MPI ERROR - %i] %s", buf, mpi_err);
            FTI_Print(str, FTI_EROR);
            free(readData);
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
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Topo        Topology metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @return     integer         FTI_SCES if successful.

  This function tries to recover the ckpt. files using the L4 ckpt. files
  stored in the PFS. If at least one ckpt. file is missing in the PFS, we
  consider this checkpoint unavailable.

 **/
/*-------------------------------------------------------------------------*/
#ifdef ENABLE_SIONLIB  // --> If SIONlib is installed
int FTI_RecoverL4Sionlib(FTIT_configuration* FTI_Conf,
     FTIT_execution* FTI_Exec, FTIT_topology* FTI_Topo,
     FTIT_checkpoint* FTI_Ckpt) {
    FTI_Print("Starting recovery L4 using Sionlib.", FTI_DBUG);
    // Create local directories
    if (mkdir(FTI_Ckpt[1].dir, 0777) == -1) {
        if (errno != EEXIST) {
            FTI_Print("Directory L1 could NOT be created.", FTI_WARN);
        }
    }

    char gfn[FTI_BUFS], lfn[FTI_BUFS];
    snprintf(lfn, FTI_BUFS, "%s/Ckpt%d-Rank%d.%s", FTI_Conf->lTmpDir,
     FTI_Exec->ckptId, FTI_Topo->myRank, FTI_Conf->suffix);
    snprintf(gfn, FTI_BUFS, "%s/Ckpt%d-sionlib.fti", FTI_Ckpt[4].dir,
     FTI_Exec->ckptId);

    int nTasks;
    MPI_Comm_size(FTI_COMM_WORLD, &nTasks);
    int numFiles = 1;
    int sid;
    int nlocaltasks = 1;
    int* file_map = calloc(nTasks, sizeof(int));
    int* ranks = talloc(int, nTasks);
    int* rank_map = talloc(int, nTasks);
    sion_int64* chunkSizes = talloc(sion_int64, nTasks);
    sion_int32 fsblksize = -1;
    int i;
    for (i = 0; i < nTasks; i++) {
        chunkSizes[i] = 1000;
        ranks[i] = FTI_Topo->splitRank;
        rank_map[i] = FTI_Topo->splitRank;
    }



    // this is done, since sionlib aborts if the file is not readable.
    if (access(gfn, F_OK) != 0) {
        return FTI_NSCS;
    } else {
        sid = sion_open(gfn, "rb,posix", &nlocaltasks , &numFiles, &chunkSizes,
         &fsblksize, &ranks, NULL);
        int correct  = 1;
        int globalcorrect = 1;
        if (sid < 0) {
            correct = 0;
        }
        MPI_Allreduce(&correct, &globalcorrect, 1, MPI_INT, MPI_BAND,
         FTI_COMM_WORLD);
        if (globalcorrect == 0) {
            return FTI_NSCS;
        }
    }


    sid = sion_paropen_mapped_mpi(gfn, "rb,posix", &numFiles,
     FTI_COMM_WORLD, &nlocaltasks, &ranks, &chunkSizes, &file_map,
      &rank_map, &fsblksize, NULL);

    if (sid < 0) {
        FTI_Print("R4 cannot open the Global ckpt. file.", FTI_WARN);
        free(file_map);
        free(ranks);
        free(rank_map);
        free(chunkSizes);
        return FTI_NSCS;
    }

    MKDIR(FTI_Conf->lTmpDir, 0777);
    FILE* lfd = fopen(lfn, "wb");
    if (lfd == NULL) {
        FTI_Print("R4 cannot open the local ckpt. file.", FTI_WARN);
        sion_parclose_mapped_mpi(sid);
        free(file_map);
        free(ranks);
        free(rank_map);
        free(chunkSizes);
        return FTI_NSCS;
    }

    int res = sion_seek(sid, FTI_Topo->splitRank, SION_CURRENT_BLK,
     SION_CURRENT_POS);
    // check if successful
    if (res != SION_SUCCESS) {
        FTI_Print("SIONlib: Could not set file pointer", FTI_EROR);
        sion_parclose_mapped_mpi(sid);
        free(file_map);
        free(ranks);
        free(rank_map);
        free(chunkSizes);
        fclose(lfd);
        return FTI_NSCS;
    }

    // Checkpoint files transfer from PFS
    while (!sion_feof(sid)) {
        int64_t fs = FTI_Exec->ckptMeta.fs;
        char *readData = talloc(char, FTI_Conf->transferSize);
        int64_t bSize = FTI_Conf->transferSize;
        int64_t pos = 0;
        // Checkpoint files transfer from PFS
        while (pos < fs) {
            if ((fs - pos) < FTI_Conf->transferSize) {
                bSize = fs - pos;
            }
            int64_t nb = sion_fread(readData, sizeof(char), bSize, sid);
            if (nb != bSize) {
                char str[FTI_BUFS];
                snprintf(str, FTI_BUFS, "SIONlib: Unable to read"
                " %ld Bytes from file", bSize);
                FTI_Print(str, FTI_EROR);
                sion_parclose_mapped_mpi(sid);
                free(file_map);
                free(ranks);
                free(rank_map);
                free(chunkSizes);
                free(readData);
                fclose(lfd);
                return FTI_NSCS;
            }

            fwrite(readData, sizeof(char), bSize, lfd);
            if (ferror(lfd)) {
                FTI_Print("R4 cannot write to the local ckpt. file.",
                 FTI_WARN);
                free(readData);
                fclose(lfd);
                sion_parclose_mapped_mpi(sid);
                free(file_map);
                free(ranks);
                free(rank_map);
                free(chunkSizes);
                return  FTI_NSCS;
            }

            pos = pos + bSize;
        }
        free(readData);
    }

    fclose(lfd);
    sion_parclose_mapped_mpi(sid);
    free(file_map);
    free(ranks);
    free(rank_map);
    free(chunkSizes);

    return FTI_SCES;
}
#endif
