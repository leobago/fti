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
#include <assert.h>
#include "interface.h"

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
               FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt, int* erased)
{
    
    //printf("rank: %i, line: %i\n", FTI_Topo->myRank, __LINE__);
    int ckptID, rank;
    sscanf(FTI_Exec->meta[3].ckptFile, "Ckpt%d-Rank%d.fti", &ckptID, &rank);
    char fn[FTI_BUFS], efn[FTI_BUFS];
    sprintf(efn, "%s/Ckpt%d-RSed%d.fti", FTI_Ckpt[3].dir, ckptID, rank);
    sprintf(fn, "%s/%s", FTI_Ckpt[3].dir, FTI_Exec->meta[3].ckptFile);
    
    int bs = FTI_Conf->blockSize;
    int k = FTI_Topo->groupSize;
    int m = k;

    long fs = FTI_Exec->meta[3].fs[0];
    
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
    
    FILE *fd, *efd;
    long maxFs = FTI_Exec->meta[3].maxFs[0];
    long ps = ((maxFs / FTI_Conf->blockSize)) * FTI_Conf->blockSize;
    if (ps < maxFs) {
        ps = ps + FTI_Conf->blockSize; // Calculating padding size
    }
    if (erased[FTI_Topo->groupRank] == 0) { // Resize and open files
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
    long pos = 0; 
    int remBsize = bs;
    
    MD5_CTX md5ctxRS;
    MD5_Init(&md5ctxRS);
    while (pos < ps) {
        
        if ((maxFs - pos) < bs) {
            remBsize = maxFs - pos;
        }
    
        //printf("rank: %i, line: %i\n", FTI_Topo->myRank, __LINE__);
        // Reading the data
        if (erased[FTI_Topo->groupRank] == 0) {
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
        else {
            bzero(data[FTI_Topo->groupRank], remBsize);
        } // Erasure found
    
        //printf("rank: %i, line: %i\n", FTI_Topo->myRank, __LINE__);

        if (erased[FTI_Topo->groupRank + FTI_Topo->groupSize] == 0) {
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
        else {
            bzero(coding[FTI_Topo->groupRank], remBsize);
        }
    
        //printf("rank: %i, line: %i\n", FTI_Topo->myRank, __LINE__);
    
        MPI_Allgather(data[FTI_Topo->groupRank] + 0, remBsize, MPI_CHAR, dataTmp, remBsize, MPI_CHAR, FTI_Exec->groupComm);
        for (i = 0; i < k; i++) {
            memcpy(data[i] + 0, &(dataTmp[i * remBsize]), sizeof(char) * remBsize);
        }

        MPI_Allgather(coding[FTI_Topo->groupRank] + 0, remBsize, MPI_CHAR, dataTmp, remBsize, MPI_CHAR, FTI_Exec->groupComm);
        for (i = 0; i < k; i++) {
            memcpy(coding[i] + 0, &(dataTmp[i * remBsize]), sizeof(char) * remBsize);
        }
    
        // Decoding the lost data work
        if (erased[FTI_Topo->groupRank]) {
            jerasure_matrix_dotprod(k, FTI_Conf->l3WordSize, decMatrix + (FTI_Topo->groupRank * k), dm_ids, FTI_Topo->groupRank, data, coding, remBsize);
        }

        MPI_Allgather(data[FTI_Topo->groupRank] + 0, remBsize, MPI_CHAR, dataTmp, remBsize, MPI_CHAR, FTI_Exec->groupComm);
        for (i = 0; i < k; i++) {
            memcpy(data[i] + 0, &(dataTmp[i * remBsize]), sizeof(char) * remBsize);
        }

        // Finally, re-encode any erased encoded checkpoint file
        if (erased[FTI_Topo->groupRank + k]) {
            jerasure_matrix_dotprod(k, FTI_Conf->l3WordSize, matrix + (FTI_Topo->groupRank * k), NULL, FTI_Topo->groupRank + k, data, coding, remBsize);
        }
        if (erased[FTI_Topo->groupRank]) {
            fwrite(data[FTI_Topo->groupRank] + 0, sizeof(char), remBsize, fd);
        }
        if (erased[FTI_Topo->groupRank + k]) {
            MD5_Update(&md5ctxRS, coding[FTI_Topo->groupRank], remBsize); 
            fwrite(coding[FTI_Topo->groupRank] + 0, sizeof(char), remBsize, efd);
        }

        pos = pos + bs;
    }
    unsigned char hashRS[MD5_DIGEST_LENGTH];
    MD5_Final( hashRS, &md5ctxRS );
    

    // Closing files
    fclose(fd);
    fclose(efd);
    
    //sleep(1);
    //exit(0);

    if ( FTI_Conf->ioMode == FTI_IO_FTIFF && erased[FTI_Topo->groupRank] ) {
        int ifd = open(fn, O_RDONLY);
        FTIFF_metaInfo *metaInfo = malloc( sizeof( FTIFF_metaInfo ) );
        long offset;
        syncfs(ifd);
        offset = lseek(ifd, 0, SEEK_SET);
        read( ifd, metaInfo, sizeof(FTIFF_metaInfo) );
        printf("rank: %i, fs: %ld, ckptSize: %ld, maxFS: %ld, offset: %ld, size metaInfo: %ld\n", 
                FTI_Topo->myRank, metaInfo->fs, metaInfo->ckptSize, maxFs, offset, sizeof(FTIFF_metaInfo));
        fs = metaInfo->fs;
        FTI_Exec->meta[3].fs[0] = fs;
        free( metaInfo );
        close( ifd );
    }            
    
    if ( FTI_Conf->ioMode == FTI_IO_FTIFF && erased[FTI_Topo->groupRank + k] ) {
        FTIFF_metaInfo *FTIFFMetaRS = malloc( sizeof( FTIFF_metaInfo) );
        // get timestamp and its hash
        MD5_CTX mdContextTS;
        MD5_Init (&mdContextTS);
        struct timespec ntime;
        clock_gettime(CLOCK_REALTIME, &ntime);
        FTIFFMetaRS->timestamp = ntime.tv_sec*1000000000 + ntime.tv_nsec;
        FTIFFMetaRS->fs = maxFs;
        FTIFFMetaRS->ckptSize = fs;
        
        char checksum[MD5_DIGEST_STRING_LENGTH];
        int ii = 0;
        for(i = 0; i < MD5_DIGEST_LENGTH; i++) {
            sprintf(&checksum[ii], "%02x", hashRS[i]);
            ii+=2;
        }

        // add RS checksum to meta data
        strcpy(FTIFFMetaRS->checksum, checksum);
        MD5_Update( &mdContextTS, FTIFFMetaRS->checksum, MD5_DIGEST_STRING_LENGTH );
        MD5_Update( &mdContextTS, &(FTIFFMetaRS->timestamp), sizeof(long) );
        MD5_Update( &mdContextTS, &(FTIFFMetaRS->ckptSize), sizeof(long) );
        MD5_Update( &mdContextTS, &(FTIFFMetaRS->fs), sizeof(long) );
        MD5_Final( FTIFFMetaRS->hashTimestamp, &mdContextTS );

        // append meta info to RS file
        int ifd = open(efn, O_APPEND);
        write( ifd, FTIFFMetaRS, sizeof(FTIFF_metaInfo) );
        close( ifd );
    }
    printf("\nfs: %ld\n\n", fs);

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
    @brief      Init of FTI-FF L1 recovery
    @param      FTI_Exec        Execution metadata.
    @param      FTI_Topo        Topology metadata.
    @param      FTI_Ckpt        Checkpoint metadata.
    @return     integer         FTI_SCES if successful.

    This function initializes the L4 checkpoint recovery. It checks for 
    erasures and loads the required meta data. 
 **/
/*-------------------------------------------------------------------------*/
int FTI_CheckL1RecoverInit( FTIT_execution* FTI_Exec, FTIT_topology* FTI_Topo, 
                  FTIT_checkpoint* FTI_Ckpt )
{
    char str[FTI_BUFS], tmpfn[FTI_BUFS];
    int fexist = 0, fileTarget, ckptID, fcount;
    struct dirent *entry = malloc(sizeof(struct dirent));
    struct stat ckptFS;
    FTIFF_metaInfo *FTIFFMetatmp = calloc( 1, sizeof(FTIFF_metaInfo) );
    MD5_CTX mdContext;
    DIR *L1CkptDir = opendir( FTI_Ckpt[1].dir );
    if(L1CkptDir) {
        while(entry = readdir(L1CkptDir)) {
            if(strcmp(entry->d_name,".") && strcmp(entry->d_name,"..")) { 
                sprintf(str, "FTI-FF: L1RecoveryInit - found file with name: %s", entry->d_name);
                FTI_Print(str, FTI_DBUG);
                sscanf(entry->d_name, "Ckpt%d-Rank%d.fti", &ckptID, &fileTarget );
                if( fileTarget == FTI_Topo->myRank ) {
                    sprintf(tmpfn, "%s/%s", FTI_Ckpt[1].dir, entry->d_name);
                    int ferr = stat(tmpfn, &ckptFS);
                    if (!ferr && S_ISREG(ckptFS.st_mode)) {
                        int fd = open(tmpfn, O_RDONLY);
                        lseek(fd, 0, SEEK_SET);
                        read( fd, FTIFFMetatmp, sizeof(FTIFF_metaInfo) );
                        unsigned char hashTScmp[MD5_DIGEST_LENGTH];
                        MD5_CTX mdContextTS;
                        MD5_Init (&mdContextTS);
                        MD5_Update( &mdContextTS, FTIFFMetatmp->checksum, MD5_DIGEST_STRING_LENGTH );
                        MD5_Update( &mdContextTS, &(FTIFFMetatmp->timestamp), sizeof(long) );
                        MD5_Update( &mdContextTS, &(FTIFFMetatmp->ckptSize), sizeof(long) );
                        MD5_Update( &mdContextTS, &(FTIFFMetatmp->fs), sizeof(long) );
                        MD5_Final( hashTScmp, &mdContextTS );
                        printf("timestamp: %ld, fs: %ld, ckptSize: \n\n", FTIFFMetatmp->timestamp, FTIFFMetatmp->fs, FTIFFMetatmp->ckptSize);
                        if ( memcmp( FTIFFMetatmp->hashTimestamp, hashTScmp, MD5_DIGEST_LENGTH ) == 0 ) {
                            long rcount = sizeof(FTIFF_metaInfo), toRead, diff;
                            int rbuffer;
                            char *buffer = malloc( CHUNK_SIZE );
                            MD5_Init (&mdContext);
                            while( rcount < FTIFFMetatmp->fs ) {
                                lseek( fd, rcount, SEEK_SET );
                                diff = FTIFFMetatmp->fs - rcount;
                                toRead = ( diff < CHUNK_SIZE ) ? diff : CHUNK_SIZE;
                                rbuffer = read( fd, buffer, toRead );
                                rcount += rbuffer;
                                MD5_Update (&mdContext, buffer, rbuffer);
                            }
                            unsigned char hash[MD5_DIGEST_LENGTH];
                            MD5_Final (hash, &mdContext);
                            int i;
                            char checksum[MD5_DIGEST_STRING_LENGTH];
                            int ii = 0;
                            for(i = 0; i < MD5_DIGEST_LENGTH; i++) {
                                sprintf(&checksum[ii], "%02x", hash[i]);
                                ii += 2;
                            }
                            printf("checksum-saved: %s, checksum-cmp: %s\n", FTIFFMetatmp->checksum, checksum);
                            if ( strcmp( checksum, FTIFFMetatmp->checksum ) == 0 ) {
                                FTI_Exec->meta[1].fs[0] = ckptFS.st_size;    
                                FTI_Exec->ckptID = ckptID;
                                strncpy(FTI_Exec->meta[1].ckptFile, entry->d_name, NAME_MAX);
                                fexist = 1;
                            }
                        }
                        close(fd);
                        break;
                    }            
                }
            }
        }
    }
    MPI_Allreduce(&fexist, &fcount, 1, MPI_INT, MPI_SUM, FTI_COMM_WORLD);
    int fneeded = FTI_Topo->nbNodes*FTI_Topo->nbApprocs;
    int res = (fcount == fneeded) ? FTI_SCES : FTI_NSCS;
    closedir(L1CkptDir);
    return res;
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
                  FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt)
{
   if (FTI_Conf->ioMode == FTI_IO_FTIFF) {
        // FTI_CheckL4RestartInit sets: 
        // FTI_Exec->meta[1].ckptFile, 
        // FTI_Exec->meta[4].ckptFile and
        // FTI_Exec->meta[4].fs[0]
        if ( FTI_CheckL1RecoverInit( FTI_Exec, FTI_Topo, FTI_Ckpt ) != FTI_SCES ) {
            FTI_Print("No restart possible from L1. Ckpt files missing.", FTI_DBUG);
            return FTI_NSCS;
        }
   } 
   
   else {
       int erased[FTI_BUFS]; // FTI_BUFS > 32*3
       if (FTI_CheckErasures(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt, erased) != FTI_SCES) {
           FTI_Print("Error checking erasures.", FTI_DBUG);
           return FTI_NSCS;
       }
       int buf = 0;
       int i;
       for (i = 0; i < FTI_Topo->groupSize; i++) {
           if (erased[i]) {
               buf++; // Counting erasures
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
                        FTIT_checkpoint* FTI_Ckpt, int destination, int ptner)
{
    long toSend ; // remaining data to send
    char filename[FTI_BUFS], str[FTI_BUFS];
    if (ptner) {    //if want to send Ptner file
        int ckptID, rank;
        sscanf(FTI_Exec->meta[2].ckptFile, "Ckpt%d-Rank%d.fti", &ckptID, &rank); //do we need this from filename?
        sprintf(filename, "%s/Ckpt%d-Pcof%d.fti", FTI_Ckpt[2].dir, ckptID, rank);
        toSend = FTI_Exec->meta[2].pfs[0];
    } else {    //if want to send Ckpt file
        sprintf(filename, "%s/%s", FTI_Ckpt[2].dir, FTI_Exec->meta[2].ckptFile);
        toSend = FTI_Exec->meta[2].fs[0];
    }

    sprintf(str, "Opening file (rb) (%s) (L2).", filename);
    FTI_Print(str, FTI_DBUG);
    FILE* fileDesc = fopen(filename, "rb");
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
        printf("until here! fs: %ld\n", toSend);

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
                        FTIT_checkpoint* FTI_Ckpt, int source, int ptner)
{
    long toRecv;    //remaining data to receive
    char filename[FTI_BUFS], str[FTI_BUFS];
    if (ptner) { //if want to receive Ptner file
        int ckptID, rank;
        sscanf(FTI_Exec->meta[2].ckptFile, "Ckpt%d-Rank%d.fti", &ckptID, &rank);
        sprintf(filename, "%s/Ckpt%d-Pcof%d.fti", FTI_Ckpt[2].dir, FTI_Exec->ckptID, rank);
        toRecv = FTI_Exec->meta[2].pfs[0];
    } else { //if want to receive Ckpt file
        sprintf(filename, "%s/%s", FTI_Ckpt[2].dir, FTI_Exec->meta[2].ckptFile);
        toRecv = FTI_Exec->meta[2].fs[0];
    }

    sprintf(str, "Opening file (wb) (%s) (L2).", filename);
    FTI_Print(str, FTI_DBUG);
    FILE* fileDesc = fopen(filename, "wb");
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
    @brief      Init of FTI-FF L2 recovery
    @param      FTI_Exec        Execution metadata.
    @param      FTI_Topo        Topology metadata.
    @param      FTI_Ckpt        Checkpoint metadata.
    @return     integer         FTI_SCES if successful.

    This function initializes the L2 checkpoint recovery. It checks for 
    erasures and loads the required meta data. 
 **/
/*-------------------------------------------------------------------------*/
int FTI_CheckL2RecoverInit( FTIT_execution* FTI_Exec, FTIT_topology* FTI_Topo, 
                  FTIT_checkpoint* FTI_Ckpt, int *exists)
{
    char dbgstr[FTI_BUFS];
    
    struct L2Info {
        int FileExists;
        int CopyExists;
    };
    
    enum {
        LEFT_FILE,  // on left node
        MY_FILE,    // on my node
        MY_COPY,    // on right node
        LEFT_COPY  // on my node
    };

    struct L2MetaInfo {
        int FileExists;
        int CopyExists;
        int ckptID;
        long fs;
        long pfs;
    };

    // create MPI datatype of L2Info
    MPI_Datatype MPI_L2Info;
    MPI_Type_contiguous( sizeof(struct L2Info), MPI_BYTE, &MPI_L2Info );
    MPI_Type_commit( &MPI_L2Info );
    
    // create MPI datatype of L2MetaInfo
    MPI_Datatype MPI_L2MetaInfo_RAW, MPI_L2MetaInfo;
    int mbrCnt = 5;
    int mbrBlkLen[] = { 1, 1, 1, 1, 1 };
    MPI_Datatype mbrTypes[] = { MPI_INT, MPI_INT, MPI_INT, MPI_LONG, MPI_LONG };
    MPI_Aint mbrDisp[] = { 
        offsetof( struct L2MetaInfo, FileExists), 
        offsetof( struct L2MetaInfo, CopyExists), 
        offsetof( struct L2MetaInfo, ckptID), 
        offsetof( struct L2MetaInfo, fs), 
        offsetof( struct L2MetaInfo, pfs) 
    }; 
    MPI_Aint lb, extent;
    MPI_Type_create_struct( mbrCnt, mbrBlkLen, mbrDisp, mbrTypes, &MPI_L2MetaInfo_RAW );
    MPI_Type_get_extent( MPI_L2MetaInfo_RAW, &lb, &extent );
    MPI_Type_create_resized( MPI_L2MetaInfo_RAW, lb, extent, &MPI_L2MetaInfo);
    MPI_Type_commit( &MPI_L2MetaInfo );

    // determine app rank representation of group ranks left and right
    MPI_Group nodesGroup;
    MPI_Comm_group(FTI_Exec->groupComm, &nodesGroup);
    MPI_Group appProcsGroup;
    MPI_Comm_group(FTI_COMM_WORLD, &appProcsGroup);
    int baseRanks[] = { FTI_Topo->left, FTI_Topo->right };
    int projRanks[2];
    MPI_Group_translate_ranks( nodesGroup, 2, baseRanks, appProcsGroup, projRanks );
    int leftIdx = projRanks[0], rightIdx = projRanks[1];
    MPI_Group_free(&nodesGroup);
    MPI_Group_free(&appProcsGroup);

    int appCommSize = FTI_Topo->nbNodes*FTI_Topo->nbApprocs;
    int fneeded = appCommSize;
    
    struct L2MetaInfo *appProcsMetaInfo = calloc( appCommSize, sizeof(struct L2MetaInfo) );
    struct L2MetaInfo *myMetaInfo = calloc( 1, sizeof(struct L2MetaInfo) );
    
    MD5_CTX mdContext;
    
    char str[FTI_BUFS], tmpfn[FTI_BUFS];
    int canRecover, fileTarget, ckptID = -1, fcount = 0, match;
    struct dirent *entry = malloc(sizeof(struct dirent));
    struct stat ckptFS;
    DIR *L2CkptDir = opendir( FTI_Ckpt[2].dir );
    
    FTIFF_metaInfo *FTIFFMetatmp = calloc( 1, sizeof(FTIFF_metaInfo) );
    if(L2CkptDir) {
        int tmpCkptID;
        while(entry = readdir(L2CkptDir)) {
            if(strcmp(entry->d_name,".") && strcmp(entry->d_name,"..")) { 
                sprintf(str, "FTI-FF: L2RecoveryInit - found file with name: %s", entry->d_name);
                FTI_Print(str, FTI_DBUG);
                tmpCkptID = ckptID;
                match = sscanf(entry->d_name, "Ckpt%d-Rank%d.fti", &ckptID, &fileTarget );
                if( match == 2 && fileTarget == FTI_Topo->myRank ) {
                    sprintf(tmpfn, "%s/%s", FTI_Ckpt[2].dir, entry->d_name);
                    int ferr = stat(tmpfn, &ckptFS);
                    // check if regular file and of reasonable size (at least must contain meta info)
                    if ( !ferr && S_ISREG(ckptFS.st_mode) && ckptFS.st_size > sizeof(FTIFF_metaInfo) ) {
                        int fd = open(tmpfn, O_RDONLY);
                        lseek(fd, 0, SEEK_SET);
                        read( fd, FTIFFMetatmp, sizeof(FTIFF_metaInfo) );
                        unsigned char hashTScmp[MD5_DIGEST_LENGTH];
                        MD5_CTX mdContextTS;
                        MD5_Init (&mdContextTS);
                        MD5_Update( &mdContextTS, FTIFFMetatmp->checksum, MD5_DIGEST_STRING_LENGTH );
                        MD5_Update( &mdContextTS, &(FTIFFMetatmp->timestamp), sizeof(long) );
                        MD5_Update( &mdContextTS, &(FTIFFMetatmp->ckptSize), sizeof(long) );
                        MD5_Update( &mdContextTS, &(FTIFFMetatmp->fs), sizeof(long) );
                        MD5_Final( hashTScmp, &mdContextTS );
                        printf("timestamp: %ld, fs: %ld, ckptSize: \n\n", FTIFFMetatmp->timestamp, FTIFFMetatmp->fs, FTIFFMetatmp->ckptSize);
                        if ( memcmp( FTIFFMetatmp->hashTimestamp, hashTScmp, MD5_DIGEST_LENGTH ) == 0 ) {
                            long rcount = sizeof(FTIFF_metaInfo), toRead, diff;
                            int rbuffer;
                            char *buffer = malloc( CHUNK_SIZE );
                            MD5_Init (&mdContext);
                            while( rcount < FTIFFMetatmp->fs ) {
                                lseek( fd, rcount, SEEK_SET );
                                diff = FTIFFMetatmp->fs - rcount;
                                toRead = ( diff < CHUNK_SIZE ) ? diff : CHUNK_SIZE;
                                rbuffer = read( fd, buffer, toRead );
                                rcount += rbuffer;
                                MD5_Update (&mdContext, buffer, rbuffer);
                            }
                            unsigned char hash[MD5_DIGEST_LENGTH];
                            MD5_Final (hash, &mdContext);
                            int i;
                            char checksum[MD5_DIGEST_STRING_LENGTH];
                            int ii = 0;
                            for(i = 0; i < MD5_DIGEST_LENGTH; i++) {
                                sprintf(&checksum[ii], "%02x", hash[i]);
                                ii += 2;
                            }
                            printf("checksum-saved: %s, checksum-cmp: %s\n", FTIFFMetatmp->checksum, checksum);
                            if ( strcmp( checksum, FTIFFMetatmp->checksum ) == 0 ) {
                                myMetaInfo->fs = FTIFFMetatmp->fs;    
                                myMetaInfo->ckptID = ckptID;    
                                myMetaInfo->FileExists = 1;
                            }
                        }
                        close(fd);
                    }            
                } else {
                    ckptID = tmpCkptID;
                }
                tmpCkptID = ckptID;
                match = sscanf(entry->d_name, "Ckpt%d-Pcof%d.fti", &ckptID, &fileTarget );
                if( match == 2 && fileTarget == FTI_Topo->myRank ) {
                    sprintf(tmpfn, "%s/%s", FTI_Ckpt[2].dir, entry->d_name);
                    int ferr = stat(tmpfn, &ckptFS);
                    if (!ferr && S_ISREG(ckptFS.st_mode) && ckptFS.st_size > sizeof(FTIFF_metaInfo)) {
                        int fd = open(tmpfn, O_RDONLY);
                        lseek(fd, 0, SEEK_SET);
                        read( fd, FTIFFMetatmp, sizeof(FTIFF_metaInfo) );
                        unsigned char hashTScmp[MD5_DIGEST_LENGTH];
                        MD5_CTX mdContextTS;
                        MD5_Init (&mdContextTS);
                        MD5_Update( &mdContextTS, FTIFFMetatmp->checksum, MD5_DIGEST_STRING_LENGTH );
                        MD5_Update( &mdContextTS, &(FTIFFMetatmp->timestamp), sizeof(long) );
                        MD5_Update( &mdContextTS, &(FTIFFMetatmp->ckptSize), sizeof(long) );
                        MD5_Update( &mdContextTS, &(FTIFFMetatmp->fs), sizeof(long) );
                        MD5_Final( hashTScmp, &mdContextTS );
                        if ( memcmp( FTIFFMetatmp->hashTimestamp, hashTScmp, MD5_DIGEST_LENGTH ) == 0 ) {
                            long rcount = sizeof(FTIFF_metaInfo), toRead, diff;
                            int rbuffer;
                            char *buffer = malloc( CHUNK_SIZE );
                            MD5_Init (&mdContext);
                            while( rcount < FTIFFMetatmp->fs ) {
                                lseek( fd, rcount, SEEK_SET );
                                diff = FTIFFMetatmp->fs - rcount;
                                toRead = ( diff < CHUNK_SIZE ) ? diff : CHUNK_SIZE;
                                rbuffer = read( fd, buffer, toRead );
                                rcount += rbuffer;
                                MD5_Update (&mdContext, buffer, rbuffer);
                            }
                            unsigned char hash[MD5_DIGEST_LENGTH];
                            MD5_Final (hash, &mdContext);
                            int i;
                            char checksum[MD5_DIGEST_STRING_LENGTH];
                            int ii = 0;
                            for(i = 0; i < MD5_DIGEST_LENGTH; i++) {
                                sprintf(&checksum[ii], "%02x", hash[i]);
                                ii += 2;
                            }
                            printf("pchecksum-saved: %s, pchecksum-cmp: %s\n", FTIFFMetatmp->checksum, checksum);
                            if ( strcmp( checksum, FTIFFMetatmp->checksum ) == 0 ) {
                                myMetaInfo->pfs = FTIFFMetatmp->fs;    
                                myMetaInfo->ckptID = ckptID;    
                                myMetaInfo->CopyExists = 1;
                            }
                        }
                        close(fd);
                    }            
                } else {
                    ckptID = tmpCkptID;
                }
            }
            if (myMetaInfo->FileExists && myMetaInfo->CopyExists) {
                break;
            }
        }
    }
    
    if(!(myMetaInfo->FileExists) && !(myMetaInfo->CopyExists)) {
        myMetaInfo->ckptID = -1;
    }
    
    // gather meta info
    MPI_Allgather( myMetaInfo, 1, MPI_L2MetaInfo, appProcsMetaInfo, 1, MPI_L2MetaInfo, FTI_COMM_WORLD);
    
    exists[LEFT_FILE] = appProcsMetaInfo[leftIdx].FileExists;
    exists[MY_FILE] = appProcsMetaInfo[FTI_Topo->splitRank].FileExists;
    exists[MY_COPY] = appProcsMetaInfo[rightIdx].CopyExists;
    exists[LEFT_COPY] = appProcsMetaInfo[FTI_Topo->splitRank].CopyExists;

    // check if recovery possible
    int i, saneCkptID = 0;
    ckptID = 0;
    for(i=0; i<appCommSize; i++) { 
        fcount += ( appProcsMetaInfo[i].FileExists || appProcsMetaInfo[rightIdx].CopyExists ) ? 1 : 0;
        if (appProcsMetaInfo[i].ckptID > 0) {
            saneCkptID++;
            ckptID += appProcsMetaInfo[i].ckptID;
        }
    }
    int res = (fcount == fneeded) ? FTI_SCES : FTI_NSCS;

    if (res == FTI_SCES) {
        FTI_Exec->ckptID = ckptID/saneCkptID;
        if (myMetaInfo->FileExists) {
            FTI_Exec->meta[2].fs[0] = myMetaInfo->fs;    
        } else {
            FTI_Exec->meta[2].fs[0] = appProcsMetaInfo[rightIdx].pfs;    
        }
        if (myMetaInfo->CopyExists) {
            FTI_Exec->meta[2].pfs[0] = myMetaInfo->pfs;    
        } else {
            FTI_Exec->meta[2].pfs[0] = appProcsMetaInfo[leftIdx].fs;    
        }
    }
    sprintf(dbgstr, "FTI-FF: L2-Recovery - rank: %i, left: %i, right: %i, fs: %ld, pfs: %ld, ckptID: %i\n",
            FTI_Topo->myRank, leftIdx, rightIdx, FTI_Exec->meta[2].fs[0], FTI_Exec->meta[2].pfs[0], FTI_Exec->ckptID);
    FTI_Print(dbgstr, FTI_DBUG);

    sprintf(FTI_Exec->meta[2].ckptFile, "Ckpt%d-Rank%d.fti", FTI_Exec->ckptID, FTI_Topo->myRank);
    
    closedir(L2CkptDir);
    free(appProcsMetaInfo);
    free(myMetaInfo);
    //free(entry);
    MPI_Type_free(&MPI_L2Info);
    
    return res;
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
                  FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt)
{
    if (mkdir(FTI_Ckpt[2].dir, 0777) == -1) {
        if (errno != EEXIST) {
            FTI_Print("Cannot create directory", FTI_EROR);
        }
    }

    int erased[FTI_BUFS];
    int source = FTI_Topo->right; //to receive Ptner file from this process (to recover)
    int destination = FTI_Topo->left; //to send Ptner file (to let him recover)
    int res;

    if (FTI_Conf->ioMode == FTI_IO_FTIFF) {
        
        enum {
            LEFT_FILE,  // on left node
            MY_FILE,    // on my node
            MY_COPY,    // on right node
            LEFT_COPY  // on my node
        };
        
        int exists[4];
        
        if ( FTI_CheckL2RecoverInit( FTI_Exec, FTI_Topo, FTI_Ckpt, exists ) != FTI_SCES ) {
            FTI_Print("No restart possible from L2. Ckpt files missing.", FTI_DBUG);
            return FTI_NSCS;
        }

        memset(erased, 0x0, FTI_BUFS*sizeof(int));

        erased[destination] = !exists[LEFT_FILE];
        printf("rank: %i, pi file: %i\n\n", FTI_Topo->myRank, !exists[LEFT_FILE]);
        erased[FTI_Topo->groupRank] = !exists[MY_FILE];
        printf("rank: %i, mi file: %i\n\n", FTI_Topo->myRank,  !exists[MY_FILE]);
        erased[source + FTI_Topo->groupSize] = !exists[MY_COPY];
        printf("rank: %i, mi copy: %i\n\n", FTI_Topo->myRank,  !exists[MY_COPY]);
        erased[FTI_Topo->groupRank + FTI_Topo->groupSize] = !exists[LEFT_COPY];
        printf("rank: %i, pi copy: %i\n\n", FTI_Topo->myRank,  !exists[LEFT_COPY]);

    } 
   
    else {
        // Checking erasures
        if (FTI_CheckErasures(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt, erased) != FTI_SCES) {
            FTI_Print("Error checking erasures.", FTI_WARN);
            return FTI_NSCS;
        }

        int i = 0;
        int j;
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

        int allRes;
        MPI_Allreduce(&res, &allRes, 1, MPI_INT, MPI_SUM, FTI_Exec->groupComm);
        if (allRes != FTI_SCES) {
            return FTI_NSCS;
        }

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
    @brief      Init of FTI-FF L2 recovery
    @param      FTI_Exec        Execution metadata.
    @param      FTI_Topo        Topology metadata.
    @param      FTI_Ckpt        Checkpoint metadata.
    @return     integer         FTI_SCES if successful.

    This function initializes the L2 checkpoint recovery. It checks for 
    erasures and loads the required meta data. 
 **/
/*-------------------------------------------------------------------------*/
int FTI_CheckL3RecoverInit( FTIT_execution* FTI_Exec, FTIT_topology* FTI_Topo, 
                  FTIT_checkpoint* FTI_Ckpt, int* erased)
{
    char dbgstr[FTI_BUFS];
    
    struct L3Info {
        int FileExists;
        int RSFileExists;
        int ckptID;
        long fs;
        long RSfs;  // maxFs
    };
    
    //struct L2MetaInfo {
    //    int canRecover;
    //    int ckptID;
    //    long fs;
    //    long pfs;
    //};

    // create MPI datatype of L2Info
    //MPI_Datatype MPI_L2Info;
    //MPI_Type_contiguous( sizeof(struct L2Info), MPI_BYTE, &MPI_L2Info );
    //MPI_Type_commit( &MPI_L2Info );
    
    // create MPI datatype of L2MetaInfo
    MPI_Datatype MPI_L3Info_RAW, MPI_L3Info;
    int mbrCnt = 5;
    int mbrBlkLen[] = { 1, 1, 1, 1, 1 };
    MPI_Datatype mbrTypes[] = { MPI_INT, MPI_INT, MPI_INT, MPI_LONG, MPI_LONG };
    MPI_Aint mbrDisp[] = { 
        offsetof( struct L3Info, FileExists), 
        offsetof( struct L3Info, RSFileExists), 
        offsetof( struct L3Info, ckptID), 
        offsetof( struct L3Info, fs), 
        offsetof( struct L3Info, RSfs), 
    }; 
    MPI_Aint lb, extent;
    MPI_Type_create_struct( mbrCnt, mbrBlkLen, mbrDisp, mbrTypes, &MPI_L3Info_RAW );
    MPI_Type_get_extent( MPI_L3Info_RAW, &lb, &extent );
    MPI_Type_create_resized( MPI_L3Info_RAW, lb, extent, &MPI_L3Info);
    MPI_Type_commit( &MPI_L3Info );
    
    //// determine app rank representation of group ranks left and right
    //MPI_Group nodesGroup;
    //MPI_Comm_group(FTI_Exec->groupComm, &nodesGroup);
    //MPI_Group appProcsGroup;
    //MPI_Comm_group(FTI_COMM_WORLD, &appProcsGroup);
    //int baseRanks[] = { FTI_Topo->left, FTI_Topo->right };
    //int projRanks[2];
    //MPI_Group_translate_ranks( nodesGroup, 2, baseRanks, appProcsGroup, projRanks );
    //int leftIdx = projRanks[0], rightIdx = projRanks[1];
    //MPI_Group_free(&nodesGroup);
    //MPI_Group_free(&appProcsGroup);

    //int appCommSize = FTI_Topo->nbNodes*FTI_Topo->nbApprocs;
    //int fneeded = appCommSize;
    
    struct L3Info *groupInfo = calloc( FTI_Topo->groupSize, sizeof(struct L3Info) );
    struct L3Info *myInfo = calloc( 1, sizeof(struct L3Info) );
    
    MD5_CTX mdContext;
    
    char str[FTI_BUFS], tmpfn[FTI_BUFS];
    int fileTarget, ckptID = -1, fcount = 0, match;
    struct dirent *entry = malloc(sizeof(struct dirent));
    struct stat ckptFS;
    DIR *L3CkptDir = opendir( FTI_Ckpt[3].dir );
    
    FTIFF_metaInfo *FTIFFMetatmp = calloc( 1, sizeof(FTIFF_metaInfo) );
    if(L3CkptDir) {
        int tmpCkptID;
        while(entry = readdir(L3CkptDir)) {
            if(strcmp(entry->d_name,".") && strcmp(entry->d_name,"..")) { 
                sprintf(str, "FTI-FF: L3RecoveryInit - found file with name: %s", entry->d_name);
                FTI_Print(str, FTI_DBUG);
                tmpCkptID = ckptID;
                match = sscanf(entry->d_name, "Ckpt%d-Rank%d.fti", &ckptID, &fileTarget );
                if( match == 2 && fileTarget == FTI_Topo->myRank ) {
                    sprintf(tmpfn, "%s/%s", FTI_Ckpt[3].dir, entry->d_name);
                    int ferr = stat(tmpfn, &ckptFS);
                    if (!ferr && S_ISREG(ckptFS.st_mode) && ckptFS.st_size > sizeof(FTIFF_metaInfo) ) {
                        int fd = open(tmpfn, O_RDONLY);
                        lseek(fd, 0, SEEK_SET);
                        read( fd, FTIFFMetatmp, sizeof(FTIFF_metaInfo) );
                        unsigned char hashTScmp[MD5_DIGEST_LENGTH];
                        MD5_CTX mdContextTS;
                        MD5_Init (&mdContextTS);
                        MD5_Update( &mdContextTS, FTIFFMetatmp->checksum, MD5_DIGEST_STRING_LENGTH );
                        MD5_Update( &mdContextTS, &(FTIFFMetatmp->timestamp), sizeof(long) );
                        MD5_Update( &mdContextTS, &(FTIFFMetatmp->ckptSize), sizeof(long) );
                        MD5_Update( &mdContextTS, &(FTIFFMetatmp->fs), sizeof(long) );
                        MD5_Final( hashTScmp, &mdContextTS );
                        if ( memcmp( FTIFFMetatmp->hashTimestamp, hashTScmp, MD5_DIGEST_LENGTH ) == 0 ) {
                            printf("%i: until here\n",__LINE__);
                            long rcount = sizeof(FTIFF_metaInfo) , toRead, diff;
                            int rbuffer;
                            char *buffer = malloc( CHUNK_SIZE );
                            MD5_Init (&mdContext);
                            while( rcount < FTIFFMetatmp->fs ) {
                                lseek( fd, rcount, SEEK_SET );
                                diff = FTIFFMetatmp->fs - rcount;
                                toRead = ( diff < CHUNK_SIZE ) ? diff : CHUNK_SIZE;
                                rbuffer = read( fd, buffer, toRead );
                                rcount += rbuffer;
                                MD5_Update (&mdContext, buffer, rbuffer);
                            }
                            unsigned char hash[MD5_DIGEST_LENGTH];
                            MD5_Final (hash, &mdContext);
                            int i;
                            char checksum[MD5_DIGEST_STRING_LENGTH];
                            int ii = 0;
                            for(i = 0; i < MD5_DIGEST_LENGTH; i++) {
                                sprintf(&checksum[ii], "%02x", hash[i]);
                                ii += 2;
                            }
                            printf("rank: %i, checksum-saved: %s, checksum-cmp: %s\n", FTI_Topo->myRank, FTIFFMetatmp->checksum, checksum);
                            if ( strcmp( checksum, FTIFFMetatmp->checksum ) == 0 ) {
                                myInfo->fs = FTIFFMetatmp->fs;    
                                myInfo->ckptID = ckptID;    
                                myInfo->FileExists = 1;
                            }
                        }
                        close(fd);
                    }            
                } else {
                    ckptID = tmpCkptID;
                }
                tmpCkptID = ckptID;
                match = sscanf(entry->d_name, "Ckpt%d-RSed%d.fti", &ckptID, &fileTarget );
                if( match == 2 && fileTarget == FTI_Topo->myRank ) {
                    sprintf(tmpfn, "%s/%s", FTI_Ckpt[3].dir, entry->d_name);
                    int ferr = stat(tmpfn, &ckptFS);
                    if (!ferr && S_ISREG(ckptFS.st_mode) && ckptFS.st_size > sizeof(FTIFF_metaInfo) ) {
                        int fd = open(tmpfn, O_RDONLY);
                        lseek(fd, -sizeof(FTIFF_metaInfo), SEEK_END);
                        read( fd, FTIFFMetatmp, sizeof(FTIFF_metaInfo) );
                        unsigned char hashTScmp[MD5_DIGEST_LENGTH];
                        MD5_CTX mdContextTS;
                        MD5_Init (&mdContextTS);
                        MD5_Update( &mdContextTS, FTIFFMetatmp->checksum, MD5_DIGEST_STRING_LENGTH );
                        MD5_Update( &mdContextTS, &(FTIFFMetatmp->timestamp), sizeof(long) );
                        MD5_Update( &mdContextTS, &(FTIFFMetatmp->ckptSize), sizeof(long) );
                        MD5_Update( &mdContextTS, &(FTIFFMetatmp->fs), sizeof(long) );
                        MD5_Final( hashTScmp, &mdContextTS );
                        if ( memcmp( FTIFFMetatmp->hashTimestamp, hashTScmp, MD5_DIGEST_LENGTH ) == 0 ) {
                            long rcount = 0, toRead, diff;
                            int rbuffer;
                            char *buffer = malloc( CHUNK_SIZE );
                            MD5_Init (&mdContext);
                            while( rcount < FTIFFMetatmp->fs ) {
                                lseek( fd, rcount, SEEK_SET );
                                diff = FTIFFMetatmp->fs - rcount;
                                toRead = ( diff < CHUNK_SIZE ) ? diff : CHUNK_SIZE;
                                rbuffer = read( fd, buffer, toRead );
                                rcount += rbuffer;
                                MD5_Update (&mdContext, buffer, rbuffer);
                            }
                            unsigned char hash[MD5_DIGEST_LENGTH];
                            MD5_Final (hash, &mdContext);
                            int i;
                            char checksum[MD5_DIGEST_STRING_LENGTH];
                            int ii = 0;
                            for(i = 0; i < MD5_DIGEST_LENGTH; i++) {
                                sprintf(&checksum[ii], "%02x", hash[i]);
                                ii += 2;
                            }
                            printf("RSchecksum-saved: %s, RSchecksum-cmp: %s\n", FTIFFMetatmp->checksum, checksum);
                            if ( strcmp( checksum, FTIFFMetatmp->checksum ) == 0 ) {
                                myInfo->RSfs = FTIFFMetatmp->fs;    
                                myInfo->ckptID = ckptID;    
                                myInfo->RSFileExists = 1;
                            }
                        }
                        close(fd);
                    }            
                } else {
                    ckptID = tmpCkptID;
                }
            }
            if (myInfo->FileExists && myInfo->RSFileExists) {
                break;
            }
        }
    }
    
    if(!(myInfo->FileExists) && !(myInfo->RSFileExists)) {
        myInfo->ckptID = -1;
    }
    
    if(!(myInfo->RSFileExists)) {
        myInfo->RSfs = -1;
    }
    
    // gather meta info
    MPI_Allgather( myInfo, 1, MPI_L3Info, groupInfo, 1, MPI_L3Info, FTI_Exec->groupComm);
    
    // check if recovery possible
    int i, saneCkptID = 0, saneMaxFs = 0;
    long maxFs = 0;
    ckptID = 0;
    for(i=0; i<FTI_Topo->groupSize; i++) { 
        erased[i]=!groupInfo[i].FileExists;
        erased[i+FTI_Topo->groupSize]=!groupInfo[i].RSFileExists;
        //printf("grank: %i, grrank: %i, ferased: %i, RSerased: %i \n", 
        //        FTI_Topo->myRank, FTI_Topo->groupRank, erased[FTI_Topo->groupRank], erased[FTI_Topo->groupRank+FTI_Topo->groupSize]);
        if (groupInfo[i].ckptID > 0) {
            saneCkptID++;
            ckptID += groupInfo[i].ckptID;
        }
        if (groupInfo[i].RSfs > 0) {
            saneMaxFs++;
            maxFs += groupInfo[i].RSfs;
        }
    }
    FTI_Exec->ckptID = ckptID/saneCkptID;
    assert( saneMaxFs != 0 );
    FTI_Exec->meta[3].maxFs[0] = maxFs/saneMaxFs;

    FTI_Exec->meta[3].fs[0] = (myInfo->FileExists) ? myInfo->fs : 0;
    
    sprintf(FTI_Exec->meta[3].ckptFile, "Ckpt%d-Rank%d.fti", FTI_Exec->ckptID, FTI_Topo->myRank);
    
    closedir(L3CkptDir);
    free(groupInfo);
    free(myInfo);
    //free(entry);
    MPI_Type_free(&MPI_L3Info);
    
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
                  FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt)
{
    if (mkdir(FTI_Ckpt[3].dir, 0777) == -1) {
        if (errno != EEXIST) {
            FTI_Print("Cannot create directory", FTI_EROR);
        }
    }

    int erased[FTI_BUFS];
    
    if (FTI_Conf->ioMode == FTI_IO_FTIFF) {
        
        if ( FTI_CheckL3RecoverInit( FTI_Exec, FTI_Topo, FTI_Ckpt, erased ) != FTI_SCES ) {
            FTI_Print("No restart possible from L3. Ckpt files missing.", FTI_DBUG);
            return FTI_NSCS;
        }

    } 
   
    else {
    
        // Checking erasures
        if (FTI_CheckErasures(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt, erased) != FTI_SCES) {
            FTI_Print("Error checking erasures.", FTI_DBUG);
            return FTI_NSCS;
        }
    }

    printf("rank: %i, line: %i\n", FTI_Topo->myRank, __LINE__);
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
        sprintf(str, "There are %d encoded/checkpoint files missing in this group.", l);
        FTI_Print(str, FTI_DBUG);
        int res = FTI_Try(FTI_Decode(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt, erased), "use RS-decoding to regenerate the missing data.");
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
                  FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt)
{
   if (!FTI_Ckpt[4].isInline || FTI_Conf->ioMode == FTI_IO_POSIX || FTI_Conf->ioMode == FTI_IO_FTIFF) {
       return FTI_RecoverL4Posix(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt);
   }
   else {
       switch(FTI_Conf->ioMode) {
          case FTI_IO_MPI:
             return FTI_RecoverL4Mpi(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt);
#ifdef ENABLE_SIONLIB // --> If SIONlib is installed
          case FTI_IO_SIONLIB:
             return FTI_RecoverL4Sionlib(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt);
#endif
       }
   }
}

/*-------------------------------------------------------------------------*/
/**
    @brief      Init of FTI-FF L4 recovery
    @param      FTI_Exec        Execution metadata.
    @param      FTI_Topo        Topology metadata.
    @param      FTI_Ckpt        Checkpoint metadata.
    @return     integer         FTI_SCES if successful.

    This function initializes the L4 checkpoint recovery. It checks for 
    erasures and loads the required meta data. 
 **/
/*-------------------------------------------------------------------------*/
int FTI_CheckL4RecoverInit( FTIT_execution* FTI_Exec, FTIT_topology* FTI_Topo, 
                  FTIT_checkpoint* FTI_Ckpt )
{
    char str[FTI_BUFS], tmpfn[FTI_BUFS];
    int fexist = 0, fileTarget, ckptID, fcount;
    struct dirent *entry = malloc(sizeof(struct dirent));
    struct stat ckptFS;
    DIR *L4CkptDir = opendir( FTI_Ckpt[4].dir );
    if(L4CkptDir) {
        while(entry = readdir(L4CkptDir)) {
            if(strcmp(entry->d_name,".") && strcmp(entry->d_name,"..")) { 
                sprintf(str, "FTI-FF: L4RecoveryInit - found file with name: %s", entry->d_name);
                FTI_Print(str, FTI_DBUG);
                sscanf(entry->d_name, "Ckpt%d-Rank%d.fti", &ckptID, &fileTarget );
                if( fileTarget == FTI_Topo->myRank ) {
                    sprintf(tmpfn, "%s/%s", FTI_Ckpt[4].dir, entry->d_name);
                    int ferr = stat(tmpfn, &ckptFS);
                    if (!ferr && S_ISREG(ckptFS.st_mode)) {
                        FTI_Exec->meta[4].fs[0] = ckptFS.st_size;    
                        FTI_Exec->ckptID = ckptID;
                        strncpy(FTI_Exec->meta[1].ckptFile, entry->d_name, NAME_MAX);
                        strncpy(FTI_Exec->meta[4].ckptFile, entry->d_name, NAME_MAX);
                        fexist = 1;
                        break;
                    }            
                }
            }
        }
    }
    MPI_Allreduce(&fexist, &fcount, 1, MPI_INT, MPI_SUM, FTI_COMM_WORLD);
    int fneeded = FTI_Topo->nbNodes*FTI_Topo->nbApprocs;
    int res = (fcount == fneeded) ? FTI_SCES : FTI_NSCS;
    closedir(L4CkptDir);
    return res;
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
                  FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt)
{
   FTI_Print("Starting recovery L4 using Posix I/O.", FTI_DBUG);
   if (mkdir(FTI_Ckpt[1].dir, 0777) == -1) {
     if (errno != EEXIST) {
         FTI_Print("Directory L1 could NOT be created.", FTI_WARN);
     }
   }

   // Checking erasures
   if (FTI_Conf->ioMode == FTI_IO_FTIFF) {
        // FTI_CheckL4RestartInit sets: 
        // FTI_Exec->meta[1].ckptFile, 
        // FTI_Exec->meta[4].ckptFile and
        // FTI_Exec->meta[4].fs[0]
        if ( FTI_CheckL4RecoverInit( FTI_Exec, FTI_Topo, FTI_Ckpt ) != FTI_SCES ) {
            FTI_Print("No restart possible from L4. Ckpt files missing.", FTI_DBUG);
            return FTI_NSCS;
        }
   } 
   
   else {
       int erased[FTI_BUFS];
       if (FTI_CheckErasures(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt, erased) != FTI_SCES) {
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

       sprintf(FTI_Exec->meta[1].ckptFile, "Ckpt%d-Rank%d.fti", FTI_Exec->ckptID, FTI_Topo->myRank);
       sprintf(FTI_Exec->meta[4].ckptFile, "Ckpt%d-Rank%d.fti", FTI_Exec->ckptID, FTI_Topo->myRank);
   }

   char gfn[FTI_BUFS], lfn[FTI_BUFS];
   sprintf(lfn, "%s/%s", FTI_Ckpt[1].dir, FTI_Exec->meta[1].ckptFile);
   sprintf(gfn, "%s/%s", FTI_Ckpt[4].dir, FTI_Exec->meta[4].ckptFile);

   FILE* gfd = fopen(gfn, "rb");
   if (gfd == NULL) {
      FTI_Print("R4 cannot open the ckpt. file in the PFS.", FTI_WARN);
      return FTI_NSCS;
   }

   FILE* lfd = fopen(lfn, "wb");
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
      FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt)
{
   FTI_Print("Starting recovery L4 using MPI-IO.", FTI_DBUG);
   // create local directories
   if (mkdir(FTI_Ckpt[1].dir, 0777) == -1) {
      if (errno != EEXIST) {
          FTI_Print("Directory L1 could NOT be created.", FTI_WARN);
      }
   }

   // TODO enable to set stripping unit in the config file (Maybe also other hints)
   // enable collective buffer optimization
   MPI_Info info;
   MPI_Info_create(&info);
   MPI_Info_set(info, "romio_cb_read", "enable");

   // set stripping unit to 4MB
   MPI_Info_set(info, "stripping_unit", "4194304");

   sprintf(FTI_Exec->meta[1].ckptFile, "Ckpt%d-Rank%d.fti", FTI_Exec->ckptID, FTI_Topo->myRank);
   sprintf(FTI_Exec->meta[4].ckptFile, "Ckpt%d-mpiio.fti", FTI_Exec->ckptID);
   char gfn[FTI_BUFS], lfn[FTI_BUFS];
   sprintf(lfn, "%s/%s", FTI_Ckpt[1].dir, FTI_Exec->meta[1].ckptFile);
   sprintf(gfn, "%s/%s", FTI_Ckpt[4].dir, FTI_Exec->meta[4].ckptFile);

   // open parallel file
   MPI_File pfh;
   int buf = MPI_File_open(FTI_COMM_WORLD, gfn, MPI_MODE_RDWR, info, &pfh);
   // check if successful
   if (buf != 0) {
      errno = 0;
      char mpi_err[FTI_BUFS];
      int reslen;
      MPI_Error_string(buf, mpi_err, &reslen);
      if (buf != MPI_ERR_NO_SUCH_FILE) {
         char str[FTI_BUFS];
         snprintf(str, FTI_BUFS, "Unable to access file [MPI ERROR - %i] %s", buf, mpi_err);
         FTI_Print(str, FTI_EROR);
      }
      return FTI_NSCS;
   }

   // collect chunksizes of other ranks
   MPI_Offset* chunkSizes = talloc(MPI_Offset, FTI_Topo->nbApprocs*FTI_Topo->nbNodes);
   MPI_Allgather(FTI_Exec->meta[4].fs, 1, MPI_OFFSET, chunkSizes, 1, MPI_OFFSET, FTI_COMM_WORLD);

   MPI_Offset offset = 0;
   // set file offset
   int i;
   for (i = 0; i < FTI_Topo->splitRank; i++) {
      offset += chunkSizes[i];
   }
   free(chunkSizes);

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
         char mpi_err[FTI_BUFS];
         char str[FTI_BUFS];
         int reslen;
         MPI_Error_string(buf, mpi_err, &reslen);
         snprintf(str, FTI_BUFS, "R4 cannot read from the ckpt. file in the PFS. [MPI ERROR - %i] %s", buf, mpi_err);
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
#ifdef ENABLE_SIONLIB // --> If SIONlib is installed
int FTI_RecoverL4Sionlib(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
      FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt)
{
   FTI_Print("Starting recovery L4 using Sionlib.", FTI_DBUG);
   //Create local directories
   if (mkdir(FTI_Ckpt[1].dir, 0777) == -1) {
       if (errno != EEXIST) {
           FTI_Print("Directory L1 could NOT be created.", FTI_WARN);
       }
   }

   sprintf(FTI_Exec->meta[1].ckptFile, "Ckpt%d-Rank%d.fti", FTI_Exec->ckptID, FTI_Topo->myRank);
   sprintf(FTI_Exec->meta[4].ckptFile, "Ckpt%d-sionlib.fti", FTI_Exec->ckptID);
   char gfn[FTI_BUFS], lfn[FTI_BUFS];
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
      sion_parclose_mapped_mpi(sid);
      free(file_map);
      free(ranks);
      free(rank_map);
      free(chunkSizes);
      return FTI_NSCS;
   }

   int res = sion_seek(sid, FTI_Topo->splitRank, SION_CURRENT_BLK, SION_CURRENT_POS);
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
            char str[FTI_BUFS];
            sprintf(str, "SIONlib: Unable to read %lu Bytes from file", bSize);
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
            FTI_Print("R4 cannot write to the local ckpt. file.", FTI_DBUG);
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
