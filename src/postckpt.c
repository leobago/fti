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
  @brief      Send Ckpt file.
  @param      fs              Ckpt file size
  @param      destination     destination group rank
  @return     integer         FTI_SCES if successful.

  This function sends ckpt file to partner process. Partner should call
  FTI_RecvPtner to receive this file.

 **/
/*-------------------------------------------------------------------------*/
int FTI_SendCkpt(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec, FTIT_checkpoint* FTI_Ckpt,
                 unsigned long fs, int destination)
{
    int bytes; //bytes read by fread
    char lfn[FTI_BUFS], str[FTI_BUFS];
    FILE* lfd;

    sprintf(lfn, "%s/%s", FTI_Conf->lTmpDir, FTI_Exec->ckptFile);

    sprintf(str, "L2 trying to access local ckpt. file (%s).", lfn);
    FTI_Print(str, FTI_DBUG);

    lfd = fopen(lfn, "rb");
    if (lfd == NULL) {
        FTI_Print("FTI failed to open L2 Ckpt. file.", FTI_DBUG);
        return FTI_NSCS;
    }

    char* buffer = talloc(char, FTI_Conf->blockSize);
    unsigned long toSend = fs; //remaining data to send
    while (toSend > 0) {
        int sendSize = (toSend > FTI_Conf->blockSize) ? FTI_Conf->blockSize : toSend;
        bytes = fread(buffer, sizeof(char), sendSize, lfd);

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
  @brief      Receive Ptner file.
  @param      pfs             Ptner file size
  @param      source          souce group rank
  @return     integer         FTI_SCES if successful.

  This function receives ckpt file from partner process and saves it as
  Ptner file. Partner should call FTI_SendCkpt to send file.

 **/
/*-------------------------------------------------------------------------*/
int FTI_RecvPtner(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec, FTIT_checkpoint* FTI_Ckpt,
                  unsigned long pfs, int source)
{
    int rank;
    char pfn[FTI_BUFS], str[FTI_BUFS];

    //heads need to use ckptFile to get ckptID and rank
    sscanf(FTI_Exec->ckptFile, "Ckpt%d-Rank%d.fti", &FTI_Exec->ckptID, &rank);
    sprintf(pfn, "%s/Ckpt%d-Pcof%d.fti", FTI_Conf->lTmpDir, FTI_Exec->ckptID, rank);
    sprintf(str, "L2 trying to access Ptner file (%s).", pfn);
    FTI_Print(str, FTI_DBUG);

    FILE* pfd = fopen(pfn, "wb");
    if (pfd == NULL) {
        FTI_Print("FTI failed to open L2 ptner file.", FTI_DBUG);
        return FTI_NSCS;
    }

    char* buffer = talloc(char, FTI_Conf->blockSize);
    unsigned long toRecv = pfs; //remaining data to receive
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
    unsigned long maxFs, fs, pfs; //ckpt file size, partner file size
    int source = FTI_Topo->left; //receive Ckpt file from this process
    int destination = FTI_Topo->right; //send Ckpt file to this process
    int res;
    char str[FTI_BUFS];

    FTI_Print("Starting checkpoint post-processing L2", FTI_DBUG);
    res = FTI_Try(FTI_GetMeta(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt, &fs, &maxFs, group, 0), "obtain metadata.");
    if (res == FTI_NSCS) {
        return FTI_NSCS;
    }

    res = FTI_GetPtnerSize(FTI_Conf, FTI_Topo, FTI_Ckpt, &pfs, group, 0);
    if (res == FTI_NSCS) {
        return FTI_NSCS;
    }

    if (FTI_Topo->groupRank % 2) { //first send, then receive
        res = FTI_SendCkpt(FTI_Conf, FTI_Exec, FTI_Ckpt, fs, destination);
        if (res != FTI_SCES) {
            return FTI_NSCS;
        }
        res = FTI_RecvPtner(FTI_Conf, FTI_Exec, FTI_Ckpt, pfs, source);
        if (res != FTI_SCES) {
            return FTI_NSCS;
        }
    } else { //first receive, then send
        res = FTI_RecvPtner(FTI_Conf, FTI_Exec, FTI_Ckpt, pfs, source);
        if (res != FTI_SCES) {
            return FTI_NSCS;
        }
        res = FTI_SendCkpt(FTI_Conf, FTI_Exec, FTI_Ckpt, fs, destination);
        if (res != FTI_SCES) {
            return FTI_NSCS;
        }
    }
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
    int *matrix, cnt, i, j, init, src, offset, dest, matVal, res, bs = FTI_Conf->blockSize, rank;
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

    sscanf(FTI_Exec->ckptFile, "Ckpt%d-Rank%d.fti", &FTI_Exec->ckptID, &rank);
    sprintf(lfn, "%s/%s", FTI_Conf->lTmpDir, FTI_Exec->ckptFile);
    sprintf(efn, "%s/Ckpt%d-RSed%d.fti", FTI_Conf->lTmpDir, FTI_Exec->ckptID, rank);

    sprintf(str, "L3 trying to access local ckpt. file (%s).", lfn);
    FTI_Print(str, FTI_DBUG);

    //all files in group must have the same size
    if (truncate(lfn, maxFs) == -1) {
        FTI_Print("Error with truncate on checkpoint file", FTI_WARN);
        return FTI_NSCS;
    }

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

    if (truncate(lfn, fs) == -1) {
        FTI_Print("Error with re-truncate on checkpoint file", FTI_WARN);
        return FTI_NSCS;
    }

    free(data);
    free(matrix);
    free(coding);
    free(myData);
    fclose(lfd);
    fclose(efd);

    //write checksum in metadata
    char checksum[MD5_DIGEST_LENGTH];
    res = FTI_Checksum(efn, checksum);
    if (res != FTI_SCES) {
        return FTI_NSCS;
    }
    res = FTI_WriteRSedChecksum(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt, rank, checksum);

    return res;
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
   int i, res, gRank, member, reslen;
   size_t fs, maxFs;
   MPI_Comm lComm;
   size_t data_written = 0;
   char lfn[FTI_BUFS], gfn[FTI_BUFS], str[FTI_BUFS], mpi_err[FTI_BUFS];
   unsigned long ps, pos = 0;
   FILE *lfd;
   FILE *gfd;
   MPI_Datatype dType;
   MPI_Status status;
   MPI_Offset *chunkSizes, *lChunkSizes, offset = 0;

   if (level == -1 || (level != -1 && FTI_Topo->amIaHead && FTI_Ckpt[4].isInline )) {
      return FTI_SCES; // Fake call for inline PFS checkpoint
   }

   FTI_Print("Starting checkpoint post-processing L4", FTI_DBUG);

   // assign index for metadata
   member = (FTI_Topo->amIaHead) ? group-1 : 0;

   // determine rank
   gRank = (FTI_Topo->amIaHead) ? FTI_Topo->body[member] : FTI_Topo->myRank;

   // determine number of members
   int nbMembers = (FTI_Topo->amIaHead) ? FTI_Topo->nbApprocs : 1;

   // determine split rank from global rank
   int splitRank = (FTI_Topo->amIaHead) ? gRank - ( (FTI_Topo->splitRank+1) * FTI_Topo->nbHeads ) : FTI_Topo->splitRank;

   // align ps to blocksize
   ps = (FTI_Exec->meta[member].maxFs / FTI_Conf->transferSize) * FTI_Conf->transferSize;
   if (ps < FTI_Exec->meta[member].maxFs) {
      ps = ps + FTI_Conf->transferSize;
   }

   switch (level) {
      case 0:
         sprintf(lfn, "%s/%s", FTI_Conf->lTmpDir, FTI_Exec->meta[member].ckptFile);
         break;
      case 1:
         sprintf(lfn, "%s/%s", FTI_Ckpt[1].dir, FTI_Exec->meta[member].ckptFile);
         break;
      case 2:
         sprintf(lfn, "%s/%s", FTI_Ckpt[2].dir, FTI_Exec->meta[member].ckptFile);
         break;
      case 3:
         sprintf(lfn, "%s/%s", FTI_Ckpt[3].dir, FTI_Exec->meta[member].ckptFile);
         break;
   }

   sprintf(str, "L4 trying to access local ckpt. file (%s).", lfn);
   FTI_Print(str, FTI_DBUG);

   // Open local file
   lfd = fopen(lfn, "rb");
   if (lfd == NULL) {
      FTI_Print("L4 cannot open the checkpoint file.", FTI_EROR);
      return FTI_NSCS;
   }

   // select behaviour depending on I/O
   switch (FTI_Conf->ioMode) {

      case FTI_IO_POSIX:

         sprintf(gfn, "%s/%s", FTI_Conf->gTmpDir, FTI_Exec->meta[member].ckptFile);
         gfd = fopen(gfn, "wb");

         if (gfd == NULL) {
            FTI_Print("L4 cannot open ckpt. file in the PFS.", FTI_EROR);
            fclose(lfd);
            return FTI_NSCS;
         }

         break;

      case FTI_IO_MPI:

         // collect chunksizes of other ranks
         lChunkSizes = talloc(MPI_Offset, nbMembers);
         for (i = 0; i < nbMembers; i++) {
            lChunkSizes[i] = FTI_Exec->meta[i].fs;
         }

         chunkSizes = talloc(MPI_Offset, FTI_Topo->nbApprocs*FTI_Topo->nbNodes);
         MPI_Allgather(lChunkSizes, nbMembers, MPI_OFFSET, chunkSizes, nbMembers, MPI_OFFSET, FTI_COMM_WORLD);

         // set file offset
         for (i=0; i<splitRank; i++) {
            offset += chunkSizes[i];
         }

         break; 

#ifdef ENABLE_SIONLIB // --> If SIONlib is installed
      case FTI_IO_SIONLIB:

         res = sion_seek(FTI_Exec->sid, gRank, SION_CURRENT_BLK, SION_CURRENT_POS);

         if (res!=SION_SUCCESS) {
            errno = 0;
            sprintf(str, "SIONlib: unable to set file pointer");
            FTI_Print(str, FTI_EROR);
            fclose(lfd);
            sion_parclose_mapped_mpi(FTI_Exec->sid);
            return FTI_NSCS;
         }

         break;
#endif
   }

   char *blBuf1 = talloc(char, FTI_Conf->transferSize);
   unsigned long bSize = FTI_Conf->transferSize;

   // Checkpoint files exchange
   while (pos < ps) {
      if ((FTI_Exec->meta[member].fs - pos) < FTI_Conf->transferSize)
         bSize = FTI_Exec->meta[member].fs - pos;

      size_t bytes = fread(blBuf1, sizeof(char), bSize, lfd);
      if (ferror(lfd)) {
         FTI_Print("L4 cannot read from the ckpt. file.", FTI_EROR);
         free(blBuf1);
         fclose(lfd);
         switch (FTI_Conf->ioMode) {
            case FTI_IO_POSIX:
               fclose(gfd);
               break;
            case FTI_IO_MPI:
               MPI_File_close(&FTI_Exec->pfh);
               break;
#ifdef ENABLE_SIONLIB // --> If SIONlib is installed
            case FTI_IO_SIONLIB:
               sion_parclose_mapped_mpi(FTI_Exec->sid);
               break;
#endif
         }
         return FTI_NSCS;
      }

      switch (FTI_Conf->ioMode) {

         case FTI_IO_POSIX:

            fwrite(blBuf1, sizeof(char), bytes, gfd);
            if (ferror(gfd)) {
               FTI_Print("L4 cannot write to the ckpt. file in the PFS.", FTI_EROR);
               free(blBuf1);
               fclose(lfd);
               fclose(gfd);
               return FTI_NSCS;
            }

            break;

         case FTI_IO_MPI:

            MPI_Type_contiguous(bytes, MPI_BYTE, &dType); 
            MPI_Type_commit(&dType);

            res = MPI_File_write_at(FTI_Exec->pfh, offset, blBuf1, 1, dType, &status);
            // check if successfull
            if (res != 0) {
               errno = 0;
               MPI_Error_string(res, mpi_err, &reslen);
               snprintf(str, FTI_BUFS, "Failed to write data to PFS durin Flush [MPI ERROR - %i] %s", res, mpi_err);
               FTI_Print(str, FTI_EROR);
               MPI_File_close(&FTI_Exec->pfh);
               fclose(lfd);
               return FTI_NSCS;
            }

            offset += bytes;
            MPI_Type_free(&dType);

            break;

#ifdef ENABLE_SIONLIB // --> If SIONlib is installed
         case FTI_IO_SIONLIB:

            data_written = sion_fwrite(blBuf1, sizeof(char), bytes, FTI_Exec->sid);

            if (data_written < 0) {
               FTI_Print("sionlib: could not write data", FTI_EROR);
               free(blBuf1);
               fclose(lfd);
               sion_parclose_mapped_mpi(FTI_Exec->sid);
               return FTI_NSCS;
            }

            break;
#endif
      }

      pos = pos + FTI_Conf->transferSize;

   }

   free(blBuf1);
   fclose(lfd);
   if (FTI_Conf->ioMode == FTI_IO_POSIX) {
      fclose(gfd);
   }

   return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
@brief      Selects I/O and prepares the flush for each I/O
@param      level           The level from which ckpt. files are flushed.
@return     integer         FTI_SCES if successful.

**/
/*-------------------------------------------------------------------------*/
int FTI_FlushInit(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
      FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt, int level)
{

   int res;
   char str[FTI_BUFS]; 
   if (level == -1) {
      return FTI_SCES; // Fake call for inline PFS checkpoint
   }

   // create global temp directory
   if (mkdir(FTI_Conf->gTmpDir, 0777) == -1) {
      if (errno != EEXIST) {
         FTI_Print("Cannot create global directory", FTI_EROR);
         return FTI_NSCS;
      }
   }

   // select IO
   switch (FTI_Conf->ioMode) {

      case FTI_IO_POSIX:

         res = FTI_FlushInitPosix(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt, level);
         break;

      case FTI_IO_MPI:

         res = FTI_FlushInitMpi(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt, level);
         break;

#ifdef ENABLE_SIONLIB // --> If SIONlib is installed
      case FTI_IO_SIONLIB:

         // write checkpoint
         res = FTI_FlushInitSionlib(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt, level);
         break;
#endif
   }

   return res;
}


/*-------------------------------------------------------------------------*/
/**
@brief      Prepares the Flush for POSIX I/O
@param      level           The level from which ckpt. files are flushed.
@return     integer         FTI_SCES if successful.

Init for the flush of locally stored checkpoint data to the PFS by POSIX

**/
/*-------------------------------------------------------------------------*/
int FTI_FlushInitPosix(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
              FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt, int level) 
{
	unsigned long maxFs, fs;
    int i, res;

    if (FTI_Topo->amIaHead) {

        for (i = 0; i<FTI_Topo->nbApprocs; i++) {

            res = FTI_Try(FTI_GetMeta(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt, &fs, &maxFs, i+1, level), "obtain metadata.");
            if (res != FTI_SCES) {
                FTI_Print("failed to obtain the metadata", FTI_EROR);
                return FTI_NSCS;
            }
            FTI_Exec->meta[i].fs = fs;
            FTI_Exec->meta[i].maxFs = maxFs;
            strcpy(FTI_Exec->meta[i].ckptFile, FTI_Exec->ckptFile);

        }

    } 
	
    else {

        res = FTI_Try(FTI_GetMeta(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt, &fs, &maxFs, FTI_Topo->nodeRank, level), "obtain metadata.");
        if (res != FTI_SCES)
            return FTI_NSCS;

        FTI_Exec->meta[0].fs = fs;
        FTI_Exec->meta[0].maxFs = maxFs;
        strcpy(FTI_Exec->meta[0].ckptFile, FTI_Exec->ckptFile);

    }

    return FTI_SCES;

}

/*-------------------------------------------------------------------------*/
/**
@brief      Prepares the Flush for MPI I/O
@param      level           The level from which ckpt. files are flushed.
@return     integer         FTI_SCES if successful.

   Init for the flush of locally stored checkpoint data to the PFS by MPI I/O

**/
/*-------------------------------------------------------------------------*/
int FTI_FlushInitMpi(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
      FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt, int level) 
{
   char str[FTI_BUFS], mpi_err[FTI_BUFS];
   unsigned long maxFs, fs;
   int i, res, reslen;
   MPI_Info info;
   MPI_Status status;

   if (FTI_Topo->amIaHead) {

      for (i = 0; i<FTI_Topo->nbApprocs; i++) {

         res = FTI_Try(FTI_GetMeta(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt, &fs, &maxFs, i+1, level), "obtain metadata.");
         if (res != FTI_SCES) {
            FTI_Print("failed to obtain the metadata", FTI_EROR);
            return FTI_NSCS;
         }
         FTI_Exec->meta[i].fs = fs;
         FTI_Exec->meta[i].maxFs = maxFs;
         strcpy(FTI_Exec->meta[i].ckptFile, FTI_Exec->ckptFile);

      }
      // set parallel file name
      snprintf(str, FTI_BUFS, "Ckpt%d-mpiio.fti", (FTI_Exec->ckptID+1));
      sprintf(FTI_Exec->fn, "%s/%s", FTI_Conf->gTmpDir, str);


   } else {

      res = FTI_Try(FTI_GetMeta(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt, &fs, &maxFs, FTI_Topo->nodeRank, level), "obtain metadata.");
      if (res != FTI_SCES)
         return FTI_NSCS;

      FTI_Exec->meta[0].fs = fs;
      FTI_Exec->meta[0].maxFs = maxFs;
      strcpy(FTI_Exec->meta[0].ckptFile, FTI_Exec->ckptFile);

      // set parallel file name
      snprintf(str, FTI_BUFS, "Ckpt%d-mpiio.fti", (FTI_Exec->ckptID));
      sprintf(FTI_Exec->fn, "%s/%s", FTI_Conf->gTmpDir, str);


   }

   // enable collective buffer optimization
   MPI_Info_create(&info);
   MPI_Info_set(info, "romio_cb_write", "enable");

   // TODO enable to set stripping unit in the config file (Maybe also other hints)
   // set stripping unit to 4MB
   MPI_Info_set(info, "stripping_unit", "4194304");

   // open parallel file (collective call)
   res = MPI_File_open(FTI_COMM_WORLD, FTI_Exec->fn, MPI_MODE_WRONLY|MPI_MODE_CREATE, info, &(FTI_Exec->pfh)); 

   // check if successfull
   if (res != 0) {
      errno = 0;
      MPI_Error_string(res, mpi_err, &reslen);
      snprintf(str, FTI_BUFS, "unable to create file during flush initialization [MPI ERROR - %i] %s", res, mpi_err);
      FTI_Print(str, FTI_EROR);
      return FTI_NSCS;
   }

   MPI_Info_free(&info);
   return FTI_SCES;

}

/*-------------------------------------------------------------------------*/
/**
@brief      Prepares the Flush for SIONlib I/O
@param      level           The level from which ckpt. files are flushed.
@return     integer         FTI_SCES if successful.

   Init for the flush of locally stored checkpoint data to the PFS by SIONlib

**/
/*-------------------------------------------------------------------------*/
#ifdef ENABLE_SIONLIB // --> If SIONlib is installed
int FTI_FlushInitSionlib(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
      FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt, int level) 
{
   unsigned long maxFs, fs;
   int i, res, numFiles = 1, fsblksize = -1, nlocaltasks = 1;
   sion_int64 *chunkSizes;
   int *gRankList;
   int *file_map;
   int *rank_map;
   char *newfname = NULL;
   char str[FTI_BUFS];
   FILE *dfp;

   if (FTI_Topo->amIaHead) {
      gRankList = talloc(int, FTI_Topo->nbApprocs+1);
      file_map = talloc(int, FTI_Topo->nbApprocs+1);
      rank_map = talloc(int, FTI_Topo->nbApprocs+1);
      chunkSizes = talloc(sion_int64, FTI_Topo->nbApprocs+1);
      nlocaltasks = FTI_Topo->nbApprocs+1;
      // gRankList has global ranks for which head is responsible.
      // file_map maps file indices to ranks. we have only one file, so file index 0 for all ranks.
      // rank_map has the ranks for the file mapping. indices are corresponding.
      // SIONlib cant map if the writing rank is excluded. hence head rank is included with chunksize = 0.
      gRankList[0] = FTI_Topo->myRank;
      file_map[0]  = 0;
      rank_map[0]  = gRankList[0];
      chunkSizes[0] = 0;
      for(i=1;i<FTI_Topo->nbApprocs+1;i++) {
         gRankList[i] = FTI_Topo->body[i-1];
         file_map[i]  = 0;
         rank_map[i]  = gRankList[i];
      }
      // get metadata
      for (i = 0; i<FTI_Topo->nbApprocs; i++) {
         res = FTI_Try(FTI_GetMeta(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt, &fs, &maxFs, i+1, level), "obtain metadata.");
         if (res != FTI_SCES) {
            FTI_Print("failed to obtain the metadata", FTI_EROR);
            return FTI_NSCS;
         }
         FTI_Exec->meta[i].fs = fs;
         FTI_Exec->meta[i].maxFs = maxFs;
         strcpy(FTI_Exec->meta[i].ckptFile, FTI_Exec->ckptFile);
         chunkSizes[i+1] = fs;
      }
      // set parallel file name
      snprintf(str, FTI_BUFS, "Ckpt%d-sionlib.fti", (FTI_Exec->ckptID+1));
      sprintf(FTI_Exec->fn, "%s/%s", FTI_Conf->gTmpDir, str);

      // open parallel file in collective call for all heads
      FTI_Exec->sid = sion_paropen_mapped_mpi(FTI_Exec->fn, "wb,posix", &numFiles, FTI_COMM_WORLD, &nlocaltasks, &gRankList, &chunkSizes, &file_map, &rank_map, &fsblksize, &dfp); 
      if (FTI_Exec->sid == -1) {
         FTI_Print("PAROPEN MAPPED ERROR", FTI_EROR);
         return FTI_NSCS;
      }
   } 

   else {
      // set parallel file name
      snprintf(str, FTI_BUFS, "Ckpt%d-sionlib.fti", FTI_Exec->ckptID);
      sprintf(FTI_Exec->fn, "%s/%s", FTI_Conf->gTmpDir, str);

      res = FTI_Try(FTI_GetMeta(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt, &fs, &maxFs, FTI_Topo->nodeRank, level), "obtain metadata.");
      if (res != FTI_SCES)
         return FTI_NSCS;

      // set parameter for paropen mapped call
      if (FTI_Topo->nbHeads == 1 && FTI_Topo->groupID == 1) {

         // for the case that we have a head
         nlocaltasks = 2;
         gRankList = talloc(int, 2);
         chunkSizes = talloc(sion_int64, 2);
         file_map = talloc(int, 2);
         rank_map = talloc(int, 2);

         chunkSizes[0] = 0;
         chunkSizes[1] = fs;
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

         *chunkSizes = fs;
         *gRankList = FTI_Topo->myRank;
         *file_map = 0;
         *rank_map = *gRankList;

      }

      FTI_Exec->meta[0].fs = fs;
      FTI_Exec->meta[0].maxFs = maxFs;
      strcpy(FTI_Exec->meta[0].ckptFile, FTI_Exec->ckptFile);

      FTI_Exec->sid = sion_paropen_mapped_mpi(FTI_Exec->fn, "wb,posix", &numFiles, FTI_COMM_WORLD, &nlocaltasks, &gRankList, &chunkSizes, &file_map, &rank_map, &fsblksize, NULL); 
      if (FTI_Exec->sid == -1) {
         FTI_Print("PAROPEN MAPPED ERROR", FTI_EROR);
         return FTI_NSCS;
      }
   }

   return FTI_SCES;
}
#endif

/*-------------------------------------------------------------------------*/
/**
@brief      Selects I/O and Finalizes Flush respectively
@param      level           The level from which ckpt. files are flushed.
@return     integer         FTI_SCES if successful.

**/
/*-------------------------------------------------------------------------*/
int FTI_FlushFinalize(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
      FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt, int level) 
{

   int res;
   char str[FTI_BUFS];

   if (level == -1 ) {
      return FTI_SCES; // Fake call for inline PFS checkpoint
   }

   // select IO
   switch(FTI_Conf->ioMode) {

      case FTI_IO_POSIX:

         res = FTI_SCES; 
         break;

      case FTI_IO_MPI:

         res = FTI_FlushFinalizeMpi(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt);
         break;

#ifdef ENABLE_SIONLIB // --> If SIONlib is installed
      case FTI_IO_SIONLIB:

         res = FTI_FlushFinalizeSionlib(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt);
         break;
#endif
   }

   return res;

}

/*-------------------------------------------------------------------------*/
/**
@brief      Finalizes flush for SIONlib I/O
@return     integer         FTI_SCES if successful.

**/
/*-------------------------------------------------------------------------*/
#ifdef ENABLE_SIONLIB // --> If SIONlib is installed
int FTI_FlushFinalizeSionlib(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
      FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt) 
{

   int res, i, j, nbSectors, save_sectorID, save_groupID;

   sion_parclose_mapped_mpi(FTI_Exec->sid); 

   if (FTI_Topo->amIaHead) {

      // set parallel file name
      snprintf(FTI_Exec->ckptFile, FTI_BUFS, "Ckpt%d-sionlib.fti", FTI_Exec->ckptID+1);

      // update meta data
      for (i = 0; i < FTI_Topo->nbApprocs; i++) {
         res = FTI_Try(FTI_CreateMetadata(FTI_Conf, FTI_Exec, FTI_Topo, 1, i), "create metadata.");
         if (res != FTI_SCES) {
            return FTI_NSCS;
         }
      }
   }

   else {

      // set parallel file name
      snprintf(FTI_Exec->ckptFile, FTI_BUFS, "Ckpt%d-sionlib.fti", FTI_Exec->ckptID);
      res = FTI_Try(FTI_CreateMetadata(FTI_Conf, FTI_Exec, FTI_Topo, 1, 0), "create metadata.");
      if (res != FTI_SCES) {
         return FTI_NSCS;
      }

   }
   return FTI_SCES;

}
#endif

/*-------------------------------------------------------------------------*/
/**
@brief      Finalizes flush for MPI I/O
@return     integer         FTI_SCES if successful.

**/
/*-------------------------------------------------------------------------*/
int FTI_FlushFinalizeMpi(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
      FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt) 
{

   int res, i, j, nbSectors, save_sectorID, save_groupID;
   unsigned long *fs, *mfs;

   MPI_File_close(&(FTI_Exec->pfh));

   if (FTI_Topo->amIaHead) {

      // set parallel file name
      snprintf(FTI_Exec->ckptFile, FTI_BUFS, "Ckpt%d-mpiio.fti", FTI_Exec->ckptID+1);

      // update meta data
      for (i = 0; i < FTI_Topo->nbApprocs; i++) {
         res = FTI_Try(FTI_CreateMetadata(FTI_Conf, FTI_Exec, FTI_Topo, 1, i), "create metadata.");
         if (res != FTI_SCES) {
            return FTI_NSCS;
         }
      }
   }

   else {

      // set parallel file name
      snprintf(FTI_Exec->ckptFile, FTI_BUFS, "Ckpt%d-mpiio.fti", FTI_Exec->ckptID);
      res = FTI_Try(FTI_CreateMetadata(FTI_Conf, FTI_Exec, FTI_Topo, 1, 0), "create metadata.");
      if (res != FTI_SCES) {
         return FTI_NSCS;
      }

   }
   return FTI_SCES;
}
