#include "interface.h"

/*-------------------------------------------------------------------------*/
/**
  @brief      It writes the checkpoint data in the target file.
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Topo        Topology metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @param      FTI_Data        Dataset metadata.
  @return     integer         FTI_SCES if successful.

  This function checks whether the checkpoint needs to be local or remote,
  opens the target file and writes dataset per dataset, the checkpoint data,
  it finally flushes and closes the checkpoint file.

 **/
/*-------------------------------------------------------------------------*/
int FTI_InitPosixICP(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt,
        FTIT_dataset* FTI_Data)
{
    char str[FTI_BUFS]; //For console output
    snprintf(str, FTI_BUFS, "Initialize incremental checkpoint (ID: %d, Lvl: %d, I/O: POSIX)",
            FTI_Exec->ckptID, FTI_Exec->ckptLvel);
    FTI_Print(str, FTI_DBUG);

    char fn[FTI_BUFS];
    int level = FTI_Exec->ckptLvel;
    if (level == 4 && FTI_Ckpt[4].isInline) { //If inline L4 save directly to global directory
        snprintf(fn, FTI_BUFS, "%s/%s", FTI_Conf->gTmpDir, FTI_Exec->meta[0].ckptFile);
    }
    else {
        snprintf(fn, FTI_BUFS, "%s/%s", FTI_Conf->lTmpDir, FTI_Exec->meta[0].ckptFile);
    }

    // open task local ckpt file
    FILE *fd = fopen(fn, "wb");
    if (fd == NULL) {
        snprintf(str, FTI_BUFS, "FTI checkpoint file (%s) could not be opened.", fn);
        FTI_Print(str, FTI_EROR);

        return FTI_NSCS;
    }
    FTI_Exec->iCPInfo.fh = (void*)fd;
    FTI_Exec->iCPInfo.result = FTI_SCES;

    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Writes ckpt to PFS using POSIX.
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Topo        Topology metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @param      FTI_Data        Dataset metadata.
  @return     integer         FTI_SCES if successful.

 **/
/*-------------------------------------------------------------------------*/
int FTI_WritePosixVar(int varID, FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt,
        FTIT_dataset* FTI_Data)
{
    FILE *fd = (FILE*)(FTI_Exec->iCPInfo.fh);
    
    char str[FTI_BUFS];

    long offset = 0;

    // write data into ckpt file
    int i;
    for (i = 0; i < FTI_Exec->nbVar; i++) {
        if( FTI_Data[i].id == varID ) {
            clearerr(fd);
            if ( fseek( fd, offset, SEEK_SET ) == -1 ) {
                FTI_Print("Error on fseek in writeposixvar", FTI_EROR );
                FTI_Exec->iCPInfo.result = FTI_NSCS;
                return FTI_NSCS;
            }
            size_t written = 0;
            int fwrite_errno;
            while (written < FTI_Data[i].count && !ferror(fd)) {
                errno = 0;
                written += fwrite(((char*)FTI_Data[i].ptr) + (FTI_Data[i].eleSize*written), FTI_Data[i].eleSize, FTI_Data[i].count - written, fd);
                fwrite_errno = errno;
            }
            if (ferror(fd)) {
                FTI_Exec->iCPInfo.result = FTI_NSCS;
                char error_msg[FTI_BUFS];
                error_msg[0] = 0;
                strerror_r(fwrite_errno, error_msg, FTI_BUFS);
                snprintf(str, FTI_BUFS, "Dataset #%d could not be written: %s.", FTI_Data[i].id, error_msg);
                FTI_Print(str, FTI_EROR);
                fclose(fd);
                return FTI_NSCS;
            }
        }
        offset += FTI_Data[i].count*FTI_Data[i].eleSize;
    }


    return FTI_SCES;

}

/*-------------------------------------------------------------------------*/
/**
  @brief      It writes the checkpoint data in the target file.
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Topo        Topology metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @param      FTI_Data        Dataset metadata.
  @return     integer         FTI_SCES if successful.

  This function checks whether the checkpoint needs to be local or remote,
  opens the target file and writes dataset per dataset, the checkpoint data,
  it finally flushes and closes the checkpoint file.

 **/
/*-------------------------------------------------------------------------*/
int FTI_FinalizePosixICP(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt,
        FTIT_dataset* FTI_Data)
{
    //Check if all processes have written correctly (every process must succeed)
    int allRes;
    FILE *fd = (FILE*)(FTI_Exec->iCPInfo.fh);
    MPI_Allreduce(&(FTI_Exec->iCPInfo.result), &allRes, 1, MPI_INT, MPI_SUM, FTI_COMM_WORLD);
    if (allRes != FTI_SCES) {
        FTI_Print("Not all variables were successfully written!.", FTI_EROR);
        fclose(fd);
        return FTI_NSCS;
    }
    DBG_MSG("until here",-1);
    
    // close file
    if (fclose(fd) != 0) {
        FTI_Print("FTI checkpoint file could not be closed.", FTI_EROR);

        return FTI_NSCS;
    }
    
    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      It writes the checkpoint data in the target file.
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Topo        Topology metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @param      FTI_Data        Dataset metadata.
  @return     integer         FTI_SCES if successful.

  This function checks whether the checkpoint needs to be local or remote,
  opens the target file and writes dataset per dataset, the checkpoint data,
  it finally flushes and closes the checkpoint file.

 **/
/*-------------------------------------------------------------------------*/
int FTI_InitMpiICP(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt,
        FTIT_dataset* FTI_Data)
{
    int res;
    FTI_Print("I/O mode: MPI-IO.", FTI_DBUG);
    char str[FTI_BUFS], mpi_err[FTI_BUFS];

    // enable collective buffer optimization
    MPI_Info info;
    MPI_Info_create(&info);
    MPI_Info_set(info, "romio_cb_write", "enable");

    // TODO enable to set stripping unit in the config file (Maybe also other hints)
    // set stripping unit to 4MB
    MPI_Info_set(info, "stripping_unit", "4194304");

    char gfn[FTI_BUFS], ckptFile[FTI_BUFS];
    snprintf(ckptFile, FTI_BUFS, "Ckpt%d-mpiio.fti", FTI_Exec->ckptID);
    snprintf(gfn, FTI_BUFS, "%s/%s", FTI_Conf->gTmpDir, ckptFile);
    // open parallel file (collective call)
    MPI_File pfh;

#ifdef LUSTRE
    if (FTI_Topo->splitRank == 0) {
        res = llapi_file_create(gfn, FTI_Conf->stripeUnit, FTI_Conf->stripeOffset, FTI_Conf->stripeFactor, 0);
        if (res) {
            char error_msg[FTI_BUFS];
            error_msg[0] = 0;
            strerror_r(-res, error_msg, FTI_BUFS);
            snprintf(str, FTI_BUFS, "[Lustre] %s.", error_msg);
            FTI_Print(str, FTI_WARN);
        } else {
            snprintf(str, FTI_BUFS, "[LUSTRE] file:%s striping_unit:%i striping_factor:%i striping_offset:%i",
                    ckptFile, FTI_Conf->stripeUnit, FTI_Conf->stripeFactor, FTI_Conf->stripeOffset);
            FTI_Print(str, FTI_DBUG);
        }
    }
#endif
    res = MPI_File_open(FTI_COMM_WORLD, gfn, MPI_MODE_WRONLY|MPI_MODE_CREATE, info, &pfh);

    // check if successful
    if (res != 0) {
        errno = 0;
        int reslen;
        MPI_Error_string(res, mpi_err, &reslen);
        snprintf(str, FTI_BUFS, "unable to create file %s [MPI ERROR - %i] %s", gfn, res, mpi_err);
        FTI_Print(str, FTI_EROR);
        return FTI_NSCS;
    }
    
    MPI_Offset chunkSize = FTI_Exec->ckptSize;

    // collect chunksizes of other ranks
    MPI_Offset* chunkSizes = talloc(MPI_Offset, FTI_Topo->nbApprocs * FTI_Topo->nbNodes);
    MPI_Allgather(&chunkSize, 1, MPI_OFFSET, chunkSizes, 1, MPI_OFFSET, FTI_COMM_WORLD);

    // set file offset
    MPI_Offset offset = 0;
    int i;
    for (i = 0; i < FTI_Topo->splitRank; i++) {
        offset += chunkSizes[i];
    }
    free(chunkSizes);
    
    FTI_Exec->iCPInfo.offset = offset;

    FTI_Exec->iCPInfo.pfh = pfh;
    MPI_Info_free(&info);

    FTI_Exec->iCPInfo.result = FTI_SCES;
    return FTI_SCES;

}

/*-------------------------------------------------------------------------*/
/**
  @brief      Writes ckpt to PFS using POSIX.
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Topo        Topology metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @param      FTI_Data        Dataset metadata.
  @return     integer         FTI_SCES if successful.

 **/
/*-------------------------------------------------------------------------*/
int FTI_WriteMpiVar(int varID, FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt,
        FTIT_dataset* FTI_Data)
{
    char mpi_err[FTI_BUFS], str[FTI_BUFS];
    
    MPI_File pfh = FTI_Exec->iCPInfo.pfh;

    MPI_Offset offset = FTI_Exec->iCPInfo.offset; 
        
    int i;
    for (i = 0; i < FTI_Exec->nbVar; i++) {
        if ( FTI_Data[i].id == varID ) {
            long pos = 0;
            long varSize = FTI_Data[i].size;
            long bSize = FTI_Conf->transferSize;
            char *data_ptr = FTI_Data[i].ptr;
            while (pos < varSize) {
                if ((varSize - pos) < FTI_Conf->transferSize) {
                    bSize = varSize - pos;
                }

                MPI_Datatype dType;
                MPI_Type_contiguous(bSize, MPI_BYTE, &dType);
                MPI_Type_commit(&dType);

                int res = MPI_File_write_at(pfh, offset, data_ptr, 1, dType, MPI_STATUS_IGNORE);
                // check if successful
                if (res != 0) {
                    FTI_Exec->iCPInfo.result = FTI_NSCS;
                    errno = 0;
                    int reslen;
                    MPI_Error_string(res, mpi_err, &reslen);
                    snprintf(str, FTI_BUFS, "Failed to write protected_var[%i] to PFS  [MPI ERROR - %i] %s", i, res, mpi_err);
                    FTI_Print(str, FTI_EROR);
                    MPI_File_close(&pfh);
                    return FTI_NSCS;
                }
                MPI_Type_free(&dType);
                data_ptr += bSize;
                offset += bSize;
                pos = pos + bSize;
            }
        }
        offset += FTI_Data[i].size;
    }

    return FTI_SCES;

}

/*-------------------------------------------------------------------------*/
/**
  @brief      It writes the checkpoint data in the target file.
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Topo        Topology metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @param      FTI_Data        Dataset metadata.
  @return     integer         FTI_SCES if successful.

  This function checks whether the checkpoint needs to be local or remote,
  opens the target file and writes dataset per dataset, the checkpoint data,
  it finally flushes and closes the checkpoint file.

 **/
/*-------------------------------------------------------------------------*/
int FTI_FinalizeMpiICP(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt,
        FTIT_dataset* FTI_Data)
{
    //Check if all processes have written correctly (every process must succeed)
    int allRes;
    MPI_File pfh = FTI_Exec->iCPInfo.pfh;
    MPI_Allreduce(&(FTI_Exec->iCPInfo.result), &allRes, 1, MPI_INT, MPI_SUM, FTI_COMM_WORLD);
    if (allRes != FTI_SCES) {
        FTI_Print("Not all variables were successfully written!.", FTI_EROR);
        MPI_File_close(&pfh);
        return FTI_NSCS;
    }
    MPI_File_close(&pfh);
    return FTI_SCES;
    
}

#ifdef ENABLE_SIONLIB // --> If SIONlib is installed
/*-------------------------------------------------------------------------*/
/**
  @brief      It writes the checkpoint data in the target file.
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Topo        Topology metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @param      FTI_Data        Dataset metadata.
  @return     integer         FTI_SCES if successful.

  This function checks whether the checkpoint needs to be local or remote,
  opens the target file and writes dataset per dataset, the checkpoint data,
  it finally flushes and closes the checkpoint file.

 **/
/*-------------------------------------------------------------------------*/
int FTI_InitSionlibICP(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt,
        FTIT_dataset* FTI_Data)
{
    int res;
    FTI_Print("I/O mode: SIONlib.", FTI_DBUG);

    int numFiles = 1;
    int nlocaltasks = 1;
    int* file_map = calloc(1, sizeof(int));
    int* ranks = talloc(int, 1);
    int* rank_map = talloc(int, 1);
    sion_int64* chunkSizes = talloc(sion_int64, 1);
    int fsblksize = -1;
    chunkSizes[0] = FTI_Exec->ckptSize;
    ranks[0] = FTI_Topo->splitRank;
    rank_map[0] = FTI_Topo->splitRank;

    // open parallel file
    char fn[FTI_BUFS], str[FTI_BUFS];
    snprintf(str, FTI_BUFS, "Ckpt%d-sionlib.fti", FTI_Exec->ckptID);
    snprintf(fn, FTI_BUFS, "%s/%s", FTI_Conf->gTmpDir, str);
    int *sid = malloc(sizeof(int));
    *sid = sion_paropen_mapped_mpi(fn, "wb,posix", &numFiles, FTI_COMM_WORLD, &nlocaltasks, &ranks, &chunkSizes, &file_map, &rank_map, &fsblksize, NULL);

    
    // check if successful
    if (*sid == -1) {
        errno = 0;
        FTI_Print("SIONlib: File could no be opened", FTI_EROR);

        free(file_map);
        free(rank_map);
        free(ranks);
        free(chunkSizes);
        return FTI_NSCS;
    }
    
    // set file pointer to corresponding block in sionlib file
    res = sion_seek(*sid, FTI_Topo->splitRank, SION_CURRENT_BLK, SION_CURRENT_POS);

    // check if successful
    if (res != SION_SUCCESS) {
        errno = 0;
        FTI_Print("SIONlib: Could not set file pointer", FTI_EROR);
        sion_parclose_mapped_mpi(*sid);
        free(file_map);
        free(rank_map);
        free(ranks);
        free(chunkSizes);
        return FTI_NSCS;
    }
    
    FTI_Exec->iCPInfo.fh = (void*) sid;

    FTI_Exec->iCPInfo.result = FTI_SCES;
    
    //free(file_map);
    //free(rank_map);
    //free(ranks);
    //free(chunkSizes);
    
    return FTI_SCES;

}

/*-------------------------------------------------------------------------*/
/**
  @brief      Writes ckpt to PFS using POSIX.
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Topo        Topology metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @param      FTI_Data        Dataset metadata.
  @return     integer         FTI_SCES if successful.

 **/
/*-------------------------------------------------------------------------*/
int FTI_WriteSionlibVar(int varID, FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt,
        FTIT_dataset* FTI_Data)
{

    int sid = *(int*)(FTI_Exec->iCPInfo.fh);

    DBG_MSG("sid: %d; *(int*)(FTI_Exec->iCPInfo.fh): %d",0, sid, *(int*)(FTI_Exec->iCPInfo.fh));
    // write datasets into file
    int i;
    for (i = 0; i < FTI_Exec->nbVar; i++) {
        
        if( FTI_Data[i].id == varID ) {
            
            // SIONlib write call
            int res = sion_fwrite(FTI_Data[i].ptr, FTI_Data[i].size, 1, sid);

            // check if successful
            if (res < 0) {
                errno = 0;
                FTI_Print("SIONlib: Data could not be written", FTI_EROR);
                res =  sion_parclose_mapped_mpi(sid);
                return FTI_NSCS;
            }
        
        }

    }
    
    DBG_MSG("sid: %d; *(int*)(FTI_Exec->iCPInfo.fh): %d",0, sid, *(int*)(FTI_Exec->iCPInfo.fh));

    return FTI_SCES;

}

/*-------------------------------------------------------------------------*/
/**
  @brief      It writes the checkpoint data in the target file.
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Topo        Topology metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @param      FTI_Data        Dataset metadata.
  @return     integer         FTI_SCES if successful.

  This function checks whether the checkpoint needs to be local or remote,
  opens the target file and writes dataset per dataset, the checkpoint data,
  it finally flushes and closes the checkpoint file.

 **/
/*-------------------------------------------------------------------------*/
int FTI_FinalizeSionlibICP(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt,
        FTIT_dataset* FTI_Data)
{
    
    int allRes;

    MPI_Allreduce(&(FTI_Exec->iCPInfo.result), &allRes, 1, MPI_INT, MPI_SUM, FTI_COMM_WORLD);
    if (allRes != FTI_SCES) {
        FTI_Print("Not all variables were successfully written!.", FTI_EROR);
        return FTI_NSCS;
    }

    int sid = *(int*)(FTI_Exec->iCPInfo.fh);
    
    // close parallel file
    if (sion_parclose_mapped_mpi(sid) == -1) {
        FTI_Print("Cannot close sionlib file.", FTI_WARN);
        return FTI_NSCS;
    }
    
    return FTI_SCES;
    
}
#endif // SIONlib enabled
