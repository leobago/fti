#include <fti-int/incremental_checkpoint.h>
#include "interface.h"
#include "utility.h"
/*-------------------------------------------------------------------------*/
/**
  @brief      Initializes iCP for POSIX I/O.
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Topo        Topology metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @param      FTI_Data        Dataset metadata.
  @return     integer         FTI_SCES if successful.

  This function takes care of the I/O specific actions needed before
  protected variables may be added to the checkpoint files.
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

  //update ckpt file name
  snprintf(FTI_Exec->meta[0].ckptFile, FTI_BUFS,
      "Ckpt%d-Rank%d.fti", FTI_Exec->ckptID, FTI_Topo->myRank);

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
  memcpy( FTI_Exec->iCPInfo.fh, &fd, sizeof(FTI_PO_FH) );

  return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Writes dataset into ckpt file using POSIX.
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
  FILE *fd;
  int res;
  memcpy( &fd, FTI_Exec->iCPInfo.fh, sizeof(FTI_PO_FH) );

  char str[FTI_BUFS];

  long offset = 0;

  // write data into ckpt file
  int i;
  for (i = 0; i < FTI_Exec->nbVar; i++) {
    if( FTI_Data[i].id == varID ) {
      clearerr(fd);
      if ( fseek( fd, offset, SEEK_SET ) == -1 ) {
        FTI_Print("Error on fseek in writeposixvar", FTI_EROR );
        return FTI_NSCS;
      }
      if ( !(FTI_Data[i].isDevicePtr) ){
        FTI_Print(str,FTI_INFO);
        res = write_posix(FTI_Data[i].ptr, FTI_Data[i].size, fd);
      }
#ifdef GPUSUPPORT            
      // if data are stored to the GPU move them from device
      // memory to cpu memory and store them.
      else {
        FTI_Print(str,FTI_INFO);
        if ((res = FTI_Try(
                FTI_pipeline_gpu_to_storage(&FTI_Data[i],  FTI_Exec, FTI_Conf, write_posix, fd),
                "moving data from GPU to storage")) != FTI_SCES) {
          snprintf(str, FTI_BUFS, "Dataset #%d could not be written.", FTI_Data[i].id);
          FTI_Print(str, FTI_EROR);
          fclose(fd);
          return FTI_NSCS;
        }
      }
#endif  
      if (ferror(fd)) {
        return FTI_NSCS;
      }
    }
    offset += FTI_Data[i].count*FTI_Data[i].eleSize;
  }

  FTI_Exec->iCPInfo.result = FTI_SCES;

  return FTI_SCES;

}

/*-------------------------------------------------------------------------*/
/**
  @brief      Finalizes iCP for POSIX I/O.
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Topo        Topology metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @param      FTI_Data        Dataset metadata.
  @return     integer         FTI_SCES if successful.

  This function takes care of the I/O specific actions needed to
  finalize iCP.
 **/
/*-------------------------------------------------------------------------*/
int FTI_FinalizePosixICP(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
    FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt,
    FTIT_dataset* FTI_Data)
{
  if ( FTI_Exec->iCPInfo.status == FTI_ICP_FAIL ) {
    return FTI_NSCS;
  }

  FILE *fd;
  memcpy( &fd, FTI_Exec->iCPInfo.fh, sizeof(FTI_PO_FH) );

  // close file
  if (fclose(fd) != 0) {
    FTI_Print("FTI checkpoint file could not be closed.", FTI_EROR);

    return FTI_NSCS;
  }

  return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Initializes iCP for MPI I/O.
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Topo        Topology metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @param      FTI_Data        Dataset metadata.
  @return     integer         FTI_SCES if successful.

  This function takes care of the I/O specific actions needed before
  protected variables may be added to the checkpoint files.
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

  /* 
   * update ckpt file name (neccessary for the restart!)
   * not very nice TODO we should think about another mechanism
   */
  snprintf(FTI_Exec->meta[0].ckptFile, FTI_BUFS,
      "Ckpt%d-Rank%d.fti", FTI_Exec->ckptID, FTI_Topo->myRank);

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

  memcpy( FTI_Exec->iCPInfo.fh, &pfh, sizeof(FTI_MI_FH) );
  MPI_Info_free(&info);

  return FTI_SCES;

}

/*-------------------------------------------------------------------------*/
/**
  @brief      Writes dataset into ckpt file using MPI-IO.
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
  char str[FTI_BUFS];
  WriteMPIInfo_t write_info;
  int res;
  memcpy( &write_info.pfh, FTI_Exec->iCPInfo.fh, sizeof(FTI_MI_FH) );

  write_info.offset = FTI_Exec->iCPInfo.offset; 
  write_info.FTI_Conf = FTI_Conf;

  int i;
  for (i = 0; i < FTI_Exec->nbVar; i++) {
    if ( FTI_Data[i].id == varID ) {
      if ( !(FTI_Data[i].isDevicePtr) ){
        FTI_Print(str,FTI_INFO);
        res = write_mpi(FTI_Data[i].ptr, FTI_Data[i].size, &write_info);
      }
#ifdef GPUSUPPORT
      // dowload data from the GPU if necessary
      // Data are stored in the GPU side.
      else {
        snprintf(str, FTI_BUFS, "Dataset #%d Writing GPU Data.", FTI_Data[i].id);
        FTI_Print(str,FTI_INFO);
        if ((res = FTI_Try(
                FTI_pipeline_gpu_to_storage(&FTI_Data[i],  FTI_Exec, FTI_Conf, write_mpi, &write_info),
                "moving data from GPU to storage")) != FTI_SCES) {
          snprintf(str, FTI_BUFS, "Dataset #%d could not be written.", FTI_Data[i].id);
          FTI_Print(str, FTI_EROR);
          MPI_File_close(&write_info.pfh);
          return res;
        }
      }
#endif
    }
    write_info.offset += FTI_Data[i].size;
  }

  FTI_Exec->iCPInfo.result = FTI_SCES;

  return FTI_SCES;

}

/*-------------------------------------------------------------------------*/
/**
  @brief      Finalizes iCP for MPI I/O.
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Topo        Topology metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @param      FTI_Data        Dataset metadata.
  @return     integer         FTI_SCES if successful.

  This function takes care of the I/O specific actions needed to
  finalize iCP.
 **/
/*-------------------------------------------------------------------------*/
int FTI_FinalizeMpiICP(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
    FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt,
    FTIT_dataset* FTI_Data)
{
  if ( FTI_Exec->iCPInfo.status == FTI_ICP_FAIL ) {
    return FTI_NSCS;
  }

  MPI_File pfh;
  memcpy( &pfh, FTI_Exec->iCPInfo.fh, sizeof(FTI_MI_FH) );

  MPI_File_close(&pfh);
  return FTI_SCES;

}

/*-------------------------------------------------------------------------*/
/**
  @brief      Initializes iCP for FTI-FF I/O.
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Topo        Topology metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @param      FTI_Data        Dataset metadata.
  @return     integer         FTI_SCES if successful.

  This function takes care of the I/O specific actions needed before
  protected variables may be added to the checkpoint files.
 **/
/*-------------------------------------------------------------------------*/
int FTI_InitFtiffICP(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
    FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt,
    FTIT_dataset* FTI_Data)
{
  FTIFF_UpdateDatastructFTIFF( FTI_Exec, FTI_Data, FTI_Conf );

  //FOR DEVELOPING 
  //FTIFF_PrintDataStructure( 0, FTI_Exec, FTI_Data );

  char fn[FTI_BUFS], strerr[FTI_BUFS];

  FTI_Print("I/O mode: FTI File Format.", FTI_DBUG);

  //update ckpt file name
  snprintf(FTI_Exec->meta[0].ckptFile, FTI_BUFS,
      "Ckpt%d-Rank%d.fti", FTI_Exec->ckptID, FTI_Topo->myRank);

  // check if metadata exists
  if( FTI_Exec->firstdb == NULL ) {
    FTI_Print("No data structure found to write data to file. Discarding checkpoint.", FTI_WARN);
    return FTI_NSCS;
  }

  // buffer for de-/serialization of meta data
  char* buffer_ser;

  //If inline L4 save directly to global directory
  int level = FTI_Exec->ckptLvel;
  if (level == 4 && FTI_Ckpt[4].isInline) { 
    if( FTI_Conf->dcpEnabled && FTI_Ckpt[4].isDcp ) {
      snprintf(fn, FTI_BUFS, "%s/%s", FTI_Ckpt[4].dcpDir, FTI_Ckpt[4].dcpName);
    } else {
      snprintf(fn, FTI_BUFS, "%s/%s", FTI_Conf->gTmpDir, FTI_Exec->meta[0].ckptFile);
    }
  } else if ( level == 4 && !FTI_Ckpt[4].isInline )
    if( FTI_Conf->dcpEnabled && FTI_Ckpt[4].isDcp ) {
      snprintf(fn, FTI_BUFS, "%s/%s", FTI_Ckpt[1].dcpDir, FTI_Ckpt[4].dcpName);
    } else {
      snprintf(fn, FTI_BUFS, "%s/%s", FTI_Conf->lTmpDir, FTI_Exec->meta[0].ckptFile);
    }
    else {
      snprintf(fn, FTI_BUFS, "%s/%s", FTI_Conf->lTmpDir, FTI_Exec->meta[0].ckptFile);
    }

  int fd;

  // for dCP: create if not exists, open if exists
  if ( FTI_Conf->dcpEnabled && FTI_Ckpt[4].isDcp ) {
    if (access(fn,R_OK) != 0) {
      fd = open( fn, O_WRONLY|O_CREAT, (mode_t) 0600 ); 
    } 
    else {
      fd = open( fn, O_WRONLY );
    }
  } else {
    fd = open( fn, O_WRONLY|O_CREAT, (mode_t) 0600 ); 
  }

  if (fd == -1) {
    snprintf(strerr, FTI_BUFS, "FTI checkpoint file (%s) could not be opened.", fn);
    FTI_Print(strerr, FTI_EROR);
    return FTI_NSCS;
  }

  // make sure that is never a null ptr. otherwise its to fix.
  assert(FTI_Exec->firstdb);
  FTIFF_db *currentdb = FTI_Exec->firstdb;
  FTIFF_dbvar *currentdbvar = NULL;
  int dbvar_idx, dbcounter=0;
  long mdoffset;
  long endoffile = FTI_filemetastructsize;

  // MD5 context for file (only data) checksum
  MD5_CTX mdContext;
  MD5_Init(&mdContext);

  int isnextdb;

  long dataSize = 0;

  // write FTI-FF meta data
  do {    

    isnextdb = 0;

    mdoffset = endoffile;

    endoffile += currentdb->dbsize;

    if (currentdb->update) {

      // serialize block meta data and write to file
      buffer_ser = (char*) malloc ( FTI_dbstructsize );
      if( buffer_ser == NULL ) {
        snprintf( strerr, FTI_BUFS, "FTI-FF: WriteFTIFF - failed to allocate %d bytes for 'buffer_ser'", FTI_dbstructsize );
        FTI_Print(strerr, FTI_EROR);
        close(fd);
        errno = 0;
        return FTI_NSCS;
      }
      if( FTIFF_SerializeDbMeta( currentdb, buffer_ser ) != FTI_SCES ) {
        FTI_Print("FTI-FF: WriteFTIFF - failed to serialize 'currentdb'", FTI_EROR);
        close(fd);
        errno = 0;
        return FTI_NSCS;
      }
      if ( lseek( fd, mdoffset, SEEK_SET ) == -1 ) {
        snprintf(strerr, FTI_BUFS, "FTI-FF: WriteFTIFF - could not seek in file: %s", fn);
        FTI_Print(strerr, FTI_EROR);
        close(fd);
        errno = 0;
        return FTI_NSCS;
      }
      write( fd, buffer_ser, FTI_dbstructsize );
      if ( fd == -1 ) {
        snprintf(strerr, FTI_BUFS, "FTI-FF: WriteFTIFF - could not write metadata in file: %s", fn);
        FTI_Print(strerr, FTI_EROR);
        errno=0;
        close(fd);
        return FTI_NSCS;
      }
      free( buffer_ser );
    }

    // advance meta data offset
    mdoffset += FTI_dbstructsize;

    for(dbvar_idx=0;dbvar_idx<currentdb->numvars;dbvar_idx++) {

      currentdbvar = &(currentdb->dbvars[dbvar_idx]);
      bool hascontent = currentdbvar->hascontent;
      FTI_ADDRVAL cbasePtr = (FTI_ADDRVAL)(FTI_Data[currentdbvar->idx].ptr) + currentdb->dbvars[dbvar_idx].dptr;
      errno = 0;

      unsigned char hashchk[MD5_DIGEST_LENGTH];
      // create datachunk hash
      if(hascontent) {
#ifndef HAVE_OPENSSL
        MD5_CTX roundContext;
        MD5_Init( &roundContext);
#endif

        dataSize += currentdbvar->chunksize;
		MD5_Update( &mdContext, (FTI_ADDRPTR) cbasePtr, currentdbvar->chunksize );

#ifndef HAVE_OPENSSL
        MD5_Update( &roundContext, (FTI_ADDRPTR) cbasePtr, currentdbvar->chunksize );
        MD5_Final( hashchk, &roundContext);
#else
        MD5( (FTI_ADDRPTR) cbasePtr, currentdbvar->chunksize, hashchk );
#endif
      }

      bool contentUpdate = 
        (memcmp(currentdbvar->hash, hashchk, MD5_DIGEST_LENGTH) == 0) ? 0 : 1;

      memcpy( currentdbvar->hash, hashchk, MD5_DIGEST_LENGTH );

      if ( currentdbvar->update || contentUpdate ) {

        // serialize data block variable meta data and write to file
        buffer_ser = (char*) malloc ( FTI_dbvarstructsize );
        if( buffer_ser == NULL ) {
          snprintf( strerr, FTI_BUFS, "FTI-FF: WriteFTIFF - failed to allocate %d bytes for 'buffer_ser'", FTI_dbvarstructsize );
          FTI_Print(strerr, FTI_EROR);
          close(fd);
          errno = 0;
          return FTI_NSCS;
        }
        if( FTIFF_SerializeDbVarMeta( currentdbvar, buffer_ser ) != FTI_SCES ) {
          FTI_Print("FTI-FF: WriteFTIFF - failed to serialize 'currentdbvar'", FTI_EROR);
          close(fd);
          errno = 0;
          return FTI_NSCS;
        }
        if ( lseek( fd, mdoffset, SEEK_SET ) == -1 ) {
          snprintf(strerr, FTI_BUFS, "FTI-FF: WriteFTIFF - could not seek in file: %s", fn);
          FTI_Print(strerr, FTI_EROR);
          close(fd);
          errno = 0;
          return FTI_NSCS;
        }
        write( fd, buffer_ser, FTI_dbvarstructsize );
        if ( fd == -1 ) {
          snprintf(strerr, FTI_BUFS, "FTI-FF: WriteFTIFF - could not write metadata in file: %s", fn);
          FTI_Print(strerr, FTI_EROR);
          errno=0;
          close(fd);
          return FTI_NSCS;
        }
        free( buffer_ser );

      }

      // advance meta data offset
      mdoffset += FTI_dbvarstructsize;

    }

    if (currentdb->next) {
      currentdb = currentdb->next;
      isnextdb = 1;
    }

    dbcounter++;

  } while( isnextdb );

  // create string of filehash and create other file meta data
  unsigned char fhash[MD5_DIGEST_LENGTH];
  MD5_Final( fhash, &mdContext );

  int ii = 0, i;
  for(i = 0; i < MD5_DIGEST_LENGTH; i++) {
    sprintf(&(FTI_Exec->FTIFFMeta.checksum[ii]), "%02x", fhash[i]);
    ii += 2;
  }

  // has to be assigned before FTIFF_CreateMetaData call!
  FTI_Exec->ckptSize = endoffile;

  if ( FTI_Try( FTIFF_CreateMetadata( FTI_Exec, FTI_Topo, FTI_Data, FTI_Conf ), "Create FTI-FF meta data" ) != FTI_SCES ) {
    return FTI_NSCS;
  }

  // serialize file meta data and write to file
  buffer_ser = (char*) malloc ( FTI_filemetastructsize );
  if( buffer_ser == NULL ) {
    snprintf( strerr, FTI_BUFS, "FTI-FF: WriteFTIFF - failed to allocate %d bytes for 'buffer_ser'", FTI_filemetastructsize );
    FTI_Print(strerr, FTI_EROR);
    close(fd);
    errno = 0;
    return FTI_NSCS;
  }
  if( FTIFF_SerializeFileMeta( &(FTI_Exec->FTIFFMeta), buffer_ser ) != FTI_SCES ) {
    FTI_Print("FTI-FF: WriteFTIFF - failed to serialize 'FTI_Exec->FTIFFMeta'", FTI_EROR);
    close(fd);
    errno = 0;
    return FTI_NSCS;
  }
  if ( lseek( fd, 0, SEEK_SET ) == -1 ) {
    snprintf(strerr, FTI_BUFS, "FTI-FF: WriteFTIFF - could not seek in file: %s", fn);
    FTI_Print(strerr, FTI_EROR);
    close(fd);
    errno = 0;
    return FTI_NSCS;
  }
  write( fd, buffer_ser, FTI_filemetastructsize );
  if ( fd == -1 ) {
    snprintf(strerr, FTI_BUFS, "FTI-FF: WriteFTIFF - could not write file metadata in file: %s", fn);
    FTI_Print(strerr, FTI_EROR);
    errno=0;
    close(fd);
    return FTI_NSCS;
  }
  free( buffer_ser );

  FTI_Exec->FTIFFMeta.dataSize = dataSize;
  FTI_Exec->FTIFFMeta.dcpSize = 0;

  memcpy( FTI_Exec->iCPInfo.fh, &fd, sizeof(FTI_FF_FH) );

  return FTI_SCES;

}

/*-------------------------------------------------------------------------*/
/**
  @brief      Writes dataset into ckpt file using FTI-FF.
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Topo        Topology metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @param      FTI_Data        Dataset metadata.
  @return     integer         FTI_SCES if successful.
 **/
/*-------------------------------------------------------------------------*/
int FTI_WriteFtiffVar(int varID, FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
    FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt,
    FTIT_dataset* FTI_Data)
{
  char str[FTI_BUFS], strerr[FTI_BUFS];

  FTIFF_db *currentdb = FTI_Exec->firstdb;
  FTIFF_dbvar *currentdbvar = NULL;
  char *dptr;
  uintptr_t fptr;
  int dbvar_idx, dbcounter=0;
  int isnextdb;
  // block size for fwrite buffer in file.
  long membs = 1024*1024*16; // 16 MB
  long cpybuf, cpynow, cpycnt;//, fptr;
  long dcpSize = 0;

  int fd;
  memcpy( &fd, FTI_Exec->iCPInfo.fh, sizeof(FTI_FF_FH) );


  // reset db pointer
  currentdb = FTI_Exec->firstdb;

  do {    

    isnextdb = 0;

    for(dbvar_idx=0;dbvar_idx<currentdb->numvars;dbvar_idx++) {

      currentdbvar = &(currentdb->dbvars[dbvar_idx]);

      if ( currentdbvar->id == varID ) {

        // get source and destination pointer
        dptr = (char*)(FTI_Data[currentdbvar->idx].ptr) + currentdb->dbvars[dbvar_idx].dptr;
        fptr = currentdbvar->fptr;
        uintptr_t chunk_addr, chunk_size, chunk_offset;

        int chunkid = 0;

        while( FTI_ReceiveDataChunk(&chunk_addr, &chunk_size, currentdbvar, FTI_Data) ) {
          chunk_offset = chunk_addr - ((FTI_ADDRVAL)(FTI_Data[currentdbvar->idx].ptr) + currentdbvar->dptr);

          dptr += chunk_offset;
          fptr = currentdbvar->fptr + chunk_offset;

          if ( lseek( fd, fptr, SEEK_SET ) == -1 ) {
            snprintf(strerr, FTI_BUFS, "FTI-FF: WriteFTIFF - could not seek in file");
            FTI_Print(strerr, FTI_EROR);
            errno=0;
            close(fd);
            return FTI_NSCS;
          }

          cpycnt = 0;
          while ( cpycnt < chunk_size ) {
            cpybuf = chunk_size - cpycnt;
            cpynow = ( cpybuf > membs ) ? membs : cpybuf;

            long WRITTEN = 0;

            int try = 0; 
            do {
              WRITTEN += write( fd, (FTI_ADDRPTR) (chunk_addr+cpycnt), cpynow );
              if ( fd == -1 ) {
                snprintf(str, FTI_BUFS, "FTI-FF: WriteFTIFF - Dataset #%d could not be written to file", currentdbvar->id);
                FTI_Print(str, FTI_EROR);
                close(fd);
                errno = 0;
                return FTI_NSCS;
              }
              try++;
            } while ((WRITTEN < cpynow) && (try < 10));

            assert( WRITTEN == cpynow );

            cpycnt += WRITTEN;
            dcpSize += WRITTEN;
          }
          assert(cpycnt == chunk_size);

          chunkid++;

        }

        // debug information
        snprintf(str, FTI_BUFS, "FTIFF: CKPT(id:%i) dataBlock:%i/dataBlockVar%i id: %i, idx: %i"
            ", dptr: %ld, fptr: %ld, chunksize: %ld, "
            "base_ptr: 0x%" PRIxPTR " ptr_pos: 0x%" PRIxPTR " ", 
            FTI_Exec->ckptID, dbcounter, dbvar_idx,  
            currentdbvar->id, currentdbvar->idx, currentdbvar->dptr,
            currentdbvar->fptr, currentdbvar->chunksize,
            (uintptr_t)FTI_Data[currentdbvar->idx].ptr, (uintptr_t)dptr);
        FTI_Print(str, FTI_DBUG);

      }

    }

    if (currentdb->next) {
      currentdb = currentdb->next;
      isnextdb = 1;
    }

    dbcounter++;

  } while( isnextdb );

  // only for printout of dCP share in FTI_Checkpoint
  FTI_Exec->FTIFFMeta.dcpSize += dcpSize;

  FTI_Exec->iCPInfo.result = FTI_SCES;

  return FTI_SCES;

}

/*-------------------------------------------------------------------------*/
/**
  @brief      Finalizes iCP for FTI-FF I/O.
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Topo        Topology metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @param      FTI_Data        Dataset metadata.
  @return     integer         FTI_SCES if successful.

  This function takes care of the I/O specific actions needed to
  finalize iCP.
 **/
/*-------------------------------------------------------------------------*/
int FTI_FinalizeFtiffICP(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
    FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt,
    FTIT_dataset* FTI_Data)
{   
  if ( FTI_Exec->iCPInfo.status == FTI_ICP_FAIL ) {
    return FTI_NSCS;
  }

  int fd; 
  memcpy( &fd, FTI_Exec->iCPInfo.fh, sizeof(FTI_FF_FH) );

  fdatasync( fd );
  close( fd );

  return FTI_SCES;

}

#ifdef ENABLE_HDF5
/*-------------------------------------------------------------------------*/
/**
  @brief      Initializes iCP for HDF5 I/O.
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Topo        Topology metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @param      FTI_Data        Dataset metadata.
  @return     integer         FTI_SCES if successful.

  This function takes care of the I/O specific actions needed before
  protected variables may be added to the checkpoint files.
 **/
/*-------------------------------------------------------------------------*/
int FTI_InitHdf5ICP(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
    FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt,
    FTIT_dataset* FTI_Data)
{
  FTI_Print("I/O mode: HDF5.", FTI_DBUG);
  char str[FTI_BUFS], fn[FTI_BUFS];

  if (FTI_Conf->ioMode == FTI_IO_HDF5) {
    snprintf(FTI_Exec->meta[0].ckptFile, FTI_BUFS,
        "Ckpt%d-Rank%d.h5", FTI_Exec->ckptID, FTI_Topo->myRank);
  }

  int level = FTI_Exec->ckptLvel;
  if (level == 4 && FTI_Ckpt[4].isInline) { //If inline L4 save directly to global directory
    snprintf(fn, FTI_BUFS, "%s/%s", FTI_Conf->gTmpDir, FTI_Exec->meta[0].ckptFile);
  }
  else {
    snprintf(fn, FTI_BUFS, "%s/%s", FTI_Conf->lTmpDir, FTI_Exec->meta[0].ckptFile);
  }

  //Creating new hdf5 file
  hid_t file_id = H5Fcreate(fn, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  if (file_id < 0) {
    sprintf(str, "FTI checkpoint file (%s) could not be opened.", fn);
    FTI_Print(str, FTI_EROR);

    return FTI_NSCS;
  }
  FTI_Exec->H5groups[0]->h5groupID = file_id;
  FTIT_H5Group* rootGroup = FTI_Exec->H5groups[0];

  int i;
  for (i = 0; i < rootGroup->childrenNo; i++) {
    FTI_CreateGroup(FTI_Exec->H5groups[rootGroup->childrenID[i]], file_id, FTI_Exec->H5groups);
  }

  memcpy( FTI_Exec->iCPInfo.fh, &file_id, sizeof(FTI_H5_FH) );

  return FTI_SCES;

}

/*-------------------------------------------------------------------------*/
/**
  @brief      Writes dataset into ckpt file using HDF5.
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Topo        Topology metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @param      FTI_Data        Dataset metadata.
  @return     integer         FTI_SCES if successful.
 **/
/*-------------------------------------------------------------------------*/
int FTI_WriteHdf5Var(int varID, FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
    FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt,
    FTIT_dataset* FTI_Data)
{

  if ( FTI_Exec->iCPInfo.status == FTI_ICP_FAIL ) {
    return FTI_NSCS;
  }

  char str[FTI_BUFS];

  hid_t file_id;
  memcpy( &file_id, FTI_Exec->iCPInfo.fh, sizeof(FTI_H5_FH) );

  FTIT_H5Group* rootGroup = FTI_Exec->H5groups[0];

  // write data into ckpt file
  int i;
  for (i = 0; i < FTI_Exec->nbVar; i++) {
    if( FTI_Data[i].id == varID ) {
      int toCommit = 0;
      if (FTI_Data[i].type->h5datatype < 0) {
        toCommit = 1;
      }
      sprintf(str, "Calling CreateComplexType [%d] with hid_t %ld", FTI_Data[i].type->id, FTI_Data[i].type->h5datatype);
      FTI_Print(str, FTI_DBUG);
      FTI_CreateComplexType(FTI_Data[i].type, FTI_Exec->FTI_Type);
      if (toCommit == 1) {
        char name[FTI_BUFS];
        if (FTI_Data[i].type->structure == NULL) {
          //this is the array of bytes with no name
          sprintf(name, "Type%d", FTI_Data[i].type->id);
        } else {
          strncpy(name, FTI_Data[i].type->structure->name, FTI_BUFS);
        }
        herr_t res = H5Tcommit(FTI_Data[i].type->h5group->h5groupID, name, FTI_Data[i].type->h5datatype, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (res < 0) {
          sprintf(str, "Datatype #%d could not be commited", FTI_Data[i].id);
          FTI_Print(str, FTI_EROR);
          int j;
          for (j = 0; j < FTI_Exec->H5groups[0]->childrenNo; j++) {
            FTI_CloseGroup(FTI_Exec->H5groups[rootGroup->childrenID[j]], FTI_Exec->H5groups);
          }
          H5Fclose(file_id);
          return FTI_NSCS;
        }
      }
      //convert dimLength array to hsize_t
      int j;
      hsize_t dimLength[32];
      for (j = 0; j < FTI_Data[i].rank; j++) {
        dimLength[j] = FTI_Data[i].dimLength[j];
      }
      herr_t res = H5LTmake_dataset(FTI_Data[i].h5group->h5groupID, FTI_Data[i].name, FTI_Data[i].rank, dimLength, FTI_Data[i].type->h5datatype, FTI_Data[i].ptr);
      if (res < 0) {
        sprintf(str, "Dataset #%d could not be written", FTI_Data[i].id);
        FTI_Print(str, FTI_EROR);
        int j;
        for (j = 0; j < FTI_Exec->H5groups[0]->childrenNo; j++) {
          FTI_CloseGroup(FTI_Exec->H5groups[rootGroup->childrenID[j]], FTI_Exec->H5groups);
        }
        H5Fclose(file_id);
        return FTI_NSCS;
      }
    }
  }

  FTI_Exec->iCPInfo.result = FTI_SCES;
  return FTI_SCES;

}

/*-------------------------------------------------------------------------*/
/**
  @brief      Finalizes iCP for HDF5 I/O.
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Topo        Topology metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @param      FTI_Data        Dataset metadata.
  @return     integer         FTI_SCES if successful.

  This function takes care of the I/O specific actions needed to
  finalize iCP.
 **/
/*-------------------------------------------------------------------------*/
int FTI_FinalizeHdf5ICP(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
    FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt,
    FTIT_dataset* FTI_Data)
{

  hid_t file_id;
  memcpy( &file_id, FTI_Exec->iCPInfo.fh, sizeof(FTI_H5_FH) );

  FTIT_H5Group* rootGroup = FTI_Exec->H5groups[0];

  int i;
  for (i = 0; i < FTI_Exec->nbVar; i++) {
    FTI_CloseComplexType(FTI_Data[i].type, FTI_Exec->FTI_Type);
  }

  int j;
  for (j = 0; j < FTI_Exec->H5groups[0]->childrenNo; j++) {
    FTI_CloseGroup(FTI_Exec->H5groups[rootGroup->childrenID[j]], FTI_Exec->H5groups);
  }

  // close file
  FTI_Exec->H5groups[0]->h5groupID = -1;
  if (H5Fclose(file_id) < 0) {
    FTI_Print("FTI checkpoint file could not be closed.", FTI_EROR);
    return FTI_NSCS;
  }


  return FTI_SCES;

}
#endif

/* 
 * As long SIONlib does not support seek in a single file
 * FTI does not support SIONlib I/O for incremental
 * checkpointing
 */
#if 0
#ifdef ENABLE_SIONLIB // --> If SIONlib is installed
/*-------------------------------------------------------------------------*/
/**
  @brief      Initializes iCP for SIONlib I/O.
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Topo        Topology metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @param      FTI_Data        Dataset metadata.
  @return     integer         FTI_SCES if successful.

  This function takes care of the I/O specific actions needed before
  protected variables may be added to the checkpoint files.
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
  int sid = sion_paropen_mapped_mpi(fn, "wb,posix", &numFiles, FTI_COMM_WORLD, &nlocaltasks, &ranks, &chunkSizes, &file_map, &rank_map, &fsblksize, NULL);


  // check if successful
  if (sid == -1) {
    errno = 0;
    FTI_Print("SIONlib: File could no be opened", FTI_EROR);

    free(file_map);
    free(rank_map);
    free(ranks);
    free(chunkSizes);
    return FTI_NSCS;
  }

  memcpy(FTI_Exec->iCPInfo.fh, &sid, sizeof(int));

  free(file_map);
  free(rank_map);
  free(ranks);
  free(chunkSizes);

  return FTI_SCES;

}

/*-------------------------------------------------------------------------*/
/**
  @brief      Writes dataset into ckpt file using SIONlib.
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

  int sid;
  memcpy( &sid, FTI_Exec->iCPInfo.fh, sizeof(FTI_SL_FH) );

  unsigned long offset = 0;
  // write datasets into file
  int i;
  for (i = 0; i < FTI_Exec->nbVar; i++) {

    if( FTI_Data[i].id == varID ) {

      // set file pointer to corresponding block in sionlib file
      int res = sion_seek(sid, FTI_Topo->splitRank, SION_CURRENT_BLK, offset);

      // check if successful
      if (res != SION_SUCCESS) {
        errno = 0;
        FTI_Print("SIONlib: Could not set file pointer", FTI_EROR);
        sion_parclose_mapped_mpi(sid);
        return FTI_NSCS;
      }

      // SIONlib write call
      res = sion_fwrite(FTI_Data[i].ptr, FTI_Data[i].size, 1, sid);

      // check if successful
      if (res < 0) {
        errno = 0;
        FTI_Print("SIONlib: Data could not be written", FTI_EROR);
        res =  sion_parclose_mapped_mpi(sid);
        return FTI_NSCS;
      }

    }

    offset += FTI_Data[i].size;

  }

  FTI_Exec->iCPInfo.result = FTI_SCES;
  return FTI_SCES;

}

/*-------------------------------------------------------------------------*/
/**
  @brief      Finalizes iCP for SIONlib I/O.
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Topo        Topology metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @param      FTI_Data        Dataset metadata.
  @return     integer         FTI_SCES if successful.

  This function takes care of the I/O specific actions needed to
  finalize iCP.
 **/
/*-------------------------------------------------------------------------*/
int FTI_FinalizeSionlibICP(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
    FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt,
    FTIT_dataset* FTI_Data)
{

  int sid;
  memcpy( &sid, FTI_Exec->iCPInfo.fh, sizeof(FTI_SL_FH) );

  // close parallel file
  if (sion_parclose_mapped_mpi(sid) == -1) {
    FTI_Print("Cannot close sionlib file.", FTI_WARN);
    return FTI_NSCS;
  }

  return FTI_SCES;

}
#endif // SIONlib enabled
#endif
