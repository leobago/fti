#include "../interface.h"


int FTI_SionClose(void *fileDesc){
    WriteSionInfo_t *fd = (WriteSionInfo_t *) fileDesc;
    if (sion_parclose_mapped_mpi(fd->sid) == -1) {
        FTI_Print("Cannot close sionlib file.", FTI_WARN);
        free(fd->file_map);
        free(fd->rank_map);
        free(fd->ranks);
        free(fd->chunkSizes);
        return FTI_NSCS;
    }
    free(fd->file_map);
    free(fd->rank_map);
    free(fd->ranks);
    free(fd->chunkSizes);
    return FTI_SCES;
}


/*-------------------------------------------------------------------------*/
/**
  @brief      Initializes the files for the upcoming checkpoint.  
  @param      FTI_Conf          Configuration of FTI 
  @param      FTI_Exec          Execution environment options 
  @param      FTI_Topo          Topology of nodes
  @param      FTI_Ckpt          Checkpoint configurations
  @param      FTI_Data          Data to be stored
  @return     void*             Return void pointer to file descriptor 

 **/
/*-------------------------------------------------------------------------*/
void *FTI_InitSion(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec, FTIT_topology* FTI_Topo, FTIT_checkpoint *FTI_Ckpt, FTIT_dataset *FTI_Data)
{
    WriteSionInfo_t *write_info = (WriteSionInfo_t *) malloc (sizeof(WriteSionInfo_t));

    write_info->loffset = 0;
    int numFiles = 1;
    int nlocaltasks = 1;
    write_info->file_map = calloc(1, sizeof(int));
    write_info->ranks = talloc(int, 1);
    write_info->rank_map = talloc(int, 1);
    write_info->chunkSizes = talloc(sion_int64, 1);
    int fsblksize = -1;
    write_info->chunkSizes[0] = FTI_Exec->ckptSize;
    write_info->ranks[0] = FTI_Topo->splitRank;
    write_info->rank_map[0] = FTI_Topo->splitRank;
    // open parallel file
    char fn[FTI_BUFS], str[FTI_BUFS];
    snprintf(str, FTI_BUFS, "Ckpt%d-sionlib.fti", FTI_Exec->ckptID);
    snprintf(fn, FTI_BUFS, "%s/%s", FTI_Conf->gTmpDir, str);

    snprintf(FTI_Exec->ckptMeta.ckptFile, FTI_BUFS, "%s",str);

    write_info->sid = sion_paropen_mapped_mpi(fn, "wb,posix", &numFiles, FTI_COMM_WORLD, &nlocaltasks, &write_info->ranks, &write_info->chunkSizes, &write_info->file_map, &write_info->rank_map, &fsblksize, NULL);

    // check if successful
    if (write_info->sid == -1) {
        errno = 0;
        FTI_Print("SIONlib: File could no be opened", FTI_EROR);

        free(write_info->file_map);
        free(write_info->rank_map);
        free(write_info->ranks);
        free(write_info->chunkSizes);
        free(write_info);
        return NULL;
    }

    // set file pointer to corresponding block in sionlib file
    int res = sion_seek(write_info->sid, FTI_Topo->splitRank, SION_CURRENT_BLK, SION_CURRENT_POS);

    // check if successful
    if (res != SION_SUCCESS) {
        errno = 0;
        FTI_Print("SIONlib: Could not set file pointer", FTI_EROR);
        sion_parclose_mapped_mpi(write_info->sid);
        free(write_info->file_map);
        free(write_info->rank_map);
        free(write_info->ranks);
        free(write_info->chunkSizes);
        free(write_info);
        return NULL;
    }
    return write_info;
}

/*-------------------------------------------------------------------------*/
/**
  @brief     Writes data to a file using the SION library
  @param     src    The location of the data to be written 
  @param     size   The number of bytes that I need to write 
  @param     opaque A pointer to the file descriptor  
  @return    integer FTI_SCES if successful.

  Writes the data to a file using the SION library. 

 **/
/*-------------------------------------------------------------------------*/
int FTI_SionWrite (void *src, size_t size, void *opaque)
{
    int *sid= (int *)opaque;
    int res = sion_fwrite(src, size, 1, *sid);
    if (res < 0 ){
        return FTI_NSCS;
    }
    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Writes ckpt to using POSIX format.
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Topo        Topology metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @param      FTI_Data        Dataset metadata.
  @return     integer         FTI_SCES if successful.

 **/
/*-------------------------------------------------------------------------*/

int FTI_WriteSionData(FTIT_dataset *FTI_DataVar, void *fd){
    WriteSionInfo_t *write_info = (WriteSionInfo_t*) fd;
    int res;
    char str[FTI_BUFS];
    FTI_Print("Writing Sion Data",FTI_INFO);
    if ( !FTI_DataVar->isDevicePtr) {
        res = FTI_SionWrite(FTI_DataVar->ptr, FTI_DataVar->size, &write_info->sid);
        if (res != FTI_SCES){
            snprintf(str, FTI_BUFS, "Dataset #%d could not be written.", FTI_DataVar->id);
            FTI_Print(str, FTI_EROR);
            errno = 0;
            FTI_Print("SIONlib: Data could not be written", FTI_EROR);
            res =  sion_parclose_mapped_mpi(write_info->sid);
            free(write_info->file_map);
            free(write_info->rank_map);
            free(write_info->ranks);
            free(write_info->chunkSizes);
            return FTI_NSCS;
        }
    }
#ifdef GPUSUPPORT            
    // if data are stored to the GPU move them from device
    // memory to cpu memory and store them.
    else {
        if ((res = FTI_Try(
                        TransferDeviceMemToFileAsync(&FTI_Data[i], FTI_SionWrite, &write_info->sid),
                        "moving data from GPU to storage")) != FTI_SCES) {
            snprintf(str, FTI_BUFS, "Dataset #%d could not be written.", FTI_DataVar->id);
            FTI_Print(str, FTI_EROR);
            errno = 0;
            FTI_Print("SIONlib: Data could not be written", FTI_EROR);
            res =  sion_parclose_mapped_mpi(write_info->sid);
            free(write_info->file_map);
            free(write_info->rank_map);
            free(write_info->ranks);
            free(write_info->chunkSizes);
            return FTI_NSCS;
        }
    }
#endif            
    write_info->loffset+= FTI_DataVar->size;
    return FTI_SCES;

}


size_t FTI_GetSionFilePos(void *fileDesc){
    WriteSionInfo_t *fd  = (WriteSionInfo_t *) fileDesc;
    return fd->loffset;
}



