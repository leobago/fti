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
 *  @file   mpio.c
 *  @date   May , 2019
 *  @brief  API functions for the FTI library.
 */


#include "../interface.h"


/*-------------------------------------------------------------------------*/
/**
  @brief      Opens and file (Only for write).
  @param      fileDesc        The file descriptor.
  @return     integer         FTI_SCES on success.
 **/
/*-------------------------------------------------------------------------*/
int FTI_MPIOOpen(char *fn, void *fileDesc){
    WriteMPIInfo_t *fd = (WriteMPIInfo_t*) fileDesc;
    int res = FTI_SCES;
    char str[FTI_BUFS], mpi_err[FTI_BUFS];
    MPI_Info_create(&(fd->info));

    if ( fd->flag == 'r' )
        MPI_Info_set(fd->info, "romio_cb_read", "enable");
    else if (fd->flag == 'w' )		
        MPI_Info_set(fd->info, "romio_cb_write", "enable");

    MPI_Info_set(fd->info, "stripping_unit", "4194304");
#ifdef LUSTRE
    if (fd->FTI_Topo->splitRank == 0) {
        res = llapi_file_create(gfn, fd->FTI_Conf->stripeUnit, fd->FTI_Conf->stripeOffset, fd->FTI_Conf->stripeFactor, 0);
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

    if ( fd->flag == 'r' )
        res = MPI_File_open(FTI_COMM_WORLD, fn, MPI_MODE_RDWR, fd->info, &(fd->pfh));
    else if (fd->flag == 'w' )		
        res = MPI_File_open(FTI_COMM_WORLD, fn, MPI_MODE_WRONLY|MPI_MODE_CREATE, fd->info, &(fd->pfh));

    if (res != 0) {
        errno = 0;
        int reslen;
        MPI_Error_string(res, mpi_err, &reslen);
        snprintf(str, FTI_BUFS, "unable to create file [MPI ERROR - %i] %s", res, mpi_err);
        FTI_Print(str, FTI_EROR);
        return FTI_NSCS;
    }
    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Closes the file  
  @param      fileDesc          The fileDescriptor 
  @return     integer         Return FTI_SCES  when successfuly write the data to the file 

 **/
/*-------------------------------------------------------------------------*/
int FTI_MPIOClose(void *fileDesc){
    WriteMPIInfo_t *fd = (WriteMPIInfo_t*) fileDesc;
    MPI_Info_free(&(fd->info));
    MPI_File_close(&(fd->pfh));
    return FTI_SCES;
}


/*-------------------------------------------------------------------------*/
/**
  @brief      Writes to the file  
  @param      src               pointer pointing to the data to be stored
  @param      size              size of the data to be written 
  @param      fileDesc          The fileDescriptor 
  @return     integer         Return FTI_SCES  when successfuly write the data to the file 

 **/
/*-------------------------------------------------------------------------*/
int FTI_MPIOWrite(void *src, size_t size, void *fileDesc)
{
    WriteMPIInfo_t *fd= (WriteMPIInfo_t *)fileDesc;
    size_t pos = 0;
    size_t bSize = fd->FTI_Conf->transferSize;
    while (pos < size) {
        if ((size - pos) < fd->FTI_Conf->transferSize) {
            bSize = size - pos;
        }

        MPI_Datatype dType;
        MPI_Type_contiguous(bSize, MPI_BYTE, &dType);
        MPI_Type_commit(&dType);

        fd->err = MPI_File_write_at(fd->pfh, fd->offset, src, 1, dType, MPI_STATUS_IGNORE);
        // check if successful
        if (fd->err != 0) {
            errno = 0;
            int reslen;
            char str[FTI_BUFS], mpi_err[FTI_BUFS];
            MPI_Error_string(fd->err, mpi_err, &reslen);
            snprintf(str, FTI_BUFS, "unable to create file [MPI ERROR - %i] %s", fd->err, mpi_err);
            FTI_Print(str, FTI_EROR);
            return FTI_NSCS;
        }
        MPI_Type_free(&dType);
        src += bSize;
        fd->offset += bSize;
        pos = pos + bSize;
    }
    return FTI_SCES;
}



/*-------------------------------------------------------------------------*/
/**
  @brief      Returns the file position.
  @param      fileDesc        The file descriptor.
  @return     integer         The position in virtual local file.
 **/
/*-------------------------------------------------------------------------*/
size_t FTI_GetMPIOFilePos(void *fileDesc){
    WriteMPIInfo_t *fd = (WriteMPIInfo_t *)fileDesc;
    return fd->loffset;
}


/*-------------------------------------------------------------------------*/
/**
  @brief      Reads from the file  
  @param      src               pointer pointing to the data to be read 
  @param      size              size of the data to be written 
  @param      fileDesc          The fileDescriptor 
  @return     integer         Return FTI_SCES  when successfuly read the data to the file 

 **/
/*-------------------------------------------------------------------------*/
int FTI_MPIORead(void *dest, size_t size, void *fileDesc){
    WriteMPIInfo_t *fd = (WriteMPIInfo_t *)fileDesc;
    return MPI_File_read_at(fd->pfh, fd->offset, dest, size, MPI_BYTE, MPI_STATUS_IGNORE);
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
void *FTI_InitMPIO(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec, FTIT_topology* FTI_Topo, FTIT_checkpoint *FTI_Ckpt, FTIT_dataset *FTI_Data){
    char gfn[FTI_BUFS], ckptFile[FTI_BUFS];
    int i;

    MPI_Offset offset = 0;
    MPI_Offset chunkSize = FTI_Exec->ckptSize;
    WriteMPIInfo_t *write_info = (WriteMPIInfo_t*) malloc (sizeof(WriteMPIInfo_t));

    write_info->FTI_Conf = FTI_Conf;
    write_info->FTI_Topo= FTI_Topo;
    write_info->loffset = 0;
    write_info->flag = 'w';

    FTI_Print("I/O mode: MPI-IO.", FTI_DBUG);
    snprintf(FTI_Exec->ckptMeta.ckptFile, FTI_BUFS, "Ckpt%d-Rank%d.fti", FTI_Exec->ckptId, FTI_Topo->myRank);
    snprintf(ckptFile, FTI_BUFS, "Ckpt%d-mpiio.fti", FTI_Exec->ckptId);
    snprintf(gfn, FTI_BUFS, "%s/%s", FTI_Conf->gTmpDir, ckptFile);
    FTI_MPIOOpen(gfn, write_info);


    // collect chunksizes of other ranks
    MPI_Offset* chunkSizes = talloc(MPI_Offset, FTI_Topo->nbApprocs * FTI_Topo->nbNodes);
    MPI_Allgather(&chunkSize, 1, MPI_OFFSET, chunkSizes, 1, MPI_OFFSET, FTI_COMM_WORLD);

    // set file offset
    for (i = 0; i < FTI_Topo->splitRank; i++) {
        offset += chunkSizes[i];
    }
    free(chunkSizes);
    write_info->offset = offset;
    return (void *)write_info;
}



/*-------------------------------------------------------------------------*/
/**
  @brief      Writes ckpt to using MPIIO file format.
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Topo        Topology metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @param      FTI_Data        Dataset metadata.
  @return     integer         FTI_SCES if successful.

 **/
/*-------------------------------------------------------------------------*/
int FTI_WriteMPIOData(FTIT_dataset * FTI_DataVar, void *fd){
    WriteMPIInfo_t *write_info = (WriteMPIInfo_t *) fd;

    char str[FTI_BUFS];
    int res;
    if ( !(FTI_DataVar->isDevicePtr) ){
        FTI_Print(str,FTI_INFO);
        res = FTI_MPIOWrite(FTI_DataVar->ptr, FTI_DataVar->size, write_info);
        if (res != FTI_SCES ){
            snprintf(str, FTI_BUFS, "Dataset #%d could not be written.", FTI_DataVar->id);
            FTI_Print(str, FTI_EROR);
            FTI_MPIOClose(write_info);
            return res;
        }
    }
#ifdef GPUSUPPORT
    // dowload data from the GPU if necessary
    // Data are stored in the GPU side.
    else {
        snprintf(str, FTI_BUFS, "Dataset #%d Writing GPU Data.", FTI_DataVar->id);
        FTI_Print(str,FTI_INFO);
        if ((res = FTI_Try(
                        FTI_TransferDeviceMemToFileAsync(FTI_DataVar, FTI_MPIOWrite, write_info),
                        "moving data from GPU to storage")) != FTI_SCES) {
            snprintf(str, FTI_BUFS, "Dataset #%d could not be written.", FTI_DataVar->id);
            FTI_Print(str, FTI_EROR);
            FTI_MPIOClose(write_info);
            return res;
        }
    }
#endif
    write_info->loffset+= FTI_DataVar->size;
    return FTI_SCES;
}
