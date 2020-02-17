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
 *  @file   posix.c
 *  @date   May, 2019
 *  @brief  Funtions to support posix checkpointing.
 */


#include "../interface.h"

int FTI_ActivateHeadsPosix(FTIT_configuration* FTI_Conf,FTIT_execution* FTI_Exec,FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt, int status)
{
    FTI_Exec->wasLastOffline = 1;
    // Head needs ckpt. ID to determine ckpt file name.
    int value = FTI_BASE + FTI_Exec->ckptLvel; //Token to send to head
    if (status != FTI_SCES) { //If Writing checkpoint failed
        value = FTI_REJW; //Send reject checkpoint token to head
    }
    MPI_Send(&value, 1, MPI_INT, FTI_Topo->headRank, FTI_Conf->ckptTag, FTI_Exec->globalComm);
    int isDCP = (int)FTI_Ckpt[4].isDcp;
    MPI_Send(&isDCP, 1, MPI_INT, FTI_Topo->headRank, FTI_Conf->ckptTag, FTI_Exec->globalComm);
    return FTI_SCES;
}
/*-------------------------------------------------------------------------*/
/**
  @brief      Opens and POSIX file (Only for write).
  @param      fileDesc        The file descriptor.
  @return     integer         FTI_SCES on success.
 **/
/*-------------------------------------------------------------------------*/
int FTI_PosixOpen(char *fn, void *fileDesc)
{
    char str[FTI_BUFS];
    WritePosixInfo_t *fd = (WritePosixInfo_t *) fileDesc;
    if ( fd->flag == 'w' )
        fd->f = fopen(fn,"wb");
    else if ( fd -> flag == 'r')
        fd->f = fopen(fn,"rb");
    else if ( fd -> flag == 'e' )
        fd->f = fopen(fn, "r+" );
    else if ( fd -> flag == 'a' )
        fd->f = fopen(fn, "a" );
    else{
        FTI_Print("Posix Open Should always indicated flag",FTI_WARN);
    }


    if ( fd->f == NULL ){
        snprintf(str, FTI_BUFS, "unable to create file [POSIX ERROR - %d] %s", errno, strerror(errno));
        FTI_Print(str,FTI_EROR);
        return FTI_NSCS;
    }
    MD5_Init(&(fd->integrity));
    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Closes the POSIX file
  @param      fileDesc          The fileDescriptor
  @return     integer         Return FTI_SCES  when successfuly write the data to the file

 **/
/*-------------------------------------------------------------------------*/
int FTI_PosixClose(void *fileDesc)
{
    WritePosixInfo_t *fd = (WritePosixInfo_t *) fileDesc;
    FTI_PosixSync(fileDesc);
    fclose(fd->f);
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
int FTI_PosixWrite(void *src, size_t size, void *fileDesc)
{
    WritePosixInfo_t *fd = (WritePosixInfo_t *)fileDesc;
    size_t written = 0;
    int fwrite_errno = 0;
    char str[FTI_BUFS];

    while (written < size && !ferror(fd->f)) {
        errno = 0;
        written += fwrite(((char *)src) + written, 1, size - written, fd->f);
        fwrite_errno = errno;
    }

    MD5_Update (&(fd->integrity), src, size);
    if (ferror(fd->f)){
        char error_msg[FTI_BUFS];
        error_msg[0] = 0;
        strerror_r(fwrite_errno, error_msg, FTI_BUFS);
        snprintf(str, FTI_BUFS, "Unable to write : [POSIX ERROR - %s.]", error_msg);
        FTI_Print(str, FTI_EROR);
        fclose(fd->f);
        return FTI_NSCS;
    }
    else
        return FTI_SCES;

}


/*-------------------------------------------------------------------------*/
/**
  @brief      Seeks into the file
  @param      pos              The new file posisiton
  @param      fileDesc          The fileDescriptor
  @return     integer         Return FTI_SCES  when successfuly write the data to the file

 **/
/*-------------------------------------------------------------------------*/
int FTI_PosixSeek(size_t pos, void *fileDesc)
{
    WritePosixInfo_t *fd = (WritePosixInfo_t *) fileDesc;
    if ( fseek( fd->f, pos, SEEK_SET ) == -1 ) {
        char error_msg[FTI_BUFS];
        sprintf(error_msg, "Unable to Seek : [POSIX ERROR -%s.]", strerror(errno));
        FTI_Print(error_msg, FTI_EROR );
        return FTI_NSCS;
    }
    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Return the current file postion
  @param      fileDesc          The fileDescriptor
  @return     size_t            Position of the file descriptor

 **/
/*-------------------------------------------------------------------------*/
size_t FTI_GetPosixFilePos(void *fileDesc)
{
    WritePosixInfo_t *fd = (WritePosixInfo_t *) fileDesc;
    return ftell(fd->f);
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
int FTI_PosixRead(void *dest, size_t size, void *fileDesc)
{
    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Synchornizes the current file
  @param      fileDesc          The fileDescriptor
  @return     int               FTI_SCES on success

 **/
/*-------------------------------------------------------------------------*/
int FTI_PosixSync(void *fileDesc)
{
    fsync(fileno(((WritePosixInfo_t *) fileDesc)->f));
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
void* FTI_InitPosix(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec, FTIT_topology* FTI_Topo, FTIT_checkpoint *FTI_Ckpt, FTIT_dataset *FTI_Data)
{

    FTI_Print("I/O mode: Posix.", FTI_DBUG);

    char fn[FTI_BUFS];
    int level = FTI_Exec->ckptLvel;

    WritePosixInfo_t *write_info = (WritePosixInfo_t *) malloc (sizeof(WritePosixInfo_t));

    snprintf(FTI_Exec->meta[0].ckptFile, FTI_BUFS, "Ckpt%d-Rank%d.fti", FTI_Exec->ckptID, FTI_Topo->myRank);

    if (level == 4 && FTI_Ckpt[4].isInline) { //If inline L4 save directly to global directory
        snprintf(fn, FTI_BUFS, "%s/%s", FTI_Conf->gTmpDir, FTI_Exec->meta[0].ckptFile);
    }
    else {
        snprintf(fn, FTI_BUFS, "%s/%s", FTI_Conf->lTmpDir, FTI_Exec->meta[0].ckptFile);
    }

    write_info->flag = 'w';
    write_info->offset = 0;
    FTI_PosixOpen(fn,write_info);
    return write_info;
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
int FTI_WritePosixData(FTIT_dataset * FTI_DataVar, void *fd)
{
    WritePosixInfo_t *write_info = (WritePosixInfo_t*) fd;
    char str[FTI_BUFS];
    int res;

    if ( !(FTI_DataVar->isDevicePtr) ){
        if (( res = FTI_Try(FTI_PosixWrite(FTI_DataVar->ptr, FTI_DataVar->size, write_info),"Storing Data to Checkpoint file")) != FTI_SCES){
            snprintf(str, FTI_BUFS, "Dataset #%d could not be written.", FTI_DataVar->id);
            FTI_Print(str, FTI_EROR);
            FTI_PosixClose(write_info);
            return FTI_NSCS;
        }
    }
#ifdef GPUSUPPORT
    // if data are stored to the GPU move them from device
    // memory to cpu memory and store them.
    else {
        if ((res = FTI_Try(
                        FTI_TransferDeviceMemToFileAsync(FTI_DataVar,  FTI_PosixWrite, write_info),
                        "moving data from GPU to storage")) != FTI_SCES) {
            snprintf(str, FTI_BUFS, "Dataset #%d could not be written.", FTI_DataVar->id);
            FTI_Print(str, FTI_EROR);
            FTI_PosixClose(write_info);
            return FTI_NSCS;
        }
    }
#endif
    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Finalizes the checksum of the file.
  @param      dest            Where to store the checksum.
  @param      md5             Md5 checksum up to now.
  @return     void.

 **/
/*-------------------------------------------------------------------------*/
void FTI_PosixMD5(unsigned char *dest, void *md5)
{
    WritePosixInfo_t *write_info =(WritePosixInfo_t *) md5;
    MD5_Final(dest,&(write_info->integrity));
}
