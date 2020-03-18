/**
 *  Copyright (c) 2017 Leonardo A. Bautista-Gomez
 *  All rights reserved
 *
 *  Copyright (c) 2020
 *  DataDirect Networks
 *
 *  See the file COPYRIGHT in the package base directory for details
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
 *  @file   ime.c
 *  @date   May, 2019
 *  @brief  Funtions to support IME checkpointing.
 */


#include "../interface.h"

/*-------------------------------------------------------------------------*/
/**
  @brief      Opens a file using IME native interface (Only for write).
  @param      fileDesc        The file descriptor.
  @return     integer         FTI_SCES on success.
 **/
/*-------------------------------------------------------------------------*/
int FTI_IMEOpen(char *fn, void *fileDesc)
{
     
    char str[FTI_BUFS];
    WriteIMEInfo_t *fd = (WriteIMEInfo_t *) fileDesc;
    if ( fd->flag == O_WRONLY )
        fd->f = ime_native_open(fn, O_CREAT | O_TRUNC | O_WRONLY, 0664);
    else if ( fd -> flag == O_RDONLY )
        fd->f = ime_native_open(fn,O_RDONLY, 0664);
    else if ( fd -> flag == O_RDWR )
        fd->f = ime_native_open(fn, O_RDWR, 0664);
    else if ( fd -> flag == O_APPEND )
        fd->f = ime_native_open(fn, O_APPEND, 0664);
    else{
        FTI_Print("IME native Open Should always indicated flag",FTI_WARN);
    }


    if ( fd->f == -1 ){
        snprintf(str, FTI_BUFS, "unable to create file [IME native ERROR - %d] %s", errno, strerror(errno));
        FTI_Print(str,FTI_EROR);
        return FTI_NSCS;
    }
    MD5_Init(&(fd->integrity));
    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Closes the IME native file
  @param      fileDesc        The fileDescriptor
  @return     integer         Return FTI_SCES  when successfuly write the data to the file

 **/
/*-------------------------------------------------------------------------*/
int FTI_IMEClose(void *fileDesc)
{
    WriteIMEInfo_t *fd = (WriteIMEInfo_t *) fileDesc;
    FTI_IMESync(fileDesc);
    ime_native_close(fd->f);
    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Writes to the file
  @param      src             pointer pointing to the data to be stored
  @param      size            size of the data to be written
  @param      fileDesc        The fileDescriptor
  @return     integer         Return FTI_SCES  when successfuly write the data to the file

 **/
/*-------------------------------------------------------------------------*/
int FTI_IMEWrite(void *src, size_t size, void *fileDesc)
{
    WriteIMEInfo_t *fd = (WriteIMEInfo_t *)fileDesc;
    size_t written = 0;
    int fwrite_errno = 0;
    char str[FTI_BUFS];

    while (written < size) { // KC FIXME && !ferror(fd->f)) {
        errno = 0;
        written += ime_native_write(fd->f, ((char *)src) + written, size - written);
        fwrite_errno = errno;
    }

    MD5_Update (&(fd->integrity), src, size);
/* KC FIXME *
    if (ferror(fd->f)){
        char error_msg[FTI_BUFS];
        error_msg[0] = 0;
        strerror_r(fwrite_errno, error_msg, FTI_BUFS);
        snprintf(str, FTI_BUFS, "Unable to write : [IME native ERROR - %s.]", error_msg);
        FTI_Print(str, FTI_EROR);
        ime_native_close(fd->f);
        return FTI_NSCS;
    }
    else
*/
        return FTI_SCES;

}

/*-------------------------------------------------------------------------*/
/**
  @brief      Seeks into the file
  @param      pos             The new file posisiton
  @param      fileDesc        The fileDescriptor
  @return     integer         Return FTI_SCES  when successfuly write the data to the file

 **/
/*-------------------------------------------------------------------------*/
int FTI_IMESeek(size_t pos, void *fileDesc)
{
    WriteIMEInfo_t *fd = (WriteIMEInfo_t *) fileDesc;
    if ( ime_native_lseek( fd->f, pos, SEEK_SET ) == -1 ) {
        char error_msg[FTI_BUFS];
        sprintf(error_msg, "Unable to Seek : [IME native ERROR -%s.]", strerror(errno));
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
size_t FTI_GetIMEFilePos(void *fileDesc)
{
    WriteIMEInfo_t *fd = (WriteIMEInfo_t *) fileDesc;
    return ime_native_lseek(fd->f, 0, SEEK_CUR);
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Synchornizes the current file
  @param      fileDesc          The fileDescriptor
  @return     int               FTI_SCES on success

 **/
/*-------------------------------------------------------------------------*/
int FTI_IMESync(void *fileDesc)
{
    // KC FIXME ime_native_fsync(fileno(((WriteIMEInfo_t *) fileDesc)->f));
    ime_native_fsync(((WriteIMEInfo_t *) fileDesc)->f);
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
void* FTI_InitIME(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec, FTIT_topology* FTI_Topo, FTIT_checkpoint *FTI_Ckpt, FTIT_dataset *FTI_Data)
{

    FTI_Print("I/O mode: IME.", FTI_DBUG);

    char fn[FTI_BUFS];
    int level = FTI_Exec->ckptMeta.level;

    WriteIMEInfo_t *write_info = (WriteIMEInfo_t *) malloc (sizeof(WriteIMEInfo_t));

    snprintf(FTI_Exec->ckptMeta.ckptFile, FTI_BUFS, "Ckpt%d-Rank%d.%s", FTI_Exec->ckptId, FTI_Topo->myRank, FTI_Conf->suffix);

    if (level == 4 && FTI_Ckpt[4].isInline) { //If inline L4 save directly to global directory
        snprintf(fn, FTI_BUFS, "%s/%s", FTI_Conf->gTmpDir, FTI_Exec->ckptMeta.ckptFile);
    }
    else {
        snprintf(fn, FTI_BUFS, "%s/%s", FTI_Conf->lTmpDir, FTI_Exec->ckptMeta.ckptFile);
    }
    
    DBG_MSG("WRITING WITH IME: fn -> '%s'", -1, fn);

    write_info->flag = O_WRONLY;
    write_info->offset = 0;
    FTI_IMEOpen(fn,write_info);
    return write_info;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Writes ckpt to using IME native format.
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Topo        Topology metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @param      FTI_Data        Dataset metadata.
  @return     integer         FTI_SCES if successful.

 **/
/*-------------------------------------------------------------------------*/
int FTI_WriteIMEData(FTIT_dataset * FTI_DataVar, void *fd)
{
    WriteIMEInfo_t *write_info = (WriteIMEInfo_t*) fd;
    char str[FTI_BUFS];
    int res;

    if ( !(FTI_DataVar->isDevicePtr) ){
        if (( res = FTI_Try(FTI_IMEWrite(FTI_DataVar->ptr, FTI_DataVar->size, write_info),"Storing Data to Checkpoint file")) != FTI_SCES){
            snprintf(str, FTI_BUFS, "Dataset #%d could not be written.", FTI_DataVar->id);
            FTI_Print(str, FTI_EROR);
            FTI_IMEClose(write_info);
            return FTI_NSCS;
        }
    }
#ifdef GPUSUPPORT
    // if data are stored to the GPU move them from device
    // memory to cpu memory and store them.
    else {
        if ((res = FTI_Try(
                        FTI_TransferDeviceMemToFileAsync(FTI_DataVar,  FTI_IMEWrite, write_info),
                        "moving data from GPU to storage")) != FTI_SCES) {
            snprintf(str, FTI_BUFS, "Dataset #%d could not be written.", FTI_DataVar->id);
            FTI_Print(str, FTI_EROR);
            FTI_IMEClose(write_info);
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
void FTI_IMEMD5(unsigned char *dest, void *md5)
{
    WriteIMEInfo_t *write_info =(WriteIMEInfo_t *) md5;
    MD5_Final(dest,&(write_info->integrity));
}
