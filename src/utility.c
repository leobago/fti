#include <string.h>

#include "interface.h"
#include "IO/ftiff.h"
#include "api_cuda.h"
#include "utility.h"


/*-------------------------------------------------------------------------*/
/**
  @brief     Writes data to a file using the posix library
  @param     src    The location of the data to be written 
  @param     size   The number of bytes that I need to write 
  @param     opaque A pointer to the File descriptor  
  @return    integer         FTI_SCES if successful.

  Writes the data to a file using the posix library. 

 **/
/*-------------------------------------------------------------------------*/
int write_posix(void *src, size_t size, void *opaque)
{
  FILE *fd = (FILE *)opaque;
  size_t written = 0;
  int fwrite_errno;
  char str[FTI_BUFS];

  while (written < size && !ferror(fd)) {
    errno = 0;
    written += fwrite(((char *)src) + written, 1, size - written, fd);
    fwrite_errno = errno;
  }

  if (ferror(fd)){
    char error_msg[FTI_BUFS];
    error_msg[0] = 0;
    strerror_r(fwrite_errno, error_msg, FTI_BUFS);
    snprintf(str, FTI_BUFS, "utility:c: (write_posix) Dataset could not be written: %s.", error_msg);
    FTI_Print(str, FTI_EROR);
    fclose(fd);
    return FTI_NSCS;
  }
  else
    return FTI_SCES;
}





#ifdef ENABLE_SIONLIB 
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
int write_sion(void *src, size_t size, void *opaque)
{
  int *sid= (int *)opaque;
  int res = sion_fwrite(src, size, 1, *sid);
  if (res < 0 ){
    return FTI_NSCS;
  }
  return FTI_SCES;
}
#endif

/*-------------------------------------------------------------------------*/
/**
  @brief     copies all data of GPU variables to a CPU memory location 
  @param     FTI_Exec Execution Meta data. 
  @param     FTI_Data        Dataset metadata.
  @return    integer FTI_SCES if successful.

  Copis data from the GPU side to the CPU memory  

 **/
/*-------------------------------------------------------------------------*/

int copyDataFromDevive(FTIT_execution* FTI_Exec, FTIT_dataset* FTI_Data){
#ifdef GPUSUPPORT
  int i;
  for (i = 0; i < FTI_Exec->nbVar; i++) {
    if ( FTI_Data[i].isDevicePtr ){
      FTI_copy_from_device( FTI_Data[i].ptr, FTI_Data[i].devicePtr,FTI_Data[i].size,FTI_Exec);
    }
  }
#endif
  return FTI_SCES;
}

