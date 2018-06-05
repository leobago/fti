/**
 *  @file   api_cuda.h
 *  @author Max M. Baird(maxbaird.gy@gmail.com)
 *  @date   May, 2018
 *  @brief  Header file for functions that add CUDA support to FTI.
 */

#ifndef _API_CUDA_H
#define _API_CUDA_H

/*---------------------------------------------------------------------------
  Defines
  ---------------------------------------------------------------------------*/

/** Used to set and check for a CPU pointer.                               */
#define CPU_POINTER 1
/** Used to set and check for a GPU pointer.                               */
#define GPU_POINTER 0

int FTI_determine_pointer_type(const void *ptr, int *pointer_type);
int FTI_copy_from_device(void* dst, void* src, long count);
int FTI_copy_to_device(void* dst, void* src, long count);

#endif
