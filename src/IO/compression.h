/**
 *  Copyright (c) 2017 Leonardo A. Bautista-Gomez
 *  All rights reserved
 *
 *  @file   compression.h
 */

#ifndef FTI_SRC_IO_COMPRESSION_H_
#define FTI_SRC_IO_COMPRESSION_H_

#ifdef __cplusplus
extern "C" {
#endif

  int FTI_InitCompression( FTIT_dataset* );
  int FTI_FiniCompression( FTIT_dataset* );
  int FTI_InitDecompression( FTIT_dataset* );
  int FTI_FiniDecompression( FTIT_dataset* );

  int64_t FTI_DoubleToFloat( FTIT_dataset* data );
  int64_t FTI_FloatToDouble( FTIT_dataset* data );

#ifdef __cplusplus
}
#endif
#endif  // FTI_SRC_IO_COMPRESSION_H_

