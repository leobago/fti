/**
 *  Copyright (c) 2017 Leonardo A. Bautista-Gomez
 *  All rights reserved
 *
 *  @file   compression.h
 */

#ifndef FTI_SRC_IO_COMPRESSION_H_
#define FTI_SRC_IO_COMPRESSION_H_

#include <fpzip.h>
#include <zfp.h>

#define  INT16_TYPE short
#define UINT16_TYPE unsigned short
#define  INT32_TYPE int
#define UINT32_TYPE unsigned int

#define HALFP_PINF ((UINT16_TYPE) 0x7C00u)  // +inf
#define HALFP_MINF ((UINT16_TYPE) 0xFC00u)  // -inf
#define HALFP_PNAN ((UINT16_TYPE) 0x7E00u)  // +nan (only is_quite bit set, no payload)
#define HALFP_MNAN ((UINT16_TYPE) 0xFE00u)  // -nan (only is_quite bit set, no payload)

/* Define our own values for rounding_mode if they aren't already defined */
#ifndef FE_TONEAREST
    #define FE_TONEAREST    0x0000
    #define FE_UPWARD       0x0100
    #define FE_DOWNWARD     0x0200
    #define FE_TOWARDZERO   0x0300
#endif
#define     FE_TONEARESTINF 0xFFFF  /* Round to nearest, ties away from zero (apparently no standard C name for this) */

#ifdef __cplusplus
extern "C" {
#endif

  int FTI_InitCompression( FTIT_dataset* );
  int FTI_FiniCompression( FTIT_dataset* );
  int FTI_InitDecompression( FTIT_dataset* );
  int FTI_FiniDecompression( FTIT_dataset* );

  int64_t FTI_DoubleToFloat16( FTIT_dataset* data );
  int FTI_Float16ToDouble( FTIT_dataset* data );
  int64_t FTI_DoubleToFloat( FTIT_dataset* data );
  int FTI_FloatToDouble( FTIT_dataset* data );
  int64_t FTI_CompressFpzip( FTIT_dataset* data );
  int FTI_DecompressFpzip( FTIT_dataset* data );
  int64_t FTI_CompressZfp( FTIT_dataset* data );
  int FTI_DecompressZfp( FTIT_dataset* data );

#ifdef __cplusplus
}
#endif
#endif  // FTI_SRC_IO_COMPRESSION_H_

