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
 *  @file   compression.c
 *  @date   May, 2019
 *  @brief  Funtions to support posix checkpointing.
 */


#include "../interface.h"

static double t0_compression;
static double t1_compression;
static double t0_decompression;
static double t1_decompression;

int FTI_InitCompression( FTIT_dataset* data )
{
  
  if ( data->compression.mode == FTI_CPC_NONE ) return FTI_SCES;
  
  t0_compression = MPI_Wtime();

  data->compression.success = true;
  data->compression.ptr = malloc(data->size);
  
  switch ( data->compression.mode ) {
    case FTI_CPC_FPZIP:
      data->compression.size = FTI_CompressFpzip( data );
      break;
    case FTI_CPC_ZFP:
      data->compression.size = FTI_CompressZfp( data );
      break;
    case FTI_CPC_SINGLE:
      data->compression.size = FTI_DoubleToFloat( data );
      break;
    case FTI_CPC_HALF:
      data->compression.size = FTI_DoubleToFloat16( data );
      break;
    case FTI_CPC_STRIP:
      break;
    default:
      free(data->compression.ptr);
      FTI_Print("Invalid compression mode!", FTI_WARN);
      return FTI_NSCS;
  }

}

int FTI_FiniCompression( FTIT_dataset* data )
{
  
  if ( data->compression.mode == FTI_CPC_NONE ) return FTI_SCES;
  
  char type[FTI_BUFS];

  switch ( data->compression.mode ) {
    case FTI_CPC_FPZIP:
      snprintf(type, FTI_BUFS, "%s", "fpzip");
      break;
    case FTI_CPC_ZFP:
      snprintf(type, FTI_BUFS, "%s", "zfp");
      break;
    case FTI_CPC_SINGLE:
      snprintf(type, FTI_BUFS, "%s", "single precision");
      break;
    case FTI_CPC_HALF:
      snprintf(type, FTI_BUFS, "%s", "half precision");
      break;
    case FTI_CPC_STRIP:
      break;
    default:
      free(data->compression.ptr);
      FTI_Print("Invalid compression mode!", FTI_WARN);
      return FTI_NSCS;
  }

  if ( data->compression.success ) free( data->compression.ptr );
  
  t1_compression = MPI_Wtime();
  
  char str[FTI_BUFS];
  
  snprintf(str, FTI_BUFS, "compression '%s' took %lf seconds", type, t1_compression - t0_compression);

  FTI_Print(str, FTI_INFO);

  return FTI_SCES;

}

int FTI_InitDecompression( FTIT_dataset* data )
{
  
  if ( data->compression.mode == FTI_CPC_NONE ) return FTI_SCES;
  
  t0_decompression = MPI_Wtime();
  
  data->compression.ptr = malloc(data->sizeStored);
  
  switch ( data->compression.mode ) {
    case FTI_CPC_FPZIP:
      break;
    case FTI_CPC_ZFP:
      break;
    case FTI_CPC_SINGLE:
      break;
    case FTI_CPC_HALF:
      break;
    case FTI_CPC_STRIP:
      break;
    default:
      free(data->compression.ptr);
      FTI_Print("Invalid compression mode!", FTI_WARN);
      return FTI_NSCS;
  }
  
}

int FTI_FiniDecompression( FTIT_dataset* data )
{
  
  if ( data->compression.mode == FTI_CPC_NONE ) return FTI_SCES;
  
  char type[FTI_BUFS];
  
  switch ( data->compression.mode ) {
    case FTI_CPC_FPZIP:
      FTI_DecompressFpzip( data );
      snprintf(type, FTI_BUFS, "%s", "fpzip");
      break;
    case FTI_CPC_ZFP:
      FTI_DecompressZfp( data );
      snprintf(type, FTI_BUFS, "%s", "zfp");
      break;
    case FTI_CPC_SINGLE:
      FTI_FloatToDouble( data );
      snprintf(type, FTI_BUFS, "%s", "single");
      break;
    case FTI_CPC_HALF:
      FTI_Float16ToDouble( data );
      snprintf(type, FTI_BUFS, "%s", "half");
      break;
    case FTI_CPC_STRIP:
      break;
    default:
      free(data->compression.ptr);
      FTI_Print("Invalid compression mode!", FTI_WARN);
      return FTI_NSCS;
  }
  
  free(data->compression.ptr);
  
  t1_decompression = MPI_Wtime();
  
  char str[FTI_BUFS];
  
  snprintf(str, FTI_BUFS, "decompression '%s' took %lf seconds", type, t1_decompression - t0_decompression);

  FTI_Print(str, FTI_INFO);
}

/**********************************************************************

		Conversion double to float

**********************************************************************/

int64_t FTI_DoubleToFloat( FTIT_dataset* data )
{
  double* values_d = (double*) data->ptr; 
  float* values_f = (float*) data->compression.ptr;
  for(int64_t i=0; i<data->count; i++) {
    values_f[i] = (float) values_d[i];
  }
  return data->count*sizeof(float);
}

int FTI_FloatToDouble( FTIT_dataset* data )
{
  double* values_d = (double*) data->ptr; 
  float* values_f = (float*) data->compression.ptr;
  for(int64_t i=0; i<data->count; i++) {
    values_d[i] = (double) values_f[i];
  }
  return FTI_SCES;
}

/**********************************************************************

		Conversion double to float16

**********************************************************************/

int64_t FTI_DoubleToFloat16( FTIT_dataset* data )
{
  int rounding_mode = FE_TONEAREST;
  UINT16_TYPE *hp = (UINT16_TYPE *) data->compression.ptr; // Type pun output as an unsigned 16-bit int
  UINT32_TYPE *xp = (UINT32_TYPE *) data->ptr; // Type pun input as an unsigned 32-bit int
  UINT16_TYPE    hs, he, hm, hq, hr;
  UINT32_TYPE x, xs, xe, xm, xn, xt, zm, zt, z1;
  int hes, N;
  ptrdiff_t i;
  static int next;  // Little Endian adjustment
  static int checkieee = 1;  // Flag to check for IEEE754, Endian, and word size
  double one = 1.0; // Used for checking IEEE754 floating point format
  UINT32_TYPE *ip; // Used for checking IEEE754 floating point format

  if( checkieee ) { // 1st call, so check for IEEE754, Endian, and word size
    ip = (UINT32_TYPE *) &one;
    if( *ip ) { // If Big Endian, then no adjustment
      next = 0;
    } else { // If Little Endian, then adjustment will be necessary
      next = 1;
      ip++;
    }
    if( *ip != 0x3FF00000u ) { // Check for exact IEEE 754 bit pattern of 1.0
      return 1;  // Floating point bit pattern is not IEEE 754
    }
    if( sizeof(INT16_TYPE) != 2 || sizeof(INT32_TYPE) != 4 ) {
      return 1;  // short is not 16-bits, or long is not 32-bits.
    }
    checkieee = 0; // Everything checks out OK
  }

  if( data->ptr == NULL || data->compression.ptr == NULL ) { // Nothing to convert (e.g., imag part of pure real)
    return 0;
  }

  hq = (UINT16_TYPE) 0x0200u;  // Force NaN results to be quiet?

#pragma omp parallel for private(x,xs,xe,xm,xt,zm,zt,z1,hs,he,hm,hr,hes,N)
  for( i=0; i<data->count; i++ ) {
    if( next ) { // Little Endian
      xn = xp[2*i];  // Lower mantissa
      x  = xp[2*i+1];  // Sign, exponent, upper mantissa
    } else { // Big Endian
      x  = xp[2*i];  // Sign, exponent, upper mantissa
      xn = xp[2*i+1];  // Lower mantissa
    }
    if( (x & 0x7FFFFFFFu) == 0 ) {  // Signed zero
      hp[i]= (UINT16_TYPE) (x >> 16);  // Return the signed zero
    } else { // Not zero
      xs = x & 0x80000000u;  // Pick off sign bit
      xe = x & 0x7FF00000u;  // Pick off exponent bits
      xm = x & 0x000FFFFFu;  // Pick off mantissa bits
      xt = x & 0x000003FFu;  // Pick off trailing 10 mantissa bits beyond the shift (used for rounding normalized determination)
      if( xe == 0 ) {  // Denormal will underflow, return a signed zero or signed smallest denormal depending on rounding_mode
        if( (rounding_mode == FE_UPWARD   && (xm || xn) && !xs) ||  // Mantissa bits are non-zero and sign bit is 0
            (rounding_mode == FE_DOWNWARD && (xm || xn) &&  xs) ) { // Mantissa bits are non-zero and sigh bit is 1
          hp[i] = (UINT16_TYPE) (xs >> 16) | (UINT16_TYPE) 1u;  // Signed smallest denormal
        } else {
          hp[i] = (UINT16_TYPE) (xs >> 16);  // Signed zero
        }
      } else if( xe == 0x7FF00000u ) {  // Inf or NaN (all the exponent bits are set)
        if( xm == 0 && xn == 0 ) { // If mantissa is zero ...
          hp[i] = (UINT16_TYPE) ((xs >> 16) | 0x7C00u); // Signed Inf
        } else {
          hm = (UINT16_TYPE) (xm >> 10); // Shift mantissa over
          if( hm ) { // If we still have some non-zero bits (payload) after the shift ...
            hp[i] = (UINT16_TYPE) ((xs >> 16) | 0x7C00u | hq | hm); // Signed NaN, shifted mantissa bits set
          } else {
            hp[i] = (UINT16_TYPE) ((xs >> 16) | 0x7E00u); // Signed NaN, only 1st mantissa bit set (quiet)
          }
        }
      } else { // Normalized number
        hs = (UINT16_TYPE) (xs >> 16); // Sign bit
        hes = ((int)(xe >> 20)) - 1023 + 15; // Exponent unbias the double, then bias the halfp
        if( hes >= 0x1F ) {  // Overflow
          hp[i] = (UINT16_TYPE) ((xs >> 16) | 0x7C00u); // Signed Inf
        } else if( hes <= 0 ) {  // Underflow exponent, so halfp will be denormal
          xm |= 0x00100000u;  // Add the hidden leading bit
          N = (11 - hes);  // Number of bits to shift mantissa to get it into halfp word
          hm = (N < 32) ? (UINT16_TYPE) (xm >> N) : (UINT16_TYPE) 0u; // Halfp mantissa
          hr = (UINT16_TYPE) 0u; // Rounding bit, default to 0 for now (this will catch FE_TOWARDZERO and other cases)
          if( N <= 21 ) {  // Mantissa bits have not shifted away from the end
            zm = (0x001FFFFFu >> N) << N;  // Halfp denormal mantissa bit mask
            zt = 0x001FFFFFu & ~zm;  // Halfp denormal trailing manntissa bits mask
            z1 = (zt >> (N-1)) << (N-1);  // First bit of trailing bit mask
            xt = xm & zt;  // Trailing mantissa bits
            if( rounding_mode == FE_TONEAREST ) {
              if( (xt > z1 || xt == z1 && xn) || (xt == z1 && !xn) && (hm & 1u) ) { // Trailing bits are more than tie, or tie and mantissa is currently odd
                hr = (UINT16_TYPE) 1u; // Rounding bit set to 1
              }
            } else if( rounding_mode == FE_TONEARESTINF ) {
              if( xt >= z1 ) { // Trailing bits are more than or equal to tie
                hr = (UINT16_TYPE) 1u; // Rounding bit set to 1
              }
            } else if( (rounding_mode == FE_UPWARD   && (xt || xn) && !xs) ||  // Trailing bits are non-zero and sign bit is 0
                (rounding_mode == FE_DOWNWARD && (xt || xn) &&  xs) ) { // Trailing bits are non-zero and sign bit is 1
              hr = (UINT16_TYPE) 1u; // Rounding bit set to 1
            }                    
          } else {  // Mantissa bits have shifted at least one bit beyond the end (ties not possible)
            if( (rounding_mode == FE_UPWARD   && (xm || xn) && !xs) ||  // Trailing bits are non-zero and sign bit is 0
                (rounding_mode == FE_DOWNWARD && (xm || xn) &&  xs) ) { // Trailing bits are non-zero and sign bit is 1
              hr = (UINT16_TYPE) 1u; // Rounding bit set to 1
            }                    
          }
          hp[i] = (hs | hm) + hr; // Combine sign bit and mantissa bits and rounding bit, biased exponent is zero
        } else {
          he = (UINT16_TYPE) (hes << 10); // Exponent
          hm = (UINT16_TYPE) (xm >> 10); // Mantissa
          hr = (UINT16_TYPE) 0u; // Rounding bit, default to 0 for now
          if( rounding_mode == FE_TONEAREST ) {
            if( (xt > 0x00000200u || xt == 0x00000200u && xn) || (xt == 0x00000200u && !xn) && (hm & 1u) ) { // Trailing bits are more than tie, or tie and mantissa is currently odd
              hr = (UINT16_TYPE) 1u; // Rounding bit set to 1
            }
          } else if( rounding_mode == FE_TONEARESTINF ) {
            if( xt >= 0x00000200u  ) { // Trailing bits are more than or equal to tie
              hr = (UINT16_TYPE) 1u; // Rounding bit set to 1
            }
          } else if( (rounding_mode == FE_UPWARD   && (xt || xn) && !xs) ||  // Trailing bits are non-zero and sign bit is 0
              (rounding_mode == FE_DOWNWARD && (xt || xn) &&  xs) ) { // Trailing bits are non-zero and sign bit is 1
            hr = (UINT16_TYPE) 1u; // Rounding bit set to 1
          }
          hp[i] = (hs | he | hm) + hr;  // Adding rounding bit might overflow into exp bits, but that is OK
        }
      }
    }
  }
  return data->count * 2;
}

int FTI_Float16ToDouble( FTIT_dataset* data )
{
  UINT16_TYPE *hp = (UINT16_TYPE *) data->compression.ptr; // Type pun input as an unsigned 16-bit int
  UINT32_TYPE *xp = (UINT32_TYPE *) data->ptr; // Type pun output as an unsigned 32-bit int
  UINT16_TYPE h, hs, he, hm;
  UINT32_TYPE xs, xe, xm, xn;
  INT32_TYPE xes;
  ptrdiff_t i, j;
  int e;
  static int next;  // Little Endian adjustment
  static int checkieee = 1;  // Flag to check for IEEE754, Endian, and word size
  double one = 1.0; // Used for checking IEEE754 floating point format
  UINT32_TYPE *ip; // Used for checking IEEE754 floating point format

  if( checkieee ) { // 1st call, so check for IEEE754, Endian, and word size
    ip = (UINT32_TYPE *) &one;
    if( *ip ) { // If Big Endian, then no adjustment
      next = 0;
    } else { // If Little Endian, then adjustment will be necessary
      next = 1;
      ip++;
    }
    if( *ip != 0x3FF00000u ) { // Check for exact IEEE 754 bit pattern of 1.0
      return 1;  // Floating point bit pattern is not IEEE 754
    }
    if( sizeof(INT16_TYPE) != 2 || sizeof(INT32_TYPE) != 4 ) {
      return 1;  // short is not 16-bits, or long is not 32-bits.
    }
    checkieee = 0; // Everything checks out OK
  }

  if( data->compression.ptr == NULL || data->ptr == NULL ) // Nothing to convert (e.g., imag part of pure real)
    return 0;

  xp += next;  // Little Endian adjustment if necessary

#pragma omp parallel for private(xs,xe,xm,h,hs,he,hm,xes,e,j)
  for( i=0; i<data->count; i++ ) {
    j = 2*i;
    if( next ) {
      xp[j-1] = 0; // Set lower mantissa bits, Little Endian
    } else {
      xp[j+1] = 0; // Set lower mantissa bits, Big Endian
    }
    h = hp[i];
    if( (h & 0x7FFFu) == 0 ) {  // Signed zero
      xp[j] = ((UINT32_TYPE) h) << 16;  // Return the signed zero
    } else { // Not zero
      hs = h & 0x8000u;  // Pick off sign bit
      he = h & 0x7C00u;  // Pick off exponent bits
      hm = h & 0x03FFu;  // Pick off mantissa bits
      if( he == 0 ) {  // Denormal will convert to normalized
        e = -1; // The following loop figures out how much extra to adjust the exponent
        do {
          e++;
          hm <<= 1;
        } while( (hm & 0x0400u) == 0 ); // Shift until leading bit overflows into exponent bit
        xs = ((UINT32_TYPE) hs) << 16; // Sign bit
        xes = ((INT32_TYPE) (he >> 10)) - 15 + 1023 - e; // Exponent unbias the halfp, then bias the double
        xe = (UINT32_TYPE) (xes << 20); // Exponent
        xm = ((UINT32_TYPE) (hm & 0x03FFu)) << 10; // Mantissa
        xp[j] = (xs | xe | xm); // Combine sign bit, exponent bits, and mantissa bits
      } else if( he == 0x7C00u ) {  // Inf or NaN (all the exponent bits are set)
        xp[j] = (((UINT32_TYPE) hs) << 16) | ((UINT32_TYPE) 0x7FF00000u) | (((UINT32_TYPE) hm) << 10); // Signed Inf or NaN
      } else {
        xs = ((UINT32_TYPE) hs) << 16; // Sign bit
        xes = ((INT32_TYPE) (he >> 10)) - 15 + 1023; // Exponent unbias the halfp, then bias the double
        xe = (UINT32_TYPE) (xes << 20); // Exponent
        xm = ((UINT32_TYPE) hm) << 10; // Mantissa
        xp[j] = (xs | xe | xm); // Combine sign bit, exponent bits, and mantissa bits
      }
    }
  }
  return 0;
}

/**********************************************************************

		Compression with fpzip

**********************************************************************/

int64_t FTI_CompressFpzip( FTIT_dataset* data )
{
	
  FPZ* fpz = (FPZ*) data->compression.context;
  int64_t result = FTI_SCES;
  int64_t size;
  char err[FTI_BUFS];
  
  data->compression.ptr = realloc( data->compression.ptr, data->size+1024 );
	
  fpz = fpzip_write_to_buffer( data->compression.ptr, data->size+1024 );
  fpz->type = FPZIP_TYPE_DOUBLE;
  fpz->prec = data->compression.parameter;
  fpz->nx = data->count;
  fpz->ny = 1;
  fpz->nz = 1;
  fpz->nf = 1;

  /* write header */
  if (!fpzip_write_header(fpz)) {
    snprintf(err, FTI_BUFS, "cannot write header: %s\n", fpzip_errstr[fpzip_errno]);
    FTI_Print( err, FTI_WARN );
    free( data->compression.ptr );
    data->compression.ptr = data->ptr;
    data->compression.size = data->size;
    data->compression.success = false;
    result = FTI_NSCS;
  }
  
  /* perform actual compression */
  if( result == FTI_SCES ) {
    size = fpzip_write(fpz, data->ptr);
    if ( !size ) {
      snprintf(err, FTI_BUFS, "compression failed: %s\n", fpzip_errstr[fpzip_errno]);
      FTI_Print( err, FTI_WARN );
      free( data->compression.ptr );
      data->compression.ptr = data->ptr;
      data->compression.size = data->size;
      data->compression.success = false;
      result = FTI_NSCS;
    }
  }
  
  fpzip_write_close(fpz);

  return (result == FTI_SCES) ? size : FTI_NSCS;

}

int FTI_DecompressFpzip( FTIT_dataset* data )
{

  FPZ* fpz = data->compression.context;

  fpz = fpzip_read_from_buffer( data->compression.ptr );

  /* read header */
  if (!fpzip_read_header(fpz)) {
    fprintf(stderr, "cannot read header: %s\n", fpzip_errstr[fpzip_errno]);
    return FTI_NSCS;
  }
  /* make sure array size stored in header matches expectations */
  if ((fpz->type == FPZIP_TYPE_FLOAT ? sizeof(float) : sizeof(double)) * fpz->nx * fpz->ny * fpz->nz * fpz->nf != data->size) {
    fprintf(stderr, "array size does not match dimensions from header\n");
    return FTI_NSCS;
  }
  /* perform actual decompression */
  if (!fpzip_read(fpz, data->ptr)) {
    fprintf(stderr, "decompression failed: %s\n", fpzip_errstr[fpzip_errno]);
    return FTI_NSCS;
  }

  return FTI_SCES;
}


/**********************************************************************

		Compression with zfp

**********************************************************************/

int64_t FTI_CompressZfp( FTIT_dataset* data )
{
	
	// initialize metadata for the 3D array a[nz][ny][nx]
	zfp_type type = zfp_type_double;                          // array scalar type
	zfp_field* field = zfp_field_1d(data->ptr, type, data->count); // array metadata

	// initialize metadata for a compressed stream
	zfp_stream* zfp = zfp_stream_open(NULL);                  // compressed stream and parameters
	zfp_stream_set_accuracy(zfp, data->compression.parameter);                  // set tolerance for fixed-accuracy mode
	//  zfp_stream_set_precision(zfp, precision);             // alternative: fixed-precision mode
	//  zfp_stream_set_rate(zfp, rate, type, 3, 0);           // alternative: fixed-rate mode

	// allocate buffer for compressed data
	size_t bufsize = zfp_stream_maximum_size(zfp, field);     // capacity of compressed buffer (conservative)
	data->compression.ptr = realloc( data->compression.ptr, bufsize );                           // storage for compressed stream

	// associate bit stream with allocated buffer
	bitstream* stream = stream_open(data->compression.ptr, bufsize);         // bit stream to compress to
	zfp_stream_set_bit_stream(zfp, stream);                   // associate with compressed stream
	zfp_stream_rewind(zfp);                                   // rewind stream to beginning

	// compress array
	return zfp_compress(zfp, field);                // return value is byte size of compressed stream

}

int FTI_DecompressZfp( FTIT_dataset* data )
{

	// initialize metadata for the 3D array a[nz][ny][nx]
	zfp_type type = zfp_type_double;                          // array scalar type
	zfp_field* field = zfp_field_1d(data->ptr, type, data->count); // array metadata

	// initialize metadata for a compressed stream
	zfp_stream* zfp = zfp_stream_open(NULL);                  // compressed stream and parameters
	zfp_stream_set_accuracy(zfp, data->compression.parameter);                  // set tolerance for fixed-accuracy mode
	//  zfp_stream_set_precision(zfp, precision);             // alternative: fixed-precision mode
	//  zfp_stream_set_rate(zfp, rate, type, 3, 0);           // alternative: fixed-rate mode

	// associate bit stream with allocated buffer
	bitstream* stream = stream_open(data->compression.ptr, data->sizeStored);         // bit stream to compress to
	zfp_stream_set_bit_stream(zfp, stream);                   // associate with compressed stream
	zfp_stream_rewind(zfp);                                   // rewind stream to beginning

	// compress array
	size_t zfpsize = zfp_decompress(zfp, field);                // return value is byte size of compressed stream
  
  return FTI_SCES;

}

