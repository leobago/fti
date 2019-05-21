#ifndef FTIFF_DEFS_H
#define FTIFF_DEFS_H

#define FTI_DCP_MODE_OFFSET 2000
#define FTI_DCP_MODE_MD5 2001
#define FTI_DCP_MODE_CRC32 2002

#define MBR_CNT(TYPE) int TYPE ## _mbrCnt
#define MBR_BLK_LEN(TYPE) int TYPE ## _mbrBlkLen[]
#define MBR_TYPES(TYPE) MPI_Datatype TYPE ## _mbrTypes[]
#define MBR_DISP(TYPE) MPI_Aint TYPE ## _mbrDisp[]

#endif // FTIFF_DEFS_H
