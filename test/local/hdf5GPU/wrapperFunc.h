#ifndef __WRAPPER__
#ifdef __cplusplus
extern "C" {
#endif

#define BLKX 16
#define BLKY 8
#define BLKZ 8
#define BLOCKSIZE (BLKX * BLKY *BLKZ)
#define GRDX 4 
#define GRDY 4 
#define GRDZ 4

#define XSIZE (BLKX*GRDX)
#define YSIZE (BLKY*GRDY)
#define ZSIZE (BLKZ*GRDZ)

typedef struct threeDdims{
  int id;
  int x;
  int y;
  int z;
}threeD;

void allocateMemory(void **ptr, size_t size);
void cudaCopy(void *str, void *dest, size_t size);
void hostCopy(void *src, void *dest, size_t size);
void executeKernel(threeD *ptr);
void freeCuda( void *ptr );
int getProperties();
void setDevice(int id);

//int getProperties();
//void setDevice(int id);
#ifdef __cplusplus
}
#endif
#endif



