#ifndef __WRAPPER__
#ifdef __cplusplus
extern "C" {
#endif

void allocateMemory(void **ptr, size_t size);
void cudaCopy(void *str, void *dest, size_t size);
void executeVecMult(int blocks, int threads, double *A, double *B, size_t size, double *ha);
void freeCuda( void *ptr );
void executeVecMultUnified(int blocks, int threads, double *A, double *B, size_t size, double *ha);
void allocateManaged( void **ptr , size_t size);
int getProperties();
void setDevice(int id);
#ifdef __cplusplus
}
#endif
#endif



