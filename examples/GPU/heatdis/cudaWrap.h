#ifndef __WRAP__
#ifdef __cplusplus
extern "C" {
#endif

void hostCopy(void *src, void *dest, size_t size);
int getProperties();
void setDevice(int n);
void deviceCopy(void *src, void *dest, size_t size);
void freeCuda( void *ptr );
void freeCudaHost(void *ptr );
void allocateMemory(void **ptr, size_t size);
void allocateSafeHost(void **ptr, size_t size);
double executekernel(long xElem, long yElem, double *in, double *out, double *halo[2], double *lErrors, double *dError, int rank);
void init(double *h, double *g, long Y, long X);
void allocateErrorMemory( void **lerror, void **derror, long xElem, long yElem);
void initStream();
void syncStream();
void destroyStream();

#ifdef __cplusplus
}
#endif

#endif
