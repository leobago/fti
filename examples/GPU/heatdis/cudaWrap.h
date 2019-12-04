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
double executekernel(long xElem, long yElem, double *in, double *out,  double *lErrors, double *dError, int rank, int deviceID);
void init(double *h, double *g, long Y, long X, int rank);
void allocateErrorMemory( void **lerror, void **derror, long xElem, long yElem, int rank);
void initStream(cudaStream_t *);
void syncStream(cudaStream_t *);
void destroyStream(cudaStream_t *);



#ifdef __cplusplus
}
#endif

#endif
