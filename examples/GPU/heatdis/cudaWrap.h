/**
 *  Copyright (c) 2017 Leonardo A. Bautista-Gomez
 *  All rights reserved
 *	@file	cudaWrap.h
 */

#ifndef FTI_EXAMPLES_GPU_HEATDIS_CUDAWRAP_H_
#define FTI_EXAMPLES_GPU_HEATDIS_CUDAWRAP_H_

#ifdef __cplusplus
extern "C" {
#endif

void hostCopy(void *src, void *dest, size_t size);
int getProperties();
void setDevice(int n);
void deviceCopy(void *src, void *dest, size_t size);
void freeCuda(void *ptr);
void freeCudaHost(void *ptr);
void allocateMemory(void **ptr, size_t size);
void allocateSafeHost(void **ptr, size_t size);
double executekernel(int32_t xElem, int32_t yElem, double *in, double *out,
 double *halo[2], double *lErrors, double *dError, int rank);
void init(double *h, double *g, int32_t Y, int32_t X);
void allocateErrorMemory(void **lerror, void **derror, int32_t xElem,
 int32_t yElem);
void initStream();
void syncStream();
void destroyStream();

#ifdef __cplusplus
}
#endif

#endif  // FTI_EXAMPLES_GPU_HEATDIS_CUDAWRAP_H_
