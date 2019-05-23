#ifndef UTILITY_H
#define UTILITY_H

int write_posix(void *src, size_t size, void *opaque);
int write_mpi(void *src, size_t size, void *opaque);
int copyDataFromDevive(FTIT_execution* FTI_Exec, FTIT_dataset* FTI_Data);

#ifdef ENABLE_SIONLIB 
int write_sion(void *src, size_t size, void *opaque);
#endif

#endif // UTILITY_H
