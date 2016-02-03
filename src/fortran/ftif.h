/**
 *  @file   ftif.h
 *  @author
 *  @date   February, 2016
 *  @brief  Header file for the FTI Fortran interface.
 */

#ifndef _FTIF_H
#define _FTIF_H

int FTI_Init_fort_wrapper(char* configFile, int* globalComm);
int FTI_InitType_wrapper(FTIT_type** type, int size);
int FTI_Protect_wrapper(int id, void* ptr, long count, FTIT_type* type);

#endif
