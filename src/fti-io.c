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
 *  @file   FTI_IO.c
 *  @date   May, 2019
 *  @brief  API functions for the FTI library.
 */



#include "interface.h"

/** Structure that stores the function pointer wrappers to perform the IO **/
FTIT_IO ftiIO[4];



/*-------------------------------------------------------------------------*/
/**
  @brief      Place holder function, it is used when the file format does not support integrity checksums.
  @param      data  Does not matter.
  @param      a does not matter.
  @return     void.

    THis function is passed as a reference when different file formats do not 
    actually compute an integrity checksum. It helps to avoid if statements in the
    code and provides a more stream line code format.
 **/
/*-------------------------------------------------------------------------*/
void FTI_dummy(unsigned char *data, void* a){
    return;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      This function initializes the FTI_IO structure with the functions that write the ckpt file.
  @param      ckptIO                File format selected by the user in the configuration file. 
  @param      FTI_Exec              Execution environment of the FTI. 
  @return     int                   On success FTI_SCES

    This function actually initializes the execution paths of the write checkpoint function.
 **/
/*-------------------------------------------------------------------------*/
int FTI_InitCheckpointWriters(int ckptIO, FTIT_execution * FTI_Exec ){
    //Initialize Local and Global writers
    switch (ckptIO) {
        case FTI_IO_POSIX:
            ftiIO[LOCAL].initCKPT = FTI_InitPosix; 
            ftiIO[LOCAL].WriteData = FTI_WritePosixData; 
            ftiIO[LOCAL].finCKPT= FTI_PosixClose; 
            ftiIO[LOCAL].getPos	= FTI_GetPosixFilePos; 
            ftiIO[LOCAL].finIntegrity = FTI_PosixMD5; 

            ftiIO[GLOBAL].initCKPT = FTI_InitPosix; 
            ftiIO[GLOBAL].WriteData = FTI_WritePosixData; 
            ftiIO[GLOBAL].finCKPT= FTI_PosixClose; 
            ftiIO[GLOBAL].getPos	= FTI_GetPosixFilePos; 
            ftiIO[GLOBAL].finIntegrity = FTI_PosixMD5; 


            ftiIO[2 + LOCAL].initCKPT = FTI_InitDCPPosix; 
            ftiIO[2 + LOCAL].WriteData = FTI_WritePosixDCPData; 
            ftiIO[2 + LOCAL].finCKPT= FTI_PosixDCPClose; 
            ftiIO[2 + LOCAL].getPos	= FTI_GetDCPPosixFilePos; 
            ftiIO[2 + LOCAL].finIntegrity = FTI_dummy; 

            ftiIO[2 + GLOBAL].initCKPT = FTI_InitDCPPosix; 
            ftiIO[2 + GLOBAL].WriteData = FTI_WritePosixDCPData; 
            ftiIO[2 + GLOBAL].finCKPT= FTI_PosixDCPClose; 
            ftiIO[2 + GLOBAL].getPos	= FTI_GetDCPPosixFilePos; 
            ftiIO[2 + GLOBAL].finIntegrity = FTI_dummy; 


            FTI_Exec->ckptFunc[GLOBAL] = FTI_Write;
            FTI_Exec->ckptFunc[LOCAL] = FTI_Write;

            FTI_Exec->initICPFunc[LOCAL] = FTI_startICP; 
            FTI_Exec->initICPFunc[GLOBAL] = FTI_startICP;

            FTI_Exec->writeVarICPFunc[LOCAL] = FTI_WriteVar;
            FTI_Exec->writeVarICPFunc[GLOBAL] = FTI_WriteVar;

            FTI_Exec->finalizeICPFunc[LOCAL] = FTI_FinishICP;
            FTI_Exec->finalizeICPFunc[GLOBAL] = FTI_FinishICP;

            break;

        case FTI_IO_MPI:
            ftiIO[LOCAL].initCKPT = FTI_InitPosix; 
            ftiIO[LOCAL].WriteData = FTI_WritePosixData; 
            ftiIO[LOCAL].finCKPT= FTI_PosixClose; 
            ftiIO[LOCAL].getPos	= FTI_GetPosixFilePos; 
            ftiIO[LOCAL].finIntegrity = FTI_PosixMD5; 

            ftiIO[GLOBAL].initCKPT = FTI_InitMPIO; 
            ftiIO[GLOBAL].WriteData = FTI_WriteMPIOData; 
            ftiIO[GLOBAL].finCKPT= FTI_MPIOClose; 
            ftiIO[GLOBAL].getPos	= FTI_GetMPIOFilePos; 
            ftiIO[GLOBAL].finIntegrity = FTI_dummy; 


            FTI_Exec->ckptFunc[GLOBAL] = FTI_Write;
            FTI_Exec->ckptFunc[LOCAL] = FTI_Write;

            FTI_Exec->initICPFunc[LOCAL] = FTI_startICP; 
            FTI_Exec->initICPFunc[GLOBAL] = FTI_startICP;

            FTI_Exec->writeVarICPFunc[LOCAL] = FTI_WriteVar;
            FTI_Exec->writeVarICPFunc[GLOBAL] = FTI_WriteVar;

            FTI_Exec->finalizeICPFunc[LOCAL] = FTI_FinishICP;
            FTI_Exec->finalizeICPFunc[GLOBAL] = FTI_FinishICP;
            break;

#ifdef ENABLE_SIONLIB //If SIONlib is installed
        case FTI_IO_SIONLIB:
            FTI_Exec->ckptFunc[LOCAL] = FTI_WritePosix;
            FTI_Exec->ckptFunc[GLOBAL] = FTI_WriteSionlib;

            FTI_Exec->initICPFunc[LOCAL] = FTI_InitPosixICP; 
            FTI_Exec->initICPFunc[GLOBAL] = FTI_InitPosixICP; 

            FTI_Exec->writeVarICPFunc[LOCAL] = FTI_WritePosixVar; 
            FTI_Exec->writeVarICPFunc[GLOBAL] = FTI_WritePosixVar;

            FTI_Exec->finalizeICPFunc[LOCAL] = FTI_FinalizePosixICP;
            FTI_Exec->finalizeICPFunc[GLOBAL] = FTI_FinalizePosixICP;

            break;
#endif
        case FTI_IO_FTIFF:
            FTI_Exec->ckptFunc[GLOBAL] = FTIFF_WriteFTIFF;
            FTI_Exec->ckptFunc[LOCAL] = FTIFF_WriteFTIFF;

            FTI_Exec->initICPFunc[LOCAL] = FTI_InitFtiffICP; 
            FTI_Exec->initICPFunc[GLOBAL] = FTI_InitFtiffICP;

            FTI_Exec->writeVarICPFunc[LOCAL] = FTI_WriteFtiffVar;
            FTI_Exec->writeVarICPFunc[GLOBAL] = FTI_WriteFtiffVar;

            FTI_Exec->finalizeICPFunc[LOCAL] = FTI_FinalizeFtiffICP;
            FTI_Exec->finalizeICPFunc[GLOBAL] = FTI_FinalizeFtiffICP;

            break;
#ifdef ENABLE_HDF5 //If HDF5 is installed
        case FTI_IO_HDF5:
            ftiIO[LOCAL].initCKPT = FTI_InitHDF5; 
            ftiIO[LOCAL].WriteData = FTI_WriteHDF5Data; 
            ftiIO[LOCAL].finCKPT= FTI_HDF5Close; 
            ftiIO[LOCAL].getPos	= FTI_GetHDF5FilePos; 
            ftiIO[LOCAL].finIntegrity = FTI_dummy; 

            ftiIO[GLOBAL].initCKPT = FTI_InitHDF5; 
            ftiIO[GLOBAL].WriteData = FTI_WriteHDF5Data; 
            ftiIO[GLOBAL].finCKPT= FTI_HDF5Close; 
            ftiIO[GLOBAL].getPos	= FTI_GetHDF5FilePos; 
            ftiIO[GLOBAL].finIntegrity = FTI_dummy; 


            FTI_Exec->ckptFunc[GLOBAL] = FTI_Write;
            FTI_Exec->ckptFunc[LOCAL] = FTI_Write;

            FTI_Exec->initICPFunc[LOCAL] = FTI_startICP; 
            FTI_Exec->initICPFunc[GLOBAL] = FTI_startICP;

            FTI_Exec->writeVarICPFunc[LOCAL] = FTI_WriteVar;
            FTI_Exec->writeVarICPFunc[GLOBAL] = FTI_WriteVar;

            FTI_Exec->finalizeICPFunc[LOCAL] = FTI_FinishICP;
            FTI_Exec->finalizeICPFunc[GLOBAL] = FTI_FinishICP;
            break;
#endif
    }
    return FTI_SCES;
}
