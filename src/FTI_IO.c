#include "FTI_IO.h"
#include "utility.h"
#include "IO/ftiff.h"
#include "interface.h"


FTIT_IO ftiIO[2];

void FTI_dummy(unsigned char *data, void* a){
    return;
}


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
