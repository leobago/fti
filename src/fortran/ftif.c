/** @brief Interface to call FTI from Fortran 
 * 
 * @file   ftif.c
 * @author Faysal Boui <faysal.boui@cea.fr>
 * @author Julien Bigot <julien.bigot@cea.fr>
 * @date 2013-08-01
 */

#include "fti.h"
#include "interface.h"
#include "ftif.h"

/** @brief Fortran wrapper for FTI_Init, Initializes FTI.
 * 
 * @return the error status of FTI
 * @param configFile (IN) the name of the configuration file as a
 *        \0 terminated string
 * @param globalComm (INOUT) the "world" communicator, FTI will replace it
 *        with a communicator where its own processes have been removed.
 */
int FTI_Init_fort_wrapper(char* configFile, int* globalComm)
{
    int ierr = FTI_Init(configFile, MPI_Comm_f2c(*globalComm));
    *globalComm = MPI_Comm_c2f(FTI_COMM_WORLD);
    return ierr;
}

/**
 *   @brief      Initializes a data type.
 *   @param      type            The data type to be intialized.
 *   @param      size            The size of the data type to be intialized.
 *   @return     integer         FTI_SCES if successful.
 *
 *   This function initalizes a data type. the only information needed is the
 *   size of the data type, the rest is black box for FTI.
 *
 **/
int FTI_InitType_wrapper(FTIT_type** type, int size)
{
    *type = talloc(FTIT_type, 1);
    return FTI_InitType(*type, size);
}

/**
 @brief      Stores or updates a pointer to a variable that needs to be protected.
 @param      id              ID for searches and update.
 @param      ptr             Pointer to the data structure.
 @param      count           Number of elements in the data structure.
 @param      type            Type of elements in the data structure.
 @return     integer         FTI_SCES if successful.

 This function stores a pointer to a data structure, its size, its ID,
 its number of elements and the type of the elements. This list of
 structures is the data that will be stored during a checkpoint and
 loaded during a recovery.

 **/
/*-------------------------------------------------------------------------*/
int FTI_Protect_wrapper(int id, void* ptr, long count, FTIT_type* type)
{
    return FTI_Protect(id, ptr, count, *type);
}
