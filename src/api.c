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
 *  @file   api.c
 *  @date   October, 2017
 *  @brief  API functions for the FTI library.
 */


#include "interface.h"

/** General configuration information used by FTI.                         */
static FTIT_configuration FTI_Conf;

/** Checkpoint information for all levels of checkpoint.                   */
static FTIT_checkpoint FTI_Ckpt[5];

/** Dynamic information for this execution.                                */
static FTIT_execution FTI_Exec;

/** Topology of the system.                                                */
static FTIT_topology FTI_Topo;

/** Array of datasets and all their internal information.                  */
static FTIT_dataset FTI_Data[FTI_BUFS];

/** SDC injection model and all the required information.                  */
static FTIT_injection FTI_Inje;

/** MPI communicator that splits the global one into app and FTI appart.   */
MPI_Comm FTI_COMM_WORLD;

/** FTI data type for chars.                                               */
FTIT_type FTI_CHAR;
/** FTI data type for short integers.                                      */
FTIT_type FTI_SHRT;
/** FTI data type for integers.                                            */
FTIT_type FTI_INTG;
/** FTI data type for long integers.                                       */
FTIT_type FTI_LONG;
/** FTI data type for unsigned chars.                                      */
FTIT_type FTI_UCHR;
/** FTI data type for unsigned short integers.                             */
FTIT_type FTI_USHT;
/** FTI data type for unsigned integers.                                   */
FTIT_type FTI_UINT;
/** FTI data type for unsigned long integers.                              */
FTIT_type FTI_ULNG;
/** FTI data type for single floating point.                               */
FTIT_type FTI_SFLT;
/** FTI data type for double floating point.                               */
FTIT_type FTI_DBLE;
/** FTI data type for long doble floating point.                           */
FTIT_type FTI_LDBE;


/*-------------------------------------------------------------------------*/
/**
    @brief      Initializes FTI.
    @param      configFile      FTI configuration file.
    @param      globalComm      Main MPI communicator of the application.
    @return     integer         FTI_SCES if successful.

    This function initializes the FTI context and prepares the heads to wait
    for checkpoints. FTI processes should never get out of this function. In
    case of a restart, checkpoint files should be recovered and in place at the
    end of this function.

 **/
/*-------------------------------------------------------------------------*/
int FTI_Init(char* configFile, MPI_Comm globalComm)
{
    FTI_InitExecVars(&FTI_Conf, &FTI_Exec, &FTI_Topo, FTI_Ckpt, &FTI_Inje);
    FTI_Exec.globalComm = globalComm;
    MPI_Comm_rank(FTI_Exec.globalComm, &FTI_Topo.myRank);
    MPI_Comm_size(FTI_Exec.globalComm, &FTI_Topo.nbProc);
    snprintf(FTI_Conf.cfgFile, FTI_BUFS, "%s", configFile);
    FTI_Conf.verbosity = 1; //Temporary needed for output in FTI_LoadConf.
    FTI_Exec.initSCES = 0;
    FTI_Inje.timer = MPI_Wtime();
    FTI_COMM_WORLD = globalComm; // Temporary before building topology. Needed in FTI_LoadConf and FTI_Topology to communicate.
    FTI_Topo.splitRank = FTI_Topo.myRank; // Temporary before building topology. Needed in FTI_Print.
    int res = FTI_Try(FTI_LoadConf(&FTI_Conf, &FTI_Exec, &FTI_Topo, FTI_Ckpt, &FTI_Inje), "load configuration.");
    if (res == FTI_NSCS) {
        return FTI_NSCS;
    }
    res = FTI_Try(FTI_Topology(&FTI_Conf, &FTI_Exec, &FTI_Topo), "build topology.");
    if (res == FTI_NSCS) {
        return FTI_NSCS;
    }
    FTI_Try(FTI_InitBasicTypes(FTI_Data), "create the basic data types.");
    if (FTI_Topo.myRank == 0) {
        FTI_Try(FTI_UpdateConf(&FTI_Conf, &FTI_Exec, FTI_Exec.reco), "update configuration file.");
    }
    MPI_Barrier(FTI_Exec.globalComm); //wait for myRank == 0 process to save config file
    FTI_MallocMeta(&FTI_Exec, &FTI_Topo);
    res = FTI_Try(FTI_LoadMeta(&FTI_Conf, &FTI_Exec, &FTI_Topo, FTI_Ckpt), "load metadata");
    if (res == FTI_NSCS) {
        FTI_FreeMeta(&FTI_Exec);
        return FTI_NSCS;
    }
    FTI_Exec.initSCES = 1;
    if (FTI_Topo.amIaHead) { // If I am a FTI dedicated process
        if (FTI_Exec.reco) {
            res = FTI_Try(FTI_RecoverFiles(&FTI_Conf, &FTI_Exec, &FTI_Topo, FTI_Ckpt), "recover the checkpoint files.");
            if (res != FTI_SCES) {
                FTI_Exec.reco = 0;
                FTI_Exec.initSCES = 2; //Could not recover all ckpt files
            }
        }
        FTI_Listen(&FTI_Conf, &FTI_Exec, &FTI_Topo, FTI_Ckpt); //infinite loop inside, can stop only by callling FTI_Finalize
    }
    else { // If I am an application process
        if (FTI_Exec.reco) {
            res = FTI_Try(FTI_RecoverFiles(&FTI_Conf, &FTI_Exec, &FTI_Topo, FTI_Ckpt), "recover the checkpoint files.");
            FTI_Exec.ckptCnt = FTI_Exec.ckptID;
            FTI_Exec.ckptCnt++;
            if (res != FTI_SCES) {
                FTI_Exec.reco = 0;
                FTI_Exec.initSCES = 2; //Could not recover all ckpt files
                FTI_Print("FTI has been initialized.", FTI_INFO);
                return FTI_NREC;
            }
        }
        FTI_Print("FTI has been initialized.", FTI_INFO);
        return FTI_SCES;
    }
}

/*-------------------------------------------------------------------------*/
/**
    @brief      It returns the current status of the recovery flag.
    @return     integer         FTI_Exec.reco.

    This function returns the current status of the recovery flag.

 **/
/*-------------------------------------------------------------------------*/
int FTI_Status()
{
    return FTI_Exec.reco;
}

/*-------------------------------------------------------------------------*/
/**
    @brief      It initializes a data type.
    @param      type            The data type to be intialized.
    @param      size            The size of the data type to be intialized.
    @return     integer         FTI_SCES if successful.

    This function initalizes a data type. the only information needed is the
    size of the data type, the rest is black box for FTI.

 **/
/*-------------------------------------------------------------------------*/
int FTI_InitType(FTIT_type* type, int size)
{
    type->id = FTI_Exec.nbType;
    type->size = size;
    FTI_Exec.nbType = FTI_Exec.nbType + 1;
    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
    @brief      It sets/resets the pointer and type to a protected variable.
    @param      id              ID for searches and update.
    @param      ptr             Pointer to the data structure.
    @param      count           Number of elements in the data structure.
    @param      type            Type of elements in the data structure.
    @return     integer         FTI_SCES if successful.

    This function stores a pointer to a data structure, its size, its ID,
    its number of elements and the type of the elements. This list of
    structures is the data that will be stored during a checkpoint and
    loaded during a recovery. It resets the pointer to a data structure,
    its size, its number of elements and the type of the elements if the
    dataset was already previously registered.

 **/
/*-------------------------------------------------------------------------*/
int FTI_Protect(int id, void* ptr, long count, FTIT_type type)
{
    if (FTI_Exec.initSCES == 0) {
        FTI_Print("FTI is not initialized.", FTI_WARN);
        return FTI_NSCS;
    }

    char str[FTI_BUFS]; //For console output

    int i;
    for (i = 0; i < FTI_BUFS; i++) {
        if (id == FTI_Data[i].id) { //Search for dataset with given id
            long prevSize = FTI_Data[i].size;
            FTI_Data[i].ptr = ptr;
            FTI_Data[i].count = count;
            FTI_Data[i].type = type;
            FTI_Data[i].eleSize = type.size;
            FTI_Data[i].size = type.size * count;
            FTI_Exec.ckptSize = FTI_Exec.ckptSize + ((type.size * count) - prevSize);
            sprintf(str, "Variable ID %d reseted. Current ckpt. size per rank is %.2fMB.", id, (float) FTI_Exec.ckptSize / (1024.0 * 1024.0));
            FTI_Print(str, FTI_DBUG);
            return FTI_SCES;
        }
    }
    //Id could not be found in datasets

    //If too many variables exit FTI.
    if (FTI_Exec.nbVar >= FTI_BUFS) {
        FTI_Print("Unable to register variable. Too many variables already registered.", FTI_WARN);
        return FTI_NSCS;
    }

    //Adding new variable to protect
    FTI_Data[FTI_Exec.nbVar].id = id;
    FTI_Data[FTI_Exec.nbVar].ptr = ptr;
    FTI_Data[FTI_Exec.nbVar].count = count;
    FTI_Data[FTI_Exec.nbVar].type = type;
    FTI_Data[FTI_Exec.nbVar].eleSize = type.size;
    FTI_Data[FTI_Exec.nbVar].size = type.size * count;
    FTI_Exec.nbVar = FTI_Exec.nbVar + 1;
    FTI_Exec.ckptSize = FTI_Exec.ckptSize + (type.size * count);
    sprintf(str, "Variable ID %d to protect. Current ckpt. size per rank is %.2fMB.", id, (float) FTI_Exec.ckptSize / (1024.0 * 1024.0));
    FTI_Print(str, FTI_INFO);
    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
    @brief      Returns size saved in metadata of variable
    @param      id              Variable ID.
    @return     long            Returns size of variable or 0 if size not saved.

    This function returns size of variable of given ID that is saved in metadata.
    This may be different from size of variable that is in the program. If this
    function it's called when recovery it returns size from metadata file, if it's
    called after checkpoint it returns size saved in temporary metadata. If there
    is no size saved in metadata it returns 0.
 **/
/*-------------------------------------------------------------------------*/
long FTI_GetStoredSize(int id)
{
    if (FTI_Exec.initSCES == 0) {
        FTI_Print("FTI is not initialized.", FTI_WARN);
        return 0;
    }

    int i;
    //Search first in temporary metadata (always the newest)
    for (i = 0; i < FTI_BUFS; i++) {
        if (FTI_Exec.meta[0].varID[i] == id) {
            if (FTI_Exec.meta[0].varSize[i] != 0) {
                return FTI_Exec.meta[0].varSize[i];
            }
            break;
        }
    }
    //If couldn't find in temporary metadata, search in last level checkpoint
    //(this means no checkpoint was taken in current execution)
    for (i = 0; i < FTI_BUFS; i++) {
        if (FTI_Exec.meta[FTI_Exec.ckptLvel].varID[i] == id) {
            return FTI_Exec.meta[FTI_Exec.ckptLvel].varSize[i];
        }
    }
    return 0;
}

/*-------------------------------------------------------------------------*/
/**
    @brief      Reallocates dataset to last checkpoint size.
    @param      id              Variable ID.
    @param      ptr             Pointer to the variable.
    @return     ptr             Pointer if successful, NULL otherwise
    This function loads the checkpoint data size from the metadata
    file, reallacates memory and updates data size information.
 **/
/*-------------------------------------------------------------------------*/
void* FTI_Realloc(int id, void* ptr) {
    if (FTI_Exec.initSCES == 0) {
        FTI_Print("FTI is not initialized.", FTI_WARN);
        return ptr;
    }

    FTI_Print("Trying to reallocate dataset.", FTI_DBUG);
    if (FTI_Exec.reco) {
        char fn[FTI_BUFS], str[FTI_BUFS];
        int i;
        for (i = 0; i < FTI_BUFS; i++) {
            if (id == FTI_Data[i].id) {
                long oldSize = FTI_Data[i].size;
                FTI_Data[i].size = FTI_Exec.meta[FTI_Exec.ckptLvel].varSize[i];
                sprintf(str, "Reallocated size: %ld", FTI_Data[i].size);
                FTI_Print(str, FTI_DBUG);
                if (FTI_Data[i].size == 0) {
                    sprintf(str, "Cannot allocate 0 size.");
                    FTI_Print(str, FTI_DBUG);
                    return ptr;
                }
                ptr = realloc (ptr, FTI_Data[i].size);
                FTI_Data[i].ptr = ptr;
                FTI_Data[i].count = FTI_Data[i].size / FTI_Data[i].eleSize;
                FTI_Exec.ckptSize += FTI_Data[i].size - oldSize;
                sprintf(str, "Dataset #%d reallocated.", FTI_Data[i].id);
                FTI_Print(str, FTI_INFO);
                break;
            }
        }
    }
    else {
        FTI_Print("This is not a recovery. Couldn't reallocate memory.", FTI_WARN);
    }
    return ptr;
}

/*-------------------------------------------------------------------------*/
/**
    @brief      It corrupts a bit of the given float.
    @param      target          Pointer to the float to corrupt.
    @param      bit             Position of the bit to corrupt.
    @return     integer         FTI_SCES if successful.

    This function filps the bit of the target float.

 **/
/*-------------------------------------------------------------------------*/
int FTI_FloatBitFlip(float* target, int bit)
{
    if (bit >= 32 || bit < 0) {
        return FTI_NSCS;
    }
    int* corIntPtr = (int*)target;
    int corInt = *corIntPtr;
    corInt = corInt ^ (1 << bit);
    corIntPtr = &corInt;
    float* fp = (float*)corIntPtr;
    *target = *fp;
    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
    @brief      It corrupts a bit of the given float.
    @param      target          Pointer to the float to corrupt.
    @param      bit             Position of the bit to corrupt.
    @return     integer         FTI_SCES if successful.

    This function filps the bit of the target float.

 **/
/*-------------------------------------------------------------------------*/
int FTI_DoubleBitFlip(double* target, int bit)
{
    if (bit >= 64 || bit < 0) {
        return FTI_NSCS;
    }
    FTIT_double myDouble;
    myDouble.value = *target;
    int bitf = (bit >= 32) ? bit - 32 : bit;
    int half = (bit >= 32) ? 1 : 0;
    FTI_FloatBitFlip(&(myDouble.floatval[half]), bitf);
    *target = myDouble.value;
    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
    @brief      Bit-flip injection following the injection instructions.
    @param      datasetID       ID of the dataset where to inject.
    @return     integer         FTI_SCES if successful.

    This function injects the given number of bit-flips, at the given
    frequency and in the given location (rank, dataset, bit position).

 **/
/*-------------------------------------------------------------------------*/
int FTI_BitFlip(int datasetID)
{
    if (FTI_Exec.initSCES == 0) {
        FTI_Print("FTI is not initialized.", FTI_WARN);
        return FTI_NSCS;
    }

    if (FTI_Inje.rank == FTI_Topo.splitRank) {
        if (datasetID >= FTI_Exec.nbVar) {
            return FTI_NSCS;
        }
        if (FTI_Inje.counter < FTI_Inje.number) {
            if ((MPI_Wtime() - FTI_Inje.timer) > FTI_Inje.frequency) {
                if (FTI_Inje.index < FTI_Data[datasetID].count) {
                    char str[FTI_BUFS];
                    if (FTI_Data[datasetID].type.id == 9) { // If it is a double
                        double* target = FTI_Data[datasetID].ptr + FTI_Inje.index;
                        double ori = *target;
                        int res = FTI_DoubleBitFlip(target, FTI_Inje.position);
                        FTI_Inje.counter = (res == FTI_SCES) ? FTI_Inje.counter + 1 : FTI_Inje.counter;
                        FTI_Inje.timer = (res == FTI_SCES) ? MPI_Wtime() : FTI_Inje.timer;
                        sprintf(str, "Injecting bit-flip in dataset %d, index %d, bit %d : %f => %f",
                            datasetID, FTI_Inje.index, FTI_Inje.position, ori, *target);
                        FTI_Print(str, FTI_WARN);
                        return res;
                    }
                    if (FTI_Data[datasetID].type.id == 8) { // If it is a float
                        float* target = FTI_Data[datasetID].ptr + FTI_Inje.index;
                        float ori = *target;
                        int res = FTI_FloatBitFlip(target, FTI_Inje.position);
                        FTI_Inje.counter = (res == FTI_SCES) ? FTI_Inje.counter + 1 : FTI_Inje.counter;
                        FTI_Inje.timer = (res == FTI_SCES) ? MPI_Wtime() : FTI_Inje.timer;
                        sprintf(str, "Injecting bit-flip in dataset %d, index %d, bit %d : %f => %f",
                            datasetID, FTI_Inje.index, FTI_Inje.position, ori, *target);
                        FTI_Print(str, FTI_WARN);
                        return res;
                    }
                }
            }
        }
    }
    return FTI_NSCS;
}

/*-------------------------------------------------------------------------*/
/**
    @brief      It takes the checkpoint and triggers the post-ckpt. work.
    @param      id              Checkpoint ID.
    @param      level           Checkpoint level.
    @return     integer         FTI_SCES if successful.

    This function starts by blocking on a receive if the previous ckpt. was
    offline. Then, it updates the ckpt. information. It writes down the ckpt.
    data, creates the metadata and the post-processing work. This function
    is complementary with the FTI_Listen function in terms of communications.

 **/
/*-------------------------------------------------------------------------*/
int FTI_Checkpoint(int id, int level)
{
    if (FTI_Exec.initSCES == 0) {
        FTI_Print("FTI is not initialized.", FTI_WARN);
        return FTI_NSCS;
    }

    if ((level < 1) || (level > 4)) {
        FTI_Print("Level of checkpoint must be 1, 2, 3 or 4.", FTI_WARN);
        return FTI_NSCS;
    }

    char str[FTI_BUFS]; //For console output
    int ckptFirst = !FTI_Exec.ckptID; //ckptID = 0 if first checkpoint
    FTI_Exec.ckptID = id;

    double t0 = MPI_Wtime(); //Start time
    if (FTI_Exec.wasLastOffline == 1) { // Block until previous checkpoint is done (Async. work)
        int lastLevel;
        MPI_Recv(&lastLevel, 1, MPI_INT, FTI_Topo.headRank, FTI_Conf.tag, FTI_Exec.globalComm, MPI_STATUS_IGNORE);
        if (lastLevel != FTI_NSCS) { //Head sends level of checkpoint if post-processing succeed, FTI_NSCS Otherwise
            FTI_Exec.lastCkptLvel = lastLevel; //Store last successful post-processing checkpoint level
            sprintf(str, "LastCkptLvel received from head: %d", lastLevel);
            FTI_Print(str, FTI_DBUG);
        } else {
            FTI_Print("Head failed to do post-processing after previous checkpoint.", FTI_WARN);
        }
    }

    double t1 = MPI_Wtime(); //Time after waiting for head to done previous post-processing
    int lastCkptLvel = FTI_Exec.ckptLvel; //Store last successful writing checkpoint level in case of failure
    FTI_Exec.ckptLvel = level; //For FTI_WriteCkpt
    int res = FTI_Try(FTI_WriteCkpt(&FTI_Conf, &FTI_Exec, &FTI_Topo, FTI_Ckpt, FTI_Data), "write the checkpoint.");
    double t2 = MPI_Wtime(); //Time after writing checkpoint
    if (!FTI_Ckpt[FTI_Exec.ckptLvel].isInline) { // If postCkpt. work is Async. then send message
        FTI_Exec.wasLastOffline = 1;
        int value = FTI_BASE + FTI_Exec.ckptLvel; //Token to send to head
        if (res != FTI_SCES) { //If Writing checkpoint failed
            FTI_Exec.ckptLvel = lastCkptLvel; //Set previous ckptLvel
            value = FTI_REJW; //Send reject checkpoint token to head
        }
        MPI_Send(&value, 1, MPI_INT, FTI_Topo.headRank, FTI_Conf.tag, FTI_Exec.globalComm);
    }
    else { //If post-processing is inline
        FTI_Exec.wasLastOffline = 0;
        if (res != FTI_SCES) { //If Writing checkpoint failed
            FTI_Exec.ckptLvel = FTI_REJW - FTI_BASE; //The same as head call FTI_PostCkpt with reject ckptLvel if not success
        }
        res = FTI_Try(FTI_PostCkpt(&FTI_Conf, &FTI_Exec, &FTI_Topo, FTI_Ckpt), "postprocess the checkpoint.");
        if (res == FTI_SCES) { //If post-processing succeed
            FTI_Exec.lastCkptLvel = FTI_Exec.ckptLvel; //Store last successful post-processing checkpoint level
        }
    }
    double t3 = MPI_Wtime(); //Time after post-processing
    if (res != FTI_SCES) {
        sprintf(str, "Checkpoint with ID %d at Level %d failed.", FTI_Exec.ckptID, FTI_Exec.ckptLvel);
        FTI_Print(str, FTI_WARN);
        return FTI_NSCS;
    }

    sprintf(str, "Ckpt. ID %d (L%d) (%.2f MB/proc) taken in %.2f sec. (Wt:%.2fs, Wr:%.2fs, Ps:%.2fs)",
            FTI_Exec.ckptID, FTI_Exec.ckptLvel, FTI_Exec.ckptSize / (1024.0 * 1024.0), t3 - t0, t1 - t0, t2 - t1, t3 - t2);
    FTI_Print(str, FTI_INFO);
    if (ckptFirst && FTI_Topo.splitRank == 0) {
        //Setting recover flag to 1 (to recover from current ckpt level)
        FTI_Try(FTI_UpdateConf(&FTI_Conf, &FTI_Exec, 1), "update configuration file.");
        FTI_Exec.initSCES = 1; //in case FTI couldn't recover all ckpt files in FTI_Init
    }
    return FTI_DONE;
}

/*-------------------------------------------------------------------------*/
/**
    @brief      It loads the checkpoint data.
    @return     integer         FTI_SCES if successful.

    This function loads the checkpoint data from the checkpoint file and
    it updates some basic checkpoint information.

 **/
/*-------------------------------------------------------------------------*/
int FTI_Recover()
{
    if (FTI_Exec.initSCES == 0) {
        FTI_Print("FTI is not initialized.", FTI_WARN);
        return FTI_NSCS;
    }
    if (FTI_Exec.initSCES == 2) {
        FTI_Print("No checkpoint files to make recovery.", FTI_WARN);
        return FTI_NSCS;
    }

    char fn[FTI_BUFS]; //Path to the checkpoint file
    char str[FTI_BUFS]; //For console output

    //Check if nubmer of protected variables matches
    if (FTI_Exec.nbVar != FTI_Exec.meta[FTI_Exec.ckptLvel].nbVar[0]) {
        sprintf(str, "Checkpoint has %d protected variables, but FTI protects %d.",
                FTI_Exec.meta[FTI_Exec.ckptLvel].nbVar[0], FTI_Exec.nbVar);
        FTI_Print(str, FTI_WARN);
        return FTI_NREC;
    }
    //Check if sizes of protected variables matches
    int i;
    for (i = 0; i < FTI_Exec.nbVar; i++) {
        if (FTI_Data[i].size != FTI_Exec.meta[FTI_Exec.ckptLvel].varSize[i]) {
            sprintf(str, "Cannot recover %ld bytes to protected variable (ID %d) size: %ld",
                    FTI_Exec.meta[FTI_Exec.ckptLvel].varSize[i], FTI_Exec.meta[FTI_Exec.ckptLvel].varID[i],
                    FTI_Data[i].size);
            FTI_Print(str, FTI_WARN);
            return FTI_NREC;
        }
    }
    //Recovering from local for L4 case in FTI_Recover
    if (FTI_Exec.ckptLvel == 4) {
        sprintf(fn, "%s/%s", FTI_Ckpt[1].dir, FTI_Exec.meta[1].ckptFile);
    }
    else {
        sprintf(fn, "%s/%s", FTI_Ckpt[FTI_Exec.ckptLvel].dir, FTI_Exec.meta[FTI_Exec.ckptLvel].ckptFile);
    }

    if (FTI_Conf.ioMode == FTI_IO_FTIFF) {

        // get filesize
        struct stat st;
        stat(fn, &st);
        int ferr;
        char strerr[FTI_BUFS];

        // block size for memcpy of pointer.
        long membs = 1024*1024*16; // 16 MB
        long cpybuf, cpynow, cpycnt;

        // open checkpoint file for read only
        int fd = open( fn, O_RDONLY, 0 );
        if (fd == -1) {
            sprintf( strerr, "FTIFF: Recovery - could not open '%s' for reading.", fn);
            FTI_Print(strerr, FTI_EROR);
            return FTI_NREC;
        }

        // map file into memory
        char* fmmap = (char*) mmap(0, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
        if (fmmap == MAP_FAILED) {
            sprintf( strerr, "FTIFF: Recovery - could not map '%s' to memory.", fn);
            FTI_Print(strerr, FTI_EROR);
            close(fd);
            return FTI_NREC;
        }

        // file is mapped, we can close it.
        close(fd);

        FTIT_db *currentdb, *nextdb;
        FTIT_dbvar *currentdbvar = NULL;
        char *destptr, *srcptr;
        int dbvar_idx, pvar_idx, dbcounter=0;

        // MD5 context for checksum of data chunks
        MD5_CTX mdContext;
        unsigned char hash[MD5_HASH_LENGTH];

        long endoffile = 0, mdoffset;

        int isnextdb;

        currentdb = (FTIT_db*) malloc( sizeof(FTIT_db) );
        if (!currentdb) {
            sprintf( strerr, "FTIFF: Recovery - failed to allocate %ld bytes for 'currentdb'", sizeof(FTIT_db));
            FTI_Print(strerr, FTI_EROR);
            return FTI_NREC;
        }

        FTI_Exec.firstdb = currentdb;
        FTI_Exec.firstdb->next = NULL;
        FTI_Exec.firstdb->previous = NULL;

        do {

            nextdb = (FTIT_db*) malloc( sizeof(FTIT_db) );
            if (!currentdb) {
                sprintf( strerr, "FTIFF: Recovery - failed to allocate %ld bytes for 'nextdb'", sizeof(FTIT_db));
                FTI_Print(strerr, FTI_EROR);
                return FTI_NREC;
            }

            isnextdb = 0;

            mdoffset = endoffile;
            
            memcpy( &(currentdb->numvars), fmmap+mdoffset, sizeof(int) ); 
            mdoffset += sizeof(int);
            memcpy( &(currentdb->dbsize), fmmap+mdoffset, sizeof(long) );
            mdoffset += sizeof(long);
        
            sprintf(str, "FTIFF: Recovery - dataBlock:%i, dbsize: %ld, numvars: %i.", 
                    dbcounter, currentdb->dbsize, currentdb->numvars);
            FTI_Print(str, FTI_DBUG);

            currentdb->dbvars = (FTIT_dbvar*) malloc( sizeof(FTIT_dbvar) * currentdb->numvars );
            if (!currentdb) {
                sprintf( strerr, "FTIFF: Recovery - failed to allocate %ld bytes for 'currentdb->dbvars'", sizeof(FTIT_dbvar));
                FTI_Print(strerr, FTI_EROR);
                return FTI_NREC;
            }

            for(dbvar_idx=0;dbvar_idx<currentdb->numvars;dbvar_idx++) {

                currentdbvar = &(currentdb->dbvars[dbvar_idx]);
                
                memcpy( currentdbvar, fmmap+mdoffset, sizeof(FTIT_dbvar) );
                mdoffset += sizeof(FTIT_dbvar);
                
                // get source and destination pointer
                destptr = (char*) FTI_Data[currentdbvar->idx].ptr + currentdbvar->dptr;
                srcptr = (char*) fmmap + currentdbvar->fptr;
                
                MD5_Init( &mdContext );
                cpycnt = 0;
                while ( cpycnt < currentdbvar->chunksize ) {
                    cpybuf = currentdbvar->chunksize - cpycnt;
                    cpynow = ( cpybuf > membs ) ? membs : cpybuf;
                    cpycnt += cpynow;
                    memcpy( destptr, srcptr, cpynow );
                    MD5_Update( &mdContext, destptr, cpynow );
                    destptr += cpynow;
                    srcptr += cpynow;
                }
                
                // debug information
                sprintf(str, "FTIFF: Recovery -  dataBlock:%i/dataBlockVar%i id: %i, idx: %i"
                        ", destptr: %ld, fptr: %ld, chunksize: %ld, "
                        "base_ptr: 0x%" PRIxPTR " ptr_pos: 0x%" PRIxPTR ".", 
                        dbcounter, dbvar_idx,  
                        currentdbvar->id, currentdbvar->idx, currentdbvar->dptr,
                        currentdbvar->fptr, currentdbvar->chunksize,
                        FTI_Data[currentdbvar->idx].ptr, destptr);
                FTI_Print(str, FTI_DBUG);

                MD5_Final( hash, &mdContext );
                
                if ( memcmp( currentdbvar->hash, hash, MD5_HASH_LENGTH ) != 0 ) {
                    sprintf( strerr, "FTIFF: Recovery - dataset with id:%i has been corrupted! Discard recovery.", currentdbvar->id);
                    FTI_Print(strerr, FTI_WARN);
                    return FTI_NREC;
                }
            
            }

            endoffile += currentdb->dbsize;

            if ( endoffile < st.st_size ) {
                memcpy( nextdb, fmmap+endoffile, FTI_dbstructsize );
                currentdb->next = nextdb;
                nextdb->previous = currentdb;
                currentdb = nextdb;
                isnextdb = 1;
            }

            dbcounter++;

        } while( isnextdb );

        FTI_Exec.lastdb = currentdb;
        FTI_Exec.lastdb->next = NULL;
       
        // unmap memory
        if ( munmap( fmmap, st.st_size ) == -1 ) {
            FTI_Print("FTIFF: Recovery - unable to unmap memory", FTI_WARN);
        }

        FTI_Exec.reco = 0;

    } else {

        FILE* fd = fopen(fn, "rb");

        sprintf(str, "Trying to load FTI checkpoint file (%s)...", fn);
        FTI_Print(str, FTI_DBUG);

        if (fd == NULL) {
            FTI_Print("Could not open FTI checkpoint file.", FTI_EROR);
            return FTI_NREC;
        }

        for (i = 0; i < FTI_Exec.nbVar; i++) {
            fread(FTI_Data[i].ptr, 1, FTI_Data[i].size, fd);
            if (ferror(fd)) {
                FTI_Print("Could not read FTI checkpoint file.", FTI_EROR);
                fclose(fd);
                return FTI_NREC;
            }
        }
        if (fclose(fd) != 0) {
            FTI_Print("Could not close FTI checkpoint file.", FTI_EROR);
            return FTI_NREC;
        }
        FTI_Exec.reco = 0;

    }

    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
    @brief      Takes an FTI snapshot or recovers the data if it is a restart.
    @return     integer         FTI_SCES if successful.

    This function loads the checkpoint data from the checkpoint file in case
    of restart. Otherwise, it checks if the current iteration requires
    checkpointing, if it does it checks which checkpoint level, write the
    data in the files and it communicates with the head of the node to inform
    that a checkpoint has been taken. Checkpoint ID and counters are updated.

 **/
/*-------------------------------------------------------------------------*/
int FTI_Snapshot()
{
    if (FTI_Exec.initSCES == 0) {
        FTI_Print("FTI is not initialized.", FTI_WARN);
        return FTI_NSCS;
    }

    int i, res, level = -1;

    if (FTI_Exec.reco) { // If this is a recovery load icheckpoint data
        res = FTI_Try(FTI_Recover(), "recover the checkpointed data.");
        if (res == FTI_NREC) {
            return FTI_NREC;
        }
    }
    else { // If it is a checkpoint test
        res = FTI_SCES;
        FTI_UpdateIterTime(&FTI_Exec);
        if (FTI_Exec.ckptNext == FTI_Exec.ckptIcnt) { // If it is time to check for possible ckpt. (every minute)
            FTI_Print("Checking if it is time to checkpoint.", FTI_DBUG);
            if (FTI_Exec.globMeanIter > 60) {
                FTI_Exec.minuteCnt = FTI_Exec.totalIterTime/60;
            }
            else {
                FTI_Exec.minuteCnt++; // Increment minute counter
            }
            for (i = 1; i < 5; i++) { // Check ckpt. level
                if (FTI_Ckpt[i].ckptIntv > 0 && FTI_Exec.minuteCnt/(FTI_Ckpt[i].ckptCnt*FTI_Ckpt[i].ckptIntv)) {
                    level = i;
                    FTI_Ckpt[i].ckptCnt++;
                }
            }
            if (level != -1) {
                res = FTI_Try(FTI_Checkpoint(FTI_Exec.ckptCnt, level), "take checkpoint.");
                if (res == FTI_DONE) {
                    FTI_Exec.ckptCnt++;
                }
            }
            FTI_Exec.ckptLast = FTI_Exec.ckptNext;
            FTI_Exec.ckptNext = FTI_Exec.ckptNext + FTI_Exec.ckptIntv;
            FTI_Exec.iterTime = MPI_Wtime(); // Reset iteration duration timer
        }
    }
    return res;
}

/*-------------------------------------------------------------------------*/
/**
    @brief      It closes FTI properly on the application processes.
    @return     integer         FTI_SCES if successful.

    This function notifies the FTI processes that the execution is over, frees
    some data structures and it closes. If this function is not called on the
    application processes the FTI processes will never finish (deadlock).

 **/
/*-------------------------------------------------------------------------*/
int FTI_Finalize()
{
    if (FTI_Exec.initSCES == 0) {
        FTI_Print("FTI is not initialized.", FTI_WARN);
        return FTI_NSCS;
    }

    if (FTI_Topo.amIaHead) {
        FTI_FreeMeta(&FTI_Exec);
        MPI_Barrier(FTI_Exec.globalComm);
        MPI_Finalize();
        exit(0);
    }

    // If there is remaining work to do for last checkpoint
    if (FTI_Exec.wasLastOffline == 1) {
        int lastLevel;
        MPI_Recv(&lastLevel, 1, MPI_INT, FTI_Topo.headRank, FTI_Conf.tag, FTI_Exec.globalComm, MPI_STATUS_IGNORE);
        if (lastLevel != FTI_NSCS) { //Head sends level of checkpoint if post-processing succeed, FTI_NSCS Otherwise
            FTI_Exec.lastCkptLvel = lastLevel;
        }
    }

    // Send notice to the head to stop listening
    if (FTI_Topo.nbHeads == 1) {
        int value = FTI_ENDW;
        MPI_Send(&value, 1, MPI_INT, FTI_Topo.headRank, FTI_Conf.tag, FTI_Exec.globalComm);
    }

    // If we need to keep the last checkpoint and there was a checkpoint
    if (FTI_Conf.saveLastCkpt && FTI_Exec.ckptID > 0) {
            if (FTI_Exec.lastCkptLvel != 4) {
                FTI_Try(FTI_Flush(&FTI_Conf, &FTI_Exec, &FTI_Topo, FTI_Ckpt, FTI_Exec.lastCkptLvel), "save the last ckpt. in the PFS.");
                MPI_Barrier(FTI_COMM_WORLD);
                if (FTI_Topo.splitRank == 0) {
                    if (access(FTI_Ckpt[4].dir, 0) == 0) {
                        FTI_RmDir(FTI_Ckpt[4].dir, 1); //Delete previous L4 checkpoint
                    }
                    if (rename(FTI_Conf.gTmpDir, FTI_Ckpt[4].dir) == -1) { //Move temporary checkpoint to L4 directory
                        FTI_Print("Cannot rename last ckpt. dir", FTI_EROR);
                    }
                    if (access(FTI_Ckpt[4].metaDir, 0) == 0) {
                        FTI_RmDir(FTI_Ckpt[4].metaDir, 1); //Delete previous L4 metadata
                    }
                    if (rename(FTI_Ckpt[FTI_Exec.ckptLvel].metaDir, FTI_Ckpt[4].metaDir) == -1) { //Move temporary metadata to L4 metadata directory
                        FTI_Print("Cannot rename last ckpt. metaDir", FTI_EROR);
                    }
                }
            }
            if (FTI_Topo.splitRank == 0) {
                //Setting recover flag to 2 (to recover from L4, keeped last checkpoint)
                FTI_Try(FTI_UpdateConf(&FTI_Conf, &FTI_Exec, 2), "update configuration file to 2.");
            }
            //Cleaning only local storage
            FTI_Try(FTI_Clean(&FTI_Conf, &FTI_Topo, FTI_Ckpt, 6), "clean local directories");
    } else {
        if (FTI_Conf.saveLastCkpt) { //if there was no saved checkpoint
            FTI_Print("No checkpoint to keep in PFS.", FTI_INFO);
        }
        if (FTI_Topo.splitRank == 0) {
            //Setting recover flag to 0 (no checkpoint files to recover from means no recovery)
            FTI_Try(FTI_UpdateConf(&FTI_Conf, &FTI_Exec, 0), "update configuration file to 0.");
        }
        //Cleaning everything
        FTI_Try(FTI_Clean(&FTI_Conf, &FTI_Topo, FTI_Ckpt, 5), "do final clean.");
    }
    FTI_FreeMeta(&FTI_Exec);
    if( FTI_Conf.ioMode == FTI_IO_FTIFF ) {
        FTI_FreeDbFTIFF(FTI_Exec.lastdb);
    }
    MPI_Barrier(FTI_Exec.globalComm);
    FTI_Print("FTI has been finalized.", FTI_INFO);
    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
    @brief      During the restart, recovers the given variable
    @param      id              Variable to recover
    @return     int             FTI_SCES if successful.

    During a restart process, this function recovers the variable specified
    by the given id. No effect during a regular execution.
    The variable must have already been protected, otherwise, FTI_NSCS is returned.
    Improvements to be done:
    - Open checkpoint file at FTI_Init, close it at FTI_Snapshot
    - Maintain a variable accumulating the offset as variable are protected during
        the restart to avoid doing the loop to calculate the offset in the
        checkpoint file.
 **/
/*-------------------------------------------------------------------------*/
int FTI_RecoverVar(int id){
    if (FTI_Exec.initSCES == 0) {
        FTI_Print("FTI is not initialized.", FTI_WARN);
        return FTI_NSCS;
    }

    if(FTI_Exec.reco==0){
        /* This is not a restart: no actions performed */
        return FTI_SCES;
    }

    if (FTI_Exec.initSCES == 2) {
        FTI_Print("No checkpoint files to make recovery.", FTI_WARN);
        return FTI_NSCS;
    }

    //Check if sizes of protected variables matches
    int i;
    for (i = 0; i < FTI_Exec.nbVar; i++) {
        if (id == FTI_Exec.meta[FTI_Exec.ckptLvel].varID[i]) {
            if (FTI_Data[i].size != FTI_Exec.meta[FTI_Exec.ckptLvel].varSize[i]) {
                char str[FTI_BUFS];
                sprintf(str, "Cannot recover %ld bytes to protected variable (ID %d) size: %ld",
                        FTI_Exec.meta[FTI_Exec.ckptLvel].varSize[i], FTI_Exec.meta[FTI_Exec.ckptLvel].varID[i],
                        FTI_Data[i].size);
                FTI_Print(str, FTI_WARN);
                return FTI_NREC;
            }
        }
    }

    char fn[FTI_BUFS]; //Path to the checkpoint file

    //Recovering from local for L4 case in FTI_Recover
    if (FTI_Exec.ckptLvel == 4) {
        sprintf(fn, "%s/%s", FTI_Ckpt[1].dir, FTI_Exec.meta[1].ckptFile);
    }
    else {
        sprintf(fn, "%s/%s", FTI_Ckpt[FTI_Exec.ckptLvel].dir, FTI_Exec.meta[FTI_Exec.ckptLvel].ckptFile);
    }

    char str[FTI_BUFS];
    sprintf(str, "Trying to load FTI checkpoint file (%s)...", fn);
    FTI_Print(str, FTI_DBUG);

    FILE* fd = fopen(fn, "rb");
    if (fd == NULL) {
        FTI_Print("Could not open FTI checkpoint file.", FTI_EROR);
        return FTI_NREC;
    }

    long offset = 0;
    for (i = 0; i < FTI_Exec.nbVar; i++) {
        if (id == FTI_Exec.meta[FTI_Exec.ckptLvel].varID[i]) {
            sprintf(str, "Recovering var %d ", id);
            FTI_Print(str, FTI_DBUG);
            fseek(fd, offset, SEEK_SET);
            fread(FTI_Data[i].ptr, 1, FTI_Data[i].size, fd);
            if (ferror(fd)) {
                FTI_Print("Could not read FTI checkpoint file.", FTI_EROR);
                fclose(fd);
                return FTI_NREC;
            }
            break;
        }
        offset += FTI_Exec.meta[FTI_Exec.ckptLvel].varSize[i];
    }

    if (i == FTI_Exec.nbVar) {
        FTI_Print("Variables must be protected before they can be recovered.", FTI_EROR);
        fclose(fd);
        return FTI_NREC;
    }
    if (fclose(fd) != 0) {
        FTI_Print("Could not close FTI checkpoint file.", FTI_EROR);
        return FTI_NREC;
    }
    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
    @brief      Prints FTI messages.
    @param      msg             Message to print.
    @param      priority        Priority of the message to be printed.
    @return     void

    This function prints messages depending on their priority and the
    verbosity level set by the user. DEBUG messages are printed by all
    processes with their rank. INFO messages are printed by one process.
    ERROR messages are printed with errno.

 **/
/*-------------------------------------------------------------------------*/
void FTI_Print(char* msg, int priority)
{
    if (priority >= FTI_Conf.verbosity) {
        if (msg != NULL) {
            switch (priority) {
                case FTI_EROR:
                    fprintf(stderr, "[ " RED "FTI Error - %06d" RESET " ] : %s : %s \n", FTI_Topo.myRank, msg, strerror(errno));
                    break;
                case FTI_WARN:
                    fprintf(stdout, "[ " ORG "FTI Warning %06d" RESET " ] : %s \n", FTI_Topo.myRank, msg);
                    break;
                case FTI_INFO:
                    if (FTI_Topo.splitRank == 0) {
                        fprintf(stdout, "[ " GRN "FTI  Information" RESET " ] : %s \n", msg);
                    }
                    break;
                case FTI_DBUG:
                    fprintf(stdout, "[FTI Debug - %06d] : %s \n", FTI_Topo.myRank, msg);
                    break;
            }
        }
    }
    fflush(stdout);
}
