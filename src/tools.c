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
 *  @file   tools.c
 *  @date   October, 2017
 *  @brief  Utility functions for the FTI library.
 */

#include "interface.h"
#include <dirent.h>

/*-------------------------------------------------------------------------*/
/**
    @brief      Init of the static variables
    @return     integer         FTI_SCES if successful.

    This function initializes all static variables to zero.

 **/
/*-------------------------------------------------------------------------*/
int FTI_InitExecVars(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
                 FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt,
                 FTIT_injection* FTI_Inje) {

// +--------- +
// | FTI_Exec |
// +--------- +

/* char[BUFS]       FTI_Exec->id */                 memset(FTI_Exec->id,0x0,FTI_BUFS);
/* int           */ FTI_Exec->ckpt                  =0;
/* int           */ FTI_Exec->reco                  =0;
/* int           */ FTI_Exec->ckptLvel              =0;
/* int           */ FTI_Exec->ckptIntv              =0;
/* int           */ FTI_Exec->lastCkptLvel          =0;
/* int           */ FTI_Exec->wasLastOffline        =0;
/* double        */ FTI_Exec->iterTime              =0;
/* double        */ FTI_Exec->lastIterTime          =0;
/* double        */ FTI_Exec->meanIterTime          =0;
/* double        */ FTI_Exec->globMeanIter          =0;
/* double        */ FTI_Exec->totalIterTime         =0;
/* unsigned int  */ FTI_Exec->syncIter              =0;
/* int           */ FTI_Exec->syncIterMax           =0;
/* unsigned int  */ FTI_Exec->minuteCnt             =0;
/* unsigned int  */ FTI_Exec->ckptCnt               =0;
/* unsigned int  */ FTI_Exec->ckptIcnt              =0;
/* unsigned int  */ FTI_Exec->ckptID                =0;
/* unsigned int  */ FTI_Exec->ckptNext              =0;
/* unsigned int  */ FTI_Exec->ckptLast              =0;
/* long          */ FTI_Exec->ckptSize              =0;
/* unsigned int  */ FTI_Exec->nbVar                 =0;
/* unsigned int  */ FTI_Exec->nbType                =0;
/* int           */ FTI_Exec->metaAlloc             =0;
/* int           */ FTI_Exec->initSCES              =0;
/* FTIT_metadata[5] FTI_Exec->meta */               memset(FTI_Exec->meta,0x0,5*sizeof(FTIT_metadata));
/* MPI_Comm      */ FTI_Exec->globalComm            =0;
/* MPI_Comm      */ FTI_Exec->groupComm             =0;

// +--------- +
// | FTI_Conf |
// +--------- +

/* char[BUFS]       FTI_Conf->cfgFile */            memset(FTI_Conf->cfgFile,0x0,FTI_BUFS);
/* int           */ FTI_Conf->saveLastCkpt          =0;
/* int           */ FTI_Conf->verbosity             =0;
/* int           */ FTI_Conf->blockSize             =0;
/* int           */ FTI_Conf->transferSize          =0;
#ifdef LUSTRE
/* int           */ FTI_Conf->stripeUnit            =0;
/* int           */ FTI_Conf->stripeOffset          =0;
/* int           */ FTI_Conf->stripeFactor          =0;
#endif
/* int           */ FTI_Conf->tag                   =0;
/* int           */ FTI_Conf->test                  =0;
/* int           */ FTI_Conf->l3WordSize            =0;
/* int           */ FTI_Conf->ioMode                =0;
/* char[BUFS]       FTI_Conf->localDir */           memset(FTI_Conf->localDir,0x0,FTI_BUFS);
/* char[BUFS]       FTI_Conf->glbalDir */           memset(FTI_Conf->glbalDir,0x0,FTI_BUFS);
/* char[BUFS]       FTI_Conf->metadDir */           memset(FTI_Conf->metadDir,0x0,FTI_BUFS);
/* char[BUFS]       FTI_Conf->lTmpDir */            memset(FTI_Conf->lTmpDir,0x0,FTI_BUFS);
/* char[BUFS]       FTI_Conf->gTmpDir */            memset(FTI_Conf->gTmpDir,0x0,FTI_BUFS);
/* char[BUFS]       FTI_Conf->mTmpDir */            memset(FTI_Conf->mTmpDir,0x0,FTI_BUFS);

// +--------- +
// | FTI_Topo |
// +--------- +

/* int           */ FTI_Topo->nbProc                =0;
/* int           */ FTI_Topo->nbNodes               =0;
/* int           */ FTI_Topo->myRank                =0;
/* int           */ FTI_Topo->splitRank             =0;
/* int           */ FTI_Topo->nodeSize              =0;
/* int           */ FTI_Topo->nbHeads               =0;
/* int           */ FTI_Topo->nbApprocs             =0;
/* int           */ FTI_Topo->groupSize             =0;
/* int           */ FTI_Topo->sectorID              =0;
/* int           */ FTI_Topo->nodeID                =0;
/* int           */ FTI_Topo->groupID               =0;
/* int           */ FTI_Topo->amIaHead              =0;
/* int           */ FTI_Topo->headRank              =0;
/* int           */ FTI_Topo->nodeRank              =0;
/* int           */ FTI_Topo->groupRank             =0;
/* int           */ FTI_Topo->right                 =0;
/* int           */ FTI_Topo->left                  =0;
/* int[BUFS]        FTI_Topo->body */               memset(FTI_Topo->body,0x0,FTI_BUFS*sizeof(int));

// +--------- +
// | FTI_Ckpt |
// +--------- +

/* char[BUFS]       FTI_Ckpt->dir */                memset(FTI_Ckpt->dir,0x0,FTI_BUFS);
/* char[BUFS]       FTI_Ckpt->metaDir */            memset(FTI_Ckpt->metaDir,0x0,FTI_BUFS);
/* int           */ FTI_Ckpt->isInline              =0;
/* int           */ FTI_Ckpt->ckptIntv              =0;
/* int           */ FTI_Ckpt->ckptCnt               =0;

// +--------- +
// | FTI_Injc |
// +--------- +

//* int           */ FTI_Injc->rank                  =0;
//* int           */ FTI_Injc->index                 =0;
//* int           */ FTI_Injc->position              =0;
//* int           */ FTI_Injc->number                =0;
//* int           */ FTI_Injc->frequency             =0;
//* int           */ FTI_Injc->counter               =0;
//* double        */ FTI_Injc->timer                 =0;

return FTI_SCES;

}


/*-------------------------------------------------------------------------*/
/**
    @brief      It calculates checksum of the checkpoint file.
    @param      FTI_Exec        Execution metadata.
    @param      FTI_Data        Dataset metadata.
    @param      checksum        Checksum that is calculated.
    @return     integer         FTI_SCES if successful.

    This function calculates checksum of the checkpoint file based on
    MD5 algorithm and saves it in checksum.

 **/
/*-------------------------------------------------------------------------*/
int FTI_Checksum(FTIT_execution* FTI_Exec, FTIT_dataset* FTI_Data, char* checksum)
{
    MD5_CTX mdContext;
    MD5_Init (&mdContext);

    int i; //iterate all variables
    for (i = 0; i < FTI_Exec->nbVar; i++) {
        MD5_Update (&mdContext, FTI_Data[i].ptr, FTI_Data[i].size);
    }

    unsigned char hash[MD5_DIGEST_LENGTH];
    MD5_Final (hash, &mdContext);

    for(i = 0; i < MD5_DIGEST_LENGTH - 1; i++)
        sprintf(&checksum[i], "%02x", hash[i]);
    checksum[i] = '\0'; //to get a proper string

    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
    @brief      It compares checksum of the checkpoint file.
    @param      fileName        Filename of the checkpoint.
    @param      checksumToCmp   Checksum to compare.
    @return     integer         FTI_SCES if successful.

    This function calculates checksum of the checkpoint file based on
    MD5 algorithm. It compares calculated hash value with the one saved
    in the file.

 **/
/*-------------------------------------------------------------------------*/
int FTI_VerifyChecksum(char* fileName, char* checksumToCmp)
{
    FILE *fd = fopen(fileName, "rb");
    if (fd == NULL) {
        char str[FTI_BUFS];
        sprintf(str, "FTI failed to open file %s to calculate checksum.", fileName);
        FTI_Print(str, FTI_WARN);
        return FTI_NSCS;
    }

    MD5_CTX mdContext;
    MD5_Init (&mdContext);

    int bytes;
    unsigned char data[CHUNK_SIZE];
    while ((bytes = fread (data, 1, CHUNK_SIZE, fd)) != 0) {
        MD5_Update (&mdContext, data, bytes);
    }
    unsigned char hash[MD5_DIGEST_LENGTH];
    MD5_Final (hash, &mdContext);

    int i;
    char checksum[MD5_DIGEST_LENGTH];   //calculated checksum
    for(i = 0; i < MD5_DIGEST_LENGTH -1; i++)
        sprintf(&checksum[i], "%02x", hash[i]);
    checksum[i] = '\0'; //to get a proper string

    if (memcmp(checksum, checksumToCmp, MD5_DIGEST_LENGTH - 1) != 0) {
        char str[FTI_BUFS];
        sprintf(str, "Checksum do not match. \"%s\" file is corrupted. %s != %s",
            fileName, checksum, checksumToCmp);
        FTI_Print(str, FTI_WARN);

        fclose (fd);

        return FTI_NSCS;
    }

    fclose (fd);

    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
    @brief      It receives the return code of a function and prints a message.
    @param      result          Result to check.
    @param      message         Message to print.
    @return     integer         The same result as passed in parameter.

    This function checks the result from a function and then decides to
    print the message either as a debug message or as a warning.

 **/
/*-------------------------------------------------------------------------*/
int FTI_Try(int result, char* message)
{
    char str[FTI_BUFS];
    if (result == FTI_SCES || result == FTI_DONE) {
        sprintf(str, "FTI succeeded to %s", message);
        FTI_Print(str, FTI_DBUG);
    }
    else {
        sprintf(str, "FTI failed to %s", message);
        FTI_Print(str, FTI_WARN);
        sprintf(str, "Error => %s", strerror(errno));
        FTI_Print(str, FTI_WARN);
    }
    return result;
}

/*-------------------------------------------------------------------------*/
/**
    @brief      It mallocs memory for the metadata.
    @param      FTI_Exec        Execution metadata.
    @param      FTI_Topo        Topology metadata.

    This function mallocs the memory used for the metadata storage.

 **/
/*-------------------------------------------------------------------------*/
void FTI_MallocMeta(FTIT_execution* FTI_Exec, FTIT_topology* FTI_Topo)
{
    int i;
    if (FTI_Topo->amIaHead) {
        for (i = 0; i < 5; i++) {
            FTI_Exec->meta[i].exists = calloc(FTI_Topo->nodeSize, sizeof(int));
            FTI_Exec->meta[i].maxFs = calloc(FTI_Topo->nodeSize, sizeof(long));
            FTI_Exec->meta[i].fs = calloc(FTI_Topo->nodeSize, sizeof(long));
            FTI_Exec->meta[i].pfs = calloc(FTI_Topo->nodeSize, sizeof(long));
            FTI_Exec->meta[i].ckptFile = calloc(FTI_BUFS * FTI_Topo->nodeSize, sizeof(char));
            FTI_Exec->meta[i].nbVar = calloc(FTI_Topo->nodeSize, sizeof(int));
            FTI_Exec->meta[i].varID = calloc(FTI_BUFS * FTI_Topo->nodeSize, sizeof(int));
            FTI_Exec->meta[i].varSize = calloc(FTI_BUFS * FTI_Topo->nodeSize, sizeof(long));
        }
    } else {
        for (i = 0; i < 5; i++) {
            FTI_Exec->meta[i].exists = calloc(1, sizeof(int));
            FTI_Exec->meta[i].maxFs = calloc(1, sizeof(long));
            FTI_Exec->meta[i].fs = calloc(1, sizeof(long));
            FTI_Exec->meta[i].pfs = calloc(1, sizeof(long));
            FTI_Exec->meta[i].ckptFile = calloc(FTI_BUFS, sizeof(char));
            FTI_Exec->meta[i].nbVar = calloc(1, sizeof(int));
            FTI_Exec->meta[i].varID = calloc(FTI_BUFS, sizeof(int));
            FTI_Exec->meta[i].varSize = calloc(FTI_BUFS, sizeof(long));
        }
    }
    FTI_Exec->metaAlloc = 1;
}

/*-------------------------------------------------------------------------*/
/**
    @brief      It frees memory for the metadata.
    @param      FTI_Exec        Execution metadata.

    This function frees the memory used for the metadata storage.

 **/
/*-------------------------------------------------------------------------*/
void FTI_FreeMeta(FTIT_execution* FTI_Exec)
{
    if (FTI_Exec->metaAlloc == 1) {
        int i;
        for (i = 0; i < 5; i++) {
            free(FTI_Exec->meta[i].exists);
            free(FTI_Exec->meta[i].maxFs);
            free(FTI_Exec->meta[i].fs);
            free(FTI_Exec->meta[i].pfs);
            free(FTI_Exec->meta[i].ckptFile);
            free(FTI_Exec->meta[i].nbVar);
            free(FTI_Exec->meta[i].varID);
            free(FTI_Exec->meta[i].varSize);
        }
        FTI_Exec->metaAlloc = 0;
    }
}

/*-------------------------------------------------------------------------*/
/**
    @brief      It creates the basic datatypes and the dataset array.
    @param      FTI_Data        Dataset metadata.
    @return     integer         FTI_SCES if successful.

    This function creates the basic data types using FTIT_Type.

 **/
/*-------------------------------------------------------------------------*/
int FTI_InitBasicTypes(FTIT_dataset* FTI_Data)
{
    int i;
    for (i = 0; i < FTI_BUFS; i++) {
        FTI_Data[i].id = -1;
    }
    FTI_InitType(&FTI_CHAR, sizeof(char));
    FTI_InitType(&FTI_SHRT, sizeof(short));
    FTI_InitType(&FTI_INTG, sizeof(int));
    FTI_InitType(&FTI_LONG, sizeof(long));
    FTI_InitType(&FTI_UCHR, sizeof(unsigned char));
    FTI_InitType(&FTI_USHT, sizeof(unsigned short));
    FTI_InitType(&FTI_UINT, sizeof(unsigned int));
    FTI_InitType(&FTI_ULNG, sizeof(unsigned long));
    FTI_InitType(&FTI_SFLT, sizeof(float));
    FTI_InitType(&FTI_DBLE, sizeof(double));
    FTI_InitType(&FTI_LDBE, sizeof(long double));
#ifdef ENABLE_HDF5
    // Maps FTI types to HDF5 types
    FTI_CHAR.h5datatype = H5T_NATIVE_CHAR;
    FTI_CHAR.h5datatype = H5T_NATIVE_CHAR;
    FTI_SHRT.h5datatype = H5T_NATIVE_SHORT;
    FTI_INTG.h5datatype = H5T_NATIVE_INT;
    FTI_LONG.h5datatype = H5T_NATIVE_LONG;
    FTI_UCHR.h5datatype = H5T_NATIVE_UCHAR;
    FTI_USHT.h5datatype = H5T_NATIVE_USHORT;
    FTI_UINT.h5datatype = H5T_NATIVE_UINT;
    FTI_ULNG.h5datatype = H5T_NATIVE_ULONG;
    FTI_SFLT.h5datatype = H5T_NATIVE_FLOAT;
    FTI_DBLE.h5datatype = H5T_NATIVE_DOUBLE;
    FTI_LDBE.h5datatype = H5T_NATIVE_LDOUBLE;
#endif
    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
    @brief      It erases a directory and all its files.
    @param      path            Path to the directory we want to erase.
    @param      flag            Set to 1 to activate.
    @return     integer         FTI_SCES if successful.

    This function erases a directory and all its files. It focusses on the
    checkpoint directories created by FTI so it does NOT handle recursive
    erasing if the given directory includes other directories.

 **/
/*-------------------------------------------------------------------------*/

int FTI_RmDir(char path[FTI_BUFS], int flag)
{
    if (flag) {
        char str[FTI_BUFS];
        sprintf(str, "Removing directory %s and its files.", path);
        FTI_Print(str, FTI_DBUG);

        DIR* dp = opendir(path);
        if (dp != NULL) {
            struct dirent* ep = NULL;
            while ((ep = readdir(dp)) != NULL) {
                char fil[FTI_BUFS];
                sprintf(fil, "%s", ep->d_name);
                if ((strcmp(fil, ".") != 0) && (strcmp(fil, "..") != 0)) {
                    char fn[FTI_BUFS];
                    sprintf(fn, "%s/%s", path, fil);
                    sprintf(str, "File %s will be removed.", fn);
                    FTI_Print(str, FTI_DBUG);
                    if (remove(fn) == -1) {
                        if (errno != ENOENT) {
                            FTI_Print("Error removing target file.", FTI_EROR);
                        }
                    }
                }
            }
        }
        else {
            if (errno != ENOENT) {
                FTI_Print("Error with opendir.", FTI_EROR);
            }
        }
        if (dp != NULL) {
            closedir(dp);
        }

        if (remove(path) == -1) {
            if (errno != ENOENT) {
                FTI_Print("Error removing target directory.", FTI_EROR);
            }
        }
    }
    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
    @brief      It erases the previous checkpoints and their metadata.
    @param      FTI_Conf        Configuration metadata.
    @param      FTI_Topo        Topology metadata.
    @param      FTI_Ckpt        Checkpoint metadata.
    @param      level           Level of cleaning.
    @return     integer         FTI_SCES if successful.

    This function erases previous checkpoint depending on the level of the
    current checkpoint. Level 5 means complete clean up. Level 6 means clean
    up local nodes but keep last checkpoint data and metadata in the PFS.

 **/
/*-------------------------------------------------------------------------*/
int FTI_Clean(FTIT_configuration* FTI_Conf, FTIT_topology* FTI_Topo,
              FTIT_checkpoint* FTI_Ckpt, int level)
{
    int nodeFlag; //only one process in the node has set it to 1
    int globalFlag = !FTI_Topo->splitRank; //only one process in the FTI_COMM_WORLD has set it to 1

    nodeFlag = (((!FTI_Topo->amIaHead) && ((FTI_Topo->nodeRank - FTI_Topo->nbHeads) == 0)) || (FTI_Topo->amIaHead)) ? 1 : 0;

    if (level == 0) {
        FTI_RmDir(FTI_Conf->mTmpDir, globalFlag);
        FTI_RmDir(FTI_Conf->gTmpDir, globalFlag);
        FTI_RmDir(FTI_Conf->lTmpDir, nodeFlag);
    }

    // Clean last checkpoint level 1
    if (level >= 1) {
        FTI_RmDir(FTI_Ckpt[1].metaDir, globalFlag);
        FTI_RmDir(FTI_Ckpt[1].dir, nodeFlag);
    }

    // Clean last checkpoint level 2
    if (level >= 2) {
        FTI_RmDir(FTI_Ckpt[2].metaDir, globalFlag);
        FTI_RmDir(FTI_Ckpt[2].dir, nodeFlag);
    }

    // Clean last checkpoint level 3
    if (level >= 3) {
        FTI_RmDir(FTI_Ckpt[3].metaDir, globalFlag);
        FTI_RmDir(FTI_Ckpt[3].dir, nodeFlag);
    }

    // Clean last checkpoint level 4
    if (level == 4 || level == 5) {
        FTI_RmDir(FTI_Ckpt[4].metaDir, globalFlag);
        FTI_RmDir(FTI_Ckpt[4].dir, globalFlag);
        rmdir(FTI_Conf->gTmpDir);
    }

    // If it is the very last cleaning and we DO NOT keep the last checkpoint
    if (level == 5) {
        rmdir(FTI_Conf->lTmpDir);
        rmdir(FTI_Conf->localDir);
        rmdir(FTI_Conf->glbalDir);
        char buf[FTI_BUFS];
        snprintf(buf, FTI_BUFS, "%s/Topology.fti", FTI_Conf->metadDir);
        if (remove(buf) == -1) {
            if (errno != ENOENT) {
                FTI_Print("Cannot remove Topology.fti", FTI_EROR);
            }
        }
        rmdir(FTI_Conf->metadDir);
    }

    // If it is the very last cleaning and we DO keep the last checkpoint
    if (level == 6) {
        rmdir(FTI_Conf->lTmpDir);
        rmdir(FTI_Conf->localDir);
    }

    return FTI_SCES;
}
