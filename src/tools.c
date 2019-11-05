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
#include <time.h>

int FTI_GetRandomHash( char* hash ) 
{
    
    if( hash == NULL )  {
        return FTI_NSCS;
    }

    void *random_buffer = malloc( sizeof(struct timespec) );
    unsigned char digest[MD5_DIGEST_LENGTH];
    clock_gettime(CLOCK_REALTIME, (struct timespec*)random_buffer);
    MD5( random_buffer, sizeof(struct timespec), digest );
    
    int i=0;
    for(; i<MD5_DIGEST_LENGTH; i++) {
        sprintf( &hash[i*2], "%02x", (unsigned int)digest[i] );
    }

}

int FTI_filemetastructsize;		        /**< size of FTIFF_db struct in file    */
int FTI_dbstructsize;		        /**< size of FTIFF_db struct in file    */
int FTI_dbvarstructsize;		        /**< size of FTIFF_db struct in file    */

#ifdef ENABLE_HDF5
int FTI_DebugCheckOpenObjects(hid_t fid, int rank) {
	ssize_t cnt;
	int howmany;
	int i;
	H5I_type_t ot;
	hid_t anobj;
	hid_t *objs;
	char name[1024];

	cnt = H5Fget_obj_count(fid, H5F_OBJ_ALL);

	if (cnt <= 0) return cnt;

	DBG_MSG("%ld object(s) open", rank, cnt);

	objs = malloc(cnt * sizeof(hid_t));

	howmany = H5Fget_obj_ids(fid, H5F_OBJ_ALL, cnt, objs);

	DBG_MSG("open objects:", rank);

	for (i = 0; i < howmany; i++ ) {
		anobj = *objs++;
		ot = H5Iget_type(anobj);
		H5Iget_name(anobj, name, 1024);
		DBG_MSG(" %d: type %d, name %s", rank, i,ot,name);
	}

	return howmany;
}
#endif

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

    // datablock size in file
    FTI_filemetastructsize
        = MD5_DIGEST_STRING_LENGTH
        + MD5_DIGEST_LENGTH
        + 7*sizeof(long)
        + sizeof(int);
    // TODO RS L3 only works for even file sizes. This accounts for many but clearly not all cases.
    // This is to fix.
    FTI_filemetastructsize += 2 - FTI_filemetastructsize%2;

    FTI_dbstructsize
        = sizeof(int)               /* numvars */
        + sizeof(long);             /* dbsize */

    FTI_dbvarstructsize
        = 3*sizeof(int)               /* numvars */
        + 2*sizeof(bool)
        + 2*sizeof(uintptr_t)
        + 2*sizeof(long)
        + MD5_DIGEST_LENGTH;

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
    /* bool          */ FTI_Exec->hasCkpt               =false;
    /* unsigned int  */ FTI_Exec->ckptCnt               =0;
    /* unsigned int  */ FTI_Exec->ckptIcnt              =0;
    /* unsigned int  */ FTI_Exec->ckptID                =0;
    /* unsigned int  */ FTI_Exec->ckptNext              =0;
    /* unsigned int  */ FTI_Exec->ckptLast              =0;
    /* long          */ FTI_Exec->ckptSize              =0;
    /* unsigned int  */ FTI_Exec->nbVar                 =0;
    /* unsigned int  */ FTI_Exec->nbVarStored           =0;
    /* unsigned int  */ FTI_Exec->nbType                =0;
    /* int           */ FTI_Exec->metaAlloc             =0;
    /* int           */ FTI_Exec->initSCES              =0;
    /* char[BUFS]       FTI_Exec->h5SingleFileLast */   memset(FTI_Exec->h5SingleFileLast,0x0,FTI_BUFS);
    /* char[BUFS]       FTI_Exec->h5SingleFileReco */   memset(FTI_Exec->h5SingleFileReco,0x0,FTI_BUFS);
    /* FTIT_iCPInfo     FTI_Exec->iCPInfo */            memset(&(FTI_Exec->iCPInfo),0x0,sizeof(FTIT_iCPInfo));
    /* FTIT_metadata[5] FTI_Exec->meta */               memset(FTI_Exec->meta,0x0,5*sizeof(FTIT_metadata));
    /* FTIFF_db      */ FTI_Exec->firstdb               =NULL;
    /* FTIFF_db      */ FTI_Exec->lastdb                =NULL;
    /* FTIT_globalDataset */ FTI_Exec->globalDatasets   =NULL;
    FTI_Exec->stageInfo             =NULL;
    /* FTIFF_metaInfo   FTI_Exec->FTIFFMeta */          memset(&(FTI_Exec->FTIFFMeta),0x0,sizeof(FTIFF_metaInfo));
    FTI_Exec->FTIFFMeta.metaSize                        = FTI_filemetastructsize;
    /* MPI_Comm      */ FTI_Exec->globalComm            =0;
    /* MPI_Comm      */ FTI_Exec->groupComm             =0;
    /* MPI_Comm      */ FTI_Exec->dcpInfoPosix.Counter  =0;
    /* MPI_Comm      */ FTI_Exec->dcpInfoPosix.FileSize =0;
    memset(FTI_Exec->dcpInfoPosix.LayerSize, 0x0, MAX_STACK_SIZE*sizeof(unsigned long));
    memset(FTI_Exec->dcpInfoPosix.LayerHash, 0x0, MAX_STACK_SIZE*MD5_DIGEST_STRING_LENGTH);

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
    /* bool          */ FTI_Conf->keepL4Ckpt            =0;
    /* bool          */ FTI_Conf->h5SingleFileEnable    =0;
    /* int           */ FTI_Conf->ckptTag               =0;
    /* int           */ FTI_Conf->stageTag              =0;
    /* int           */ FTI_Conf->finalTag              =0;
    /* int           */ FTI_Conf->generalTag            =0;
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

    /* FTIT_Ckpt[5]     FTI_Ckpt Array */               memset(FTI_Ckpt,0x0,sizeof(FTIT_checkpoint)*5);
    /* int           */ FTI_Ckpt[4].isDcp               =false;
    /* int           */ FTI_Ckpt[4].hasDcp              =false;

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
int FTI_Checksum(FTIT_execution* FTI_Exec, FTIT_dataset* FTI_Data,
        FTIT_configuration* FTI_Conf, char* checksum)
{
    int i;
    int ii = 0;

    for(i = 0; i < MD5_DIGEST_LENGTH; i++) {
        sprintf(&checksum[ii], "%02x", FTI_Exec->integrity[i]);
        ii += 2;
    }
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
    char checksum[MD5_DIGEST_STRING_LENGTH];   //calculated checksum
    int ii = 0;
    for(i = 0; i < MD5_DIGEST_LENGTH; i++) {
        sprintf(&checksum[ii], "%02x", hash[i]);
        ii += 2;
    }

    if (strcmp(checksum, checksumToCmp) != 0) {
        char str[FTI_BUFS];
        sprintf(str, "TOOLS: Checksum do not match. \"%s\" file is corrupted. %s != %s",
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
            FTI_Exec->meta[i].currentL4CkptFile = calloc(FTI_BUFS * FTI_Topo->nodeSize, sizeof(char));
            FTI_Exec->meta[i].nbVar = calloc(FTI_Topo->nodeSize, sizeof(int));
            FTI_Exec->meta[i].varID = calloc(FTI_BUFS * FTI_Topo->nodeSize, sizeof(int));
            FTI_Exec->meta[i].varSize = calloc(FTI_BUFS * FTI_Topo->nodeSize, sizeof(long));
            FTI_Exec->meta[i].filePos = calloc(FTI_BUFS * FTI_Topo->nodeSize, sizeof(long));
        }
    } else {
        for (i = 0; i < 5; i++) {
            FTI_Exec->meta[i].exists = calloc(1, sizeof(int));
            FTI_Exec->meta[i].maxFs = calloc(1, sizeof(long));
            FTI_Exec->meta[i].fs = calloc(1, sizeof(long));
            FTI_Exec->meta[i].pfs = calloc(1, sizeof(long));
            FTI_Exec->meta[i].ckptFile = calloc(FTI_BUFS, sizeof(char));
            FTI_Exec->meta[i].currentL4CkptFile = calloc(FTI_BUFS, sizeof(char));
            FTI_Exec->meta[i].nbVar = calloc(1, sizeof(int));
            FTI_Exec->meta[i].varID = calloc(FTI_BUFS, sizeof(int));
            FTI_Exec->meta[i].varSize = calloc(FTI_BUFS, sizeof(long));
            FTI_Exec->meta[i].filePos= calloc(FTI_BUFS, sizeof(long));
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
            free(FTI_Exec->meta[i].filePos);
        }
        FTI_Exec->metaAlloc = 0;
    }
}

/*-------------------------------------------------------------------------*/
/**
  @brief      It mallocs memory for the metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Topo        Topology metadata.

  This function mallocs the memory used for the metadata storage.

 **/
/*-------------------------------------------------------------------------*/
int FTI_InitGroupsAndTypes(FTIT_execution* FTI_Exec) 
{
    FTI_Exec->FTI_Type = malloc(sizeof(FTIT_type*) * FTI_BUFS);
    if (FTI_Exec->FTI_Type == NULL) {
        return FTI_NSCS;
    }

    FTI_Exec->H5groups = malloc(sizeof(FTIT_H5Group*) * FTI_BUFS);
    if (FTI_Exec->H5groups == NULL) {
        return FTI_NSCS;
    }

    FTI_Exec->H5groups[0] = malloc(sizeof(FTIT_H5Group));
    if (FTI_Exec->H5groups[0] == NULL) {
        return FTI_NSCS;
    }

    FTI_Exec->H5groups[0]->id = 0;
    FTI_Exec->H5groups[0]->childrenNo = 0;
    sprintf(FTI_Exec->H5groups[0]->name, "/");
    FTI_Exec->H5groups[0]->fullName[0] = '\0';
    FTI_Exec->nbGroup = 1;
    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      It frees memory for the types.
  @param      FTI_Exec        Execution metadata.

  This function frees the memory used for the type storage.

 **/
/*-------------------------------------------------------------------------*/
void FTI_FreeTypesAndGroups(FTIT_execution* FTI_Exec) 
{
    int i;
    for (i = 0; i < FTI_Exec->nbType; i++) {
        if (FTI_Exec->FTI_Type[i]->structure != NULL) {
            //if complex type and have structure
            free(FTI_Exec->FTI_Type[i]->structure);
        }
        free(FTI_Exec->FTI_Type[i]);
    }
    free(FTI_Exec->FTI_Type);
    for (i = 0; i < FTI_Exec->nbGroup; i++) {
        free(FTI_Exec->H5groups[i]);
    }
    free(FTI_Exec->H5groups);
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
                FTI_Print(fil, FTI_DBUG);
                if ((strcmp(fil, ".") != 0) && (strcmp(fil, "..") != 0)) {
                    char fn[FTI_BUFS];
                    snprintf(fn,FTI_BUFS, "%s/%s", path, fil);
                    snprintf(str,FTI_BUFS, "File %s will be removed.", fn);
                    FTI_Print(str, FTI_DBUG);
                    if (remove(fn) == -1) {
                        if (errno != ENOENT) {
                            snprintf(str, FTI_BUFS, "Error removing target file (%s).", fn);
                            FTI_Print(str, FTI_EROR);
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

    bool notDcpFtiff = !(FTI_Ckpt[4].isDcp && FTI_Conf->dcpFtiff); 
    bool notDcp = !FTI_Ckpt[4].isDcp;   

    if (level == 0) {
        FTI_RmDir(FTI_Conf->mTmpDir, globalFlag && notDcpFtiff );
        FTI_RmDir(FTI_Conf->gTmpDir, globalFlag && notDcp );
        FTI_RmDir(FTI_Conf->lTmpDir, nodeFlag && notDcp );
    }

    // Clean last checkpoint level 1
    if (level >= 1) {
        FTI_RmDir(FTI_Ckpt[1].metaDir, globalFlag && notDcpFtiff );
        FTI_RmDir(FTI_Ckpt[1].dir, nodeFlag && notDcp );
    }

    // Clean last checkpoint level 2
    if (level >= 2) {
        FTI_RmDir(FTI_Ckpt[2].metaDir, globalFlag && notDcpFtiff );
        FTI_RmDir(FTI_Ckpt[2].dir, nodeFlag && notDcp );
    }

    // Clean last checkpoint level 3
    if (level >= 3) {
        FTI_RmDir(FTI_Ckpt[3].metaDir, globalFlag && notDcpFtiff );
        FTI_RmDir(FTI_Ckpt[3].dir, nodeFlag && notDcp );
    }

    // Clean last checkpoint level 4
    if ( level == 4 || level == 5 ) {
        FTI_RmDir(FTI_Ckpt[4].metaDir, globalFlag && notDcpFtiff );
        FTI_RmDir(FTI_Ckpt[4].dir, globalFlag && notDcp );
        rmdir(FTI_Conf->gTmpDir);
    }
    if ( (FTI_Conf->dcpPosix || FTI_Conf->dcpFtiff) && level == 5 ) {
        FTI_RmDir(FTI_Ckpt[4].dcpDir, !FTI_Topo->splitRank);
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
        snprintf(buf, FTI_BUFS, "%s/Checkpoint.fti", FTI_Conf->metadDir);
        if (remove(buf) == -1) {
            if (errno != ENOENT) {
                FTI_Print("Cannot remove Checkpoint.fti", FTI_EROR);
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

/*-------------------------------------------------------------------------*/
/**
  @brief      generates hex string representation of hash digest.
  @param      char*             hash digest 
  @param      int               digest width 
  @return     char*             hex string of hash
 **/
/*-------------------------------------------------------------------------*/
char* FTI_GetHashHexStr( unsigned char* hash, int digestWidth, char* hashHexStr )
{       
    static char hashHexStatic[MD5_DIGEST_STRING_LENGTH];
    if( hashHexStr == NULL ) {
        hashHexStr = hashHexStatic;
    }

    int i;
    for(i = 0; i < digestWidth; i++) {
        sprintf(&hashHexStr[2*i], "%02x", hash[i]);
    }

    return hashHexStr;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Finds the the id of a protected variable in the metadata file .
  @param      FTI_Exec*         Checkpoint execution data 
  @param      FTIT_dataset      Information regarding the protected variables.
  @param      int id            Variable id we are searching for
  @param      int *currentIndex The variable index of the found variable in FTI_Data structure
  @param      int *oldIndex     The variable index of the found variable in FTI_metadata structure
  @return     int               FTI_SCES on succefully finding a variable 

  This function matches the index of a current protected variable with the index of a 
  variable protected on the previous execution (Recovery)
 **/
/*-------------------------------------------------------------------------*/

int FTI_FindVarInMeta(FTIT_execution *FTI_Exec, FTIT_dataset *FTI_Data, int id, int *currentIndex, int *oldIndex){
    int i,j;
    for (i = 0; i < FTI_Exec->meta[FTI_Exec->ckptLvel].nbVar[0]; i++) {
        if (id == FTI_Exec->meta[FTI_Exec->ckptLvel].varID[i]) {
            *oldIndex = i;
            for ( j = 0 ; j < FTI_Exec->nbVar; j++){
                if ( id == FTI_Data[j].id){
                    *currentIndex = j;
                    break;
                }
            }		
            if ( j == FTI_Exec->nbVar){
                FTI_Print("Variables must be protected before theiy can be revoered.", FTI_EROR);
                return FTI_NREC;
            }

            if (FTI_Data[j].size != FTI_Exec->meta[FTI_Exec->ckptLvel].varSize[i]) {
                char str[FTI_BUFS];
                sprintf(str, "Cannot recover %ld bytes to protected variable (ID %d) size: %ld",
                        FTI_Exec->meta[FTI_Exec->ckptLvel].varSize[i], FTI_Exec->meta[FTI_Exec->ckptLvel].varID[i],
                        FTI_Data[j].size);
                FTI_Print(str, FTI_WARN);
                return FTI_NREC;
            }
        }
    }
    return FTI_SCES;
}
