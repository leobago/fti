/**
 *  @file   tools.c
 *  @author Leonardo A. Bautista Gomez (leobago@gmail.com)
 *  @date   December, 2013
 *  @brief  Utility functions for the FTI library.
 */

#include "interface.h"
#include <dirent.h>

/*-------------------------------------------------------------------------*/
/**
    @brief      Receive the return code of a function and print a message.
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
    @brief      It creates the basic datatypes and the dataset array.
    @param      FTI_Data        Dataset array.
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
    @param      flag            set to 1 to activate.
    @return     integer         FTI_SCES if successful.

    This function erases a directory and all its files. It focusses on the
    checkpoint directories created by FTI so it does NOT handle recursive
    erasing if the given directory includes other directories.

 **/
/*-------------------------------------------------------------------------*/

int FTI_RmDir(char path[FTI_BUFS], int flag)
{
    if (flag) {
        char buf[FTI_BUFS], fn[FTI_BUFS], fil[FTI_BUFS];
        DIR* dp = NULL;
        struct dirent* ep = NULL;

        sprintf(buf, "Removing directory %s and its files.", path);
        FTI_Print(buf, FTI_DBUG);

        dp = opendir(path);
        if (dp != NULL) {
            while ((ep = readdir(dp)) != NULL) {
                sprintf(fil, "%s", ep->d_name);
                if ((strcmp(fil, ".") != 0) && (strcmp(fil, "..") != 0)) {
                    sprintf(fn, "%s/%s", path, fil);
                    sprintf(buf, "File %s will be removed.", fn);
                    FTI_Print(buf, FTI_DBUG);
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
    @param      level           Level of cleaning.
    @param      group           Group ID of the cleaning target process.
    @param      rank            Rank of the cleaning target process.
    @return     integer         FTI_SCES if successful.

    This function erases previous checkpoint depending on the level of the
    current checkpoint. Level 5 means complete clean up. Level 6 means clean
    up local nodes but keep last checkpoint data and metadata in the PFS.

 **/
/*-------------------------------------------------------------------------*/
int FTI_Clean(FTIT_configuration* FTI_Conf, FTIT_topology* FTI_Topo,
              FTIT_checkpoint* FTI_Ckpt, int level, int group, int rank)
{
    char buf[FTI_BUFS];
    int nodeFlag, globalFlag = !FTI_Topo->splitRank;

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
