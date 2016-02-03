/**
 *  @file   tools.c
 *  @author Leonardo A. Bautista Gomez (leobago@gmail.com)
 *  @date   December, 2013
 *  @brief  Utility functions for the FTI library.
 */

#include "interface.h"
#include <dirent.h>

int FTI_Clean(int level, int group, int rank);

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
                fprintf(stderr, "[FTI Error - %06d] : %s : %s \n", FTI_Topo.myRank, msg, strerror(errno));
                break;
            case FTI_WARN:
                fprintf(stdout, "[FTI Warning %06d] : %s \n", FTI_Topo.myRank, msg);
                break;
            case FTI_INFO:
                if (FTI_Topo.splitRank == 0)
                    fprintf(stdout, "[ FTI  Information ] : %s \n", msg);
                break;
            case FTI_DBUG:
                fprintf(stdout, "[FTI Debug - %06d] : %s \n", FTI_Topo.myRank, msg);
                break;
            default:
                break;
            }
        }
    }
    fflush(stdout);
}

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
int FTI_InitBasicTypes(FTIT_dataset FTI_Data[FTI_BUFS])
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
    if (flag && (!access(path, R_OK))) {
        DIR* dp;
        char buf[FTI_BUFS], fn[FTI_BUFS], fil[FTI_BUFS];
        struct dirent* ep;
        dp = opendir(path);
        sprintf(buf, "Removing directory %s and its files.", path);
        FTI_Print(buf, FTI_DBUG);
        if (dp != NULL) {
            while (ep = readdir(dp)) {
                sprintf(fil, "%s", ep->d_name);
                if ((strcmp(fil, ".") != 0) && (strcmp(fil, "..") != 0)) {
                    sprintf(fn, "%s/%s", path, fil);
                    sprintf(buf, "File %s will be removed.", fn);
                    FTI_Print(buf, FTI_DBUG);
                    if (remove(fn) != 0)
                        FTI_Print("Error removing target file.", FTI_EROR);
                }
            }
        }
        else {
            FTI_Print("Error with opendir.", FTI_EROR);
        }
        closedir(dp);
        if (remove(path) != 0)
            FTI_Print("Error removing target directory.", FTI_EROR);
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
int FTI_Clean(int level, int group, int rank)
{
    char buf[FTI_BUFS];
    int nodeFlag, globalFlag = !FTI_Topo.splitRank;
    nodeFlag = (((!FTI_Topo.amIaHead) && (FTI_Topo.nodeRank == 0)) || (FTI_Topo.amIaHead)) ? 1 : 0;
    if (level == 0) {
        FTI_RmDir(FTI_Conf.mTmpDir, globalFlag);
        FTI_RmDir(FTI_Conf.gTmpDir, globalFlag);
        FTI_RmDir(FTI_Conf.lTmpDir, nodeFlag);
    }
    if (level >= 1) { // Clean last checkpoint level 1
        FTI_RmDir(FTI_Ckpt[1].metaDir, globalFlag);
        FTI_RmDir(FTI_Ckpt[1].dir, nodeFlag);
    }
    if (level >= 2) { // Clean last checkpoint level 2
        FTI_RmDir(FTI_Ckpt[2].metaDir, globalFlag);
        FTI_RmDir(FTI_Ckpt[2].dir, nodeFlag);
    }
    if (level >= 3) { // Clean last checkpoint level 3
        FTI_RmDir(FTI_Ckpt[3].metaDir, globalFlag);
        FTI_RmDir(FTI_Ckpt[3].dir, nodeFlag);
    }
    if (level == 4 || level == 5) { // Clean last checkpoint level 4
        FTI_RmDir(FTI_Ckpt[4].metaDir, globalFlag);
        FTI_RmDir(FTI_Ckpt[4].dir, globalFlag);
        rmdir(FTI_Conf.gTmpDir);
    }
    if (level == 5) { // If it is the very last cleaning and we DO NOT keep the last checkpoint
        rmdir(FTI_Conf.lTmpDir);
        rmdir(FTI_Conf.localDir);
        rmdir(FTI_Conf.glbalDir);
        snprintf(buf, FTI_BUFS, "%s/Topology.fti", FTI_Conf.metadDir);
        remove(buf);
        rmdir(FTI_Conf.metadDir);
    }
    if (level == 6) { // If it is the very last cleaning and we DO keep the last checkpoint
        rmdir(FTI_Conf.lTmpDir);
        rmdir(FTI_Conf.localDir);
    }
    return FTI_SCES;
}
