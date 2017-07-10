/**
 *  @file   tools.c
 *  @author Leonardo A. Bautista Gomez (leobago@gmail.com)
 *  @date   December, 2013
 *  @brief  Utility functions for the FTI library.
 */

#include "interface.h"
#include <dirent.h>
#define CHUNK_SIZE 4096    /**< MD5 algorithm chunk size.      */

/*-------------------------------------------------------------------------*/
/**
    @brief      It calculates checksum of the checkpoint file.
    @param      fileName        filename of the checkpoint
    @param      checksumToCmp   checksum that is calculated
    @return     integer         FTI_SCES if successful

    This function calculates checksum of the checkpoint file based on
    MD5 algorithm and saves it in checksum.

 **/
/*-------------------------------------------------------------------------*/
int FTI_Checksum(char* fileName, char* checksum)
{
    MD5_CTX mdContext;
    unsigned char data[CHUNK_SIZE];
    unsigned char hash[MD5_DIGEST_LENGTH];
    int bytes;
    char str[FTI_BUFS];
    double startTime = MPI_Wtime();
    int i;

    FILE *fd = fopen(fileName, "rb");
    if (fd == NULL) {
        sprintf(str, "FTI failed to open file %s to calculate checksum.", fileName);
        FTI_Print(str, FTI_WARN);
        return FTI_NSCS;
    }

    MD5_Init (&mdContext);
    while ((bytes = fread (data, 1, CHUNK_SIZE, fd)) != 0) {
        MD5_Update (&mdContext, data, bytes);
    }
    MD5_Final (hash, &mdContext);

    for(i = 0; i < MD5_DIGEST_LENGTH -1; i++)
        sprintf(&checksum[i], "%02x", hash[i]);
    checksum[i] = '\0'; //to get a proper string

    sprintf(str, "Checksum took %.2f sec.", MPI_Wtime() - startTime);
    FTI_Print(str, FTI_DBUG);

    fclose (fd);

    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
    @brief      It compares checksum of the checkpoint file.
    @param      fileName        filename of the checkpoint
    @param      checksumToCmp   checksum to compare
    @return     integer         FTI_SCES if successful

    This function calculates checksum of the checkpoint file based on
    MD5 algorithm. It compares calculated hash value with the one saved
    in the file.

 **/
/*-------------------------------------------------------------------------*/
int FTI_VerifyChecksum(char* fileName, char* checksumToCmp)
{
    MD5_CTX mdContext;
    unsigned char data[CHUNK_SIZE];
    unsigned char hash[MD5_DIGEST_LENGTH];
    char checksum[MD5_DIGEST_LENGTH];   //calculated checksum
    int bytes;
    char str[FTI_BUFS];
    int i;

    FILE *fd = fopen(fileName, "rb");
    if (fd == NULL) {
        sprintf(str, "FTI failed to open file %s to calculate checksum.", fileName);
        FTI_Print(str, FTI_WARN);
        return FTI_NSCS;
    }

    MD5_Init (&mdContext);
    while ((bytes = fread (data, 1, CHUNK_SIZE, fd)) != 0) {
        MD5_Update (&mdContext, data, bytes);
    }
    MD5_Final (hash, &mdContext);

    for(i = 0; i < MD5_DIGEST_LENGTH -1; i++)
        sprintf(&checksum[i], "%02x", hash[i]);
    checksum[i] = '\0'; //to get a proper string

    if (memcmp(checksum, checksumToCmp, MD5_DIGEST_LENGTH - 1) != 0) {
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
    @brief      It receives the return code of a init critical function.
    @param      result          Result to check.
    @param      message         Message to print.

    This function checks the result from a critical function which is in FTI_Init
    where heads are not in FTI_Listen yet and if result is not success terminates
    all processes, otherwise print the debug message.

 **/
/*-------------------------------------------------------------------------*/
void FTI_InitCritical(int result, char* message, FTIT_execution* FTI_Exec)
{
    char str[FTI_BUFS];
    if (result == FTI_SCES) {
        sprintf(str, "FTI succeeded to %s", message);
        FTI_Print(str, FTI_DBUG);
    }
    else {
        sprintf(str, "FTI failed to %s", message);
        int allResults;
        MPI_Allreduce(&result, &allResults, 1, MPI_INT, MPI_SUM, FTI_Exec->globalComm);
        if (allResults != FTI_SCES) {
            FTI_Print("Exiting with status 1.", FTI_DBUG);
            exit(1);
        }
    }
}

/*-------------------------------------------------------------------------*/
/**
    @brief      It receives the return code of a critical function.
    @param      result          Result to check.
    @param      message         Message to print.

    This function checks the result from a critical function and if result is
    not success terminates all processes, otherwise print the debug message.

 **/
/*-------------------------------------------------------------------------*/
void FTI_Critical(int result, char* message, FTIT_configuration* FTI_Conf,
                    FTIT_execution* FTI_Exec, FTIT_topology* FTI_Topo)
{
    char str[FTI_BUFS];
    if (result == FTI_SCES || result == FTI_DONE) {
        sprintf(str, "FTI succeeded to %s", message);
        FTI_Print(str, FTI_DBUG);
    }
    else {
        sprintf(str, "FTI failed to %s", message);
        int allResults, endWork = FTI_ENDW;
        MPI_Allreduce(&result, &allResults, 1, MPI_INT, MPI_SUM, FTI_COMM_WORLD);
        if (allResults != FTI_SCES) {
            if (FTI_Topo->nbHeads == 1) {
                FTI_Print("Sending FTI_ENDW to the head process", FTI_DBUG);
                MPI_Send(&endWork, 1, MPI_INT, FTI_Topo->headRank, FTI_Conf->tag, FTI_Exec->globalComm);
            }
            FTI_Print("Exiting with status 1.", FTI_DBUG);
            exit(1);
        }
    }
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
    @return     integer         FTI_SCES if successful.

    This function erases previous checkpoint depending on the level of the
    current checkpoint. Level 5 means complete clean up. Level 6 means clean
    up local nodes but keep last checkpoint data and metadata in the PFS.

 **/
/*-------------------------------------------------------------------------*/
int FTI_Clean(FTIT_configuration* FTI_Conf, FTIT_topology* FTI_Topo,
              FTIT_checkpoint* FTI_Ckpt, int level)
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
