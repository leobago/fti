/**
 *  @file   recover.c
 *  @author Leonardo A. Bautista Gomez (leobago@gmail.com)
 *  @date   January, 2014
 *  @brief  Recovery functions for the FTI library.
 */

#include "interface.h"

/*-------------------------------------------------------------------------*/
/**
    @brief      It checks if a file exist and that its size is 'correct'.
    @param      fn              The ckpt. file name to check.
    @param      fs              The ckpt. file size to check.
    @param      checksum        The file checksum to check.
    @return     integer         0 if file exists, 1 if not or wrong size.

    This function checks whether a file exist or not and if its size is
    the expected one.

 **/
/*-------------------------------------------------------------------------*/
int FTI_CheckFile(char* fn, unsigned long fs, char* checksum)
{
    struct stat fileStatus;
    char str[FTI_BUFS];
    if (access(fn, F_OK) == 0) {
        if (stat(fn, &fileStatus) == 0) {
            if (fileStatus.st_size == fs) {
                if (strlen(checksum)) {
                    int res = FTI_VerifyChecksum(fn, checksum);
                    if (res != FTI_SCES) {
                        return 1;
                    }
                    return 0;
                }
                return 0;
            }
            else {
                return 1;
            }
        }
        else {
            return 1;
        }
    }
    else {
        sprintf(str, "Missing file: \"%s\"", fn);
        FTI_Print(str, FTI_WARN);
        return 1;
    }
}

/*-------------------------------------------------------------------------*/
/**
    @brief      It detects all the erasures for a particular level.
    @param      FTI_Conf        Configuration metadata.
    @param      FTI_Exec        Execution metadata.
    @param      FTI_Topo        Topology metadata.
    @param      FTI_Ckpt        Checkpoint metadata.
    @param      erased          The array of erasures to fill.
    @return     integer         FTI_SCES if successful.

    This function detects all the erasures for L1, L2 and L3. It return the
    results in the erased array. The search for erasures is done at the
    three levels independently on the current recovery level.

 **/
/*-------------------------------------------------------------------------*/
int FTI_CheckErasures(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
                      FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt,
                      int *erased)
{
    int buf;
    int ckptID, rank;
    int level = FTI_Exec->ckptLvel;
    char fn[FTI_BUFS];
    char checksum[MD5_DIGEST_LENGTH], ptnerChecksum[MD5_DIGEST_LENGTH], rsChecksum[MD5_DIGEST_LENGTH];
    long fs = FTI_Exec->meta[level].fs[0];
    long pfs = FTI_Exec->meta[level].pfs[0];
    long maxFs = FTI_Exec->meta[level].maxFs[0];
    char ckptFile[FTI_BUFS];
    strcpy(ckptFile, FTI_Exec->meta[level].ckptFile);

    // TODO Checksums only local currently
    if ( level > 0 && level < 4 ) {
        FTI_GetChecksums(FTI_Conf, FTI_Topo, FTI_Ckpt, checksum, ptnerChecksum, rsChecksum, FTI_Topo->groupID, level);
    }
    sprintf(fn, "Checking file %s and its erasures.", ckptFile);
    FTI_Print(fn, FTI_DBUG);
    switch (level) {
        case 1:
            sprintf(fn, "%s/%s", FTI_Ckpt[1].dir, ckptFile);
            buf = FTI_CheckFile(fn, fs, checksum);
            MPI_Allgather(&buf, 1, MPI_INT, erased, 1, MPI_INT, FTI_Exec->groupComm);
            break;
        case 2:
            sprintf(fn, "%s/%s", FTI_Ckpt[2].dir, ckptFile);
            buf = FTI_CheckFile(fn, fs, checksum);
            MPI_Allgather(&buf, 1, MPI_INT, erased, 1, MPI_INT, FTI_Exec->groupComm);
            sscanf(ckptFile, "Ckpt%d-Rank%d.fti", &ckptID, &rank);
            sprintf(fn, "%s/Ckpt%d-Pcof%d.fti", FTI_Ckpt[2].dir, ckptID, rank);
            buf = FTI_CheckFile(fn, pfs, ptnerChecksum);
            MPI_Allgather(&buf, 1, MPI_INT, erased + FTI_Topo->groupSize, 1, MPI_INT, FTI_Exec->groupComm);
            break;
        case 3:
            sprintf(fn, "%s/%s", FTI_Ckpt[3].dir, ckptFile);
            buf = FTI_CheckFile(fn, fs, checksum);
            MPI_Allgather(&buf, 1, MPI_INT, erased, 1, MPI_INT, FTI_Exec->groupComm);
            sscanf(ckptFile, "Ckpt%d-Rank%d.fti", &ckptID, &rank);
            sprintf(fn, "%s/Ckpt%d-RSed%d.fti", FTI_Ckpt[3].dir, ckptID, rank);
            buf = FTI_CheckFile(fn, maxFs, rsChecksum);
            MPI_Allgather(&buf, 1, MPI_INT, erased + FTI_Topo->groupSize, 1, MPI_INT, FTI_Exec->groupComm);
            break;
        case 4:
            sprintf(fn, "%s/%s", FTI_Ckpt[4].dir, ckptFile);
            buf = FTI_CheckFile(fn, fs, "");
            MPI_Allgather(&buf, 1, MPI_INT, erased, 1, MPI_INT, FTI_Exec->groupComm);
            break;
    }
    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
    @brief      It decides wich action take depending on the restart level.
    @param      FTI_Conf        Configuration metadata.
    @param      FTI_Exec        Execution metadata.
    @param      FTI_Topo        Topology metadata.
    @param      FTI_Ckpt        Checkpoint metadata.
    @return     integer         FTI_SCES if successful.

    This function launches the required action depending on the recovery
    level. The recovery level is detected from the checkpoint ID of the
    last checkpoint taken.

 **/
/*-------------------------------------------------------------------------*/
int FTI_RecoverFiles(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
                     FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt)
{
   char str[FTI_BUFS]; //For console output

   if (!FTI_Topo->amIaHead) { //Application process
       int levelCkptID[5];
       //Find last checkpoint (biggest ckptID)
       int i;
       for (i = 1; i < 5; i++) { //For every level
           int id = 0;
           if (FTI_Exec->meta[i].exists[0]) {
               sscanf(FTI_Exec->meta[i].ckptFile, "Ckpt%d", &id);
           }
           levelCkptID[i] = id;
       }
       for (i = 1; i < 5; i++) {  //For every level
           int biggestCkptID = 0;
           int level = 0;
           //Find biggest ckptID
           int j;
           for (j = 1; j < 5; j++) {
               if (levelCkptID[j] > biggestCkptID) {
                   biggestCkptID = levelCkptID[j];
                   level = j;
               }
           }
           if (level == 0) {
               //Cannot recover from any level
               break;
           }
           levelCkptID[level] = 0; //Reset level's ckptID to make other levels in next iteration

           sprintf(str, "Trying recovery with Ckpt. %d at level %d.", biggestCkptID, level);
           FTI_Print(str, FTI_DBUG);

           //Try to recover from level
           FTI_Exec->ckptID = biggestCkptID; //Needed for FTI_RecoverL# to get correct file name
           FTI_Exec->ckptLvel = level; //Needed fir FTI_CheckErasures to check correct level
           int res;
           switch (level) {
               case 4:
                    FTI_Clean(FTI_Conf, FTI_Topo, FTI_Ckpt, 1, FTI_Topo->groupID, FTI_Topo->myRank);
                    MPI_Barrier(FTI_COMM_WORLD);
                    res = FTI_RecoverL4(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt);
                    break;
               case 3:
                    res = FTI_RecoverL3(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt);
                    break;
               case 2:
                    res = FTI_RecoverL2(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt);
                    break;
               case 1:
                    res = FTI_RecoverL1(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt);
                    break;
           }
           int allRes;
           MPI_Allreduce(&res, &allRes, 1, MPI_INT, MPI_SUM, FTI_COMM_WORLD);
           if (allRes == FTI_SCES) {
              //Inform heads that recovered successfully
              MPI_Allreduce(&res, &allRes, 1, MPI_INT, MPI_SUM, FTI_Exec->globalComm);

              sprintf(str, "Recovering successfully from level %d with Ckpt. %d.", level, biggestCkptID);
              FTI_Print(str, FTI_INFO);
              //This is to enable recovering from local for L4 case in FTI_Recover
              if (level == 4) {
                  FTI_Exec->ckptLvel = 1;
                  if (FTI_Topo->splitRank == 0) {
                      if (rename(FTI_Ckpt[4].metaDir, FTI_Ckpt[1].metaDir) == -1) {
                          FTI_Print("Cannot rename L4 metadata folder to L1", FTI_WARN);
                      }
                  }
              }
              return FTI_SCES; //Recovered successfully
          }
          else {
              sprintf(str, "Recover failed from level %d with Ckpt. %d.", level, biggestCkptID);
              FTI_Print(str, FTI_INFO);
          }
       }
       //Looped all levels with no success
       FTI_Print("Cannot recover from any checkpoint level.", FTI_INFO);

       //Inform heads that cannot recover
       int res = FTI_NSCS, allRes;
       MPI_Allreduce(&res, &allRes, 1, MPI_INT, MPI_SUM, FTI_Exec->globalComm);

       //Reset ckptID and ckptLvel
       FTI_Exec->ckptID = 0;
       FTI_Exec->ckptLvel = 0;
       return FTI_NSCS;
   }
   else { //Head process
       int res = FTI_SCES, allRes;
       MPI_Allreduce(&res, &allRes, 1, MPI_INT, MPI_SUM, FTI_Exec->globalComm);
       if (allRes != FTI_SCES) {
           //Recover not successful
           return FTI_NSCS;
       }
       return FTI_SCES;
   }
}
