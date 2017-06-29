/**
 *  @file   recover.c
 *  @author Leonardo A. Bautista Gomez (leobago@gmail.com)
 *  @date   January, 2014
 *  @brief  Recovery functions for the FTI library.
 */

#include "interface.h"

/*-------------------------------------------------------------------------*/
/**
    @brief      Check if a file exist and that its size is 'correct'.
    @param      fn              The ckpt. file name to check.
    @param      fs              The ckpt. file size tocheck.
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
    @brief      Detects all the erasures for a particular level.
    @param      fs              The ckpt. file size for this process.
    @param      maxFs           The max. ckpt. file size in the group.
    @param      group           The group ID.
    @param      erased          The array of erasures to fill.
    @param      level           The ckpt. level to check for erasures.
    @return     integer         FTI_SCES if successful.

    This function detects all the erasures for L1, L2 and L3. It return the
    results in the erased array. The search for erasures is done at the
    three levels independently on the current recovery level.

 **/
/*-------------------------------------------------------------------------*/
int FTI_CheckErasures(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
                      FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt,
                      unsigned long *fs, unsigned long *maxFs, int group,
                      int *erased, int level)
{
    int buf;
    char fn[FTI_BUFS];
    char checksum[MD5_DIGEST_LENGTH], ptnerChecksum[MD5_DIGEST_LENGTH], rsChecksum[MD5_DIGEST_LENGTH];
    unsigned long pfs;
    if (FTI_GetMeta(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt, fs, maxFs, group, level) == FTI_SCES) {
        FTI_Print("Metadata obtained.", FTI_DBUG);
    }
    else {
        FTI_Print("Error getting metadata.", FTI_WARN);
        return FTI_NSCS;
    }
    // TODO Checksums only local currently
    if ( level > 0 && level < 4 ) {
        FTI_GetChecksums(FTI_Conf, FTI_Topo, FTI_Ckpt, checksum, ptnerChecksum, rsChecksum, group, level);
        sprintf(fn, "Checking file %s and its erasures.", FTI_Exec->ckptFile);
        FTI_Print(fn, FTI_DBUG);
    }
    switch (level) {
        case 1:
            sprintf(fn, "%s/%s", FTI_Ckpt[1].dir, FTI_Exec->ckptFile);
            buf = FTI_CheckFile(fn, *fs, checksum);
            MPI_Allgather(&buf, 1, MPI_INT, erased, 1, MPI_INT, FTI_Exec->groupComm);
            break;
        case 2:
            sprintf(fn, "%s/%s", FTI_Ckpt[2].dir, FTI_Exec->ckptFile);
            buf = FTI_CheckFile(fn, *fs, checksum);
            MPI_Allgather(&buf, 1, MPI_INT, erased, 1, MPI_INT, FTI_Exec->groupComm);

            if (FTI_GetPtnerSize(FTI_Conf, FTI_Topo, FTI_Ckpt, &pfs, group, 2) == FTI_SCES) {
                FTI_Print("PtnerSize obtained.", FTI_DBUG);
            }
            else {
                FTI_Print("Error getting ptnerSize.", FTI_WARN);
                return FTI_NSCS;
            }
            sscanf(FTI_Exec->ckptFile, "Ckpt%d-Rank%d.fti", &FTI_Exec->ckptID, &buf);
            sprintf(fn, "%s/Ckpt%d-Pcof%d.fti", FTI_Ckpt[2].dir, FTI_Exec->ckptID, buf);
            buf = FTI_CheckFile(fn, pfs, ptnerChecksum);
            MPI_Allgather(&buf, 1, MPI_INT, erased + FTI_Topo->groupSize, 1, MPI_INT, FTI_Exec->groupComm);
            break;
        case 3:
            sprintf(fn, "%s/%s", FTI_Ckpt[3].dir, FTI_Exec->ckptFile);
            buf = FTI_CheckFile(fn, *fs, checksum);
            MPI_Allgather(&buf, 1, MPI_INT, erased, 1, MPI_INT, FTI_Exec->groupComm);
            sscanf(FTI_Exec->ckptFile, "Ckpt%d-Rank%d.fti", &FTI_Exec->ckptID, &buf);
            sprintf(fn, "%s/Ckpt%d-RSed%d.fti", FTI_Ckpt[3].dir, FTI_Exec->ckptID, buf);
            buf = FTI_CheckFile(fn, *maxFs, rsChecksum);
            MPI_Allgather(&buf, 1, MPI_INT, erased + FTI_Topo->groupSize, 1, MPI_INT, FTI_Exec->groupComm);
            break;
        case 4:
            sprintf(fn, "%s/%s", FTI_Ckpt[4].dir, FTI_Exec->ckptFile);
            buf = FTI_CheckFile(fn, *fs, "");
            MPI_Allgather(&buf, 1, MPI_INT, erased, 1, MPI_INT, FTI_Exec->groupComm);
            break;
    }
    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
    @brief      Decides wich action take depending on the restart level.
    @return     integer         FTI_SCES if successful.

    This function launchs the required action depending on the recovery
    level. The recovery level is detected from the checkpoint ID of the
    last checkpoint taken.

 **/
/*-------------------------------------------------------------------------*/
int FTI_RecoverFiles(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
                     FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt)
{
   int f, r, tres = FTI_SCES, id, level = 1;
   unsigned long fs, maxFs;
   char str[FTI_BUFS];
   if (!FTI_Topo->amIaHead) {
      while (level < 5) {
         if ((FTI_Exec->reco == 2) && (level != 4)) {
            tres = FTI_NSCS;
         }
         else {
            if (FTI_GetMeta(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt, &fs, &maxFs, FTI_Topo->groupID, level) != FTI_SCES) {
               tres = FTI_NSCS;
            }
            else {
               FTI_Exec->meta[0].fs = fs;
               FTI_Exec->meta[0].maxFs = maxFs;
               FTI_Exec->ckptLvel = level;
               switch (FTI_Exec->ckptLvel) {
                  case 4:
                     FTI_Clean(FTI_Conf, FTI_Topo, FTI_Ckpt, 1, FTI_Topo->groupID, FTI_Topo->myRank);
                     MPI_Barrier(FTI_COMM_WORLD);
                     switch(FTI_Conf->ioMode) {

                        case FTI_IO_POSIX:

                           sscanf(FTI_Exec->ckptFile, "Ckpt%d-Rank%d.fti", &id, &r);
                           break;

                        case FTI_IO_MPI:

                           sscanf(FTI_Exec->ckptFile, "Ckpt%d-mpiio.fti", &id);
                           break;
#ifdef ENABLE_SIONLIB // --> If SIONlib is installed
                        case FTI_IO_SIONLIB:

                           sscanf(FTI_Exec->ckptFile, "Ckpt%d-sionlib.fti", &id);
                           break;
#endif
                     }
                     sprintf(str, "Trying recovery with Ckpt. %d at level %d.", id, level);
                     FTI_Print(str, FTI_DBUG);
                     FTI_Exec->ckptID = id;
                     FTI_Exec->lastCkptLvel = FTI_Exec->ckptLvel;
                     r = FTI_RecoverL4(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt, FTI_Topo->groupID);
                     break;
                  case 3:
                     sscanf(FTI_Exec->ckptFile, "Ckpt%d-Rank%d.fti", &id, &r);
                     sprintf(str, "Trying recovery with Ckpt. %d at level %d.", id, level);
                     FTI_Print(str, FTI_DBUG);
                     FTI_Exec->ckptID = id;
                     FTI_Exec->lastCkptLvel = FTI_Exec->ckptLvel;
                     r = FTI_RecoverL3(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt, FTI_Topo->groupID);
                     break;
                  case 2:
                     sscanf(FTI_Exec->ckptFile, "Ckpt%d-Rank%d.fti", &id, &r);
                     sprintf(str, "Trying recovery with Ckpt. %d at level %d.", id, level);
                     FTI_Print(str, FTI_DBUG);
                     FTI_Exec->ckptID = id;
                     FTI_Exec->lastCkptLvel = FTI_Exec->ckptLvel;
                     r = FTI_RecoverL2(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt, FTI_Topo->groupID);
                     break;
                  case 1:
                     sscanf(FTI_Exec->ckptFile, "Ckpt%d-Rank%d.fti", &id, &r);
                     sprintf(str, "Trying recovery with Ckpt. %d at level %d.", id, level);
                     FTI_Print(str, FTI_DBUG);
                     FTI_Exec->ckptID = id;
                     FTI_Exec->lastCkptLvel = FTI_Exec->ckptLvel;
                     r = FTI_RecoverL1(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt, FTI_Topo->groupID);
                     break;
               }
               MPI_Allreduce(&r, &tres, 1, MPI_INT, MPI_SUM, FTI_COMM_WORLD);
            }
         }

         if (tres == FTI_SCES) {
            sprintf(str, "Recovering successfully from level %d.", level);
            FTI_Print(str, FTI_INFO);
            // This is to enable recovering from local for L4 case in FTI_Recover
            if (level == 4) {
                FTI_Exec->ckptLvel = 1;
                if (FTI_Topo->splitRank == 0) {
                    if (rename(FTI_Ckpt[4].metaDir, FTI_Ckpt[1].metaDir) == -1) {
                        FTI_Print("Cannot rename L4 metadata folder to L1", FTI_WARN);
                    }
                }
            }
            break;
         }
         else {
            sprintf(str, "No possible to restart from level %d.", level);
            FTI_Print(str, FTI_INFO);
            level++;
         }
      }
   }
   fs = tres;
   MPI_Allreduce(&fs, &tres, 1, MPI_INT, MPI_SUM, FTI_Exec->globalComm);
   MPI_Barrier(FTI_Exec->globalComm);
   sleep(1); // Global barrier and sleep for clearer output
   return tres;
}
