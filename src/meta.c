/**
 *  @file   meta.c
 *  @author Leonardo A. Bautista Gomez (leobago@gmail.com)
 *  @date   December, 2013
 *  @brief  Metadata functions for the FTI library.
 */

#include "interface.h"

/*-------------------------------------------------------------------------*/
/**
    @brief      It gets the checksums from metadata.
    @param      checksum        Pointer to fill the checkpoint checksum.
    @param      ptnerChecksum   Pointer to fill the ptner file checksum.
    @param      rsChecksum      Pointer to fill the RS file checksum.
    @param      group           The group in the node.
    @param      level           The level of the ckpt or 0 if tmp.
    @return     integer         FTI_SCES if successfull.

    This function read the metadata file created during checkpointing and
    recover the checkpoint checksum. If there is no RS file, rsChecksum
    string length is 0.

 **/
/*-------------------------------------------------------------------------*/
int FTI_GetChecksums(FTIT_configuration* FTI_Conf, FTIT_topology* FTI_Topo,
                    FTIT_checkpoint* FTI_Ckpt, char* checksum, char* ptnerChecksum,
                    char* rsChecksum, int group, int level)
{
    dictionary* ini;
    char* checksumTemp;
    char mfn[FTI_BUFS], str[FTI_BUFS];
    if (level == 0) {
        sprintf(mfn, "%s/sector%d-group%d.fti", FTI_Conf->mTmpDir, FTI_Topo->sectorID, group);
    }
    else {
        sprintf(mfn, "%s/sector%d-group%d.fti", FTI_Ckpt[level].metaDir, FTI_Topo->sectorID, group);
    }

    sprintf(str, "Getting FTI metadata file (%s)...", mfn);
    FTI_Print(str, FTI_DBUG);
    if (access(mfn, R_OK) != 0) {
        FTI_Print("FTI metadata file NOT accessible.", FTI_WARN);
        return FTI_NSCS;
    }
    ini = iniparser_load(mfn);
    if (ini == NULL) {
        FTI_Print("Iniparser failed to parse the metadata file.", FTI_WARN);
        return FTI_NSCS;
    }

    sprintf(str, "%d:Ckpt_checksum", FTI_Topo->groupRank);
    checksumTemp = iniparser_getstring(ini, str, NULL);
    strncpy(checksum, checksumTemp, MD5_DIGEST_LENGTH);
    sprintf(str, "%d:Ckpt_checksum", (FTI_Topo->groupRank + FTI_Topo->groupSize - 1) % FTI_Topo->groupSize);
    checksumTemp = iniparser_getstring(ini, str, NULL);
    strncpy(ptnerChecksum, checksumTemp, MD5_DIGEST_LENGTH);
    sprintf(str, "%d:RSed_checksum", FTI_Topo->groupRank);
    checksumTemp = iniparser_getstring(ini, str, NULL);
    if (checksumTemp != NULL) {
        strncpy(rsChecksum, checksumTemp, MD5_DIGEST_LENGTH);
    } else {
        rsChecksum[0] = '\0';
    }

    iniparser_freedict(ini);

    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
    @brief      It writes the RSed file checksum to metadata.
    @param      rank            global rank of the process
    @param      checksum        Pointer to the checksum.
    @return     integer         FTI_SCES if successfull.

    This function should be executed only by one process per group. It
    writes the RSed checksum to the metadata file.

 **/
/*-------------------------------------------------------------------------*/
int FTI_WriteRSedChecksum(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
                            FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt,
                             int rank, char* checksum)
{
    char str[FTI_BUFS], buf[FTI_BUFS], fileName[FTI_BUFS];
    dictionary* ini;
    char* checksums;
    int i;
    int sectorID = rank / (FTI_Topo->groupSize * FTI_Topo->nodeSize);
    int node = rank / FTI_Topo->nodeSize;
    int rankInGroup = node - (sectorID * FTI_Topo->groupSize);
    int groupID = rank % FTI_Topo->nodeSize;

    checksums = talloc(char, FTI_Topo->groupSize * MD5_DIGEST_LENGTH);
    MPI_Allgather(checksum, MD5_DIGEST_LENGTH, MPI_CHAR, checksums, MD5_DIGEST_LENGTH, MPI_CHAR, FTI_Exec->groupComm);
    if (rankInGroup) {
        free(checksums);
        return FTI_SCES;
    }

    sprintf(fileName, "%s/sector%d-group%d.fti", FTI_Conf->mTmpDir, FTI_Topo->sectorID, groupID);
    ini = iniparser_load(fileName);
    if (ini == NULL) {
        FTI_Print("Temporary metadata file could NOT be parsed", FTI_WARN);
        return FTI_NSCS;
    }
    // Add metadata to dictionary
    for (i = 0; i < FTI_Topo->groupSize; i++) {
        strncpy(buf, checksums + (i * MD5_DIGEST_LENGTH), MD5_DIGEST_LENGTH);
        sprintf(str, "%d:RSed_checksum", i);
        iniparser_set(ini, str, buf);
    }
    free(checksums);

    sprintf(str, "Recreating metadata file (%s)...", buf);
    FTI_Print(str, FTI_DBUG);

    FILE* fd = fopen(fileName, "w");
    if (fd == NULL) {
        FTI_Print("Metadata file could NOT be opened.", FTI_WARN);

        iniparser_freedict(ini);

        return FTI_NSCS;
    }

    // Write metadata
    iniparser_dump_ini(ini, fd);

    if (fflush(fd) != 0) {
        FTI_Print("Metadata file could NOT be flushed.", FTI_WARN);

        iniparser_freedict(ini);
        fclose(fd);

        return FTI_NSCS;
    }
    if (fclose(fd) != 0) {
        FTI_Print("Metadata file could NOT be closed.", FTI_WARN);

        iniparser_freedict(ini);

        return FTI_NSCS;
    }

    iniparser_freedict(ini);

    return FTI_SCES;
}
    
/*-------------------------------------------------------------------------*/
/**
    @brief      It gets the ptner file size from metadata.
    @param      pfs             Pointer to fill the ptner file size.
    @param      group           The group in the node.
    @param      level           The level of the ckpt or 0 if tmp.
    @return     integer         FTI_SCES if successfull.

    This function read the metadata file created during checkpointing and
    reads partner file size.

 **/
/*-------------------------------------------------------------------------*/
int FTI_GetPtnerSize(FTIT_configuration* FTI_Conf, FTIT_topology* FTI_Topo,
                      FTIT_checkpoint* FTI_Ckpt, unsigned long* pfs, int group, int level)
{
    dictionary* ini;
    char mfn[FTI_BUFS], str[FTI_BUFS], *cfn;
    if (level == 0) {
        sprintf(mfn, "%s/sector%d-group%d.fti", FTI_Conf->mTmpDir, FTI_Topo->sectorID, group);
    }
    else {
        sprintf(mfn, "%s/sector%d-group%d.fti", FTI_Ckpt[level].metaDir, FTI_Topo->sectorID, group);
    }
    sprintf(str, "Getting Ptner file size (%s)...", mfn);
    FTI_Print(str, FTI_DBUG);
    if (access(mfn, R_OK) != 0) {
        FTI_Print("FTI metadata file NOT accessible.", FTI_WARN);
        return FTI_NSCS;
    }
    ini = iniparser_load(mfn);
    if (ini == NULL) {
        FTI_Print("Iniparser failed to parse the metadata file.", FTI_WARN);
        return FTI_NSCS;
    }
    //get Ptner file size
    sprintf(str, "%d:Ckpt_file_size", (FTI_Topo->groupRank + FTI_Topo->groupSize - 1) % FTI_Topo->groupSize);
    *pfs = iniparser_getlint(ini, str, -1);
    iniparser_freedict(ini);

    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
    @brief      It gets the metadata to recover the data after a failure.
    @param      fs              Pointer to fill the checkpoint file size.
    @param      mfs             Pointer to fill the maximum file size.
    @param      group           The group in the node.
    @param      level           The level of the ckpt or 0 if tmp.
    @return     integer         FTI_SCES if successfull.

    This function read the metadata file created during checkpointing and
    recover the checkpoint file name, file size and the size of the largest
    file in the group (for padding if ncessary during decoding).

 **/
/*-------------------------------------------------------------------------*/
int FTI_GetMeta(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
                FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt,
                unsigned long* fs, unsigned long* mfs, int group, int level)
{
    dictionary* ini;
    char mfn[FTI_BUFS], str[FTI_BUFS], *cfn;
    if (level == 0) {
        sprintf(mfn, "%s/sector%d-group%d.fti", FTI_Conf->mTmpDir, FTI_Topo->sectorID, group);
    }
    else {
        sprintf(mfn, "%s/sector%d-group%d.fti", FTI_Ckpt[level].metaDir, FTI_Topo->sectorID, group);
    }
    sprintf(str, "Getting FTI metadata file (%s)...", mfn);
    FTI_Print(str, FTI_DBUG);
    if (access(mfn, R_OK) != 0) {
        FTI_Print("FTI metadata file NOT accessible.", FTI_DBUG);
        return FTI_NSCS;
    }
    ini = iniparser_load(mfn);
    if (ini == NULL) {
        FTI_Print("Iniparser failed to parse the metadata file.", FTI_WARN);
        return FTI_NSCS;
    }
    sprintf(str, "%d:Ckpt_file_name", FTI_Topo->groupRank);
    cfn = iniparser_getstring(ini, str, NULL);
    snprintf(FTI_Exec->ckptFile, FTI_BUFS, "%s", cfn);
    sprintf(str, "%d:Ckpt_file_size", FTI_Topo->groupRank);
    *fs = iniparser_getlint(ini, str, -1);
    sprintf(str, "%d:Ckpt_file_maxs", FTI_Topo->groupRank);
    *mfs = iniparser_getlint(ini, str, -1);

    iniparser_freedict(ini);

    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
    @brief      It writes the metadata to recover the data after a failure.
    @param      fs              Pointer to the list of checkpoint sizes.
    @param      mfs             The maximum checkpoint file size.
    @param      fnl             Pointer to the list of checkpoint names.
    @param      member          0 if application process groupID of
                                respective application process if head.
    @return     integer         FTI_SCES if successfull.

    This function should be executed only by one process per group. It
    writes the metadata file used to recover in case of failure.

 **/
/*-------------------------------------------------------------------------*/
int FTI_WriteMetadata(FTIT_configuration* FTI_Conf, FTIT_topology* FTI_Topo,
                      unsigned long* fs, unsigned long mfs, char* fnl, char* checksums, int member)
{
    char str[FTI_BUFS], buf[FTI_BUFS];
    dictionary* ini;
    int i, groupID;

    snprintf(buf, FTI_BUFS, "%s/Topology.fti", FTI_Conf->metadDir);
    sprintf(str, "Temporary load of topology file (%s)...", buf);
    FTI_Print(str, FTI_DBUG);

    groupID = (FTI_Topo->amIaHead) ? member + 1 : FTI_Topo->groupID;

    // To bypass iniparser bug while empty dict.
    ini = iniparser_load(buf);
    if (ini == NULL) {
        FTI_Print("Temporary topology file could NOT be parsed", FTI_WARN);
        return FTI_NSCS;
    }

    // Add metadata to dictionary
    for (i = 0; i < FTI_Topo->groupSize; i++) {
        strncpy(buf, fnl + (i * FTI_BUFS), FTI_BUFS - 1);
        sprintf(str, "%d", i);
        iniparser_set(ini, str, NULL);
        sprintf(str, "%d:Ckpt_file_name", i);
        iniparser_set(ini, str, buf);
        sprintf(str, "%d:Ckpt_file_size", i);
        sprintf(buf, "%ld", fs[i]);
        iniparser_set(ini, str, buf);
        sprintf(str, "%d:Ckpt_file_maxs", i);
        sprintf(buf, "%ld", mfs);
        iniparser_set(ini, str, buf);
        // TODO Checksums only local currently
        if (strlen(checksums)) {
            strncpy(buf, checksums + (i * MD5_DIGEST_LENGTH), MD5_DIGEST_LENGTH);
            sprintf(str, "%d:Ckpt_checksum", i);
            iniparser_set(ini, str, buf);
        }
    }

    // Remove topology section
    iniparser_unset(ini, "topology");
    if (mkdir(FTI_Conf->mTmpDir, 0777) == -1) {
        if (errno != EEXIST) {
            FTI_Print("Cannot create directory", FTI_EROR);
        }
    }

    sprintf(buf, "%s/sector%d-group%d.fti", FTI_Conf->mTmpDir, FTI_Topo->sectorID, groupID);
    if (remove(buf) == -1) {
        if (errno != ENOENT) {
            FTI_Print("Cannot remove sector-group.fti", FTI_EROR);
        }
    }

    sprintf(str, "Creating metadata file (%s)...", buf);
    FTI_Print(str, FTI_DBUG);

    FILE* fd = fopen(buf, "w");
    if (fd == NULL) {
        FTI_Print("Metadata file could NOT be opened.", FTI_WARN);

        iniparser_freedict(ini);

        return FTI_NSCS;
    }

    // Write metadata
    iniparser_dump_ini(ini, fd);

    if (fflush(fd) != 0) {
        FTI_Print("Metadata file could NOT be flushed.", FTI_WARN);

        iniparser_freedict(ini);
        fclose(fd);

        return FTI_NSCS;
    }
    if (fclose(fd) != 0) {
        FTI_Print("Metadata file could NOT be closed.", FTI_WARN);

        iniparser_freedict(ini);

        return FTI_NSCS;
    }

    iniparser_freedict(ini);

    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
    @brief      It writes the metadata to recover the data after a failure.
    @param      globalTmp       1 if using global temporary directory.
    @param      member          0 if application process groupID of
                                respective application process if head.
    @return     integer         FTI_SCES if successfull.

    This function gathers information about the checkpoint files in the
    group (name and sizes), and creates the metadata file used to recover in
    case of failure.

 **/
/*-------------------------------------------------------------------------*/
int FTI_CreateMetadata(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
                       FTIT_topology* FTI_Topo, int globalTmp, int member)
{
    char* fnl = talloc(char, FTI_Topo->groupSize* FTI_BUFS);
    unsigned long fs[FTI_BUFS], mfs, tmpo;
    char str[FTI_BUFS], buf[FTI_BUFS], checksum[MD5_DIGEST_LENGTH];
    struct stat fileStatus;
    int i, res;
    if (globalTmp) {
        sprintf(buf, "%s/%s", FTI_Conf->gTmpDir, FTI_Exec->ckptFile);
    }
    else {
        sprintf(buf, "%s/%s", FTI_Conf->lTmpDir, FTI_Exec->ckptFile);
    }
    if (stat(buf, &fileStatus) == 0) { // Getting size of files
        fs[FTI_Topo->groupRank] = (unsigned long)FTI_Exec->meta[member].fs;
    }
    else {
        FTI_Print("Error with stat on the checkpoint file.", FTI_WARN);

        free(fnl);

        return FTI_NSCS;
    }
    sprintf(str, "Checkpoint file size : %ld bytes.", fs[FTI_Topo->groupRank]);
    FTI_Print(str, FTI_DBUG);
    
    sprintf(fnl + (FTI_Topo->groupRank * FTI_BUFS), "%s", FTI_Exec->ckptFile);
    tmpo = fs[FTI_Topo->groupRank]; // Gather all the file sizes
    MPI_Allgather(&tmpo, 1, MPI_UNSIGNED_LONG, fs, 1, MPI_UNSIGNED_LONG, FTI_Exec->groupComm);
    strncpy(str, fnl + (FTI_Topo->groupRank * FTI_BUFS), FTI_BUFS - 1); // Gather all the file names
    MPI_Allgather(str, FTI_BUFS, MPI_CHAR, fnl, FTI_BUFS, MPI_CHAR, FTI_Exec->groupComm);
    mfs = 0;
    for (i = 0; i < FTI_Topo->groupSize; i++) {
        if (fs[i] > mfs) {
            mfs = fs[i]; // Search max. size
        }
    }
    FTI_Exec->meta[0].maxFs; 
    sprintf(str, "Max. file size %ld.", mfs);
    FTI_Print(str, FTI_DBUG);
    
    // TODO Checksums only local currently
    char* checksums;
    if (!globalTmp) {
        res = FTI_Checksum(buf, checksum);
        checksums = talloc(char, FTI_Topo->groupSize * MD5_DIGEST_LENGTH);
        MPI_Allgather(checksum, MD5_DIGEST_LENGTH, MPI_CHAR, checksums, MD5_DIGEST_LENGTH, MPI_CHAR, FTI_Exec->groupComm);
    } else {
        checksums = talloc(char, FTI_BUFS);
        checksums[0]=0;
    }

    if (res == FTI_NSCS) {
        free(fnl);

        return FTI_NSCS;
    }

    if (FTI_Topo->groupRank == 0) { // Only one process in the group create the metadata
        res = FTI_Try(FTI_WriteMetadata(FTI_Conf, FTI_Topo, fs, mfs, fnl, checksums, member), "write the metadata.");
        if (res == FTI_NSCS) {
            free(fnl);
            free(checksums);

            return FTI_NSCS;
        }
    }

    free(fnl);
    free(checksums);

    return FTI_SCES;
}
