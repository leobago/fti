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
    @param      FTI_Conf        Configuration metadata.
    @param      FTI_Topo        Topology metadata.
    @param      FTI_Ckpt        Checkpoint metadata.
    @param      checksum        Pointer to fill the checkpoint checksum.
    @param      ptnerChecksum   Pointer to fill the ptner file checksum.
    @param      rsChecksum      Pointer to fill the RS file checksum.
    @param      group           The group in the node.
    @param      level           The level of the ckpt or 0 if tmp.
    @return     integer         FTI_SCES if successful.

    This function reads the metadata file created during checkpointing and
    recovers the checkpoint checksum. If there is no RS file, rsChecksum
    string length is 0.

 **/
/*-------------------------------------------------------------------------*/
int FTI_GetChecksums(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
                    FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt,
                    char* checksum, char* ptnerChecksum, char* rsChecksum)
{
    char mfn[FTI_BUFS]; //Path to the metadata file
    char str[FTI_BUFS]; //For console output
    if (FTI_Exec->ckptLvel == 0) {
        sprintf(mfn, "%s/sector%d-group%d.fti", FTI_Conf->mTmpDir, FTI_Topo->sectorID, FTI_Topo->groupID);
    }
    else {
        sprintf(mfn, "%s/sector%d-group%d.fti", FTI_Ckpt[FTI_Exec->ckptLvel].metaDir, FTI_Topo->sectorID, FTI_Topo->groupID);
    }

    sprintf(str, "Getting FTI metadata file (%s)...", mfn);
    FTI_Print(str, FTI_DBUG);
    if (access(mfn, R_OK) != 0) {
        FTI_Print("FTI metadata file NOT accessible.", FTI_WARN);
        return FTI_NSCS;
    }
    dictionary* ini = iniparser_load(mfn);
    if (ini == NULL) {
        FTI_Print("Iniparser failed to parse the metadata file.", FTI_WARN);
        return FTI_NSCS;
    }

    //Get checksum of checkpoint file
    sprintf(str, "%d:Ckpt_checksum", FTI_Topo->groupRank);
    char* checksumTemp = iniparser_getstring(ini, str, NULL);
    strncpy(checksum, checksumTemp, MD5_DIGEST_LENGTH);

    //Get checksum of partner checkpoint file
    sprintf(str, "%d:Ckpt_checksum", (FTI_Topo->groupRank + FTI_Topo->groupSize - 1) % FTI_Topo->groupSize);
    checksumTemp = iniparser_getstring(ini, str, NULL);
    strncpy(ptnerChecksum, checksumTemp, MD5_DIGEST_LENGTH);

    //Get checksum of Reed-Salomon file
    sprintf(str, "%d:RSed_checksum", FTI_Topo->groupRank);
    checksumTemp = iniparser_getstring(ini, str, NULL);

    //If RS checksum don't exists length set to 0;
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
    @param      FTI_Conf        Configuration metadata.
    @param      FTI_Exec        Execution metadata.
    @param      FTI_Topo        Topology metadata.
    @param      FTI_Ckpt        Checkpoint metadata.
    @param      rank            global rank of the process
    @param      checksum        Pointer to the checksum.
    @return     integer         FTI_SCES if successful.

    This function should be executed only by one process per group. It
    writes the RSed checksum to the metadata file.

 **/
/*-------------------------------------------------------------------------*/
int FTI_WriteRSedChecksum(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
                            FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt,
                             int rank, char* checksum)
{
    char str[FTI_BUFS], fileName[FTI_BUFS];

    //Calcuate which groupID rank belongs
    int sectorID = rank / (FTI_Topo->groupSize * FTI_Topo->nodeSize);
    int node = rank / FTI_Topo->nodeSize;
    int rankInGroup = node - (sectorID * FTI_Topo->groupSize);
    int groupID = rank % FTI_Topo->nodeSize;

    char* checksums = talloc(char, FTI_Topo->groupSize * MD5_DIGEST_LENGTH);
    MPI_Allgather(checksum, MD5_DIGEST_LENGTH, MPI_CHAR, checksums, MD5_DIGEST_LENGTH, MPI_CHAR, FTI_Exec->groupComm);

    //Only first process in group save RS checksum
    if (rankInGroup) {
        free(checksums);
        return FTI_SCES;
    }

    sprintf(fileName, "%s/sector%d-group%d.fti", FTI_Conf->mTmpDir, FTI_Topo->sectorID, groupID);
    dictionary* ini = iniparser_load(fileName);
    if (ini == NULL) {
        FTI_Print("Temporary metadata file could NOT be parsed", FTI_WARN);
        free(checksums);
        return FTI_NSCS;
    }
    // Add metadata to dictionary
    int i;
    for (i = 0; i < FTI_Topo->groupSize; i++) {
        char buf[FTI_BUFS];
        strncpy(buf, checksums + (i * MD5_DIGEST_LENGTH), MD5_DIGEST_LENGTH);
        sprintf(str, "%d:RSed_checksum", i);
        iniparser_set(ini, str, buf);
    }
    free(checksums);

    sprintf(str, "Recreating metadata file (%s)...", fileName);
    FTI_Print(str, FTI_DBUG);

    FILE* fd = fopen(fileName, "w");
    if (fd == NULL) {
        FTI_Print("Metadata file could NOT be opened.", FTI_WARN);

        iniparser_freedict(ini);

        return FTI_NSCS;
    }

    // Write metadata
    iniparser_dump_ini(ini, fd);

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
    @brief      It gets the temporary metadata.
    @param      FTI_Conf        Configuration metadata.
    @param      FTI_Exec        Execution metadata.
    @param      FTI_Topo        Topology metadata.
    @param      FTI_Ckpt        Checkpoint metadata.
    @return     integer         FTI_SCES if successful.

    This function reads the temporary metadata file created during checkpointing and
    recovers the checkpoint file name, file size, partner file size and the size
    of the largest file in the group (for padding if necessary during decoding).

 **/
/*-------------------------------------------------------------------------*/
int FTI_LoadTmpMeta(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
                FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt)
{
    if (FTI_Topo->amIaHead) { //I am a head
        int j, biggestCkptID = 0; //Need to find biggest CkptID
        for (j = 1; j < FTI_Topo->nodeSize; j++) { //all body processes
            char metaFileName[FTI_BUFS], str[FTI_BUFS];
            sprintf(metaFileName, "%s/sector%d-group%d.fti", FTI_Conf->mTmpDir, FTI_Topo->sectorID, j);
            sprintf(str, "Getting FTI metadata file (%s)...", metaFileName);
            FTI_Print(str, FTI_DBUG);
            if (access(metaFileName, R_OK) == 0) {
                dictionary* ini = iniparser_load(metaFileName);
                if (ini == NULL) {
                    FTI_Print("Iniparser failed to parse the metadata file.", FTI_WARN);
                    return FTI_NSCS;
                }
                else {
                    FTI_Exec->meta[0].exists[j] = 1;

                    sprintf(str, "%d:Ckpt_file_name", FTI_Topo->groupRank);
                    char* ckptFileName = iniparser_getstring(ini, str, NULL);
                    snprintf(&FTI_Exec->meta[0].ckptFile[j * FTI_BUFS], FTI_BUFS, "%s", ckptFileName);

                    //update head's ckptID
                    sscanf(&FTI_Exec->meta[0].ckptFile[j * FTI_BUFS], "Ckpt%d", &FTI_Exec->ckptID);
                    if (FTI_Exec->ckptID < biggestCkptID) {
                        FTI_Exec->ckptID = biggestCkptID;
                    }

                    sprintf(str, "%d:Ckpt_file_size", FTI_Topo->groupRank);
                    FTI_Exec->meta[0].fs[j] = iniparser_getlint(ini, str, -1);

                    sprintf(str, "%d:Ckpt_file_size", (FTI_Topo->groupRank + FTI_Topo->groupSize - 1) % FTI_Topo->groupSize);
                    FTI_Exec->meta[0].pfs[j] = iniparser_getlint(ini, str, -1);

                    FTI_Exec->meta[0].maxFs[j] = iniparser_getlint(ini, "0:Ckpt_file_maxs", -1);

                    int k;
                    for (k = 0; k < FTI_BUFS; k++) {
                        sprintf(str, "%d:Var%d_id", FTI_Topo->groupRank, k);
                        int id = iniparser_getint(ini, str, -1);
                        if (id == -1) {
                            //No more variables
                            break;
                        }
                        //Variable exists
                        FTI_Exec->meta[0].varID[j * FTI_BUFS + k] = id;

                        sprintf(str, "%d:Var%d_size", FTI_Topo->groupRank, k);
                        FTI_Exec->meta[0].varSize[j * FTI_BUFS + k] = iniparser_getlint(ini, str, -1);
                    }
                    //Save number of variables in metadata
                    FTI_Exec->meta[0].nbVar[j] = k;

                    iniparser_freedict(ini);
                }
            }
            else {
                sprintf(str, "Temporary metadata do not exist for node process %d.", j);
                FTI_Print(str, FTI_WARN);
                return FTI_NSCS;
            }
        }
    }
    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
    @brief      It gets the metadata to recover the data after a failure.
    @param      FTI_Conf        Configuration metadata.
    @param      FTI_Exec        Execution metadata.
    @param      FTI_Topo        Topology metadata.
    @param      FTI_Ckpt        Checkpoint metadata.
    @return     integer         FTI_SCES if successful.

    This function reads the metadata file created during checkpointing and
    recovers the checkpoint file name, file size, partner file size and the size
    of the largest file in the group (for padding if necessary during decoding).

 **/
/*-------------------------------------------------------------------------*/
int FTI_LoadMeta(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
                FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt)
{
    if (!FTI_Topo->amIaHead) {
        int i;
        for (i = 0; i < 5; i++) { //for each level
            char metaFileName[FTI_BUFS], str[FTI_BUFS];
            if (i == 0) {
                sprintf(metaFileName, "%s/sector%d-group%d.fti", FTI_Conf->mTmpDir, FTI_Topo->sectorID, FTI_Topo->groupID);
            } else {
                sprintf(metaFileName, "%s/sector%d-group%d.fti", FTI_Ckpt[i].metaDir, FTI_Topo->sectorID, FTI_Topo->groupID);
            }
            sprintf(str, "Getting FTI metadata file (%s)...", metaFileName);
            FTI_Print(str, FTI_DBUG);
            if (access(metaFileName, R_OK) == 0) {
                dictionary* ini = iniparser_load(metaFileName);
                if (ini == NULL) {
                    FTI_Print("Iniparser failed to parse the metadata file.", FTI_WARN);
                    return FTI_NSCS;
                }
                else {
                    sprintf(str, "Meta for level %d exists.", i);
                    FTI_Print(str, FTI_DBUG);
                    FTI_Exec->meta[i].exists[0] = 1;

                    sprintf(str, "%d:Ckpt_file_name", FTI_Topo->groupRank);
                    char* ckptFileName = iniparser_getstring(ini, str, NULL);
                    snprintf(FTI_Exec->meta[i].ckptFile, FTI_BUFS, "%s", ckptFileName);

                    sprintf(str, "%d:Ckpt_file_size", FTI_Topo->groupRank);
                    FTI_Exec->meta[i].fs[0] = iniparser_getlint(ini, str, -1);

                    sprintf(str, "%d:Ckpt_file_size", (FTI_Topo->groupRank + FTI_Topo->groupSize - 1) % FTI_Topo->groupSize);
                    FTI_Exec->meta[i].pfs[0] = iniparser_getlint(ini, str, -1);

                    FTI_Exec->meta[i].maxFs[0] = iniparser_getlint(ini, "0:Ckpt_file_maxs", -1);

                    int k;
                    for (k = 0; k < FTI_BUFS; k++) {
                        sprintf(str, "%d:Var%d_id", FTI_Topo->groupRank, k);
                        int id = iniparser_getint(ini, str, -1);
                        if (id == -1) {
                            //No more variables
                            break;
                        }
                        //Variable exists
                        FTI_Exec->meta[i].varID[k] = id;

                        sprintf(str, "%d:Var%d_size", FTI_Topo->groupRank, k);
                        FTI_Exec->meta[i].varSize[k] = iniparser_getlint(ini, str, -1);
                    }
                    //Save number of variables in metadata
                    FTI_Exec->meta[i].nbVar[0] = k;

                    iniparser_freedict(ini);
                }
            }
        }
    }
    else { //I am a head
        int biggestCkptID = 0;
        int i;
        for (i = 0; i < 5; i++) {        //for each level
            int j;
            for (j = 1; j < FTI_Topo->nodeSize; j++) { //for all body processes
                dictionary* ini;
                char metaFileName[FTI_BUFS], str[FTI_BUFS];
                if (i == 0) {
                    sprintf(metaFileName, "%s/sector%d-group%d.fti", FTI_Conf->mTmpDir, FTI_Topo->sectorID, j);
                } else {
                    sprintf(metaFileName, "%s/sector%d-group%d.fti", FTI_Ckpt[i].metaDir, FTI_Topo->sectorID, j);
                }
                sprintf(str, "Getting FTI metadata file (%s)...", metaFileName);
                FTI_Print(str, FTI_DBUG);
                if (access(metaFileName, R_OK) == 0) {
                    ini = iniparser_load(metaFileName);
                    if (ini == NULL) {
                        FTI_Print("Iniparser failed to parse the metadata file.", FTI_WARN);
                        return FTI_NSCS;
                    }
                    else {
                        sprintf(str, "Meta for level %d exists.", i);
                        FTI_Print(str, FTI_DBUG);
                        FTI_Exec->meta[i].exists[j] = 1;

                        sprintf(str, "%d:Ckpt_file_name", FTI_Topo->groupRank);
                        char* ckptFileName = iniparser_getstring(ini, str, NULL);
                        snprintf(&FTI_Exec->meta[i].ckptFile[j * FTI_BUFS], FTI_BUFS, "%s", ckptFileName);

                        //update heads ckptID
                        sscanf(&FTI_Exec->meta[i].ckptFile[j * FTI_BUFS], "Ckpt%d", &FTI_Exec->ckptID);
                        if (FTI_Exec->ckptID < biggestCkptID) {
                            FTI_Exec->ckptID = biggestCkptID;
                        }

                        sprintf(str, "%d:Ckpt_file_size", FTI_Topo->groupRank);
                        FTI_Exec->meta[i].fs[j] = iniparser_getlint(ini, str, -1);

                        sprintf(str, "%d:Ckpt_file_size", (FTI_Topo->groupRank + FTI_Topo->groupSize - 1) % FTI_Topo->groupSize);
                        FTI_Exec->meta[i].pfs[j] = iniparser_getlint(ini, str, -1);

                        FTI_Exec->meta[i].maxFs[j] = iniparser_getlint(ini, "0:Ckpt_file_maxs", -1);
                        int k;
                        for (k = 0; k < FTI_BUFS; k++) {
                            sprintf(str, "%d:Var%d_id", FTI_Topo->groupRank, k);
                            int id = iniparser_getint(ini, str, -1);
                            if (id == -1) {
                                //No more variables
                                break;
                            }
                            //Variable exists
                            FTI_Exec->meta[i].varID[j * FTI_BUFS + k] = id;

                            sprintf(str, "%d:Var%d_size", FTI_Topo->groupRank, k);
                            FTI_Exec->meta[i].varSize[j * FTI_BUFS + k] = iniparser_getlint(ini, str, -1);
                        }
                        //Save number of variables in metadata
                        FTI_Exec->meta[i].nbVar[j] = k;

                        iniparser_freedict(ini);
                    }
                }
            }
        }
    }
    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
    @brief      It writes the metadata to recover the data after a failure.
    @param      FTI_Conf        Configuration metadata.
    @param      FTI_Topo        Topology metadata.
    @param      fs              Pointer to the list of checkpoint sizes.
    @param      mfs             The maximum checkpoint file size.
    @param      fnl             Pointer to the list of checkpoint names.
    @param      checksums       Checksums array.
    @return     integer         FTI_SCES if successful.

    This function should be executed only by one process per group. It
    writes the metadata file used to recover in case of failure.

 **/
/*-------------------------------------------------------------------------*/
int FTI_WriteMetadata(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
                      FTIT_topology* FTI_Topo, long* fs, long mfs, char* fnl,
                      char* checksums, int* allVarIDs, long* allVarSizes)
{
    char str[FTI_BUFS], buf[FTI_BUFS];
    snprintf(buf, FTI_BUFS, "%s/Topology.fti", FTI_Conf->metadDir);
    sprintf(str, "Temporary load of topology file (%s)...", buf);
    FTI_Print(str, FTI_DBUG);

    // To bypass iniparser bug while empty dict.
    dictionary* ini = iniparser_load(buf);
    if (ini == NULL) {
        FTI_Print("Temporary topology file could NOT be parsed", FTI_WARN);
        return FTI_NSCS;
    }

    // Add metadata to dictionary
    int i;
    for (i = 0; i < FTI_Topo->groupSize; i++) {
        strncpy(buf, fnl + (i * FTI_BUFS), FTI_BUFS - 1);
        sprintf(str, "%d", i);
        iniparser_set(ini, str, NULL);
        sprintf(str, "%d:Ckpt_file_name", i);
        iniparser_set(ini, str, buf);
        sprintf(str, "%d:Ckpt_file_size", i);
        sprintf(buf, "%lu", fs[i]);
        iniparser_set(ini, str, buf);
        sprintf(str, "%d:Ckpt_file_maxs", i);
        sprintf(buf, "%lu", mfs);
        iniparser_set(ini, str, buf);
        // TODO Checksums only local currently
        if (strlen(checksums)) {
            strncpy(buf, checksums + (i * MD5_DIGEST_LENGTH), MD5_DIGEST_LENGTH);
            sprintf(str, "%d:Ckpt_checksum", i);
            iniparser_set(ini, str, buf);
        }
        int j;
        for (j = 0; j < FTI_Exec->nbVar; j++) {
            //Save id of variable
            sprintf(str, "%d:Var%d_id", i, j);
            sprintf(buf, "%d", allVarIDs[i * FTI_Exec->nbVar + j]);
            iniparser_set(ini, str, buf);

            //Save size of variable
            sprintf(str, "%d:Var%d_size", i, j);
            sprintf(buf, "%ld", allVarSizes[i * FTI_Exec->nbVar + j]);
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

    sprintf(buf, "%s/sector%d-group%d.fti", FTI_Conf->mTmpDir, FTI_Topo->sectorID, FTI_Topo->groupID);
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
    @param      FTI_Conf        Configuration metadata.
    @param      FTI_Exec        Execution metadata.
    @param      FTI_Topo        Topology metadata.
    @param      FTI_Ckpt        Checkpoint metadata.
    @return     integer         FTI_SCES if successful.

    This function gathers information about the checkpoint files in the
    group (name and sizes), and creates the metadata file used to recover in
    case of failure.

 **/
/*-------------------------------------------------------------------------*/
int FTI_CreateMetadata(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
                       FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt,
                       FTIT_dataset* FTI_Data)
{
    FTI_Exec->meta[0].fs[0] = FTI_Exec->ckptSize;
    long fs = FTI_Exec->meta[0].fs[0]; // Gather all the file sizes
    long fileSizes[FTI_BUFS];
    MPI_Allgather(&fs, 1, MPI_LONG, fileSizes, 1, MPI_LONG, FTI_Exec->groupComm);

    //update partner file size:
    if (FTI_Exec->ckptLvel == 2) {
        int ptnerGroupRank = (FTI_Topo->groupRank + FTI_Topo->groupSize - 1) % FTI_Topo->groupSize;
        FTI_Exec->meta[0].pfs[0] = fileSizes[ptnerGroupRank];
    }

    long mfs = 0; //Max file size in group
    int i;
    for (i = 0; i < FTI_Topo->groupSize; i++) {
        if (fileSizes[i] > mfs) {
            mfs = fileSizes[i]; // Search max. size
        }
    }
    FTI_Exec->meta[0].maxFs[0] = mfs;
    char str[FTI_BUFS]; //For console output
    sprintf(str, "Max. file size in group %lu.", mfs);
    FTI_Print(str, FTI_DBUG);

    char* ckptFileNames = talloc(char, FTI_Topo->groupSize * FTI_BUFS);
    strcpy(str, FTI_Exec->meta[0].ckptFile); // Gather all the file names
    MPI_Gather(str, FTI_BUFS, MPI_CHAR, ckptFileNames, FTI_BUFS, MPI_CHAR, 0, FTI_Exec->groupComm);

    // TODO Checksums only local currently
    char* checksums;
    if (!(FTI_Exec->ckptLvel == 4 && FTI_Ckpt[4].isInline)) {
        char lfn[FTI_BUFS];
        sprintf(lfn, "%s/%s", FTI_Conf->lTmpDir, FTI_Exec->meta[0].ckptFile);
        char checksum[MD5_DIGEST_LENGTH];
        int res = FTI_Checksum(lfn, checksum);
        checksums = talloc(char, FTI_Topo->groupSize * MD5_DIGEST_LENGTH);
        MPI_Allgather(checksum, MD5_DIGEST_LENGTH, MPI_CHAR, checksums, MD5_DIGEST_LENGTH, MPI_CHAR, FTI_Exec->groupComm);
        if (res == FTI_NSCS) {
            free(ckptFileNames);
            free(checksums);

            return FTI_NSCS;
        }
    } else {
        checksums = talloc(char, FTI_BUFS);
        checksums[0]=0;
    }

    //Every process has the same number of protected variables
    int* myVarIDs = talloc(int, FTI_Exec->nbVar);
    int* allVarIDs = talloc(int, FTI_Topo->groupSize * FTI_Exec->nbVar);
    long* myVarSizes = talloc(long, FTI_Exec->nbVar);
    long* allVarSizes = talloc(long, FTI_Topo->groupSize * FTI_Exec->nbVar);
    for (i = 0; i < FTI_Exec->nbVar; i++) {
        myVarIDs[i] = FTI_Data[i].id;
        myVarSizes[i] =  FTI_Data[i].size;
    }
    //Gather variables IDs
    MPI_Gather(myVarIDs, FTI_Exec->nbVar, MPI_INT, allVarIDs, FTI_Exec->nbVar, MPI_INT, 0, FTI_Exec->groupComm);
    //Gather variables sizes
    MPI_Gather(myVarSizes, FTI_Exec->nbVar, MPI_LONG, allVarSizes, FTI_Exec->nbVar, MPI_LONG, 0, FTI_Exec->groupComm);

    free(myVarIDs);
    free(myVarSizes);

    if (FTI_Topo->groupRank == 0) { // Only one process in the group create the metadata
        int res = FTI_Try(FTI_WriteMetadata(FTI_Conf, FTI_Exec, FTI_Topo, fileSizes, mfs,
                    ckptFileNames, checksums, allVarIDs, allVarSizes), "write the metadata.");
        free(allVarIDs);
        free(allVarSizes);
        if (res == FTI_NSCS) {
            free(ckptFileNames);
            free(checksums);

            return FTI_NSCS;
        }
    } else {
        free(allVarIDs);
        free(allVarSizes);
    }

    //Flush metadata in case postCkpt done inline
    FTI_Exec->meta[FTI_Exec->ckptLvel].fs[0] = FTI_Exec->meta[0].fs[0];
    FTI_Exec->meta[FTI_Exec->ckptLvel].pfs[0] = FTI_Exec->meta[0].pfs[0];
    FTI_Exec->meta[FTI_Exec->ckptLvel].maxFs[0] = FTI_Exec->meta[0].maxFs[0];
    strcpy(FTI_Exec->meta[FTI_Exec->ckptLvel].ckptFile, FTI_Exec->meta[0].ckptFile);
    for (i = 0; i < FTI_Exec->nbVar; i++) {
        FTI_Exec->meta[0].varID[i] = FTI_Data[i].id;
        FTI_Exec->meta[0].varSize[i] = FTI_Data[i].size;
    }

    free(ckptFileNames);
    free(checksums);

    return FTI_SCES;
}
