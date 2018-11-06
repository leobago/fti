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
 *  @file   meta.c
 *  @date   October, 2017
 *  @brief  Metadata functions for the FTI library.
 */

#include "interface.h"

/*-------------------------------------------------------------------------*/
/**
  @brief      It gets the checksums from metadata.
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Topo        Topology metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @param      checksum        Pointer to fill the checkpoint checksum.
  @param      ptnerChecksum   Pointer to fill the ptner file checksum.
  @param      rsChecksum      Pointer to fill the RS file checksum.
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
        snprintf(mfn, FTI_BUFS, "%s/sector%d-group%d.fti", FTI_Conf->mTmpDir, FTI_Topo->sectorID, FTI_Topo->groupID);
    }
    else {
        snprintf(mfn, FTI_BUFS, "%s/sector%d-group%d.fti", FTI_Ckpt[FTI_Exec->ckptLvel].metaDir, FTI_Topo->sectorID, FTI_Topo->groupID);
    }

    snprintf(str, FTI_BUFS, "Getting FTI metadata file (%s)...", mfn);
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
    snprintf(str, FTI_BUFS, "%d:Ckpt_checksum", FTI_Topo->groupRank);
    char* checksumTemp = iniparser_getstring(ini, str, "");
    strncpy(checksum, checksumTemp, MD5_DIGEST_STRING_LENGTH);

    //Get checksum of partner checkpoint file
    snprintf(str, FTI_BUFS, "%d:Ckpt_checksum", (FTI_Topo->groupRank + FTI_Topo->groupSize - 1) % FTI_Topo->groupSize);
    checksumTemp = iniparser_getstring(ini, str, "");
    strncpy(ptnerChecksum, checksumTemp, MD5_DIGEST_STRING_LENGTH);

    //Get checksum of Reed-Salomon file
    snprintf(str, FTI_BUFS, "%d:RSed_checksum", FTI_Topo->groupRank);
    checksumTemp = iniparser_getstring(ini, str, "");
    strncpy(rsChecksum, checksumTemp, MD5_DIGEST_STRING_LENGTH);

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
    // Fake call for FTI-FF. checksum is done for the datasets.
    if (FTI_Conf->ioMode == FTI_IO_FTIFF) {return FTI_SCES;}

    char str[FTI_BUFS], fileName[FTI_BUFS];

    //Calcuate which groupID rank belongs
    int sectorID = rank / (FTI_Topo->groupSize * FTI_Topo->nodeSize);
    int node = rank / FTI_Topo->nodeSize;
    int rankInGroup = node - (sectorID * FTI_Topo->groupSize);
    int groupID = rank % FTI_Topo->nodeSize;

    char* checksums = talloc(char, FTI_Topo->groupSize * MD5_DIGEST_STRING_LENGTH);
    MPI_Allgather(checksum, MD5_DIGEST_STRING_LENGTH, MPI_CHAR, checksums, MD5_DIGEST_STRING_LENGTH, MPI_CHAR, FTI_Exec->groupComm);

    //Only first process in group save RS checksum
    if (rankInGroup) {
        free(checksums);
        return FTI_SCES;
    }

    snprintf(fileName, FTI_BUFS, "%s/sector%d-group%d.fti", FTI_Conf->mTmpDir, FTI_Topo->sectorID, groupID);
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
        strncpy(buf, checksums + (i * MD5_DIGEST_STRING_LENGTH), MD5_DIGEST_STRING_LENGTH);
        snprintf(str, FTI_BUFS, "%d:RSed_checksum", i);
        iniparser_set(ini, str, buf);
    }
    free(checksums);

    snprintf(str, FTI_BUFS, "Recreating metadata file (%s)...", fileName);
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
    // no metadata files for FTI-FF
    if ( FTI_Conf->ioMode == FTI_IO_FTIFF ) { return FTI_SCES; }
    if (FTI_Topo->amIaHead) { //I am a head
        int j, biggestCkptID = 0; //Need to find biggest CkptID
        for (j = 1; j < FTI_Topo->nodeSize; j++) { //all body processes
            char metaFileName[FTI_BUFS], str[FTI_BUFS];
            snprintf(metaFileName, FTI_BUFS, "%s/sector%d-group%d.fti", FTI_Conf->mTmpDir, FTI_Topo->sectorID, j);
            snprintf(str, FTI_BUFS, "Getting FTI metadata file (%s)...", metaFileName);
            FTI_Print(str, FTI_DBUG);
            if (access(metaFileName, R_OK) == 0) {
                dictionary* ini = iniparser_load(metaFileName);
                if (ini == NULL) {
                    FTI_Print("Iniparser failed to parse the metadata file.", FTI_WARN);
                    return FTI_NSCS;
                }
                else {
                    FTI_Exec->meta[0].exists[j] = 1;

                    snprintf(str, FTI_BUFS, "%d:Ckpt_file_name", FTI_Topo->groupRank);
                    char* ckptFileName = iniparser_getstring(ini, str, NULL);
                    snprintf(&FTI_Exec->meta[0].ckptFile[j * FTI_BUFS], FTI_BUFS, "%s", ckptFileName);

                    //update head's ckptID
                    sscanf(&FTI_Exec->meta[0].ckptFile[j * FTI_BUFS], "Ckpt%d", &FTI_Exec->ckptID);
                    if (FTI_Exec->ckptID < biggestCkptID) {
                        FTI_Exec->ckptID = biggestCkptID;
                    }

                    snprintf(str, FTI_BUFS, "%d:Ckpt_file_size", FTI_Topo->groupRank);
                    FTI_Exec->meta[0].fs[j] = iniparser_getlint(ini, str, -1);

                    snprintf(str, FTI_BUFS, "%d:Ckpt_file_size", (FTI_Topo->groupRank + FTI_Topo->groupSize - 1) % FTI_Topo->groupSize);
                    FTI_Exec->meta[0].pfs[j] = iniparser_getlint(ini, str, -1);

                    FTI_Exec->meta[0].maxFs[j] = iniparser_getlint(ini, "0:Ckpt_file_maxs", -1);

                    int k;
                    for (k = 0; k < FTI_BUFS; k++) {
                        snprintf(str, FTI_BUFS, "%d:Var%d_id", FTI_Topo->groupRank, k);
                        int id = iniparser_getint(ini, str, -1);
                        if (id == -1) {
                            //No more variables
                            break;
                        }
                        //Variable exists
                        FTI_Exec->meta[0].varID[j * FTI_BUFS + k] = id;

                        snprintf(str, FTI_BUFS, "%d:Var%d_size", FTI_Topo->groupRank, k);
                        FTI_Exec->meta[0].varSize[j * FTI_BUFS + k] = iniparser_getlint(ini, str, -1);
                    }
                    //Save number of variables in metadata
                    FTI_Exec->meta[0].nbVar[j] = k;

                    iniparser_freedict(ini);
                }
            }
            else {
                snprintf(str, FTI_BUFS, "Temporary metadata do not exist for node process %d.", j);
                FTI_Print(str, FTI_WARN);
                return FTI_NSCS;
            }
        }
    }
    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      It gets the GPU metadata to recover the data after a failure.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Topo        Topology metadata.
  @param      ini             The ini dictionary from which to load GPU metadata.
  @return     integer         FTI_SCES if successful.

  This function reads the metadata file created during checkpointing to recover
  the metadata necessary to re-initialize the GpuInfo member of FTI_Exec.

 **/
/*-------------------------------------------------------------------------*/
static int FTI_LoadGpuMetadata(FTIT_execution* FTI_Exec, FTIT_topology* FTI_Topo, dictionary *ini){
  char str[FTI_BUFS];
  char kernelInfoSection[FTI_BUFS];
  char gpuInfoSection[FTI_BUFS];

  snprintf(kernelInfoSection, FTI_BUFS, "Kernel Info");
  snprintf(str, FTI_BUFS, "%s:nbkernels", kernelInfoSection); 
  FTI_Exec->nbKernels = iniparser_getint(ini, str, -1);

  int i = 0;
  for(i = 0; i < FTI_Exec->nbKernels; i++){

    /* Allocations */
    int *id                       = talloc(int, 1);  
    size_t *block_amt             = talloc(size_t, 1);
    bool *complete                = talloc(bool, 1);
    unsigned int *quantum         = talloc(unsigned int, 1);
    FTI_Exec->gpuInfo[i].all_done = talloc(bool, FTI_Topo->nbProc);

    /* Return if any allocations failed */
    if(id                             == NULL){return FTI_NSCS;}
    if(block_amt                      == NULL){return FTI_NSCS;}
    if(complete                       == NULL){return FTI_NSCS;}
    if(quantum                        == NULL){return FTI_NSCS;}
    if(FTI_Exec->gpuInfo[i].all_done  == NULL){return FTI_NSCS;}

    snprintf(str, FTI_BUFS, "%s:id%d", kernelInfoSection, i);

    *id = iniparser_getint(ini, str, -1);
    FTI_Exec->gpuInfo[i].id = id;

    snprintf(gpuInfoSection, FTI_BUFS, "%dGPU Info%d", FTI_Topo->groupRank, *FTI_Exec->gpuInfo[i].id);
    snprintf(str, FTI_BUFS, "%s:block_amt", gpuInfoSection);
    char *str_block_amt = iniparser_getstring(ini, str, NULL);

    sscanf(str_block_amt, "%zu", block_amt);
    FTI_Exec->gpuInfo[i].block_amt = block_amt;

    snprintf(str, FTI_BUFS, "%s:complete", gpuInfoSection);
    *complete = iniparser_getboolean(ini, str, -1);
    FTI_Exec->gpuInfo[i].complete = complete;

    snprintf(str, FTI_BUFS, "%s:quantum", gpuInfoSection);

    char *str_quantum = iniparser_getstring(ini, str, NULL); 
    sscanf(str_quantum, "%u", quantum);
    FTI_Exec->gpuInfo[i].quantum = quantum;

    size_t idx = 0;
    for(idx = 0; idx < FTI_Topo->nbProc; idx++){
      snprintf(str, FTI_BUFS, "%s:all_done%zu", gpuInfoSection, idx);
      FTI_Exec->gpuInfo[i].all_done[idx] = iniparser_getboolean(ini, str, -1);
    }

    FTI_Exec->gpuInfo[i].h_is_block_executed = malloc(sizeof(bool) * *FTI_Exec->gpuInfo[i].block_amt);

    if(FTI_Exec->gpuInfo[i].h_is_block_executed == NULL){
      return FTI_NSCS;
    }

    for(idx = 0; idx < *FTI_Exec->gpuInfo[i].block_amt; idx++){
      snprintf(str, FTI_BUFS, "%s:block%zu", gpuInfoSection, idx);
      FTI_Exec->gpuInfo[i].h_is_block_executed[idx] = iniparser_getboolean(ini, str, -1);
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
    // no metadata files for FTI-FF
    if ( FTI_Conf->ioMode == FTI_IO_FTIFF ) { return FTI_SCES; }
    if (!FTI_Topo->amIaHead) {
        int i;
        for (i = 0; i < 5; i++) { //for each level
            char metaFileName[FTI_BUFS], str[FTI_BUFS];
            if (i == 0) {
                snprintf(metaFileName, FTI_BUFS, "%s/sector%d-group%d.fti", FTI_Conf->mTmpDir, FTI_Topo->sectorID, FTI_Topo->groupID);
            } else {
                snprintf(metaFileName, FTI_BUFS, "%s/sector%d-group%d.fti", FTI_Ckpt[i].metaDir, FTI_Topo->sectorID, FTI_Topo->groupID);
            }
            snprintf(str, FTI_BUFS, "Getting FTI metadata file (%s)...", metaFileName);
            FTI_Print(str, FTI_DBUG);
            if (access(metaFileName, R_OK) == 0) {
                dictionary* ini = iniparser_load(metaFileName);
                if (ini == NULL) {
                    FTI_Print("Iniparser failed to parse the metadata file.", FTI_WARN);
                    return FTI_NSCS;
                }
                else {
                    snprintf(str, FTI_BUFS, "Meta for level %d exists.", i);
                    FTI_Print(str, FTI_DBUG);
                    FTI_Exec->meta[i].exists[0] = 1;

                    snprintf(str, FTI_BUFS, "%d:Ckpt_file_name", FTI_Topo->groupRank);
                    char* ckptFileName = iniparser_getstring(ini, str, NULL);
                    snprintf(FTI_Exec->meta[i].ckptFile, FTI_BUFS, "%s", ckptFileName);

                    snprintf(str, FTI_BUFS, "%d:Ckpt_file_size", FTI_Topo->groupRank);
                    FTI_Exec->meta[i].fs[0] = iniparser_getlint(ini, str, -1);

                    snprintf(str, FTI_BUFS, "%d:Ckpt_file_size", (FTI_Topo->groupRank + FTI_Topo->groupSize - 1) % FTI_Topo->groupSize);
                    FTI_Exec->meta[i].pfs[0] = iniparser_getlint(ini, str, -1);

                    FTI_Exec->meta[i].maxFs[0] = iniparser_getlint(ini, "0:Ckpt_file_maxs", -1);

                    int k;
                    for (k = 0; k < FTI_BUFS; k++) {
                        snprintf(str, FTI_BUFS, "%d:Var%d_id", FTI_Topo->groupRank, k);
                        int id = iniparser_getint(ini, str, -1);
                        if (id == -1) {
                            //No more variables
                            break;
                        }
                        //Variable exists
                        FTI_Exec->meta[i].varID[k] = id;

                        snprintf(str, FTI_BUFS, "%d:Var%d_size", FTI_Topo->groupRank, k);
                        FTI_Exec->meta[i].varSize[k] = iniparser_getlint(ini, str, -1);
                    }
                    //Save number of variables in metadata
                    FTI_Exec->meta[i].nbVar[0] = k;
                    int sectionExists = iniparser_find_entry(ini, "Kernel Info");

                      if(sectionExists){
                        int res = FTI_Try(FTI_LoadGpuMetadata(FTI_Exec, FTI_Topo, ini), "load gpu metadata");

                        if(res == FTI_NSCS){
                          return FTI_NSCS;
                        }
                      }
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
                    snprintf(metaFileName, FTI_BUFS, "%s/sector%d-group%d.fti", FTI_Conf->mTmpDir, FTI_Topo->sectorID, j);
                } else {
                    snprintf(metaFileName, FTI_BUFS, "%s/sector%d-group%d.fti", FTI_Ckpt[i].metaDir, FTI_Topo->sectorID, j);
                }
                snprintf(str, FTI_BUFS, "Getting FTI metadata file (%s)...", metaFileName);
                FTI_Print(str, FTI_DBUG);
                if (access(metaFileName, R_OK) == 0) {
                    ini = iniparser_load(metaFileName);
                    if (ini == NULL) {
                        FTI_Print("Iniparser failed to parse the metadata file.", FTI_WARN);
                        return FTI_NSCS;
                    }
                    else {
                        snprintf(str, FTI_BUFS, "Meta for level %d exists.", i);
                        FTI_Print(str, FTI_DBUG);
                        FTI_Exec->meta[i].exists[j] = 1;

                        snprintf(str, FTI_BUFS, "%d:Ckpt_file_name", FTI_Topo->groupRank);
                        char* ckptFileName = iniparser_getstring(ini, str, NULL);
                        snprintf(&FTI_Exec->meta[i].ckptFile[j * FTI_BUFS], FTI_BUFS, "%s", ckptFileName);

                        //update heads ckptID
                        sscanf(&FTI_Exec->meta[i].ckptFile[j * FTI_BUFS], "Ckpt%d", &FTI_Exec->ckptID);
                        if (FTI_Exec->ckptID < biggestCkptID) {
                            FTI_Exec->ckptID = biggestCkptID;
                        }

                        snprintf(str, FTI_BUFS, "%d:Ckpt_file_size", FTI_Topo->groupRank);
                        FTI_Exec->meta[i].fs[j] = iniparser_getlint(ini, str, -1);

                        snprintf(str, FTI_BUFS, "%d:Ckpt_file_size", (FTI_Topo->groupRank + FTI_Topo->groupSize - 1) % FTI_Topo->groupSize);
                        FTI_Exec->meta[i].pfs[j] = iniparser_getlint(ini, str, -1);

                        FTI_Exec->meta[i].maxFs[j] = iniparser_getlint(ini, "0:Ckpt_file_maxs", -1);
                        int k;
                        for (k = 0; k < FTI_BUFS; k++) {
                            snprintf(str, FTI_BUFS, "%d:Var%d_id", FTI_Topo->groupRank, k);
                            int id = iniparser_getint(ini, str, -1);
                            if (id == -1) {
                                //No more variables
                                break;
                            }
                            //Variable exists
                            FTI_Exec->meta[i].varID[j * FTI_BUFS + k] = id;

                            snprintf(str, FTI_BUFS, "%d:Var%d_size", FTI_Topo->groupRank, k);
                            FTI_Exec->meta[i].varSize[j * FTI_BUFS + k] = iniparser_getlint(ini, str, -1);
                        }
                        //Save number of variables in metadata
                        FTI_Exec->meta[i].nbVar[j] = k;
                        int sectionExists = iniparser_find_entry(ini, "Kernel Info");

                        if(sectionExists){
                          int res = FTI_Try(FTI_LoadGpuMetadata(FTI_Exec, FTI_Topo, ini), "load gpu metadata");

                          if(res == FTI_NSCS){
                            return FTI_NSCS;
                          }
                        }
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
  @brief      It writes the GPU metadata to recover the data after a failure.
  @param      FTI_Exec              Execution metadata.
  @param      FTI_Topo              Topology metadata.
  @param      FTI_GpuInfoMetadata   GPU metadata.
  @param      ini                   The metadata ini dictionary to write to.
  @param      groupIdx              The process's group index to which the GPU information belongs.
  @return     integer               FTI_SCES if successful.

  This function should only be called in FTI_Writemetadata. It writes the
  metadata to recover partially executed GPU kernels.
 **/
/*-------------------------------------------------------------------------*/
static int FTI_WriteGpuMetadata(FTIT_execution* FTI_Exec, FTIT_topology* FTI_Topo, FTIT_gpuInfoMetadata* FTI_GpuInfoMetadata, dictionary *ini, int groupIdx){
	char str[FTI_BUFS]; 
  char buf[FTI_BUFS];
  char kernelInfoSection[FTI_BUFS];
  char gpuInfoSection[FTI_BUFS];

  //Save GPU Info
  int i = 0;
  for (i = 0; i < FTI_Exec->nbKernels; i++)
  {
    snprintf(kernelInfoSection, FTI_BUFS, "Kernel Info");
    iniparser_set(ini, kernelInfoSection, NULL);

    snprintf(str, FTI_BUFS, "%s:nbKernels", kernelInfoSection);
    snprintf(buf, FTI_BUFS, "%u", FTI_Exec->nbKernels);
    iniparser_set(ini, str, buf);

    snprintf(str, FTI_BUFS, "%s:id%d", kernelInfoSection, i);
    snprintf(buf, FTI_BUFS, "%d", *FTI_GpuInfoMetadata[groupIdx].FTI_GpuInfo[i].id);
    iniparser_set(ini, str, buf);

    snprintf(gpuInfoSection, FTI_BUFS, "%dGPU Info%d", FTI_GpuInfoMetadata[groupIdx].groupRank, *FTI_GpuInfoMetadata[groupIdx].FTI_GpuInfo[i].id);
    iniparser_set(ini, gpuInfoSection, NULL);

    snprintf(str, FTI_BUFS, "%s:block_amt", gpuInfoSection); 
    snprintf(buf, FTI_BUFS, "%zu", *FTI_GpuInfoMetadata[groupIdx].FTI_GpuInfo[i].block_amt);
    iniparser_set(ini, str, buf);

    snprintf(str, FTI_BUFS, "%s:complete", gpuInfoSection);
    snprintf(buf, FTI_BUFS, "%s", *FTI_GpuInfoMetadata[groupIdx].FTI_GpuInfo[i].complete ? "T" : "F");
    iniparser_set(ini, str, buf);

    snprintf(str, FTI_BUFS, "%s:quantum", gpuInfoSection);
    snprintf(buf, FTI_BUFS, "%u", *FTI_GpuInfoMetadata[groupIdx].FTI_GpuInfo[i].quantum);
    iniparser_set(ini, str, buf);

    int j = 0;
    for(j = 0; j < FTI_Topo->nbProc; j++){
      snprintf(str, FTI_BUFS, "%s:all_done%d", gpuInfoSection, j); 
      snprintf(buf, FTI_BUFS, "%s", FTI_GpuInfoMetadata[groupIdx].FTI_GpuInfo[i].all_done[j] ? "T" : "F");
      iniparser_set(ini, str, buf);
    }

    size_t k = 0;
    for(k = 0; k < *FTI_GpuInfoMetadata[groupIdx].FTI_GpuInfo[i].block_amt; k++){
      snprintf(str, FTI_BUFS, "%s:block%zu", gpuInfoSection, k);
      snprintf(buf, FTI_BUFS, "%s", FTI_GpuInfoMetadata[groupIdx].FTI_GpuInfo[i].h_is_block_executed[k] ? "T" : "F");
      iniparser_set(ini, str, buf);
    }
  }
  return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      It writes the metadata to recover the data after a failure.
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Topo        Topology metadata.
  @param      fs              Pointer to the list of checkpoint sizes.
  @param      mfs             The maximum checkpoint file size.
  @param      fnl             Pointer to the list of checkpoint names.
  @param      checksums       Checksums array.
  @param      allVarIDs       IDs of vars from all processes in group.
  @param      allVarSizes     Sizes of vars from all processes in group.
  @return     integer         FTI_SCES if successful.

  This function should be executed only by one process per group. It
  writes the metadata file used to recover in case of failure.

 **/
/*-------------------------------------------------------------------------*/
int FTI_WriteMetadata(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_gpuInfoMetadata* FTI_GpuInfoMetadata, long* fs, long mfs, char* fnl,
        char* checksums, int* allVarIDs, long* allVarSizes)
{
    // no metadata files for FTI-FF
    if ( FTI_Conf->ioMode == FTI_IO_FTIFF ) { return FTI_SCES; }

	char str[FTI_BUFS], buf[FTI_BUFS];
    snprintf(buf, FTI_BUFS, "%s/Topology.fti", FTI_Conf->metadDir);
    snprintf(str, FTI_BUFS, "Temporary load of topology file (%s)...", buf);
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
        snprintf(str, FTI_BUFS, "%d", i);
        iniparser_set(ini, str, NULL);
        snprintf(str, FTI_BUFS, "%d:Ckpt_file_name", i);
        iniparser_set(ini, str, buf);
        snprintf(str, FTI_BUFS, "%d:Ckpt_file_size", i);
        snprintf(buf, FTI_BUFS, "%lu", fs[i]);
        iniparser_set(ini, str, buf);
        snprintf(str, FTI_BUFS, "%d:Ckpt_file_maxs", i);
        snprintf(buf, FTI_BUFS, "%lu", mfs);
        iniparser_set(ini, str, buf);
        strncpy(buf, checksums + (i * MD5_DIGEST_STRING_LENGTH), MD5_DIGEST_STRING_LENGTH);
        snprintf(str, FTI_BUFS, "%d:Ckpt_checksum", i);
        iniparser_set(ini, str, buf);
        int j;
        for (j = 0; j < FTI_Exec->nbVar; j++) {
            //Save id of variable
            snprintf(str, FTI_BUFS, "%d:Var%d_id", i, j);
            snprintf(buf, FTI_BUFS, "%d", allVarIDs[i * FTI_Exec->nbVar + j]);
            iniparser_set(ini, str, buf);

            //Save size of variable
            snprintf(str, FTI_BUFS, "%d:Var%d_size", i, j);
            snprintf(buf, FTI_BUFS, "%ld", allVarSizes[i * FTI_Exec->nbVar + j]);
            iniparser_set(ini, str, buf);
        }

        if(FTI_GpuInfoMetadata != NULL){
          FTI_WriteGpuMetadata(FTI_Exec, FTI_Topo, FTI_GpuInfoMetadata, ini, i);
        }
    }

    // Remove topology section
    iniparser_unset(ini, "topology");
    if (mkdir(FTI_Conf->mTmpDir, 0777) == -1) {
        if (errno != EEXIST) {
            FTI_Print("Cannot create directory", FTI_EROR);
        }
    }

    snprintf(buf, FTI_BUFS, "%s/sector%d-group%d.fti", FTI_Conf->mTmpDir, FTI_Topo->sectorID, FTI_Topo->groupID);
    if (remove(buf) == -1) {
        if (errno != ENOENT) {
            FTI_Print("Cannot remove sector-group.fti", FTI_EROR);
        }
    }

    snprintf(str, FTI_BUFS, "Creating metadata file (%s)...", buf);
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
  @param      FTI_GpuInfoMetadata   Gpu metadata.
  @param      FTI_Topo              Topology metadata.
  @param      FTI_Exec              Execution metadata.
  @return     integer               FTI_SCES if successful.

  This function should be executed only by one process per group. It
  writes the metadata file used to recover in case of failure.

 **/
/*-------------------------------------------------------------------------*/
int FTI_FreeGpuMetadata(FTIT_gpuInfoMetadata* FTI_GpuInfoMetadata, FTIT_topology* FTI_Topo, FTIT_execution *FTI_Exec){
  if(FTI_GpuInfoMetadata != NULL){
    unsigned int i = 0;
    unsigned int j = 0;
    /* Only free FTI_GpuInfo for group head */
    free(FTI_GpuInfoMetadata[FTI_Topo->groupRank].FTI_GpuInfo);

    /* Free other allocations used to receive data from other processes */
    for(i = 1; i < FTI_Topo->groupSize; i++){
      for(j = 0; j < FTI_Exec->nbKernels; j++){
        free((void *)FTI_GpuInfoMetadata[i].FTI_GpuInfo[j].id);
        free((void *)FTI_GpuInfoMetadata[i].FTI_GpuInfo[j].block_amt);
        free((void *)FTI_GpuInfoMetadata[i].FTI_GpuInfo[j].all_done);
        free((void *)FTI_GpuInfoMetadata[i].FTI_GpuInfo[j].complete);
        free((void *)FTI_GpuInfoMetadata[i].FTI_GpuInfo[j].h_is_block_executed);
        free((void *)FTI_GpuInfoMetadata[i].FTI_GpuInfo[j].quantum);
      }
      free(FTI_GpuInfoMetadata[i].FTI_GpuInfo);
    } 
    free(FTI_GpuInfoMetadata);
    FTI_GpuInfoMetadata = NULL;
  }
  return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      It creates the GPU metadata to recover the data after a failure.
  @param      FTI_Exec              Execution metadata.
  @param      FTI_Conf              Configuration metadata.
  @param      FTI_Topo              Topology metadata.
  @param      FTI_gpuInfoMetadata   GPU metadata structure to populate with GPU metadata
  @return     integer               FTI_SCES if successful.

  This function sends all kernel data from each process to the head process of the group.
  The kernel data from each process is received by the head process andthen used to 
  populate FTI_GpuInfoMetadata.
 **/
/*-------------------------------------------------------------------------*/
static int FTI_CreateGpuMetadata(FTIT_execution* FTI_Exec, FTIT_topology* FTI_Topo, FTIT_gpuInfoMetadata *FTI_GpuInfoMetadata){
  unsigned int i = 0;
  unsigned int j = 0;

  if(FTI_Topo->groupRank != 0){
    int tag = FTI_Topo->groupRank;
    int dest= 0; //Send to head of group

    for(i = 0; i < FTI_Exec->nbKernels; i++){  
      MPI_Send((const void*)FTI_Exec->gpuInfo[i].id, 1, MPI_INT, dest, tag, FTI_Exec->groupComm);
      MPI_Send((const void*)FTI_Exec->gpuInfo[i].block_amt, 1, MPI_UNSIGNED_LONG_LONG, dest, tag, FTI_Exec->groupComm);
      MPI_Send((const void*)FTI_Exec->gpuInfo[i].all_done, FTI_Topo->nbProc, MPI_C_BOOL, dest, tag, FTI_Exec->groupComm);
      MPI_Send((const void*)FTI_Exec->gpuInfo[i].complete, 1, MPI_C_BOOL, dest, tag, FTI_Exec->groupComm);
      MPI_Send((const void*)FTI_Exec->gpuInfo[i].h_is_block_executed, *FTI_Exec->gpuInfo[i].block_amt, MPI_C_BOOL, dest, tag, FTI_Exec->groupComm);
      MPI_Send((const void*)FTI_Exec->gpuInfo[i].quantum, 1, MPI_UNSIGNED, dest, tag, FTI_Exec->groupComm);
    }
  }
  else{
    /* Gather data from head of group (i.e FTI_Topo->groupRank = 0) */
    FTI_GpuInfoMetadata[FTI_Topo->groupRank].FTI_GpuInfo = talloc(FTIT_gpuInfo, FTI_Exec->nbKernels);
    if(FTI_GpuInfoMetadata[FTI_Topo->groupRank].FTI_GpuInfo == NULL){return FTI_NSCS;}

    for(i = 0; i < FTI_Exec->nbKernels; i++){
      FTI_GpuInfoMetadata[FTI_Topo->groupRank].FTI_GpuInfo[i].id                  = FTI_Exec->gpuInfo[i].id;
      FTI_GpuInfoMetadata[FTI_Topo->groupRank].FTI_GpuInfo[i].block_amt           = FTI_Exec->gpuInfo[i].block_amt;
      FTI_GpuInfoMetadata[FTI_Topo->groupRank].FTI_GpuInfo[i].all_done            = FTI_Exec->gpuInfo[i].all_done;
      FTI_GpuInfoMetadata[FTI_Topo->groupRank].FTI_GpuInfo[i].complete            = FTI_Exec->gpuInfo[i].complete;
      FTI_GpuInfoMetadata[FTI_Topo->groupRank].FTI_GpuInfo[i].h_is_block_executed = FTI_Exec->gpuInfo[i].h_is_block_executed;
      FTI_GpuInfoMetadata[FTI_Topo->groupRank].FTI_GpuInfo[i].quantum             = FTI_Exec->gpuInfo[i].quantum;
    }
  }

  if(FTI_Topo->groupRank == 0){
    /* init loop to receive kernel info from other processes */
    for(i = 1; i < FTI_Topo->groupSize; i++){
      FTI_GpuInfoMetadata[i].FTI_GpuInfo = talloc(FTIT_gpuInfo, FTI_Exec->nbKernels);
      if(FTI_GpuInfoMetadata[FTI_Topo->groupRank].FTI_GpuInfo == NULL){return FTI_NSCS;}
      for(j = 0; j < FTI_Exec->nbKernels; j++){
        FTI_GpuInfoMetadata[i].FTI_GpuInfo[j].id                   = talloc(int, FTI_Exec->nbKernels);
        FTI_GpuInfoMetadata[i].FTI_GpuInfo[j].block_amt            = talloc(size_t, FTI_Exec->nbKernels);
        FTI_GpuInfoMetadata[i].FTI_GpuInfo[j].all_done             = talloc(bool, FTI_Topo->nbProc * FTI_Exec->nbKernels);
        FTI_GpuInfoMetadata[i].FTI_GpuInfo[j].complete             = talloc(bool, FTI_Exec->nbKernels);
        FTI_GpuInfoMetadata[i].FTI_GpuInfo[j].h_is_block_executed  = talloc(bool, *FTI_Exec->gpuInfo[j].block_amt);
        FTI_GpuInfoMetadata[i].FTI_GpuInfo[j].quantum              = talloc(unsigned int, FTI_Exec->nbKernels);

        if(FTI_GpuInfoMetadata[i].FTI_GpuInfo[j].id                   == NULL){return FTI_NSCS;}
        if(FTI_GpuInfoMetadata[i].FTI_GpuInfo[j].block_amt            == NULL){return FTI_NSCS;}
        if(FTI_GpuInfoMetadata[i].FTI_GpuInfo[j].all_done             == NULL){return FTI_NSCS;}
        if(FTI_GpuInfoMetadata[i].FTI_GpuInfo[j].complete             == NULL){return FTI_NSCS;}
        if(FTI_GpuInfoMetadata[i].FTI_GpuInfo[j].h_is_block_executed  == NULL){return FTI_NSCS;}
        if(FTI_GpuInfoMetadata[i].FTI_GpuInfo[j].quantum              == NULL){return FTI_NSCS;}
      }
    }

    /* Now receive data */
    int src = 0;
    int tag = 0;
    FTI_GpuInfoMetadata[FTI_Topo->groupRank].groupRank = FTI_Topo->groupRank; /* Necessary so that kernel information is mapped to rank of process in group */

    for(i = 1; i < FTI_Topo->groupSize; i++){
      src = i; /* Rank of process in group from which to receive data */
      tag = i; /* Processes use their group rank as the tag when sending */

      FTI_GpuInfoMetadata[i].groupRank = tag; /* Maps incoming kernel data to group rank of process sending data */

      for(j = 0; j < FTI_Exec->nbKernels; j++){
        MPI_Recv((void *)FTI_GpuInfoMetadata[i].FTI_GpuInfo[j].id, 1, MPI_INT, src, tag, FTI_Exec->groupComm, MPI_STATUS_IGNORE);
        MPI_Recv((void *)FTI_GpuInfoMetadata[i].FTI_GpuInfo[j].block_amt, 1, MPI_UNSIGNED_LONG_LONG, src, tag, FTI_Exec->groupComm, MPI_STATUS_IGNORE);
        MPI_Recv((void *)FTI_GpuInfoMetadata[i].FTI_GpuInfo[j].all_done, FTI_Topo->nbProc, MPI_C_BOOL, src, tag, FTI_Exec->groupComm, MPI_STATUS_IGNORE);
        MPI_Recv((void *)FTI_GpuInfoMetadata[i].FTI_GpuInfo[j].complete, 1, MPI_C_BOOL, src, tag, FTI_Exec->groupComm, MPI_STATUS_IGNORE);
        MPI_Recv((void *)FTI_GpuInfoMetadata[i].FTI_GpuInfo[j].h_is_block_executed, *FTI_Exec->gpuInfo[j].block_amt, MPI_C_BOOL, src, tag, FTI_Exec->groupComm, MPI_STATUS_IGNORE);
        MPI_Recv((void *)FTI_GpuInfoMetadata[i].FTI_GpuInfo[j].quantum, 1, MPI_UNSIGNED, src, tag, FTI_Exec->groupComm, MPI_STATUS_IGNORE);
      }
    }
  }

  return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      It writes the metadata to recover the data after a failure.
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Topo        Topology metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @param      FTI_Data        Dataset metadata.
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
    // metadata is created before for FTI-FF
    if ( FTI_Conf->ioMode == FTI_IO_FTIFF ) { return FTI_SCES; }

    FTI_Exec->meta[0].fs[0] = FTI_Exec->ckptSize;
    FTI_Exec->meta[0].nbVar[0] = FTI_Exec->nbVar;

#ifdef ENABLE_HDF5
    char fn[FTI_BUFS];
    if (FTI_Exec->ckptLvel == 4 && FTI_Ckpt[4].isInline) { //If inline L4 save directly to global directory
        snprintf(fn, FTI_BUFS, "%s/%s", FTI_Conf->gTmpDir, FTI_Exec->meta[0].ckptFile);
    }
    else {
        snprintf(fn, FTI_BUFS, "%s/%s", FTI_Conf->lTmpDir, FTI_Exec->meta[0].ckptFile);
    }
    if (access(fn, F_OK) == 0) {
        struct stat fileStatus;
        if (stat(fn, &fileStatus) == 0) {
            FTI_Exec->meta[0].fs[0] = fileStatus.st_size;
        }
        else {
            char str[FTI_BUFS];
            snprintf(str, FTI_BUFS, "FTI couldn't get ckpt file size. (%s)", fn);
            FTI_Print(str, FTI_WARN);
        }
    }
    else {
        char str[FTI_BUFS];
        snprintf(str, FTI_BUFS, "FTI couldn't access file ckpt file. (%s)", fn);
        snprintf(str, FTI_BUFS, "FTI couldn't acces file ckpt file. (%s)", fn);
        FTI_Print(str, FTI_WARN);
    }
#endif

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
    snprintf(str, FTI_BUFS, "Max. file size in group %lu.", mfs);
    FTI_Print(str, FTI_DBUG);

    char* ckptFileNames;
    if (FTI_Topo->groupRank == 0) {
        ckptFileNames = talloc(char, FTI_Topo->groupSize * FTI_BUFS);
    }
    strncpy(str, FTI_Exec->meta[0].ckptFile, FTI_BUFS); // Gather all the file names
    MPI_Gather(str, FTI_BUFS, MPI_CHAR, ckptFileNames, FTI_BUFS, MPI_CHAR, 0, FTI_Exec->groupComm);

    char checksum[MD5_DIGEST_STRING_LENGTH];
    FTI_Checksum(FTI_Exec, FTI_Data, FTI_Conf, checksum);

    //TODO checksums of HDF5 files
#ifdef ENABLE_HDF5
    if (FTI_Conf->ioMode == FTI_IO_HDF5) {
        checksum[0] = '\0';
    }
#endif

    char* checksums;
    if (FTI_Topo->groupRank == 0) {
        checksums = talloc(char, FTI_Topo->groupSize * MD5_DIGEST_STRING_LENGTH);
    }
    MPI_Gather(checksum, MD5_DIGEST_STRING_LENGTH, MPI_CHAR, checksums, MD5_DIGEST_STRING_LENGTH, MPI_CHAR, 0, FTI_Exec->groupComm);
 
    FTIT_gpuInfoMetadata *FTI_GpuInfoMetadata = NULL;
    if(FTI_Exec->nbKernels > 0){
      FTI_GpuInfoMetadata = talloc(FTIT_gpuInfoMetadata, FTI_Topo->groupSize); 
      if(FTI_GpuInfoMetadata == NULL){
        FTI_Print("Failed to allocate memory for gpu metadata", FTI_WARN);
        return FTI_NSCS;
      }
      int res = FTI_Try(FTI_CreateGpuMetadata(FTI_Exec, FTI_Topo, FTI_GpuInfoMetadata), "create GPU metadata");

      if(res == FTI_NSCS){
        return FTI_NSCS;
      }
    }

    //Every process has the same number of protected variables
    int* allVarIDs;
    long* allVarSizes;
    if (FTI_Topo->groupRank == 0) {
        allVarIDs = talloc(int, FTI_Topo->groupSize * FTI_Exec->nbVar);
        allVarSizes = talloc(long, FTI_Topo->groupSize * FTI_Exec->nbVar);
    }
    int* myVarIDs = talloc(int, FTI_Exec->nbVar);
    long* myVarSizes = talloc(long, FTI_Exec->nbVar);
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
        int res = FTI_Try(FTI_WriteMetadata(FTI_Conf, FTI_Exec, FTI_Topo, FTI_GpuInfoMetadata, fileSizes, mfs,
                    ckptFileNames, checksums, allVarIDs, allVarSizes), "write the metadata.");
        free(allVarIDs);
        free(allVarSizes);
        free(ckptFileNames);
        free(checksums);

        FTI_FreeGpuMetadata(FTI_GpuInfoMetadata, FTI_Topo, FTI_Exec);

        if (res == FTI_NSCS) {
            return FTI_NSCS;
        }
    }

    //Flush metadata in case postCkpt done inline
    FTI_Exec->meta[FTI_Exec->ckptLvel].fs[0] = FTI_Exec->meta[0].fs[0];
    FTI_Exec->meta[FTI_Exec->ckptLvel].pfs[0] = FTI_Exec->meta[0].pfs[0];
    FTI_Exec->meta[FTI_Exec->ckptLvel].maxFs[0] = FTI_Exec->meta[0].maxFs[0];
    FTI_Exec->meta[FTI_Exec->ckptLvel].nbVar[0] = FTI_Exec->meta[0].nbVar[0];
    strncpy(FTI_Exec->meta[FTI_Exec->ckptLvel].ckptFile, FTI_Exec->meta[0].ckptFile, FTI_BUFS);
    for (i = 0; i < FTI_Exec->nbVar; i++) {
        FTI_Exec->meta[0].varID[i] = FTI_Data[i].id;
        FTI_Exec->meta[0].varSize[i] = FTI_Data[i].size;
        FTI_Exec->meta[FTI_Exec->ckptLvel].varID[i] = FTI_Data[i].id;
        FTI_Exec->meta[FTI_Exec->ckptLvel].varSize[i] = FTI_Data[i].size;
    }

    return FTI_SCES;
}
