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

#include <time.h>
#include <stdint.h>

#include "meta.h"

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
        char* checksum, char* ptnerChecksum, char* rsChecksum) {
    char mfn[FTI_BUFS];  // Path to the metadata file
    char str[FTI_BUFS];  // For console output
    if (FTI_Exec->ckptMeta.level == 0) {
        snprintf(mfn, FTI_BUFS, "%s/sector%d-group%d.fti",
         FTI_Conf->mTmpDir, FTI_Topo->sectorID, FTI_Topo->groupID);
    } else {
        snprintf(mfn, FTI_BUFS, "%s/sector%d-group%d.fti",
         FTI_Ckpt[FTI_Exec->ckptMeta.level].metaDir, FTI_Topo->sectorID,
         FTI_Topo->groupID);
    }

    snprintf(str, FTI_BUFS, "Getting FTI metadata file (%s)...", mfn);
    FTI_Print(str, FTI_DBUG);

    FTIT_iniparser ini;
    if (FTI_Iniparser(&ini, mfn, FTI_INI_OPEN) != FTI_SCES) {
        FTI_Print("Iniparser failed to parse the metadata file.", FTI_WARN);
        return FTI_NSCS;
    }

    // Get checksum of checkpoint file
    snprintf(str, FTI_BUFS, "%d:Ckpt_checksum", FTI_Topo->groupRank);
    strncpy(checksum, ini.getString(&ini, str), MD5_DIGEST_STRING_LENGTH);

    // Get checksum of partner checkpoint file
    snprintf(str, FTI_BUFS, "%d:Ckpt_checksum",
     (FTI_Topo->groupRank + FTI_Topo->groupSize - 1) % FTI_Topo->groupSize);
    strncpy(ptnerChecksum, ini.getString(&ini, str), MD5_DIGEST_STRING_LENGTH);

    // Get checksum of Reed-Salomon file
    snprintf(str, FTI_BUFS, "%d:RSed_checksum", FTI_Topo->groupRank);
    strncpy(rsChecksum, ini.getString(&ini, str), MD5_DIGEST_STRING_LENGTH);

    ini.clear(&ini);

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
int FTI_WriteRSedChecksum(FTIT_configuration* FTI_Conf,
        FTIT_execution* FTI_Exec, FTIT_topology* FTI_Topo,
        FTIT_checkpoint* FTI_Ckpt, int rank, char* checksum) {
    // Fake call for FTI-FF. checksum is done for the datasets.
    if (FTI_Conf->ioMode == FTI_IO_FTIFF) {return FTI_SCES;}

    char str[FTI_BUFS], fileName[FTI_BUFS];

    // Calcuate which groupID rank belongs
    int sectorID = rank / (FTI_Topo->groupSize * FTI_Topo->nodeSize);
    int node = rank / FTI_Topo->nodeSize;
    int rankInGroup = node - (sectorID * FTI_Topo->groupSize);
    int groupID = rank % FTI_Topo->nodeSize;

    char* checksums = talloc(char,
     FTI_Topo->groupSize * MD5_DIGEST_STRING_LENGTH);
    MPI_Allgather(checksum, MD5_DIGEST_STRING_LENGTH, MPI_CHAR, checksums,
     MD5_DIGEST_STRING_LENGTH, MPI_CHAR, FTI_Exec->groupComm);

    // Only first process in group save RS checksum
    if (rankInGroup) {
        free(checksums);
        return FTI_SCES;
    }

    snprintf(fileName, FTI_BUFS, "%s/sector%d-group%d.fti", FTI_Conf->mTmpDir,
     FTI_Topo->sectorID, groupID);

    FTIT_iniparser ini;
    if (FTI_Iniparser(&ini, fileName, FTI_INI_OPEN) != FTI_SCES) {
        FTI_Print("Temporary metadata file could NOT be parsed", FTI_WARN);
        free(checksums);
        return FTI_NSCS;
    }
    // Add metadata to dictionary
    int i;
    for (i = 0; i < FTI_Topo->groupSize; i++) {
        char buf[FTI_BUFS];
        strncpy(buf, checksums + (i * MD5_DIGEST_STRING_LENGTH),
         MD5_DIGEST_STRING_LENGTH);
        snprintf(str, FTI_BUFS, "%d:RSed_checksum", i);
        ini.set(&ini, str, buf);
    }
    free(checksums);

    snprintf(str, FTI_BUFS, "Recreating metadata file (%s)...", fileName);
    FTI_Print(str, FTI_DBUG);

    ini.dump(&ini);
    ini.clear(&ini);

    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Loads metadata from current checkpoint for the postprocessing.
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Topo        Topology metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @return     integer         FTI_SCES if successful.

  This function loads the metadata file created during the preprocessing
  of a checkpoint if the postprocessing is not performed inline. The function
  is only called by the FTI head processes. The heads load the checkpoint 
  file name, file size, partner file size and the size of the largest file 
  in the group (for padding if necessary during decoding). That is, the 
  information needed to perform the postprocessing from the plain checkpoint
  files.

 **/
/*-------------------------------------------------------------------------*/
int FTI_LoadMetaPostprocessing(FTIT_configuration* FTI_Conf,
     FTIT_execution* FTI_Exec, FTIT_topology* FTI_Topo,
     FTIT_checkpoint* FTI_Ckpt, int proc) {
    // no metadata files for FTI-FF
    if (!FTI_Topo->amIaHead) return FTI_SCES;
    if (FTI_Conf->ioMode == FTI_IO_FTIFF)
      return FTIFF_LoadMetaPostprocessing(FTI_Exec, FTI_Topo, FTI_Ckpt,
       FTI_Conf, proc);

    char metaFileName[FTI_BUFS], str[FTI_BUFS];
    snprintf(metaFileName, FTI_BUFS, "%s/sector%d-group%d.fti",
     FTI_Conf->mTmpDir, FTI_Topo->sectorID, proc);
    snprintf(str, FTI_BUFS, "Getting FTI metadata file (%s)...", metaFileName);
    FTI_Print(str, FTI_DBUG);

    FTIT_iniparser ini;
    if (FTI_Iniparser(&ini, metaFileName, FTI_INI_OPEN) != FTI_SCES)
      return FTI_NSCS;

    snprintf(str, FTI_BUFS, "%d:Ckpt_file_name", FTI_Topo->groupRank);
    snprintf(FTI_Exec->ckptMeta.ckptFile, FTI_BUFS, "%s",
     ini.getString(&ini, str));

    // update head's ckptId
    sscanf(FTI_Exec->ckptMeta.ckptFile, "Ckpt%d", &FTI_Exec->ckptMeta.ckptId);

    snprintf(str, FTI_BUFS, "%d:Ckpt_file_size", FTI_Topo->groupRank);
    FTI_Exec->ckptMeta.fs = ini.getLong(&ini, str);

    snprintf(str, FTI_BUFS, "%d:Ckpt_file_size",
     (FTI_Topo->groupRank + FTI_Topo->groupSize - 1) % FTI_Topo->groupSize);
    FTI_Exec->ckptMeta.pfs = ini.getLong(&ini, str);

    FTI_Exec->ckptMeta.maxFs = ini.getLong(&ini, "0:Ckpt_file_maxs");

    ini.clear(&ini);

    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Loads the metadata from the meta data files that are needed to
  perform the recovery on a restart.
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Topo        Topology metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @return     integer         FTI_SCES if successful.

  This function reads the metadata file that corresponds to the available
  checkpoints and loads all nessecary information into the checkpoint 
  runtime metadata. The metadata is stored inside a fifo queue in increasing
  order of reliability levels. That is because the lower the reliability, the 
  more recent the checkpoint creation. On a restart, FTI tries to recover from 
  the most recent checkpoint. If the recovery failes, the first element of 
  the queue is removed and the next checkpoint i used for the recovery.

 **/
/*-------------------------------------------------------------------------*/
int FTI_LoadMetaRecovery(FTIT_configuration* FTI_Conf,
        FTIT_execution* FTI_Exec, FTIT_topology* FTI_Topo,
         FTIT_checkpoint* FTI_Ckpt) {
    // no metadata files for FTI-FF
    if (FTI_Conf->ioMode == FTI_IO_FTIFF) {
        if (FTI_Conf->dcpFtiff) {
            FTI_LoadCkptMetaData(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt);
        }

        int i=4; for (; i > 0; i--) {  // for each level
            FTIT_metadata meta;
            meta.level = i;
            FTI_Exec->mqueue.push(&FTI_Exec->mqueue, meta);
        }
        return FTI_SCES;
    }

    FTIT_iniparser ini;

    int i=4; for (; i > 0; i--) {  // for each level
        FTIT_metadata meta;

        meta.level = i;

        char metaFileName[FTI_BUFS], str[FTI_BUFS];

        if (i == 0)
         snprintf(metaFileName, FTI_BUFS, "%s/sector%d-group%d.fti",
         FTI_Conf->mTmpDir, FTI_Topo->sectorID, FTI_Topo->groupID);
        else
         snprintf(metaFileName, FTI_BUFS, "%s/sector%d-group%d.fti",
         FTI_Ckpt[i].metaDir, FTI_Topo->sectorID, FTI_Topo->groupID);

        snprintf(str, FTI_BUFS, "Getting FTI metadata file (%s)...",
         metaFileName);
        FTI_Print(str, FTI_DBUG);

        if (FTI_Iniparser(&ini, metaFileName, FTI_INI_OPEN) != FTI_SCES)
          continue;

        snprintf(str, FTI_BUFS, "Meta for level %d exists.", i);
        FTI_Print(str, FTI_DBUG);

        // check for dcp
        FTI_Ckpt[i].recoIsDcp = !strcmp("dcp",
         ini.getString(&ini, "ckpt_info:ckpt_type"));

        // get ckptId
        FTI_Exec->ckptId = ini.getInt(&ini, "ckpt_info:ckpt_id");

        snprintf(str, FTI_BUFS, "%d:Ckpt_file_name", FTI_Topo->groupRank);
        snprintf(meta.ckptFile, FTI_BUFS, "%s", ini.getString(&ini, str));

        snprintf(str, FTI_BUFS, "%d:Ckpt_file_size", FTI_Topo->groupRank);
        meta.fs = ini.getLongLong(&ini, str);
        FTI_Exec->dcpInfoPosix.FileSize = meta.fs;

        snprintf(str, FTI_BUFS, "%d:Ckpt_file_size",
         (FTI_Topo->groupRank + FTI_Topo->groupSize - 1) %
          FTI_Topo->groupSize);
        meta.pfs = ini.getLongLong(&ini, str);

        meta.maxFs = ini.getLongLong(&ini, "0:Ckpt_file_maxs");

        FTI_Exec->mqueue.push(&FTI_Exec->mqueue, meta);

        ini.clear(&ini);
    }

    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Loads the metadata needed to recover from a posix dCP 
  checkpoint file.
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Topo        Topology metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @return     integer         FTI_SCES if successful.

  This function reads the metadata which is necessary to restart from a dCP 
  checkpoint file.

 **/
/*-------------------------------------------------------------------------*/
int FTI_LoadMetaDcp(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt) {
    // no metadata files for FTI-FF
    if (FTI_Conf->ioMode == FTI_IO_FTIFF) { return FTI_SCES; }

    char metaFileName[FTI_BUFS], str[FTI_BUFS];

    int level = FTI_Exec->ckptMeta.level;

    if (level == 0)
      snprintf(metaFileName, FTI_BUFS, "%s/sector%d-group%d.fti",
     FTI_Conf->mTmpDir, FTI_Topo->sectorID, FTI_Topo->groupID);

    else
      snprintf(metaFileName, FTI_BUFS, "%s/sector%d-group%d.fti",
     FTI_Ckpt[level].metaDir, FTI_Topo->sectorID, FTI_Topo->groupID);

    snprintf(str, FTI_BUFS, "Getting FTI metadata file (%s)...", metaFileName);
    FTI_Print(str, FTI_DBUG);

    FTIT_iniparser ini;
    if (FTI_Iniparser(&ini, metaFileName, FTI_INI_OPEN) != FTI_SCES)
      return FTI_NSCS;

    int k; for (k = 0; k < MAX_STACK_SIZE; k++) {
        snprintf(str, FTI_BUFS, "%d:dcp_layer%d_size", FTI_Topo->groupRank, k);
        int64_t LayerSize = ini.getLongLong(&ini, str);
        if (LayerSize == -1) {
            // No more variables
            break;
        }
        FTI_Exec->dcpInfoPosix.LayerSize[k] = LayerSize;

        snprintf(str, FTI_BUFS, "%d:dcp_layer%d_hash", FTI_Topo->groupRank, k);
        snprintf(&FTI_Exec->dcpInfoPosix.LayerHash[k*MD5_DIGEST_STRING_LENGTH],
         MD5_DIGEST_STRING_LENGTH, "%s", ini.getString(&ini, str));
        int j;
        for (j=0; j < FTI_BUFS; j++) {
            snprintf(str, FTI_BUFS, "%d:dcp_layer%d_var%d_id",
             FTI_Topo->groupRank, k, j);
            int varID = ini.getInt(&ini, str);
            if (varID == -1) {
                break;
            }
            FTI_Exec->dcpInfoPosix.datasetInfo[k][j].varID = varID;
            snprintf(str, FTI_BUFS, "%d:dcp_layer%d_var%d_size",
             FTI_Topo->groupRank, k, j);
            int64_t varSize = ini.getLongLong(&ini, str);
            if (varID < 0) {
                break;
            }
            FTI_Exec->dcpInfoPosix.datasetInfo[k][j].varSize =
             (int64_t) varSize;
        }
    }

    ini.clear(&ini);

    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Loads the metadata for the protected datasets.
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Topo        Topology metadata.
  @param      FTI_Ckpt        Checkpoint metadata.
  @return     integer         FTI_SCES if successful.

  This function loads the metadata from the checkpoint file of the last 
  successful recovery and loads the data corresponding to FTI protected
  buffers.

 **/
/*-------------------------------------------------------------------------*/
int FTI_LoadMetaDataset(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt,
        FTIT_keymap* FTI_Data) {
    // no metadata files for FTI-FF
    if (FTI_Conf->ioMode == FTI_IO_FTIFF) { return FTI_SCES; }

    char metaFileName[FTI_BUFS], str[FTI_BUFS];

    int level = FTI_Exec->ckptLvel;

    if (level == 0)
      snprintf(metaFileName, FTI_BUFS, "%s/sector%d-group%d.fti",
        FTI_Conf->mTmpDir, FTI_Topo->sectorID, FTI_Topo->groupID);

    else
      snprintf(metaFileName, FTI_BUFS, "%s/sector%d-group%d.fti",
       FTI_Ckpt[level].metaDir, FTI_Topo->sectorID, FTI_Topo->groupID);

    snprintf(str, FTI_BUFS, "Getting FTI metadata file (%s)...", metaFileName);
    FTI_Print(str, FTI_DBUG);

    FTIT_iniparser ini;
    if (FTI_Iniparser(&ini, metaFileName, FTI_INI_OPEN) != FTI_SCES)
      return FTI_NSCS;

    int k; for (k = 0; k < FTI_Conf->maxVarId; k++) {
        snprintf(str, FTI_BUFS, "%d:Var%d_id", FTI_Topo->groupRank, k);
        int id = ini.getInt(&ini, str);
        if (id == -1) {
            // No more variables
            break;
        }

        FTIT_dataset data; FTI_InitDataset(FTI_Exec, &data, id);

        snprintf(str, FTI_BUFS, "%d:Var%d_size", FTI_Topo->groupRank, k);
        data.sizeStored = ini.getLongLong(&ini, str);

        snprintf(str, FTI_BUFS, "%d:Var%d_pos", FTI_Topo->groupRank, k);
        data.filePos = ini.getLongLong(&ini, str);

        snprintf(str, FTI_BUFS, "%d:Var%d_idChar", FTI_Topo->groupRank, k);
        strncpy(data.idChar, ini.getString(&ini, str), FTI_BUFS);

        FTI_Exec->ckptSize = FTI_Exec->ckptSize + data.size;

        data.recovered = true;

        FTI_Data->push_back(&data, data.id);
    }

    // Save number of variables in metadata
    FTI_Exec->nbVarStored = k;

    ini.clear(&ini);

    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Loads relevant data from checkpoint meta data 
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Topo        Topology metadata.
  @param      FTI_Ckpt        Checkpoint metadata.

 **/
/*-------------------------------------------------------------------------*/
int FTI_LoadL4CkptMetaData(FTIT_configuration* FTI_Conf,
      FTIT_execution* FTI_Exec, FTIT_topology* FTI_Topo,
      FTIT_checkpoint* FTI_Ckpt) {
    char fn[FTI_BUFS];
    snprintf(fn, FTI_BUFS, "%s/Checkpoint.fti", FTI_Conf->metadDir);

    FTIT_iniparser ctx_ini;
    if (FTI_Iniparser(&ctx_ini, fn, FTI_INI_OPEN) != FTI_SCES) {
        FTI_Print("Failed to parse checkpoint meta data file.", FTI_EROR);
        return FTI_NSCS;
    }

    dictionary* ini = ctx_ini.dict;

    if (((int)(ini->n)-6) < 0) {
        FTI_Print("Unexpected checkpoint meta data file structure.", FTI_EROR);
        return FTI_NSCS;
    }

    char currCkpt[FTI_BUFS];

    // ini->key[2] must be the level field
    int ckptId;
    int i = 0;
    bool hasL4Ckpt = false;
    for (; i < ini->n; i+=6) {
        memset(currCkpt, 0x0, FTI_BUFS);
        strncpy(currCkpt, ini->key[i], FTI_BUFS-1);
        char level_key[FTI_BUFS];
        snprintf(level_key, FTI_BUFS, "%s:level", currCkpt);
        if (iniparser_getint(ini, level_key, -1) == 4) {
            hasL4Ckpt = true;
            sscanf(currCkpt, "checkpoint_id.%d", &ckptId);
        }
    }

    ctx_ini.clear(&ctx_ini);

    // if we find a level 4 checkpoint return the id if not return -1
    return (hasL4Ckpt) ? ckptId : -1;
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Loads relevant data from checkpoint meta data 
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Topo        Topology metadata.
  @param      FTI_Ckpt        Checkpoint metadata.

 **/
/*-------------------------------------------------------------------------*/
int FTI_LoadCkptMetaData(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt) {
    char fn[FTI_BUFS];
    snprintf(fn, FTI_BUFS, "%s/Checkpoint.fti", FTI_Conf->metadDir);

    FTIT_iniparser ctx_ini;
    if (FTI_Iniparser(&ctx_ini, fn, FTI_INI_OPEN) != FTI_SCES) {
        FTI_Print("Failed to parse checkpoint meta data file.", FTI_EROR);
        return FTI_NSCS;
    }

    dictionary* ini = ctx_ini.dict;

    if (((int)(ini->n)-6) < 0) {
        FTI_Print("Unexpected checkpoint meta data file structure.", FTI_EROR);
        return FTI_NSCS;
    }

    char lastCkpt[FTI_BUFS];

    memset(lastCkpt, 0x0, FTI_BUFS);
    strncpy(lastCkpt, ini->key[ini->n-6], FTI_BUFS-1);

    int ckptId;
    sscanf(lastCkpt, "checkpoint_id.%d", &ckptId);

    char key[FTI_BUFS];
    snprintf(key, FTI_BUFS, "%s:level", lastCkpt);
    int ckptLvel = iniparser_getint(ini, key, -1);

    if (ckptLvel == -1) {
        FTI_Print("Unable to read checkpoint level from checkpoint "
          "meta data file", FTI_EROR);
        dictionary_del(ini);
        return FTI_NSCS;
    }

    if (ckptLvel == 4) {
        snprintf(key, FTI_BUFS, "%s:is_dcp", lastCkpt);
        int isDcp = iniparser_getboolean(ini, key, -1);
        if (isDcp == -1) {
            FTI_Print("Unable to identify if dCP from checkpoint "
              "meta data file", FTI_EROR);
            dictionary_del(ini);
            return FTI_NSCS;
        } else {
            FTI_Ckpt[4].recoIsDcp = (bool) isDcp;
        }
    }

    FTI_Exec->ckptId = ckptId;

    ctx_ini.clear(&ctx_ini);

    return FTI_SCES;
}


/*-------------------------------------------------------------------------*/
/**
  @brief      Creates or updates checkpoint meta data 
  @param      FTI_Conf        Configuration metadata.
  @param      FTI_Exec        Execution metadata.
  @param      FTI_Topo        Topology metadata.
  @param      FTI_Ckpt        Checkpoint metadata.

  Writes checkpoint meta data in checkpoint meta data file.
  - timestamp
  - level
  - number of processes participating in the checkpoint
  - I/O mode
  - dCP enabled/disabled
 **/
/*-------------------------------------------------------------------------*/
int FTI_WriteCkptMetaData(FTIT_configuration* FTI_Conf,
      FTIT_execution* FTI_Exec, FTIT_topology* FTI_Topo,
      FTIT_checkpoint* FTI_Ckpt) {
    char fn[FTI_BUFS];
    snprintf(fn, FTI_BUFS, "%s/Checkpoint.fti", FTI_Conf->metadDir);

    FTIT_iniparser ini;
    if (FTI_Iniparser(&ini, fn, FTI_INI_APPEND) != FTI_SCES) {
        FTI_Print("FTI failed to load the checkpoint meta data", FTI_WARN);
        return FTI_NSCS;
    }

    char section[FTI_BUFS], key[FTI_BUFS], value[FTI_BUFS];
    snprintf(section, FTI_BUFS, "checkpoint_id.%d", FTI_Exec->ckptMeta.ckptId);
    ini.set(&ini, section, NULL);
    time_t time_ctx;
    struct tm * time_info;
    time(&time_ctx);
    // time_info = localtime(&time_ctx);
    struct tm local_tm;
    time_info = localtime_r(&time_ctx, &local_tm);
    char timestr[FTI_BUFS];
    strftime(timestr, FTI_BUFS, "%A %x - %H:%M:%S", time_info);
    snprintf(key, FTI_BUFS, "%s:timestamp", section);
    snprintf(value, FTI_BUFS, "%s", timestr);
    ini.set(&ini, key, value);
    snprintf(key, FTI_BUFS, "%s:level", section);
    snprintf(value, FTI_BUFS, "%d", FTI_Exec->ckptMeta.level);
    ini.set(&ini, key, value);
    snprintf(key, FTI_BUFS, "%s:nb_procs", section);
    snprintf(value, FTI_BUFS, "%d", FTI_Topo->nbApprocs * FTI_Topo->nbNodes);
    ini.set(&ini, key, value);
    snprintf(key, FTI_BUFS, "%s:io_mode", section);
    snprintf(value, FTI_BUFS, "%d", FTI_Conf->ioMode);
    ini.set(&ini, key, value);
    snprintf(key, FTI_BUFS, "%s:is_dcp", section);
    snprintf(value, FTI_BUFS, "%s",
     (FTI_Ckpt[FTI_Exec->ckptMeta.level].isDcp) ? "true" : "false");
    ini.set(&ini, key, value);

    ini.dump(&ini);
    ini.clear(&ini);

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
  @param      allLayerSizes   Sizes of all layers used in dcp.
  @param      allLayerHashes  Hashes of all layers used in dcp.
  @param      allVarPositions Positions of variables stored in dCP.
  @return     integer         FTI_SCES if successful.

  This function should be executed only by one process per group. It
  writes the metadata file used to recover in case of failure.

 **/
/*-------------------------------------------------------------------------*/
int FTI_WriteMetadata(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt, int64_t* fs,
        int64_t mfs, char* fnl, char* checksums, int* allVarIDs,
        int* allRanks, uint64_t* allCounts,
        int* allVarTypeIDs, int64_t* allVarTypeSizes,
        int64_t* allVarSizes, int64_t* allLayerSizes, char* allLayerHashes,
        int64_t *allVarPositions, char *allNames, char *allCharIds) {
    // no metadata files for FTI-FF
    if (FTI_Conf->ioMode == FTI_IO_FTIFF) { return FTI_SCES; }

    char fn[FTI_BUFS], key[FTI_BUFS], val[FTI_BUFS];

    snprintf(fn, FTI_BUFS, "%s/sector%d-group%d.fti",
     FTI_Conf->mTmpDir, FTI_Topo->sectorID, FTI_Topo->groupID);

    // To bypass iniparser bug while empty dict.
    FTIT_iniparser ini;
    if (FTI_Iniparser(&ini, fn, FTI_INI_CREATE) != FTI_SCES) {
        FTI_Print("failed to write the metadata", FTI_WARN);
        return FTI_NSCS;
    }

    // Add dcp POSIX meta data
    ini.set(&ini, "ckpt_info", NULL);
    switch (FTI_Ckpt[FTI_Exec->ckptMeta.level].isDcp) {
        case 0:
            ini.set(&ini, "ckpt_info:ckpt_type", "full");
            break;
        case 1:
            ini.set(&ini, "ckpt_info:ckpt_type", "dcp");
            break;
    }

    // add checkpoint id
    snprintf(val, FTI_BUFS, "%d", FTI_Exec->ckptMeta.ckptId);
    ini.set(&ini, "ckpt_info:ckpt_id", val);

    // Add metadata to dictionary
    int i;
    for (i = 0; i < FTI_Topo->groupSize; i++) {
        strncpy(val, fnl + (i * FTI_BUFS), FTI_BUFS - 1);
        snprintf(key, FTI_BUFS, "%d", i);
        ini.set(&ini, key, NULL);
        snprintf(key, FTI_BUFS, "%d:Ckpt_file_name", i);
        ini.set(&ini, key, val);
        snprintf(key, FTI_BUFS, "%d:Ckpt_file_size", i);
        snprintf(val, FTI_BUFS, "%ld", fs[i]);
        ini.set(&ini, key, val);
        snprintf(key, FTI_BUFS, "%d:Ckpt_file_maxs", i);
        snprintf(val, FTI_BUFS, "%ld", mfs);
        ini.set(&ini, key, val);
        strncpy(val, checksums + (i * MD5_DIGEST_STRING_LENGTH),
         MD5_DIGEST_STRING_LENGTH);
        snprintf(key, FTI_BUFS, "%d:Ckpt_checksum", i);
        ini.set(&ini, key, val);
        int j;
        for (j = 0; j < FTI_Exec->nbVar; j++) {
            // Save id of variable
            snprintf(key, FTI_BUFS, "%d:Var%d_id", i, j);
            snprintf(val, FTI_BUFS, "%d", allVarIDs[i * FTI_Exec->nbVar + j]);
            ini.set(&ini, key, val);

            // Save id of type
            snprintf(key, FTI_BUFS, "%d:Var%d_typeId", i, j);
            snprintf(val, FTI_BUFS, "%d",
                    allVarTypeIDs[i * FTI_Exec->nbVar + j]);
            ini.set(&ini, key, val);

            // Save size of type
            snprintf(key, FTI_BUFS, "%d:Var%d_typeSize", i, j);
            snprintf(val, FTI_BUFS, "%ld",
                    allVarTypeSizes[i * FTI_Exec->nbVar + j]);
            ini.set(&ini, key, val);

            // Save size of variable
            snprintf(key, FTI_BUFS, "%d:Var%d_size", i, j);
            snprintf(val, FTI_BUFS, "%ld",
             allVarSizes[i * FTI_Exec->nbVar + j]);
            ini.set(&ini, key, val);

            snprintf(key, FTI_BUFS, "%d:Var%d_pos", i, j);
            snprintf(val, FTI_BUFS, "%ld",
             allVarPositions[i * FTI_Exec->nbVar + j]);
            ini.set(&ini, key, val);

            snprintf(key, FTI_BUFS, "%d:Var%d_name", i, j);
            snprintf(val, FTI_BUFS, "%s",
             &allNames[(i*FTI_Exec->nbVar*FTI_BUFS) +j*FTI_BUFS]);
            ini.set(&ini, key, val);

            snprintf(key, FTI_BUFS, "%d:Var%d_idchar", i, j);
            snprintf(val, FTI_BUFS, "%s",
             &allCharIds[(i*FTI_Exec->nbVar*FTI_BUFS) +j*FTI_BUFS]);
            ini.set(&ini, key, val);

            // Save rank of variable
            snprintf(key, FTI_BUFS, "%d:Var%d_ndims", i, j);
            snprintf(val, FTI_BUFS, "%d",
                    allRanks[i * FTI_Exec->nbVar + j]);
            ini.set(&ini, key, val);

            int r = 0; for (; r < allRanks[i * FTI_Exec->nbVar + j]; r++) {
                // Save rank of variable
                snprintf(key, FTI_BUFS, "%d:Var%d_dim%d", i, j, r);
                snprintf(val, FTI_BUFS, "%lu",
                        allCounts[i * 32 * FTI_Exec->nbVar + j * 32 + r]);
                ini.set(&ini, key, val);
            }
        }
        if (FTI_Ckpt[FTI_Exec->ckptMeta.level].isDcp) {
            int nbLayer = ((FTI_Exec->dcpInfoPosix.Counter-1) %
             FTI_Conf->dcpInfoPosix.StackSize) + 1;
            for (j=0; j < nbLayer; j++) {
                snprintf(key, FTI_BUFS, "%d:dcp_layer%d_size", i, j);
                snprintf(val, FTI_BUFS, "%ld", allLayerSizes[i * nbLayer + j]);
                ini.set(&ini, key, val);

                snprintf(key, FTI_BUFS, "%d:dcp_layer%d_hash", i, j);
                ini.set(&ini, key,
                 &allLayerHashes[i*nbLayer*MD5_DIGEST_STRING_LENGTH +
                  j*MD5_DIGEST_STRING_LENGTH]);
                int k;
                for (k = 0; k < FTI_Exec->nbVar; k++) {
                    // Save id of variable
                    snprintf(key, FTI_BUFS, "%d:dcp_layer%d_var%d_id",
                     i, j, k);
                    snprintf(val, FTI_BUFS, "%d",
                     allVarIDs[i * FTI_Exec->nbVar + k]);
                    ini.set(&ini, key, val);
                    // Save size of variable
                    snprintf(key, FTI_BUFS, "%d:dcp_layer%d_var%d_size",
                     i, j, k);
                    snprintf(val, FTI_BUFS, "%ld",
                     allVarSizes[i * FTI_Exec->nbVar + k]);
                    ini.set(&ini, key, val);
                }
            }
        }
    }
    // Remove topology section
    MKDIR(FTI_Conf->mTmpDir, 0777);

    ini.dump(&ini);
    ini.clear(&ini);

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
        FTIT_keymap* FTI_Data) {
    // metadata is created before for FTI-FF
    if (FTI_Conf->ioMode == FTI_IO_FTIFF) { return FTI_SCES; }

    FTI_Exec->ckptMeta.fs = (FTI_Ckpt[FTI_Exec->ckptMeta.level].isDcp) ?
     FTI_Exec->dcpInfoPosix.FileSize : FTI_Exec->ckptSize;

#ifdef ENABLE_HDF5
    if (FTI_Conf->ioMode == FTI_IO_HDF5) {
        char fn[FTI_BUFS];
        // If inline L4 save directly to global directory
        if (FTI_Exec->ckptMeta.level == 4 && FTI_Ckpt[4].isInline) {
            snprintf(fn, FTI_BUFS, "%s/%s", FTI_Conf->gTmpDir,
             FTI_Exec->ckptMeta.ckptFile);
        } else {
            snprintf(fn, FTI_BUFS, "%s/%s", FTI_Conf->lTmpDir,
             FTI_Exec->ckptMeta.ckptFile);
        }
        if (access(fn, F_OK) == 0) {
            struct stat fileStatus;
            if (stat(fn, &fileStatus) == 0) {
                FTI_Exec->ckptMeta.fs = fileStatus.st_size;
            } else {
                char str[FTI_BUFS];
                snprintf(str, FTI_BUFS,
                 "FTI couldn't get ckpt file size. (%s)", fn);
                FTI_Print(str, FTI_WARN);
            }
        } else {
            char str[FTI_BUFS];
            snprintf(str, FTI_BUFS,
             "FTI couldn't access file ckpt file. (%s)", fn);
            snprintf(str, FTI_BUFS,
             "FTI couldn't acces file ckpt file. (%s)", fn);
            FTI_Print(str, FTI_WARN);
        }
    }
#endif

    int64_t fileSizes[FTI_BUFS];
    MPI_Allgather(&FTI_Exec->ckptMeta.fs, 1, MPI_INT64_T,
            fileSizes, 1, MPI_INT64_T, FTI_Exec->groupComm);

    // update partner file size:
    if (FTI_Exec->ckptMeta.level == 2) {
        int ptnerGroupRank = (FTI_Topo->groupRank + FTI_Topo->groupSize - 1) %
         FTI_Topo->groupSize;
        FTI_Exec->ckptMeta.pfs = fileSizes[ptnerGroupRank];
    }

    int64_t mfs = 0;  // Max file size in group
    int i;
    for (i = 0; i < FTI_Topo->groupSize; i++) {
        if (fileSizes[i] > mfs) {
            mfs = fileSizes[i];  // Search max. size
        }
    }
    FTI_Exec->ckptMeta.maxFs = mfs;
    char str[FTI_BUFS];  // For console output
    snprintf(str, FTI_BUFS, "Max. file size in group %ld.", mfs);
    FTI_Print(str, FTI_DBUG);

    char* ckptFileNames = NULL;
    if (FTI_Topo->groupRank == 0) {
        ckptFileNames = talloc(char, FTI_Topo->groupSize * FTI_BUFS);
    }
    strncpy(str, FTI_Exec->ckptMeta.ckptFile, FTI_BUFS);
    // Gather all the file names
    MPI_Gather(str, FTI_BUFS, MPI_CHAR, ckptFileNames, FTI_BUFS, MPI_CHAR, 0,
     FTI_Exec->groupComm);

    char checksum[MD5_DIGEST_STRING_LENGTH];
    FTI_Checksum(FTI_Exec, FTI_Data, FTI_Conf, checksum);

    // TODO(leobago) checksums of HDF5 files
#ifdef ENABLE_HDF5
    if (FTI_Conf->ioMode == FTI_IO_HDF5) {
        checksum[0] = '\0';
    }
#endif

    char* checksums = NULL;
    if (FTI_Topo->groupRank == 0) {
        checksums = talloc(char, FTI_Topo->groupSize*MD5_DIGEST_STRING_LENGTH);
    }
    MPI_Gather(checksum, MD5_DIGEST_STRING_LENGTH, MPI_CHAR, checksums,
     MD5_DIGEST_STRING_LENGTH, MPI_CHAR, 0, FTI_Exec->groupComm);


    // Every process has the same number of protected variables

    int* allVarIDs = NULL;
    int* allVarTypeIDs = NULL;
    int64_t* allVarTypeSizes = NULL;
    int64_t* allVarSizes = NULL;
    int64_t *allVarPositions = NULL;

    // for posix dcp
    int64_t* allLayerSizes = NULL;
    char* allLayerHashes = NULL;
    char* allCharIds = NULL;
    char* allNames = NULL;
    uint64_t* allCounts = NULL;
    int* allRanks = NULL;

    int nbLayer = ((FTI_Exec->dcpInfoPosix.Counter-1) %
     FTI_Conf->dcpInfoPosix.StackSize) + 1;

    if (FTI_Topo->groupRank == 0) {
        allRanks = talloc(int, FTI_Topo->groupSize * FTI_Exec->nbVar);
        allCounts = talloc(uint64_t, 32 * FTI_Topo->groupSize *
                FTI_Exec->nbVar);
        allVarIDs = talloc(int, FTI_Topo->groupSize * FTI_Exec->nbVar);
        allVarTypeIDs = talloc(int, FTI_Topo->groupSize * FTI_Exec->nbVar);
        allVarTypeSizes = talloc(int64_t, FTI_Topo->groupSize * FTI_Exec->nbVar);
        allVarSizes = talloc(int64_t, FTI_Topo->groupSize * FTI_Exec->nbVar);
        allVarPositions = talloc(int64_t,
         FTI_Topo->groupSize * FTI_Exec->nbVar);
        allCharIds = (char *)malloc(sizeof(char)*FTI_BUFS*
          FTI_Exec->nbVar*FTI_Topo->groupSize);
        allNames = (char *)malloc(sizeof(char)*FTI_BUFS*
          FTI_Exec->nbVar*FTI_Topo->groupSize);
        if (FTI_Ckpt[FTI_Exec->ckptMeta.level].isDcp) {
            allLayerSizes = talloc(int64_t,
             FTI_Topo->groupSize * nbLayer);
            allLayerHashes = talloc(char,
             FTI_Topo->groupSize * nbLayer * MD5_DIGEST_STRING_LENGTH);
        }
    }

    int* myVarIDs = talloc(int, FTI_Exec->nbVar);
    int* myRanks = talloc(int, FTI_Exec->nbVar);
    uint64_t* myCounts = talloc(uint64_t, 32 * FTI_Exec->nbVar);
    int* myVarTypeIDs = talloc(int, FTI_Exec->nbVar);
    int64_t* myVarTypeSizes = talloc(int64_t, FTI_Exec->nbVar);
    int64_t* myVarSizes = talloc(int64_t, FTI_Exec->nbVar);
    int64_t* myVarPositions = talloc(int64_t, FTI_Exec->nbVar);
    char *ArrayOfIdChars = (char *)malloc(FTI_Exec->nbVar *
     sizeof(char*) *FTI_BUFS);
    char *ArrayOfNames = (char *)malloc(FTI_Exec->nbVar *
     sizeof(char*) *FTI_BUFS);

    FTIT_dataset* data;
    if (FTI_Data->data(&data, FTI_Exec->nbVar) != FTI_SCES) return FTI_NSCS;

    for (i = 0; i < FTI_Exec->nbVar; i++) {
        int typeID = data[i].type->id - FTI_Exec->datatypes.primitive_offset;
        myVarIDs[i] = data[i].id;
        myVarTypeIDs[i] = (typeID < FTI_Exec->datatypes.nprimitives) ?
          typeID : -1;
        myVarTypeSizes[i] = data[i].type->size;
        myVarSizes[i] =  data[i].size;
        myVarIDs[i] =  data[i].id;
        myRanks[i] =  data[i].attribute.dim.ndims;
        memcpy(&myCounts[i*32], &data[i].attribute.dim.count,
                32 * sizeof(uint64_t));
        myVarPositions[i] = data[i].filePos;
        strncpy(&ArrayOfIdChars[i*FTI_BUFS], data[i].idChar, FTI_BUFS);
        strncpy(&ArrayOfNames[i*FTI_BUFS], data[i].attribute.name, FTI_BUFS);
    }

    // Gather variables IDs
    MPI_Gather(myVarIDs, FTI_Exec->nbVar, MPI_INT, allVarIDs, FTI_Exec->nbVar,
     MPI_INT, 0, FTI_Exec->groupComm);
    // Gather variables Type IDs
    MPI_Gather(myVarTypeIDs, FTI_Exec->nbVar, MPI_INT, allVarTypeIDs,
            FTI_Exec->nbVar,    MPI_INT, 0, FTI_Exec->groupComm);
    // Gather variables Type Sizes
    MPI_Gather(myVarTypeSizes, FTI_Exec->nbVar, MPI_INT64_T, allVarTypeSizes,
            FTI_Exec->nbVar,     MPI_INT64_T, 0, FTI_Exec->groupComm);
    // Gather variables Ranks
    MPI_Gather(myRanks, FTI_Exec->nbVar, MPI_INT, allRanks, FTI_Exec->nbVar,
     MPI_INT, 0, FTI_Exec->groupComm);
    // Gather variables Counts
    MPI_Gather(myCounts, 32*FTI_Exec->nbVar, MPI_INT64_T,
            allCounts, 32*FTI_Exec->nbVar, MPI_INT64_T, 0, FTI_Exec->groupComm);
    // Gather variables sizes
    MPI_Gather(myVarSizes, FTI_Exec->nbVar, MPI_INT64_T, allVarSizes,
     FTI_Exec->nbVar, MPI_INT64_T, 0, FTI_Exec->groupComm);
    // Gather variables file positions
    MPI_Gather(myVarPositions, FTI_Exec->nbVar, MPI_INT64_T, allVarPositions,
     FTI_Exec->nbVar, MPI_INT64_T, 0, FTI_Exec->groupComm);
    // Gather all variable idchars
    MPI_Gather(ArrayOfIdChars, FTI_Exec->nbVar*FTI_BUFS, MPI_CHAR,
      allCharIds, FTI_Exec->nbVar*FTI_BUFS, MPI_CHAR, 0, FTI_Exec->groupComm);
    // gather all variable names
    MPI_Gather(ArrayOfNames, FTI_Exec->nbVar*FTI_BUFS, MPI_CHAR,
      allNames, FTI_Exec->nbVar*FTI_BUFS, MPI_CHAR, 0, FTI_Exec->groupComm);


    if (FTI_Ckpt[FTI_Exec->ckptMeta.level].isDcp) {
        // Gather dcp layer sizes
        MPI_Gather(FTI_Exec->dcpInfoPosix.LayerSize, nbLayer, MPI_INT64_T,
         allLayerSizes, nbLayer, MPI_INT64_T, 0, FTI_Exec->groupComm);
        // Gather dcp layer hashes
        MPI_Gather(FTI_Exec->dcpInfoPosix.LayerHash,
         nbLayer * MD5_DIGEST_STRING_LENGTH, MPI_CHAR, allLayerHashes,
         nbLayer * MD5_DIGEST_STRING_LENGTH, MPI_CHAR, 0, FTI_Exec->groupComm);
    }

    free(myVarIDs);
    free(myRanks);
    free(myCounts);
    free(myVarTypeIDs);
    free(myVarTypeSizes);
    free(myVarSizes);
    free(myVarPositions);
    free(ArrayOfIdChars);
    free(ArrayOfNames);

    // Only one process in the group create the metadata
    if (FTI_Topo->groupRank == 0) {
        int res = FTI_Try(FTI_WriteMetadata(FTI_Conf, FTI_Exec, FTI_Topo,
         FTI_Ckpt, fileSizes, mfs, ckptFileNames, checksums, allVarIDs,
         allRanks, allCounts,
         allVarTypeIDs, allVarTypeSizes,
         allVarSizes, allLayerSizes, allLayerHashes, allVarPositions,
          allNames, allCharIds), "write the metadata.");
        free(allVarIDs);
        free(allVarTypeIDs);
        free(allVarTypeSizes);
        free(allVarSizes);
        free(allCharIds);
        free(allNames);
        free(allCounts);
        free(allRanks);
        if (FTI_Ckpt[FTI_Exec->ckptMeta.level].isDcp) {
            free(allLayerSizes);
            free(allLayerHashes);
        }
        free(ckptFileNames);
        free(checksums);
        free(allVarPositions);
        if (res == FTI_NSCS) {
            return FTI_NSCS;
        }
    }

    for (i = 0; i < FTI_Exec->nbVar; i++) {
        data[i].sizeStored =  data[i].size;
    }

    return FTI_SCES;
}
