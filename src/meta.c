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
#include <time.h>

static FTIT_mqueue* mqueue;

int FTI_MetadataQueue( FTIT_mqueue* q )
{
    if( q == NULL ) {
        FTI_Print("metadata queue context is NULL", FTI_WARN);
        return FTI_NSCS;
    }
    
    q->_front = NULL;
    
    q->push = FTI_MetadataQueuePush;
    q->pop = FTI_MetadataQueuePop;
    q->empty = FTI_MetadataQueueEmpty;
    q->clear = FTI_MetadataQueueClear;

    mqueue = q;
}

int FTI_MetadataQueuePush( FTIT_metadata data )
{
    
    if( mqueue == NULL ) {
        FTI_Print("metadata queue context is NULL", FTI_WARN);
        return FTI_NSCS;
    }
    
    FTIT_mnode* old = mqueue->_front;
    FTIT_mnode* new = malloc( sizeof(FTIT_mnode) );
    
    new->_data = malloc( sizeof(FTIT_metadata) );
    memcpy( new->_data, &data, sizeof(FTIT_metadata) );
    new->_next = NULL; 

    mqueue->_front = new;
    mqueue->_front->_next = old;

    return FTI_SCES;

}

int FTI_MetadataQueuePop( FTIT_metadata* data )
{

    if( !mqueue ) return FTI_NSCS;
    if( !mqueue->_front ) return FTI_NSCS;
    if( !mqueue->_front->_data ) return FTI_NSCS;
    
    if( data )
        memcpy( data, mqueue->_front->_data, sizeof(FTIT_metadata) );
    
    FTIT_mnode* pop = mqueue->_front;

    mqueue->_front = mqueue->_front->_next;
    
    free(pop->_data);
    free(pop);

    return FTI_SCES;

}

bool FTI_MetadataQueueEmpty()
{
    if( !mqueue ) return true;
    return (mqueue->_front == NULL);
}

int FTI_MetadataQueueClear()
{
    while( !mqueue->empty() )
        mqueue->pop( NULL );
}

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
int FTI_LoadMetaPostprocessing(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt, int proc )
{
    // no metadata files for FTI-FF
    if ( FTI_Conf->ioMode == FTI_IO_FTIFF ) return FTI_SCES;
    if (!FTI_Topo->amIaHead) return FTI_SCES;

    char metaFileName[FTI_BUFS], str[FTI_BUFS];
    snprintf(metaFileName, FTI_BUFS, "%s/sector%d-group%d.fti", FTI_Conf->mTmpDir, FTI_Topo->sectorID, proc);
    snprintf(str, FTI_BUFS, "Getting FTI metadata file (%s)...", metaFileName);
    FTI_Print(str, FTI_DBUG);

    FTIT_iniparser ini; if( FTI_Iniparser( &ini, metaFileName ) != FTI_SCES ) return FTI_NSCS;

    snprintf(str, FTI_BUFS, "%d:Ckpt_file_name", FTI_Topo->groupRank);
    snprintf(FTI_Exec->ckptMeta.ckptFile, FTI_BUFS, "%s", ini.getString(&ini, str));

    //update head's ckptID
    sscanf(FTI_Exec->ckptMeta.ckptFile, "Ckpt%d", &FTI_Exec->ckptID);

    snprintf(str, FTI_BUFS, "%d:Ckpt_file_size", FTI_Topo->groupRank);
    FTI_Exec->ckptMeta.fs = ini.getLong(&ini, str);

    snprintf(str, FTI_BUFS, "%d:Ckpt_file_size", (FTI_Topo->groupRank + FTI_Topo->groupSize - 1) % FTI_Topo->groupSize);
    FTI_Exec->ckptMeta.pfs = ini.getLong(&ini, str);

    FTI_Exec->ckptMeta.maxFs = ini.getLong(&ini, "0:Ckpt_file_maxs");

    ini.clear( &ini );

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
int FTI_LoadMetaRecovery(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt)
{

    // no metadata files for FTI-FF
    if ( FTI_Conf->ioMode == FTI_IO_FTIFF ) { return FTI_SCES; }
    
    FTIT_iniparser ini;
    FTI_MetadataQueue( &FTI_Exec->mqueue ); 


    int i=4; for (; i > -1; i--) { //for each level

        FTIT_metadata meta;

        meta.level = i;

        char metaFileName[FTI_BUFS], str[FTI_BUFS];

        if (i == 0) snprintf(metaFileName, FTI_BUFS, "%s/sector%d-group%d.fti", FTI_Conf->mTmpDir, FTI_Topo->sectorID, FTI_Topo->groupID);
        else snprintf(metaFileName, FTI_BUFS, "%s/sector%d-group%d.fti", FTI_Ckpt[i].metaDir, FTI_Topo->sectorID, FTI_Topo->groupID);

        snprintf(str, FTI_BUFS, "Getting FTI metadata file (%s)...", metaFileName);
        FTI_Print(str, FTI_DBUG);

        if( FTI_Iniparser( &ini, metaFileName ) != FTI_SCES ) continue;

        snprintf(str, FTI_BUFS, "Meta for level %d exists.", i);
        FTI_Print(str, FTI_DBUG);

        // check for dcp
        FTI_Ckpt[i].recoIsDcp = !strcmp( "dcp", ini.getString( &ini, "ckpt_info:ckpt_type" ) );

        // get ckptID
        FTI_Exec->ckptID = ini.getInt( &ini, "ckpt_info:ckpt_id" );

        snprintf(str, FTI_BUFS, "%d:Ckpt_file_name", FTI_Topo->groupRank);
        snprintf( meta.ckptFile, FTI_BUFS, "%s", ini.getString( &ini, str ) );

        snprintf(str, FTI_BUFS, "%d:Ckpt_file_size", FTI_Topo->groupRank);
        meta.fs = ini.getLong( &ini, str );
        FTI_Exec->dcpInfoPosix.FileSize = meta.fs;

        snprintf(str, FTI_BUFS, "%d:Ckpt_file_size", (FTI_Topo->groupRank + FTI_Topo->groupSize - 1) % FTI_Topo->groupSize);
        meta.pfs = ini.getLong( &ini, str );

        meta.maxFs = ini.getLong( &ini, "0:Ckpt_file_maxs" );

        FTI_Exec->mqueue.push( meta );

        ini.clear( &ini );

    }
    
    return FTI_SCES;
}

int FTI_LoadMetaDcp(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt)
{

    // no metadata files for FTI-FF
    if ( FTI_Conf->ioMode == FTI_IO_FTIFF ) { return FTI_SCES; }
    
    char metaFileName[FTI_BUFS], str[FTI_BUFS];
    
    int level = FTI_Exec->ckptLvel;
    
    if ( level == 0 ) snprintf(metaFileName, FTI_BUFS, "%s/sector%d-group%d.fti", FTI_Conf->mTmpDir, FTI_Topo->sectorID, FTI_Topo->groupID);
    
    else snprintf(metaFileName, FTI_BUFS, "%s/sector%d-group%d.fti", FTI_Ckpt[level].metaDir, FTI_Topo->sectorID, FTI_Topo->groupID);

    snprintf(str, FTI_BUFS, "Getting FTI metadata file (%s)...", metaFileName);
    FTI_Print(str, FTI_DBUG);

    FTIT_iniparser ini; if( FTI_Iniparser( &ini, metaFileName ) != FTI_SCES ) return FTI_NSCS;

    int k; for (k = 0; k < MAX_STACK_SIZE; k++) {
        snprintf(str, FTI_BUFS, "%d:dcp_layer%d_size", FTI_Topo->groupRank, k);
        unsigned long LayerSize = ini.getLong( &ini, str );
        if (LayerSize == -1) {
            //No more variables
            break;
        }
        FTI_Exec->dcpInfoPosix.LayerSize[k] = LayerSize;

        snprintf(str, FTI_BUFS, "%d:dcp_layer%d_hash", FTI_Topo->groupRank, k);
        snprintf( &FTI_Exec->dcpInfoPosix.LayerHash[k*MD5_DIGEST_STRING_LENGTH], MD5_DIGEST_STRING_LENGTH, "%s", ini.getString( &ini, str ) );
        int j; for( j=0; j<FTI_BUFS; j++ )
        {
            snprintf( str, FTI_BUFS, "%d:dcp_layer%d_var%d_id", FTI_Topo->groupRank, k, j );
            int varID = ini.getInt( &ini, str );
            if( varID == -1 ) {
                break;
            }
            FTI_Exec->dcpInfoPosix.datasetInfo[k][j].varID = varID;
            snprintf( str, FTI_BUFS, "%d:dcp_layer%d_var%d_size", FTI_Topo->groupRank, k, j );
            long varSize = ini.getLong( &ini, str );
            if( varID < 0 ) {
                break;
            }
            FTI_Exec->dcpInfoPosix.datasetInfo[k][j].varSize = (unsigned long) varSize;
        }
    }

    ini.clear( &ini );
    
    return FTI_SCES;

}

int FTI_LoadMetaDataset(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt, FTIT_dataset* FTI_Data)
{

    // no metadata files for FTI-FF
    if ( FTI_Conf->ioMode == FTI_IO_FTIFF ) { return FTI_SCES; }
    
    char metaFileName[FTI_BUFS], str[FTI_BUFS];
    
    int level = FTI_Exec->ckptLvelReco;
    
    if ( level == 0 ) snprintf(metaFileName, FTI_BUFS, "%s/sector%d-group%d.fti", FTI_Conf->mTmpDir, FTI_Topo->sectorID, FTI_Topo->groupID);
    
    else snprintf(metaFileName, FTI_BUFS, "%s/sector%d-group%d.fti", FTI_Ckpt[level].metaDir, FTI_Topo->sectorID, FTI_Topo->groupID);

    snprintf(str, FTI_BUFS, "Getting FTI metadata file (%s)...", metaFileName);
    FTI_Print(str, FTI_DBUG);

    FTIT_iniparser ini; if( FTI_Iniparser( &ini, metaFileName ) != FTI_SCES ) return FTI_NSCS;

    int k,j; for (k = 0; k < FTI_BUFS; k++) {
        snprintf(str, FTI_BUFS, "%d:Var%d_id", FTI_Topo->groupRank, k);
        int id = ini.getInt( &ini, str );
        if (id == -1) {
            //No more variables
            break;
        }
        //Variable exists
        FTI_Data[k].id = id;

        snprintf(str, FTI_BUFS, "%d:Var%d_size", FTI_Topo->groupRank, k);
        FTI_Data[k].size = 0;//ini.getLong( &ini, str );
        FTI_Data[k].storedSize = ini.getLong( &ini, str );

        snprintf(str, FTI_BUFS, "%d:Var%d_pos", FTI_Topo->groupRank, k);
        FTI_Data[k].filePos = ini.getLong( &ini, str );
        FTI_Data[k].storedFilePos = ini.getLong( &ini, str );

        snprintf(str, FTI_BUFS, "%d:Var%d_name", FTI_Topo->groupRank, k);
        strncpy(FTI_Data[k].idChar, ini.getString( &ini, str ), FTI_BUFS);
    
        // Important assignment, we use realloc!
        FTI_Data[k].sharedData.dataset = NULL;
        FTI_Data[k].rank = 1;
        FTI_Data[k].h5group = FTI_Exec->H5groups[0];
        sprintf(FTI_Data[k].name, "Dataset_%d", id);
        FTI_Exec->ckptSize = FTI_Exec->ckptSize + FTI_Data[k].size;

        if ( FTI_Conf->dcpPosix ){
            FTI_Data[k].dcpInfoPosix.hashDataSize = 0;
            FTI_Data[k].dcpInfoPosix.currentHashArray=NULL;
            FTI_Data[k].dcpInfoPosix.oldHashArray=NULL;
        }

        FTI_Data[k].recovered = true;
        
    }

    //Save number of variables in metadata
    FTI_Exec->nbVarStored = k;

    ini.clear( &ini );
    
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
int FTI_LoadL4CkptMetaData(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt )
{
    char str[FTI_BUFS], fn[FTI_BUFS];
    snprintf(fn, FTI_BUFS, "%s/Checkpoint.fti", FTI_Conf->metadDir);

    dictionary* ini;

    if ( access( fn, F_OK ) != 0 ) { 
        snprintf(str, FTI_BUFS, "Could not access checkpoint metadata file (%s)...", fn);
        FTI_Print(str, FTI_EROR);
    }     

    // initialize dictionary
    ini = iniparser_load(fn);
    if ( ini == NULL ) {
        FTI_Print("Failed to load dictionary for checkpoint meta data file.", FTI_EROR);
        return FTI_NSCS;
    }

    if ( ((int)(ini->n)-6) < 0 ) {
        FTI_Print("Unexpected checkpoint meta data file structure.", FTI_EROR);
        return FTI_NSCS;
    }

    char currCkpt[FTI_BUFS];

    // ini->key[2] must be the level field
    int ckptID;
    int i=0;
    bool hasL4Ckpt = false;
    for(; i<ini->n; i+=6) {
        memset( currCkpt, 0x0, FTI_BUFS );
        strncpy( currCkpt, ini->key[i], FTI_BUFS-1 );
        char level_key[FTI_BUFS];
        snprintf( level_key, FTI_BUFS, "%s:level", currCkpt );
        if( iniparser_getint( ini, level_key, -1) == 4 ){
            hasL4Ckpt = true;
            sscanf( currCkpt, "checkpoint_id.%d", &ckptID );
        }
    }

    // if we find a level 4 checkpoint return the id if not return -1
    return (hasL4Ckpt) ? ckptID : -1;    

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
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt )
{
    char str[FTI_BUFS], fn[FTI_BUFS];
    snprintf(fn, FTI_BUFS, "%s/Checkpoint.fti", FTI_Conf->metadDir);

    dictionary* ini;

    if ( access( fn, F_OK ) != 0 ) { 
        snprintf(str, FTI_BUFS, "Could not access checkpoint metadata file (%s)...", fn);
        FTI_Print(str, FTI_EROR);
    }     

    // initialize dictionary
    ini = iniparser_load(fn);
    if ( ini == NULL ) {
        FTI_Print("Failed to load dictionary for checkpoint meta data file.", FTI_EROR);
        return FTI_NSCS;
    }

    char lastCkpt[FTI_BUFS];
    if ( ((int)(ini->n)-6) < 0 ) {
        FTI_Print("Unexpected checkpoint meta data file structure.", FTI_EROR);
        return FTI_NSCS;
    }

    memset( lastCkpt, 0x0, FTI_BUFS );
    strncpy( lastCkpt, ini->key[ini->n-6], FTI_BUFS-1);

    int ckptID;
    sscanf(lastCkpt, "checkpoint_id.%d", &ckptID ); 

    char key[FTI_BUFS];
    snprintf( key, FTI_BUFS, "%s:level", lastCkpt );
    int ckptLvel = iniparser_getint( ini, key, -1);

    if ( ckptLvel == -1 ) {
        FTI_Print( "Unable to read checkpoint level from checkpoint meta data file", FTI_EROR );
        dictionary_del(ini);
        return FTI_NSCS;
    }

    if ( ckptLvel == 4 ) {
        snprintf( key, FTI_BUFS, "%s:is_dcp", lastCkpt );
        int isDcp = iniparser_getboolean( ini, key, -1);
        if ( isDcp == -1 ) {
            FTI_Print( "Unable to identify if dCP from checkpoint meta data file", FTI_EROR );
            dictionary_del(ini);
            return FTI_NSCS;
        } else {    
            FTI_Ckpt[4].recoIsDcp = (bool) isDcp;
        }
    }

    //FTI_Exec->ckptLvel = ckptLvel;
    FTI_Exec->ckptID = ckptID;

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
int FTI_WriteCkptMetaData(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt )
{

    char str[FTI_BUFS], fn[FTI_BUFS], strErr[FTI_BUFS];
    snprintf(fn, FTI_BUFS, "%s/Checkpoint.fti", FTI_Conf->metadDir);

    FILE* fstream;
    dictionary* ini;

    // initialize dictionary
    // [A] - create empty if no ckpt meta file exists
    if ( access( fn, F_OK ) != 0 ) { 
        snprintf(str, FTI_BUFS, "Creating checkpoint metadata file (%s)...", fn);
        FTI_Print(str, FTI_DBUG);
        ini = dictionary_new(0);
        if ( ini == NULL ) {
            FTI_Print("Failed to allocate dictionary for checkpoint meta data file.", FTI_EROR);
            return FTI_NSCS;
        }
        // [B] - initialize with data from ckpt meta file
    } else {
        snprintf(str, FTI_BUFS, "Updating checkpoint metadata file (%s)...", fn);
        FTI_Print(str, FTI_DBUG);
        ini = iniparser_load(fn);
        if ( ini == NULL ) {
            FTI_Print("Failed to load dictionary for checkpoint meta data file.", FTI_EROR);
            return FTI_NSCS;
        }
    }

    char section[FTI_BUFS], key[FTI_BUFS], value[FTI_BUFS];
    snprintf( section, FTI_BUFS, "checkpoint_id.%d", FTI_Exec->ckptID );
    iniparser_set( ini, section, NULL ); 
    time_t time_ctx;
    struct tm * time_info;
    time( &time_ctx );
    time_info = localtime( &time_ctx );
    char timestr[FTI_BUFS];
    strftime(timestr,FTI_BUFS,"%A %x - %H:%M:%S", time_info);
    snprintf( key, FTI_BUFS, "%s:timestamp", section );
    snprintf( value, FTI_BUFS, "%s", timestr );
    iniparser_set( ini, key, value ); 
    snprintf( key, FTI_BUFS, "%s:level", section );
    snprintf( value, FTI_BUFS, "%d", FTI_Exec->ckptLvel );
    iniparser_set( ini, key, value ); 
    snprintf( key, FTI_BUFS, "%s:nb_procs", section );
    snprintf( value, FTI_BUFS, "%d", FTI_Topo->nbApprocs * FTI_Topo->nbNodes );
    iniparser_set( ini, key, value ); 
    snprintf( key, FTI_BUFS, "%s:io_mode", section );
    snprintf( value, FTI_BUFS, "%d", FTI_Conf->ioMode );
    iniparser_set( ini, key, value ); 
    snprintf( key, FTI_BUFS, "%s:is_dcp", section );
    snprintf( value, FTI_BUFS, "%s", (FTI_Ckpt[FTI_Exec->ckptLvel].isDcp) ? "true" : "false" );
    iniparser_set( ini, key, value ); 

    fstream = fopen( fn, "w" );
    if ( fstream == NULL ) {
        snprintf( strErr, FTI_BUFS, "Failed to write checkpoint meta data to file (%s).",fn );
        FTI_Print( strErr, FTI_EROR );
        dictionary_del(ini);
    }

    iniparser_dump_ini( ini, fstream );

    if ( fclose( fstream ) != 0 ) {
        snprintf( strErr, FTI_BUFS, "Failed to close checkpoint meta data file (%s)", fn );
        FTI_Print( strErr, FTI_WARN );
    }

    dictionary_del(ini);

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
        FTIT_topology* FTI_Topo, FTIT_checkpoint* FTI_Ckpt, long* fs, long mfs, char* fnl,
        char* checksums, int* allVarIDs, long* allVarSizes, unsigned long* allLayerSizes,
        char* allLayerHashes,long *allVarPositions, char *allCharIds )
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

    // Add dcp POSIX meta data
    iniparser_set( ini, "ckpt_info", NULL );
    switch( FTI_Ckpt[FTI_Exec->ckptLvel].isDcp ) {
        case 0:
            iniparser_set( ini, "ckpt_info:ckpt_type", "full" );
            break;
        case 1:
            iniparser_set( ini, "ckpt_info:ckpt_type", "dcp" );
            break;
    } 

    // add checkpoint id
    snprintf( buf, FTI_BUFS, "%d", FTI_Exec->ckptID );
    iniparser_set( ini, "ckpt_info:ckpt_id", buf ); 

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

            snprintf(str, FTI_BUFS, "%d:Var%d_pos", i, j);
            snprintf(buf, FTI_BUFS, "%ld", allVarPositions[i * FTI_Exec->nbVar + j]);
            iniparser_set(ini, str, buf);

            snprintf(str, FTI_BUFS, "%d:Var%d_name", i,j);
            snprintf(buf, FTI_BUFS, "%s", &allCharIds[ (i*FTI_Exec->nbVar*FTI_BUFS) +j*FTI_BUFS]);
            iniparser_set(ini,str,buf);
        }
        if( FTI_Ckpt[FTI_Exec->ckptLvel].isDcp ) {
            int nbLayer = ((FTI_Exec->dcpInfoPosix.Counter-1) % FTI_Conf->dcpInfoPosix.StackSize) + 1;
            for( j=0; j<nbLayer; j++ ) {
                snprintf(str, FTI_BUFS, "%d:dcp_layer%d_size", i, j);
                snprintf(buf, FTI_BUFS, "%lu", allLayerSizes[i * nbLayer + j]);
                iniparser_set(ini, str, buf);

                snprintf(str, FTI_BUFS, "%d:dcp_layer%d_hash", i, j);
                iniparser_set(ini, str, &allLayerHashes[i*nbLayer*MD5_DIGEST_STRING_LENGTH + j*MD5_DIGEST_STRING_LENGTH]);
                int k;
                for (k = 0; k < FTI_Exec->nbVar; k++) {
                    //Save id of variable
                    snprintf(str, FTI_BUFS, "%d:dcp_layer%d_var%d_id", i, j, k);
                    snprintf(buf, FTI_BUFS, "%d", allVarIDs[i * FTI_Exec->nbVar + k]);
                    iniparser_set(ini, str, buf);

                    //Save size of variable
                    snprintf(str, FTI_BUFS, "%d:dcp_layer%d_var%d_size", i, j, k);
                    snprintf(buf, FTI_BUFS, "%ld", allVarSizes[i * FTI_Exec->nbVar + k]);
                    iniparser_set(ini, str, buf);
                }
            }
        }
    }

    // Remove topology section
    iniparser_unset(ini, "topology");
    MKDIR(FTI_Conf->mTmpDir,0777);

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

    FTI_Exec->ckptMeta.fs = (FTI_Ckpt[FTI_Exec->ckptLvel].isDcp) ? FTI_Exec->dcpInfoPosix.FileSize : FTI_Exec->ckptSize;

#ifdef ENABLE_HDF5
    if( FTI_Conf->ioMode == FTI_IO_HDF5 ) {
        char fn[FTI_BUFS];
        if (FTI_Exec->ckptLvel == 4 && FTI_Ckpt[4].isInline) { //If inline L4 save directly to global directory
            snprintf(fn, FTI_BUFS, "%s/%s", FTI_Conf->gTmpDir, FTI_Exec->ckptMeta.ckptFile);
        }
        else {
            snprintf(fn, FTI_BUFS, "%s/%s", FTI_Conf->lTmpDir, FTI_Exec->ckptMeta.ckptFile);
        }
        if (access(fn, F_OK) == 0) {
            struct stat fileStatus;
            if (stat(fn, &fileStatus) == 0) {
                FTI_Exec->ckptMeta.fs = fileStatus.st_size;
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
    }
#endif

    long fileSizes[FTI_BUFS];
    MPI_Allgather(&FTI_Exec->ckptMeta.fs, 1, MPI_LONG, fileSizes, 1, MPI_LONG, FTI_Exec->groupComm);

    //update partner file size:
    if (FTI_Exec->ckptLvel == 2) {
        int ptnerGroupRank = (FTI_Topo->groupRank + FTI_Topo->groupSize - 1) % FTI_Topo->groupSize;
        FTI_Exec->ckptMeta.pfs = fileSizes[ptnerGroupRank];
    }

    long mfs = 0; //Max file size in group
    int i;
    for (i = 0; i < FTI_Topo->groupSize; i++) {
        if (fileSizes[i] > mfs) {
            mfs = fileSizes[i]; // Search max. size
        }
    }
    FTI_Exec->ckptMeta.maxFs = mfs;
    char str[FTI_BUFS]; //For console output
    snprintf(str, FTI_BUFS, "Max. file size in group %lu.", mfs);
    FTI_Print(str, FTI_DBUG);

    char* ckptFileNames = NULL;
    if (FTI_Topo->groupRank == 0) {
        ckptFileNames = talloc(char, FTI_Topo->groupSize * FTI_BUFS);
    }
    strncpy(str, FTI_Exec->ckptMeta.ckptFile, FTI_BUFS); // Gather all the file names
    MPI_Gather(str, FTI_BUFS, MPI_CHAR, ckptFileNames, FTI_BUFS, MPI_CHAR, 0, FTI_Exec->groupComm);

    char checksum[MD5_DIGEST_STRING_LENGTH];
    FTI_Checksum(FTI_Exec, FTI_Data, FTI_Conf, checksum);

    //TODO checksums of HDF5 files
#ifdef ENABLE_HDF5
    if (FTI_Conf->ioMode == FTI_IO_HDF5) {
        checksum[0] = '\0';
    }
#endif

    char* checksums = NULL;
    if (FTI_Topo->groupRank == 0) {
        checksums = talloc(char, FTI_Topo->groupSize * MD5_DIGEST_STRING_LENGTH);
    }
    MPI_Gather(checksum, MD5_DIGEST_STRING_LENGTH, MPI_CHAR, checksums, MD5_DIGEST_STRING_LENGTH, MPI_CHAR, 0, FTI_Exec->groupComm);


    //Every process has the same number of protected variables

    int* allVarIDs = NULL;
    long* allVarSizes = NULL;
    long *allVarPositions = NULL;

    // for posix dcp
    unsigned long* allLayerSizes = NULL;
    char* allLayerHashes = NULL;
    char* allCharIds = NULL;

    int nbLayer = ((FTI_Exec->dcpInfoPosix.Counter-1) % FTI_Conf->dcpInfoPosix.StackSize) + 1;

    if (FTI_Topo->groupRank == 0) {
        allVarIDs = talloc(int, FTI_Topo->groupSize * FTI_Exec->nbVar);
        allVarSizes = talloc(long, FTI_Topo->groupSize * FTI_Exec->nbVar);
        allVarPositions = talloc(long, FTI_Topo->groupSize * FTI_Exec->nbVar);
        allCharIds = (char *) malloc( sizeof(char)*FTI_BUFS*FTI_Exec->nbVar*FTI_Topo->groupSize);
        if( FTI_Ckpt[FTI_Exec->ckptLvel].isDcp ) {
            allLayerSizes = talloc( unsigned long, FTI_Topo->groupSize * nbLayer );
            allLayerHashes = talloc( char, FTI_Topo->groupSize * nbLayer * MD5_DIGEST_STRING_LENGTH );
        }
    }

    int* myVarIDs = talloc(int, FTI_Exec->nbVar);
    long* myVarSizes = talloc(long, FTI_Exec->nbVar);
    long* myVarPositions = talloc(long, FTI_Exec->nbVar);
    char *ArrayOfStrings = ( char *) malloc (FTI_Exec->nbVar * sizeof(char*) *FTI_BUFS);

    for (i = 0; i < FTI_Exec->nbVar; i++) {
        myVarIDs[i] = FTI_Data[i].id;
        myVarSizes[i] =  FTI_Data[i].size;
        myVarPositions[i] = FTI_Data[i].filePos;
        strncpy(&ArrayOfStrings[i*FTI_BUFS], FTI_Data[i].idChar, FTI_BUFS);
    }

    //Gather variables IDs
    MPI_Gather(myVarIDs, FTI_Exec->nbVar, MPI_INT, allVarIDs, FTI_Exec->nbVar, MPI_INT, 0, FTI_Exec->groupComm);
    //Gather variables sizes
    MPI_Gather(myVarSizes, FTI_Exec->nbVar, MPI_LONG, allVarSizes, FTI_Exec->nbVar, MPI_LONG, 0, FTI_Exec->groupComm);
    //Gather variables file positions
    MPI_Gather(myVarPositions, FTI_Exec->nbVar, MPI_LONG, allVarPositions, FTI_Exec->nbVar, MPI_LONG, 0, FTI_Exec->groupComm);
    //Gather all variable names
    MPI_Gather(ArrayOfStrings, FTI_Exec->nbVar*FTI_BUFS, MPI_CHAR, 
        allCharIds, FTI_Exec->nbVar*FTI_BUFS, MPI_CHAR, 0, FTI_Exec->groupComm);

    if( FTI_Ckpt[FTI_Exec->ckptLvel].isDcp ) {
        // Gather dcp layer sizes
        MPI_Gather(FTI_Exec->dcpInfoPosix.LayerSize, nbLayer, MPI_UNSIGNED_LONG, 
                allLayerSizes, nbLayer, MPI_UNSIGNED_LONG, 0, FTI_Exec->groupComm);
        // Gather dcp layer hashes
        MPI_Gather(FTI_Exec->dcpInfoPosix.LayerHash, nbLayer * MD5_DIGEST_STRING_LENGTH, MPI_CHAR, 
                allLayerHashes, nbLayer * MD5_DIGEST_STRING_LENGTH, MPI_CHAR, 0, FTI_Exec->groupComm);
    }

    free(myVarIDs);
    free(myVarSizes);
    free(myVarPositions);
    free(ArrayOfStrings);

    if (FTI_Topo->groupRank == 0) { // Only one process in the group create the metadata
        int res = FTI_Try(FTI_WriteMetadata(FTI_Conf, FTI_Exec, FTI_Topo, FTI_Ckpt, fileSizes, mfs,
                    ckptFileNames, checksums, allVarIDs, allVarSizes, allLayerSizes, allLayerHashes,allVarPositions, allCharIds), "write the metadata.");
        free(allVarIDs);
        free(allVarSizes);
        free(allCharIds);
        if( FTI_Ckpt[FTI_Exec->ckptLvel].isDcp ) {
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
        FTI_Data[i].storedSize =  FTI_Data[i].size;
    }

    return FTI_SCES;
}
