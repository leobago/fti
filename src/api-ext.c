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
 *  @file   api-ext.c
 *  @date   July, 2022
 *  @brief  API functions for the extended FTI functionality.
 */


#define FTIX_PREREC(RETURN_VAL) do {                                          \
  if ( !FTI_INITIALIZED ) {                                                   \
    FTI_Print("[missing prerequisite] FTI is not Initialized!\n", FTI_WARN);  \
    return RETURN_VAL;                                                        \
  }                                                                           \
} while ( false )

#include "fti-ext.h"
#include "fti-kernel.h"
#include "interface.h"

int FTIX_Load( const char* ckptDir ) {
  FTIX_PREREC(-1);
  char mfn[FTI_BUFS];
  char key[FTI_BUFS];
  snprintf(mfn, FTI_BUFS, "%s/sector%d-group%d.fti", ckptDir, FTI_Topo.sectorID, FTI_Topo.groupID);
  FTIT_iniparser ini;
  if (FTI_Iniparser(&ini, mfn, FTI_INI_OPEN) != FTI_SCES) {
      FTI_Print("Iniparser failed to parse the metadata file.", FTI_WARN);
      return FTI_NSCS;
  }
  snprintf( key, FTI_BUFS, "ckpt_info:ckpt_id" );
  int ckptId = ini.getInt(&ini, key);

  typedef struct dlist dlist;
  struct dlist {
    int64_t filePos;
    int64_t size;
    void* dataPtr;
    dlist* next;
  };

  char fn[FTI_BUFS];
  snprintf( key, FTI_BUFS, "%d:ckpt_file_name", FTI_Topo.groupRank );
  snprintf(fn, FTI_BUFS, "%s/%s", ckptDir, ini.getString(&ini, key));

  dlist* dnode = (dlist*) malloc(sizeof(dlist));
  dlist* dfirst = dnode;

  FTIT_dataset data_stored; 
  FTIT_dataset* data;
  int id; 
  int i; 
  
  for (i = 0; i < FTI_Conf.maxVarId; i++) {

    snprintf(key, FTI_BUFS, "%d:Var%d_id", FTI_Topo.groupRank, i);
    if ( (id = ini.getInt(&ini, key)) == -1 ) break;
    
    if (FTI_Data->get(&data, id) != FTI_SCES) {
        FTI_Print("failed to set attribute: could not query dataset", FTI_WARN);
        return FTI_NSCS;
    }

    FTI_InitDataset(&FTI_Exec, &data_stored, id);

    snprintf(key, FTI_BUFS, "%d:Var%d_size", FTI_Topo.groupRank, i);
    data_stored.size = ini.getLong(&ini, key);
    
    if ( data_stored.size != data->size ) {
      char wstr[FTI_BUFS];
      snprintf( wstr, FTI_BUFS, 
          "Load failed for id '%d': stored size (%ld Bytes) and current size (%ld Bytes) differ", 
          id, data_stored.size, data->size);
      FTI_Print(wstr, FTI_WARN);
      return FTI_NSCS;
    }

    snprintf(key, FTI_BUFS, "%d:Var%d_pos", FTI_Topo.groupRank, i);
    data_stored.filePos = ini.getLong(&ini, key);
    
    if( i > 0 ) {
      
      dnode->next = (dlist*) malloc(sizeof(dlist));
      dnode = dnode->next;
    
    }

    dnode->dataPtr = data->ptr;
    dnode->size = data_stored.size;
    dnode->filePos = data_stored.filePos;
    dnode->next = NULL;

  }
  
  char estr[FTI_BUFS];
  FILE *fd = fopen(fn, "rb");
  if (fd == NULL) {
    snprintf(estr, FTI_BUFS, "FTI failed to open '%s'.", fn);
    FTI_Print(estr, FTI_WARN);
    return FTI_NSCS;
  }

  dnode = dfirst;

  do {
    dlist* dfree = dnode;
    
    fseek(fd, dnode->filePos, SEEK_SET);
    
    int64_t bytes = 0;
    while( bytes < dnode->size ) {
      int64_t block = FTI_MIN( dnode->size - bytes, FTI_Conf.transferSize );
      bytes += fread( dnode->dataPtr + bytes, 1, block, fd );
      if (feof(fd)) break;
      if (ferror(fd)) {
        snprintf(estr, FTI_BUFS, "FTI failed to read from '%s'.", fn);
        FTI_Print(estr, FTI_WARN);
        return FTI_NSCS;
      }
    }
    
    if ( bytes != dnode->size ) {
        FTI_Print("Wrong size", FTI_EROR);
        return FTI_NSCS;
    }
    dnode = dnode->next;
    free(dfree);
  } while ( dnode );
  
  fclose(fd);

  return ckptId;

}

int FTIX_TopoGet_nbProc(){ 
  FTIX_PREREC(-1);
  return FTI_Topo.nbProc;
}

int FTIX_TopoGet_nbNodes(){ 
  FTIX_PREREC(-1);
  return FTI_Topo.nbNodes;
}

int FTIX_TopoGet_myRank(){ 
  FTIX_PREREC(-1);
  return FTI_Topo.myRank;
}

int FTIX_TopoGet_splitRank(){ 
  FTIX_PREREC(-1);
  return FTI_Topo.splitRank;
}

int FTIX_TopoGet_nodeSize(){ 
  FTIX_PREREC(-1);
  return FTI_Topo.nodeSize;
}

int FTIX_TopoGet_nbHeads(){ 
  FTIX_PREREC(-1);
  return FTI_Topo.nbHeads;
}

int FTIX_TopoGet_nbApprocs(){ 
  FTIX_PREREC(-1);
  return FTI_Topo.nbApprocs;
}

int FTIX_TopoGet_groupSize(){ 
  FTIX_PREREC(-1);
  return FTI_Topo.groupSize;
}

int FTIX_TopoGet_sectorID(){ 
  FTIX_PREREC(-1);
  return FTI_Topo.sectorID;
}

int FTIX_TopoGet_nodeID(){ 
  FTIX_PREREC(-1);
  return FTI_Topo.nodeID;
}

int FTIX_TopoGet_groupID(){ 
  FTIX_PREREC(-1);
  return FTI_Topo.groupID;
}

bool FTIX_TopoGet_amIaHead(){ 
  FTIX_PREREC(-1);
  return FTI_Topo.amIaHead;
}

int FTIX_TopoGet_headRank(){ 
  FTIX_PREREC(-1);
  return FTI_Topo.headRank;
}

int FTIX_TopoGet_headRankNode(){ 
  FTIX_PREREC(-1);
  return FTI_Topo.headRankNode;
}

int FTIX_TopoGet_nodeRank(){ 
  FTIX_PREREC(-1);
  return FTI_Topo.nodeRank;
}

int FTIX_TopoGet_groupRank(){ 
  FTIX_PREREC(-1);
  return FTI_Topo.groupRank;
}

int FTIX_TopoGet_right(){ 
  FTIX_PREREC(-1);
  return FTI_Topo.right;
}

int FTIX_TopoGet_left(){ 
  FTIX_PREREC(-1);
  return FTI_Topo.left;
}

int* FTIX_TopoGet_body( int* body, int* len, int maxlen ){ 
  FTIX_PREREC(NULL);
  *len = FTI_Topo.nodeSize - FTI_Topo.nbHeads;
  if ( maxlen < *len ) {
    FTI_Print("not enough space in array", FTI_WARN);
    return NULL;
  }
  memset(body, 0x0, sizeof(int)*maxlen);
  memcpy(body, FTI_Topo.body, sizeof(int)*(*len));
  return body;
}

char* FTIX_ExecGet_id( char* id, int* len, int maxlen ){ 
  FTIX_PREREC(NULL);
  *len = strlen(FTI_Exec.id);
  if ( maxlen < *len ) {
    FTI_Print("not enough space in array", FTI_WARN);
    return NULL;
  }
  memset(id, 0x0, sizeof(char)*maxlen);
  memcpy(id, FTI_Exec.id, sizeof(char)*(*len));
  return id;
}

MPI_Comm FTIX_ExecGet_globalComm(){ 
  FTIX_PREREC(MPI_COMM_NULL);
  return FTI_Exec.globalComm;
}

int FTIX_ConfGet_blockSize(){ 
  FTIX_PREREC(-1);
  return FTI_Conf.blockSize;
}

int FTIX_ConfGet_transferSize(){ 
  FTIX_PREREC(-1);
  return FTI_Conf.transferSize;
}

int FTIX_ConfGet_ckptTag(){ 
  FTIX_PREREC(-1);
  return FTI_Conf.ckptTag;
}

int FTIX_ConfGet_stageTag(){ 
  FTIX_PREREC(-1);
  return FTI_Conf.stageTag;
}

int FTIX_ConfGet_finalTag(){ 
  FTIX_PREREC(-1);
  return FTI_Conf.finalTag;
}

int FTIX_ConfGet_generalTag(){ 
  FTIX_PREREC(-1);
  return FTI_Conf.generalTag;
}

char* FTIX_ConfGet_cfgFile( char* cfgFile, int* len, int maxlen ){ 
  FTIX_PREREC(NULL);
  *len = strlen(FTI_Conf.cfgFile);
  if ( maxlen < *len ) {
    FTI_Print("not enough space in array", FTI_WARN);
    return NULL;
  }
  memset(cfgFile, 0x0, sizeof(char)*maxlen);
  memcpy(cfgFile, FTI_Conf.cfgFile, sizeof(char)*(*len));
  return cfgFile;
}

char* FTIX_ConfGet_localDir( char* localDir, int* len, int maxlen ){ 
  FTIX_PREREC(NULL);
  *len = strlen(FTI_Conf.localDir);
  if ( maxlen < *len ) {
    FTI_Print("not enough space in array", FTI_WARN);
    return NULL;
  }
  memset(localDir, 0x0, sizeof(char)*maxlen);
  memcpy(localDir, FTI_Conf.localDir, sizeof(char)*(*len));
  return localDir;
}

char* FTIX_ConfGet_glbalDir( char* glbalDir, int* len, int maxlen ){ 
  FTIX_PREREC(NULL);
  *len = strlen(FTI_Conf.glbalDir);
  if ( maxlen < *len ) {
    FTI_Print("not enough space in array", FTI_WARN);
    return NULL;
  }
  memset(glbalDir, 0x0, sizeof(char)*maxlen);
  memcpy(glbalDir, FTI_Conf.glbalDir, sizeof(char)*(*len));
  return glbalDir;
}

char* FTIX_ConfGet_metadDir( char* metadDir, int* len, int maxlen ){ 
  FTIX_PREREC(NULL);
  *len = strlen(FTI_Conf.metadDir);
  if ( maxlen < *len ) {
    FTI_Print("not enough space in array", FTI_WARN);
    return NULL;
  }
  memset(metadDir, 0x0, sizeof(char)*maxlen);
  memcpy(metadDir, FTI_Conf.metadDir, sizeof(char)*(*len));
  return metadDir;
}

char* FTIX_ConfGet_lTmpDir( char* lTmpDir, int* len, int maxlen ){ 
  FTIX_PREREC(NULL);
  *len = strlen(FTI_Conf.lTmpDir);
  if ( maxlen < *len ) {
    FTI_Print("not enough space in array", FTI_WARN);
    return NULL;
  }
  memset(lTmpDir, 0x0, sizeof(char)*maxlen);
  memcpy(lTmpDir, FTI_Conf.lTmpDir, sizeof(char)*(*len));
  return lTmpDir;
}

char* FTIX_ConfGet_gTmpDir( char* gTmpDir, int* len, int maxlen ){ 
  FTIX_PREREC(NULL);
  *len = strlen(FTI_Conf.gTmpDir);
  if ( maxlen < *len ) {
    FTI_Print("not enough space in array", FTI_WARN);
    return NULL;
  }
  memset(gTmpDir, 0x0, sizeof(char)*maxlen);
  memcpy(gTmpDir, FTI_Conf.gTmpDir, sizeof(char)*(*len));
  return gTmpDir;
}

char* FTIX_ConfGet_mTmpDir( char* mTmpDir, int* len, int maxlen ){ 
  FTIX_PREREC(NULL);
  *len = strlen(FTI_Conf.mTmpDir);
  if ( maxlen < *len ) {
    FTI_Print("not enough space in array", FTI_WARN);
    return NULL;
  }
  memset(mTmpDir, 0x0, sizeof(char)*maxlen);
  memcpy(mTmpDir, FTI_Conf.mTmpDir, sizeof(char)*(*len));
  return mTmpDir;
}

char* FTIX_ConfGet_suffix( char* suffix, int* len, int maxlen ){ 
  FTIX_PREREC(NULL);
  *len = strlen(FTI_Conf.suffix);
  if ( maxlen < *len ) {
    FTI_Print("not enough space in array", FTI_WARN);
    return NULL;
  }
  memset(suffix, 0x0, sizeof(char)*maxlen);
  memcpy(suffix, FTI_Conf.suffix, sizeof(char)*(*len));
  return suffix;
}
