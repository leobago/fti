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
 *  @author Kai Keller (kellekai@gmx.de)
 *  @file   api-ext.c
 *  @date   July, 2022
 *  @brief  API functions for the extended FTI functionality.
 */

#define FTI_NODEFLAG \
    ( (((!FTI_Topo.amIaHead) && ((FTI_Topo.nodeRank - FTI_Topo.nbHeads) == 0)) || (FTI_Topo.amIaHead)) ? 1 : 0 )

#define FTIX_PREREC(RETURN_VAL) do {                                          \
  if ( FTI_Exec.initSCES == 0 ) {                                             \
    FTI_Print("[missing prerequisite] FTI is not Initialized!\n", FTI_WARN);  \
    return RETURN_VAL;                                                        \
  }                                                                           \
} while ( false )

#include "fti-ext.h"
#include "fti-kernel.h"
#include "interface.h"
#include <glob.h>
#include <libgen.h>

/* TODO:
 *
 * Create MPI Window that holds a hash-table that keeps
 * all existing stashes. We can keep there all local and
 * global stashes. In that way we can mix creating stashes
 * from local and head ranks. We can also easily check if
 * a stash already exists.
 *
 * We can then also provide a means of synchronization.
 * currently the user has to ensure consistency.
 *
 */

int FTIX_StashPull( uint64_t stashId ) {
  FTIX_PREREC(-1);

  char err[FTI_BUFS], info[FTI_BUFS];
  int retVal, error=0, exists = 0, errorAll, existsAll;

  if( stashId > INT64_MAX ) {
    FTI_Print("stash ID must not be larger than INT64_MAX", FTI_WARN);
    error = 1;
  }

  char stashLocal[FTI_BUFS];
  if ( !error ) {
    snprintf( stashLocal, FTI_BUFS, "%s/%lu", FTI_Conf.stashDir, stashId );
    retVal = FTI_CheckDirectory( stashLocal ); 
    if ( retVal == 1 ) {
      snprintf( err, FTI_BUFS, "stash with id '%lu' already exists", stashId ); 
      FTI_Print( err, FTI_WARN );
      exists = 1;
    } else if ( retVal == -1 ) {
      error = 1;
    }
  }

  char stashGlobal[FTI_BUFS];
  if ( !error ) {
    snprintf( stashGlobal, FTI_BUFS, "%s/%lu", FTI_Conf.stashDirGlobal, stashId );
    retVal = FTI_CheckDirectory( stashGlobal ); 
    if ( retVal == 0 ) {
      snprintf( info, FTI_BUFS, "stash with id '%lu' does not exists on global directory '%s'", stashId, stashGlobal ); 
      FTI_Print( err, FTI_WARN );
      error = 1;
    } else if ( retVal == -1 ) {
      error = 1;
    }
  }
  
  MPI_Allreduce( &error, &errorAll, 1, MPI_INT, MPI_SUM, FTI_COMM_WORLD ); 
  MPI_Allreduce( &exists, &existsAll, 1, MPI_INT, MPI_SUM, FTI_COMM_WORLD ); 
  if ( errorAll ) { 
    snprintf( err, FTI_BUFS, "failed to pull stash with id '%lu'", stashId ); 
    FTI_Print( err, FTI_WARN );
    return FTI_NSCS;
  } else if ( (existsAll < FTI_Topo.splitSize) &&  (existsAll > 0) ) { 
    snprintf( err, FTI_BUFS, "failed to pull stash with id '%lu' (some exist local some not)", stashId ); 
    FTI_Print( err, FTI_WARN );
    return FTI_SCES;
  } else if ( existsAll == FTI_Topo.splitSize ) { 
    snprintf( info, FTI_BUFS, "Succsessfully pulled stashed checkpoint (stashId: %lu)", stashId );
    FTI_Print( info, FTI_INFO );
    return FTI_SCES;
  }

  if ( FTI_CreateDirectory( &FTI_Topo, stashLocal, FTI_FS_LOCAL ) != FTI_SCES ) {
    snprintf( err, FTI_BUFS, "failed to pull stash with id '%lu'", stashId ); 
    FTI_Print( err, FTI_WARN );
    return FTI_NSCS;
  }
  
  MPI_Barrier(FTI_COMM_WORLD);
  
  error = 0;

  char fi[FTI_BUFS], fo[FTI_BUFS];
  
  glob_t glb_ctx;

  char pattern[FTI_BUFS];
  if ( FTI_Topo.masterLocal ) {
    snprintf( pattern , FTI_BUFS, "%s/sector%d-group*.fti", stashGlobal, FTI_Topo.sectorID );
    if ( glob( pattern, GLOB_ERR|GLOB_NOCHECK, NULL, &glb_ctx ) != GLOB_NOMATCH ) { 
      int i=0; for(; i<glb_ctx.gl_pathc; i++) { 
        snprintf( fi, FTI_BUFS, "%s/%s", stashGlobal, basename(glb_ctx.gl_pathv[i]) );
        snprintf( fo, FTI_BUFS, "%s/%s", stashLocal, basename(glb_ctx.gl_pathv[i]) );
        if ( FTI_FileCopy( fi, fo ) != FTI_SCES ) error = 1;
      }
    }
  }
  
  if ( !error ) {
    if( FTI_Topo.amIaHead ) {
      int i=0; for(; i<FTI_Topo.nodeSize-FTI_Topo.nbHeads; i++) { 
        snprintf( pattern , FTI_BUFS, "%s/Ckpt*-Rank%d.fti", stashGlobal, FTI_Topo.body[i] );
        if ( glob( pattern, GLOB_ERR|GLOB_NOCHECK, NULL, &glb_ctx ) != GLOB_NOMATCH ) { 
          snprintf( fi, FTI_BUFS, "%s/%s", stashGlobal, basename(*glb_ctx.gl_pathv) );
          snprintf( fo, FTI_BUFS, "%s/%s", stashLocal, basename(*glb_ctx.gl_pathv) );
          if ( FTI_FileCopy( fi, fo ) != FTI_SCES ) {
            error = 1;
            break;
          }
        }
      }
    } else {
      snprintf( pattern , FTI_BUFS, "%s/Ckpt*-Rank%d.fti", stashGlobal, FTI_Topo.myRank);
      if ( glob( pattern, GLOB_ERR|GLOB_NOCHECK, NULL, &glb_ctx ) != GLOB_NOMATCH ) { 
        snprintf( fi, FTI_BUFS, "%s/%s", stashGlobal, basename(*glb_ctx.gl_pathv) );
        snprintf( fo, FTI_BUFS, "%s/%s", stashLocal, basename(*glb_ctx.gl_pathv) );
        if ( FTI_FileCopy( fi, fo ) != FTI_SCES ) error = 1;
      }
    }
  }

  MPI_Allreduce( &error, &errorAll, 1, MPI_INT, MPI_SUM, FTI_COMM_WORLD );

  if ( !errorAll ) {
    snprintf( info, FTI_BUFS, "Succsessfully pulled stashed checkpoint (stashId: %lu)", stashId );
    FTI_Print( info, FTI_INFO );
  } else {
    snprintf( err, FTI_BUFS, "failed to pull stash with id '%lu'", stashId ); 
    FTI_Print( err, FTI_WARN );
  }
  
  return ( errorAll ) ? FTI_NSCS : FTI_SCES;

}

int FTIX_StashPush( uint64_t stashId ) {
  FTIX_PREREC(-1);

  char err[FTI_BUFS], info[FTI_BUFS];
  char stashLocal[FTI_BUFS], stashGlobal[FTI_BUFS];
  int retVal, error=0, exists = 0, errorAll, existsAll;

  if( stashId > INT64_MAX ) {
    FTI_Print("stash ID must not be larger than INT64_MAX", FTI_WARN);
    error = 1;
  }

  if ( !error ) {
    snprintf( stashLocal, FTI_BUFS, "%s/%lu", FTI_Conf.stashDir, stashId );
    retVal = FTI_CheckDirectory( stashLocal ); 
    if ( retVal == 0 ) {
      snprintf( err, FTI_BUFS, "stash with id '%lu' does not exist", stashId ); 
      FTI_Print( err, FTI_WARN );
      error = 1;
    } else if ( retVal == -1 ) {
      error = 1;
    }
  }

  if ( !error ) {
    snprintf( stashGlobal, FTI_BUFS, "%s/%lu", FTI_Conf.stashDirGlobal, stashId );
    retVal = FTI_CheckDirectory( stashGlobal ); 
    if ( retVal == 1 ) {
      snprintf( info, FTI_BUFS, "stash with id '%lu' already exists", stashId ); 
      FTI_Print( err, FTI_WARN );
      exists = 1;
    } else if ( retVal == -1 ) {
      error = 1;
    }
  }
  
  MPI_Allreduce( &error, &errorAll, 1, MPI_INT, MPI_SUM, FTI_COMM_WORLD ); 
  MPI_Allreduce( &exists, &existsAll, 1, MPI_INT, MPI_SUM, FTI_COMM_WORLD ); 
  if ( errorAll ) { 
    return FTI_NSCS;
  } else if ( existsAll ) { 
    return FTI_SCES;
  }

  if ( FTI_CreateDirectory( &FTI_Topo, stashGlobal, FTI_FS_GLOBAL ) != FTI_SCES ) {
    snprintf( err, FTI_BUFS, "failed to push stash with id '%lu'", stashId ); 
    FTI_Print( err, FTI_WARN );
    return FTI_NSCS;
  }
  
  MPI_Barrier(FTI_COMM_WORLD);
  
  error = 0;

  char fi[FTI_BUFS], fo[FTI_BUFS];
  
  glob_t glb_ctx;

  char pattern[FTI_BUFS];
  if ( FTI_Topo.groupRank == 0 ) {
    snprintf( pattern , FTI_BUFS, "%s/%s", stashLocal, "sector*-group*.fti");
    if ( glob( pattern, GLOB_ERR|GLOB_NOCHECK, NULL, &glb_ctx ) != GLOB_NOMATCH ) { 
      int i=0; for(; i<glb_ctx.gl_pathc; i++) { 
        snprintf( fi, FTI_BUFS, "%s/%s", stashLocal, basename(glb_ctx.gl_pathv[i]) );
        snprintf( fo, FTI_BUFS, "%s/%s", stashGlobal, basename(glb_ctx.gl_pathv[i]) );
        if ( FTI_FileCopy( fi, fo ) != FTI_SCES ) error = 1;
      }
    }
  }
  
  if ( !error ) {
    if( FTI_Topo.amIaHead ) {
      snprintf( pattern , FTI_BUFS, "%s/%s", stashLocal, "Ckpt*-Rank*.fti");
      if ( glob( pattern, GLOB_ERR|GLOB_NOCHECK, NULL, &glb_ctx ) != GLOB_NOMATCH ) { 
        int i=0; for(; i<glb_ctx.gl_pathc; i++) { 
          snprintf( fi, FTI_BUFS, "%s/%s", stashLocal, basename(glb_ctx.gl_pathv[i]) );
          snprintf( fo, FTI_BUFS, "%s/%s", stashGlobal, basename(glb_ctx.gl_pathv[i]) );
          if ( FTI_FileCopy( fi, fo ) != FTI_SCES ) {
            error = 1;
            break;
          }
        }
      }
    } else {
      snprintf( pattern , FTI_BUFS, "%s/Ckpt*-Rank%d.fti", stashLocal, FTI_Topo.myRank);
      if ( glob( pattern, GLOB_ERR|GLOB_NOCHECK, NULL, &glb_ctx ) != GLOB_NOMATCH ) { 
        snprintf( fi, FTI_BUFS, "%s/%s", stashLocal, basename(*glb_ctx.gl_pathv) );
        snprintf( fo, FTI_BUFS, "%s/%s", stashGlobal, basename(*glb_ctx.gl_pathv) );
        if ( FTI_FileCopy( fi, fo ) != FTI_SCES ) error = 1;
      }
    }
  }

  MPI_Allreduce( &error, &errorAll, 1, MPI_INT, MPI_SUM, FTI_COMM_WORLD );

  if ( !errorAll ) {
    snprintf( info, FTI_BUFS, "Succsessfully pushed stashed checkpoint (stashId: %lu)", stashId );
    FTI_Print( info, FTI_INFO );
  } else {
    snprintf( err, FTI_BUFS, "failed to push stash with id '%lu'", stashId ); 
    FTI_Print( err, FTI_WARN );
  }
  
  return ( errorAll ) ? FTI_NSCS : FTI_SCES;

}

int64_t FTIX_Stash( int ckptId, uint64_t stashId ) {
  FTIX_PREREC(-1);
 
  int allRes, success = 0, failure = 1;

  if( stashId > INT64_MAX ) {
    FTI_Print("stash ID must not be larger than INT64_MAX", FTI_WARN);
    MPI_Allreduce( &failure, &allRes, 1, MPI_INT, MPI_SUM, FTI_COMM_WORLD ); 
    return FTI_NSCS; 
  }

  char dir[FTI_BUFS];
  snprintf( dir, FTI_BUFS, "%s/%lu", FTI_Conf.stashDir, stashId );
  if ( FTI_CreateDirectory( &FTI_Topo, dir, FTI_FS_LOCAL ) != FTI_SCES ) {
    FTI_Print("failed to stash checkpoint", FTI_WARN);
    MPI_Allreduce( &failure, &allRes, 1, MPI_INT, MPI_SUM, FTI_COMM_WORLD ); 
    return FTI_NSCS; 
  }

  int proclist[FTI_BUFS] = { FTI_Topo.myRank };
  int groupID[FTI_BUFS] = { FTI_Topo.groupID };
  
  if( FTI_Topo.amIaHead ) {
    memcpy( proclist, FTI_Topo.body, (FTI_Topo.nodeSize - FTI_Topo.nbHeads) * sizeof(int));
    int i=1; for(; i<FTI_Topo.nodeSize; i++) groupID[i-1] = i;
  }

  char fi[FTI_BUFS], fo[FTI_BUFS];
  int i=0, end = (FTI_Topo.amIaHead) ? (FTI_Topo.nodeSize - FTI_Topo.nbHeads) : 1;
  for(; i<end; i++) {
    snprintf( fi, FTI_BUFS, "%s/Ckpt%d-Rank%d.%s",
        FTI_Ckpt[1].dir, ckptId, proclist[i], FTI_Conf.suffix );
    snprintf( fo, FTI_BUFS, "%s/Ckpt%d-Rank%d.%s",
        dir, ckptId, proclist[i], FTI_Conf.suffix );
    if ( FTI_FileCopy( fi, fo ) != FTI_SCES ) {
      FTI_Print("failed to stash checkpoint", FTI_WARN);
      MPI_Allreduce( &failure, &allRes, 1, MPI_INT, MPI_SUM, FTI_COMM_WORLD ); 
      return FTI_NSCS;
    }
    snprintf( fi, FTI_BUFS, "%s/sector%d-group%d.fti",
        FTI_Ckpt[1].metaDir, FTI_Topo.sectorID, groupID[i] );
    snprintf( fo, FTI_BUFS, "%s/sector%d-group%d.fti",
        dir, FTI_Topo.sectorID, groupID[i] );
    if ( FTI_FileCopy( fi, fo ) != FTI_SCES ) {
      FTI_Print("failed to stash metadata", FTI_WARN);
      MPI_Allreduce( &failure, &allRes, 1, MPI_INT, MPI_SUM, FTI_COMM_WORLD ); 
      return FTI_NSCS;
    }
  }

  MPI_Allreduce( &success, &allRes, 1, MPI_INT, MPI_SUM, FTI_COMM_WORLD );

  char info[FTI_BUFS];
  snprintf( info, FTI_BUFS, "Succsessfully stashed checkpoint (ckptId: %d, stashId: %lu)", ckptId, stashId );
  FTI_Print( info, FTI_INFO );
  
  return ( allRes == success ) ? stashId : FTI_NSCS ;
  
}

int FTIX_StashLoad( uint64_t stashId ) {
  FTIX_PREREC(-1);
  
  int allRes, success = 0, failure = 1;
  
  char ckptDir[FTI_BUFS];
  snprintf( ckptDir, FTI_BUFS, "%s/%lu", FTI_Conf.stashDir, stashId );

  char mfn[FTI_BUFS];
  char key[FTI_BUFS];
  snprintf(mfn, FTI_BUFS, "%s/sector%d-group%d.fti", ckptDir, FTI_Topo.sectorID, FTI_Topo.groupID);
  FTIT_iniparser ini;
  if (FTI_Iniparser(&ini, mfn, FTI_INI_OPEN) != FTI_SCES) {
      FTI_Print("Iniparser failed to parse the metadata file.", FTI_WARN);
      MPI_Allreduce( &failure, &allRes, 1, MPI_INT, MPI_SUM, FTI_COMM_WORLD ); 
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
        MPI_Allreduce( &failure, &allRes, 1, MPI_INT, MPI_SUM, FTI_COMM_WORLD ); 
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
      MPI_Allreduce( &failure, &allRes, 1, MPI_INT, MPI_SUM, FTI_COMM_WORLD ); 
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
    MPI_Allreduce( &failure, &allRes, 1, MPI_INT, MPI_SUM, FTI_COMM_WORLD ); 
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
        MPI_Allreduce( &failure, &allRes, 1, MPI_INT, MPI_SUM, FTI_COMM_WORLD ); 
        return FTI_NSCS;
      }
    }
    
    if ( bytes != dnode->size ) {
        FTI_Print("Wrong size", FTI_EROR);
        MPI_Allreduce( &failure, &allRes, 1, MPI_INT, MPI_SUM, FTI_COMM_WORLD ); 
        return FTI_NSCS;
    }
    dnode = dnode->next;
    free(dfree);
  } while ( dnode );
  
  fclose(fd);

  MPI_Allreduce( &success, &allRes, 1, MPI_INT, MPI_SUM, FTI_COMM_WORLD );

  char info[FTI_BUFS];
  snprintf( info, FTI_BUFS, "Succsessfully loaded stashed checkpoint (ckptId: %d, stashId: %lu)", ckptId, stashId );
  FTI_Print( info, FTI_INFO );
  
  return ( allRes == success ) ? ckptId : FTI_NSCS ;

}

int FTIX_StashDrop( uint64_t stashId, int where ) {
  FTIX_PREREC(-1);
   
  char ckptDir[FTI_BUFS];

  int nodeFlag = FTI_Topo.masterLocal;
  int globalFlag = FTI_Topo.masterGlobal;

  if ( where == FTI_FS_LOCAL ) {
    snprintf( ckptDir, FTI_BUFS, "%s/%lu", FTI_Conf.stashDir, stashId );
    FTI_RmDir( ckptDir, nodeFlag );
  } else if ( where == FTI_FS_GLOBAL ) {
    snprintf( ckptDir, FTI_BUFS, "%s/%lu", FTI_Conf.stashDirGlobal, stashId );
    FTI_RmDir( ckptDir, globalFlag );
  }

  MPI_Barrier(FTI_COMM_WORLD);

  return FTI_SCES;

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

bool FTIX_TopoGet_masterLocal(){ 
  FTIX_PREREC(-1);
  return FTI_Topo.masterLocal;
}

bool FTIX_TopoGet_masterGlobal(){ 
  FTIX_PREREC(-1);
  return FTI_Topo.masterGlobal;
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
