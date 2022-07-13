/**
 *  Copyright (c) 2017 Leonardo A. Bautista-Gomez
 *  All rights reserved
 *
 *  @file   fti-ext.h
 *  @author Kai Keller (kai.rasmus.keller@gmail.com)
 *  @date   July, 2022
 *  @brief  Header file for the FTI library extensions.
 */

#ifndef FTI_INCLUDE_FTI_EXT_H_
#define FTI_INCLUDE_FTI_EXT_H_

#include <stdbool.h>
#include <stdint.h>
#include <mpi.h>

#ifdef __cplusplus
extern "C" {
#endif
  
#ifdef FTIX_CALLBACK
  void FTIX_Callback(void);
  void (*__ftix_callback)(void) = FTIX_Callback;
#else
  void (*__ftix_callback)(void) = NULL;
#endif


  /*==================================================================*/
	/*  Stash mechanism                                                 */
	/*==================================================================*/
	


  /**--------------------------------------------------------------------------
    
    
    @brief        Copies checkpoint with 'ckptId' to local stash directory.
  
    @param        ckptId[in]   <b> int </b> ID of local (L1) checkpoint to copy
    @param        stashId[in]  <b> uint64_t </b> unique ID of the stashed item
    
    @return                     stashId on success.    
                                \ref FTI_NSCS upon failure.  
     

    Copies local (L1) checkpoint with 'ckptId' to the local stash folder. The 
    checkpoint remains unchanged. The stashed checkpoint will be stored to
    "<local_path>/stash/<stashId>".  
    

    The function can be called from both the heads and application processes.
  
  --------------------------------------------------------------------------**/
  int64_t FTIX_Stash( int ckptId, uint64_t stashId );
  
  /**--------------------------------------------------------------------------
    
    
    @brief        Loads the stashed chackpoint data to protected buffers.
  
    @param        stashId[in]  <b> uint64_t </b> unique ID of the stashed item
    
    @return                     ckptId upon success.  
                                \ref FTI_NSCS upon failure.  
      

    Loads the data from the stashed checkpoint with 'stashId' to the protected
    buffers. The ID of the stashed checkpoint is returned upon success. 
    

    No FTI meta-data will be changed.
  
  --------------------------------------------------------------------------**/
  int FTIX_StashLoad( uint64_t stashId );
  
  /**--------------------------------------------------------------------------
    
    
    @brief        Erases stashed item with 'stashId' from 'layer'
  
    @param        stashId[in]  <b> uint64_t </b> unique ID of the stashed item
    @param        layer[in]    <b> int </b> file-system layer. \ref FTI_FS_LOCAL
    if item locally and \ref FTI_FS_GLOBAL if globally.
    
    @return                     \ref FTI_SCES upon success.  
                                \ref FTI_NSCS upon failure.  
      

    Removes the stashed item from local or global file-system layer. If the
    item shall be removed locally, 'layer' must be \ref FTI_FS_LOCAL. If the
    item shall be removed globally, 'layer' must be \ref FTI_FS_GLOBAL.
      

    The function can be called from both the heads and application processes.

  --------------------------------------------------------------------------**/
  int FTIX_StashDrop( uint64_t stashId, int layer );
  
  /**--------------------------------------------------------------------------
    
    
    @brief        Copies the local stashed item to the global file-system layer 
  
    @param        stashId[in]  <b> uint64_t </b> unique ID of the stashed item
    
    @return                     \ref FTI_SCES upon success.  
                                \ref FTI_NSCS upon failure.  
      

    Copies stashed item with ID 'stashId' from the local stash directory to the 
    global stash directory.  
      
        
    The function can be called from both the heads and application processes.
  
  --------------------------------------------------------------------------**/
  int FTIX_StashPush( uint64_t stashId );
  
  /**--------------------------------------------------------------------------
    
    
    @brief        Copies the global stashed item to the local file-system layer   
  
    @param        stashId[in]  <b> uint64_t </b> unique ID of the stashed item  
    
    @return                     \ref FTI_SCES upon success.  
                                \ref FTI_NSCS upon failure.  
      

    Copies stashed item with ID 'stashId' from the global stash directory to the 
    local stash directory.  

      
    The function can be called from both the heads and application processes.  
  
  --------------------------------------------------------------------------**/
  int FTIX_StashPull( uint64_t stashId );
 

  /*==================================================================*/
	/*  Expose internal FTI information [TOPOLOGY]                      */
	/*==================================================================*/
	


  /**-------------------------------------------------------------------
    @brief Get total global number of processes
  -------------------------------------------------------------------**/
  int FTIX_TopoGet_nbProc();
  
  /**-------------------------------------------------------------------
    @brief Get Total global number of nodes
  -------------------------------------------------------------------**/
  int FTIX_TopoGet_nbNodes();
  
  /**-------------------------------------------------------------------
    @brief Get My rank on the global comm
  -------------------------------------------------------------------**/
  int FTIX_TopoGet_myRank();
  
  /**-------------------------------------------------------------------
    @brief returns TRUE if rank is master in FTI_COMM_WORLD
  -------------------------------------------------------------------**/
  bool FTIX_TopoGet_masterGlobal();
  
  /**-------------------------------------------------------------------
    @brief returns TRUE if rank is master in node for FTI_COMM_WORLD
  -------------------------------------------------------------------**/
  bool FTIX_TopoGet_masterLocal();
  
  /**-------------------------------------------------------------------
    @brief Get My rank on the FTI comm
  -------------------------------------------------------------------**/
  int FTIX_TopoGet_splitRank();
  
  /**-------------------------------------------------------------------
    @brief Get Total number of pro. per node
  -------------------------------------------------------------------**/
  int FTIX_TopoGet_nodeSize();
  
  /**-------------------------------------------------------------------
    @brief Get Number of FTI proc. per node
  -------------------------------------------------------------------**/
  int FTIX_TopoGet_nbHeads();
  
  /**-------------------------------------------------------------------
    @brief Get Number of app. proc. per node
  -------------------------------------------------------------------**/
  int FTIX_TopoGet_nbApprocs();
  
  /**-------------------------------------------------------------------
    @brief Get Group size for L2 and L3
  -------------------------------------------------------------------**/
  int FTIX_TopoGet_groupSize();
  
  /**-------------------------------------------------------------------
    @brief Get Sector ID in the system
  -------------------------------------------------------------------**/
  int FTIX_TopoGet_sectorID();
  
  /**-------------------------------------------------------------------
    @brief Get Node ID in the system
  -------------------------------------------------------------------**/
  int FTIX_TopoGet_nodeID();
  
  /**-------------------------------------------------------------------
    @brief Get Group ID in the node
  -------------------------------------------------------------------**/
  int FTIX_TopoGet_groupID();
  
  /**-------------------------------------------------------------------
    @brief Returns TRUE if FTI process
  -------------------------------------------------------------------**/
  bool FTIX_TopoGet_amIaHead();
  
  /**-------------------------------------------------------------------
    @brief Get Rank of the head in this node
  -------------------------------------------------------------------**/
  int FTIX_TopoGet_headRank();
  
  /**-------------------------------------------------------------------
    @brief Get Rank of the head in node comm
  -------------------------------------------------------------------**/
  int FTIX_TopoGet_headRankNode();
  
  /**-------------------------------------------------------------------
    @brief Get Rank of the node
  -------------------------------------------------------------------**/
  int FTIX_TopoGet_nodeRank();
  
  /**-------------------------------------------------------------------
    @brief Get My rank in the group comm
  -------------------------------------------------------------------**/
  int FTIX_TopoGet_groupRank();
  
  /**-------------------------------------------------------------------
    @brief Get Proc. on the right of the ring
  -------------------------------------------------------------------**/
  int FTIX_TopoGet_right();
  
  /**-------------------------------------------------------------------
    @brief Get Proc. on the left of the ring
  -------------------------------------------------------------------**/
  int FTIX_TopoGet_left();
  
  /**-------------------------------------------------------------------
    @brief Get List of app. proc. in the node
  -------------------------------------------------------------------**/
  int* FTIX_TopoGet_body( int* body, int* len, int );
	


	/*==================================================================*/
	/*  Expose internal FTI information [CONFIGURATION]                 */
	/*==================================================================*/
  


  /**-------------------------------------------------------------------
    @brief Get FTI execution id
  -------------------------------------------------------------------**/
  char* FTIX_ExecGet_id( char*, int* len, int );
  
  /**-------------------------------------------------------------------
    @brief Get global communicator
  -------------------------------------------------------------------**/
  MPI_Comm FTIX_ExecGet_globalComm();
 


	/*==================================================================*/
	/*  Expose internal FTI information [CONFIGURATION]                 */
	/*==================================================================*/
  


  /**-------------------------------------------------------------------
    @brief Get global communicator
  -------------------------------------------------------------------**/
  int FTIX_ConfGet_blockSize();
  
  /**-------------------------------------------------------------------
    @brief Get Transfer size local to PFS
  -------------------------------------------------------------------**/
  int FTIX_ConfGet_transferSize();
  
  /**-------------------------------------------------------------------
    @brief Get MPI tag for ckpt requests
  -------------------------------------------------------------------**/
  int FTIX_ConfGet_ckptTag();
  
  /**-------------------------------------------------------------------
    @brief Get MPI tag for staging comm
  -------------------------------------------------------------------**/
  int FTIX_ConfGet_stageTag();
  
  /**-------------------------------------------------------------------
    @brief Get MPI tag for finalize comm
  -------------------------------------------------------------------**/
  int FTIX_ConfGet_finalTag();
  
  /**-------------------------------------------------------------------
    @brief Get MPI tag for general comm
  -------------------------------------------------------------------**/
  int FTIX_ConfGet_generalTag();
  
  /**-------------------------------------------------------------------
    @brief Get Configuration file name
  -------------------------------------------------------------------**/
  char* FTIX_ConfGet_cfgFile( char*, int*, int );
  
  /**-------------------------------------------------------------------
    @brief Get Local directory
  -------------------------------------------------------------------**/
  char* FTIX_ConfGet_localDir( char*, int*, int );
  
  /**-------------------------------------------------------------------
    @brief Get Global directory
  -------------------------------------------------------------------**/
  char* FTIX_ConfGet_glbalDir( char*, int*, int );
  
  /**-------------------------------------------------------------------
    @brief Get Metadata directory
  -------------------------------------------------------------------**/
  char* FTIX_ConfGet_metadDir( char*, int*, int );
  
  /**-------------------------------------------------------------------
    @brief Get Local temporary directory
  -------------------------------------------------------------------**/
  char* FTIX_ConfGet_lTmpDir( char*, int*, int );
  
  /**-------------------------------------------------------------------
    @brief Get Global temporary directory
  -------------------------------------------------------------------**/
  char* FTIX_ConfGet_gTmpDir( char*, int*, int );
  
  /**-------------------------------------------------------------------
    @brief Get Metadata temporary directory
  -------------------------------------------------------------------**/
  char* FTIX_ConfGet_mTmpDir( char*, int*, int );
  
  /**-------------------------------------------------------------------
    @brief Get Suffix of the checkpoint files
  -------------------------------------------------------------------**/
  char* FTIX_ConfGet_suffix( char*, int*, int );


#ifdef __cplusplus
}
#endif

#endif  // FTI_INCLUDE_FTI_EXT_H_
