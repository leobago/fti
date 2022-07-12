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

  int64_t FTIX_StashDump( int, uint64_t );
  int FTIX_StashLoad( uint64_t );
  int FTIX_StashDrop( uint64_t );
  
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
