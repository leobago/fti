/** Copyright (c) 2017 Leonardo A. Bautista-Gomez All rights reserved
 *
 *  FTI - A multi-level checkpointing library for C/C++/Fortran
 *  applications
 *
 *  Revision 1.0 : Fault Tolerance Interface (FTI)
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *  1. Redistributions of source code must retain the above copyright
 *  notice, this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright
 *  notice, this list of conditions and the following disclaimer in the
 *  documentation and/or other materials provided with the distribution.
 *
 *  3. Neither the name of the copyright holder nor the names of its
 *  contributors may be used to endorse or promote products derived from
 *  this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *  @author Kai Keller (kellekai@gmx.de) 
 *  @file   stage.c 
 *  @date   July, 2018 
 *  @brief  header for stage.c
 */

#ifndef FTI_SRC_STAGE_H_
#define FTI_SRC_STAGE_H_

#include "interface.h"

/** Maximum amount of concurrent active staging requests                   
  @note leads to 2.5MB for the application processes as minimum memory
  allocated
 **/
#define FTI_SI_MAX_NUM (512L*1024L)

// 1 bit field
#define FTI_SI_NAVL 0x1
#define FTI_SI_IAVL 0x0

#define FTI_SI_IALL 0x1
#define FTI_SI_NALL 0x0

#define FTI_SI_APTR(ptr) ((FTIT_StageAppInfo*)ptr)
#define FTI_SI_HPTR(ptr) ((FTIT_StageHeadInfo*)ptr)

#define FTI_SI_MAX_ID (0x7ffff)

#define FTI_DISABLE_STAGING do {*enableStagingPtr = false;} while (0)
#define FTI_SI_ENABLED (*(bool*)enableStagingPtr)

/** @typedef    FTIT_StatusField
 *  @brief      valid fields of 'status'.
 * 
 *  enum that keeps the particular field identifiers for the 'status'
 *  field.
 */
typedef enum {
    FTI_SIF_AVL = 0,
    FTI_SIF_VAL,
} FTIT_StatusField;

/** @typedef    FTIT_RequestField
 *  @brief      valid fields of 'idxRequest'.
 * 
 *  enum that keeps the particular field identifiers for the
 *  'idxRequest' field.
 */
typedef enum {
    FTI_SIF_ALL = 0,
    FTI_SIF_IDX
} FTIT_RequestField;

/** @typedef    FTIT_StageHeadInfo
 *  @brief      Head rank staging meta info.
 */
typedef struct FTIT_StageHeadInfo {
    char lpath[FTI_BUFS];           /**< file path                      */
    char rpath[FTI_BUFS];           /**< file name                      */
    size_t offset;                  /**< current offset of file pointer */
    size_t size;                    /**< file size                      */
    int ID;                         /**< ID of request                  */
} FTIT_StageHeadInfo;

/** @typedef    FTIT_StageAppInfo
 *  @brief      Application rank staging meta info.
 */
typedef struct FTIT_StageAppInfo {
    void *sendBuf;                  /**< send buffer of MPI_Isend       */
    MPI_Request mpiReq;             /**< MPI_Request of MPI_Isend       */
    int ID;                         /**< ID of request                  */
} FTIT_StageAppInfo;


int FTI_GetRequestID(FTIT_execution *FTI_Exec, FTIT_topology *FTI_Topo);
int FTI_InitStage(FTIT_execution *FTI_Exec, FTIT_configuration *FTI_Conf,
        FTIT_topology *FTI_Topo);
int FTI_InitStageRequestApp(FTIT_execution *FTI_Exec, FTIT_topology *FTI_Topo,
        uint32_t ID);
int FTI_AsyncStage(const char *lpath, const char *rpath, FTIT_configuration *FTI_Conf,
        FTIT_execution *FTI_Exec, FTIT_topology *FTI_Topo, int ID);
int FTI_InitStageRequestHead(char* lpath, char *rpath, FTIT_execution *FTI_Exec,
        FTIT_topology *FTI_Topo, int source, uint32_t ID);
int FTI_SyncStage(const char* lpath, const char *rpath, FTIT_execution *FTI_Exec,
        FTIT_topology *FTI_Topo, FTIT_configuration *FTI_Conf, uint32_t ID);
int FTI_HandleStageRequest(FTIT_configuration* FTI_Conf,
        FTIT_execution* FTI_Exec, FTIT_topology* FTI_Topo,
        FTIT_checkpoint* FTI_Ckpt, int source);
int FTI_GetStatusField(FTIT_execution *FTI_Exec, FTIT_topology *FTI_Topo,
 int ID, FTIT_StatusField val, int source);
int FTI_SetStatusField(FTIT_execution *FTI_Exec, FTIT_topology *FTI_Topo,
 int ID, uint8_t entry, FTIT_StatusField val, int source);
int FTI_GetRequestField(int ID, FTIT_RequestField val);
int FTI_SetRequestField(int ID, uint32_t entry, FTIT_RequestField val);
int FTI_FreeStageRequest(FTIT_execution *FTI_Exec, FTIT_topology *FTI_Topo,
 int ID, int source);
void FTI_PrintStageStatus(FTIT_execution *FTI_Exec, FTIT_topology *FTI_Topo,
 int ID, int source);
int FTI_GetRequestIdx(int ID);
void FTI_FinalizeStage(FTIT_execution *FTI_Exec, FTIT_topology *FTI_Topo,
 FTIT_configuration *FTI_Conf);

#endif  // FTI_SRC_STAGE_H_
