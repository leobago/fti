#ifndef STAGE_TYPES_H
#define STAGE_TYPES_H

/** @typedef    FTIT_StageInfo
 *  @brief      Staging meta info.
 *  
 *  The request pointer is void in order to allow the structure to
 *  keep the head rank staging info if used by a head process or the
 *  application rank staging info otherwise. The cast is performed
 *  via the macros 'FTI_SI_HPTR( ptr )' for the head processes and
 *  'FTI_SI_APTR( ptr )' for the application processes.
 */
typedef struct FTIT_StageInfo {
    int nbRequest;  /**< Number of allocated request info structures        */
    void *request;  /**< pointer to request meta info array                 */
} FTIT_StageInfo;

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

#endif // STAGE_TYPES_H
