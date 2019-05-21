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

#endif // STAGE_TYPES_H
