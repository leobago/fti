#ifndef ICP_TYPES_H
#define ICP_TYPES_H

#include <fti-int/defs.h>

/** @typedef    FTIT_iCPInfo
 *  @brief      Meta Information needed for iCP.
 *  
 *  The member fh is a generic file handle container large enough to hold any
 *  file handle type of I/O modes that are used within FTI.
 */
typedef struct FTIT_iCPInfo {
    bool isFirstCp;             /**< TRUE if first cp in run                */
    short status;               /**< holds status (active,failed) of iCP    */
    int  result;                /**< holds result of I/O specific write     */
    int lastCkptLvel;           /**< holds last successful cp level         */
    int lastCkptID;             /**< holds last successful cp ID            */
    int countVar;               /**< counts datasets written                */
    int isWritten[FTI_BUFS];    /**< holds IDs of datasets in cp file       */
    double t0;                  /**< timing for CP statistics               */
    double t1;                  /**< timing for CP statistics               */
    char fh[FTI_ICP_FH_SIZE];   /**< generic fh container                   */
    char fn[FTI_BUFS];          /**< Name of the checkpoint file            */
    unsigned long long offset;  /**< file offset (for MPI-IO only)          */
} FTIT_iCPInfo;


#endif // ICP_TYPES_H
