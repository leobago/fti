#ifndef FTI_DEFS_H
#define FTI_DEFS_H

/** Standard size of buffer and max node size.                             */
#define FTI_BUFS 256
/** Word size used during RS encoding.                                     */
#define FTI_WORD 16
/** Token returned when FTI performs a checkpoint.                         */
#define FTI_DONE 1
/** Token returned if a FTI function succeeds.                             */
#define FTI_SCES 0
/** Token returned if a FTI function fails.                                */
#define FTI_NSCS -1
/** Token returned if recovery fails.                                      */
#define FTI_NREC -2
/** Token that indicates a head process in user space                      */
#define FTI_HEAD 2

#endif // FTI_DEFS_H
