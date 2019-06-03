#ifndef __KERNEL_H__
#define __KERNEL_H__

#include "interface.h"

/** General configuration information used by FTI.                         */
extern FTIT_configuration FTI_Conf;

/** Checkpoint information for all levels of checkpoint.                   */
extern FTIT_checkpoint FTI_Ckpt[5];

/** Dynamic information for this execution.                                */
extern FTIT_execution FTI_Exec;

/** Topology of the system.                                                */
extern FTIT_topology FTI_Topo;

/** Array of datasets and all their internal information.                  */
extern FTIT_dataset FTI_Data[FTI_BUFS];

/** SDC injection model and all the required information.                  */
extern FTIT_injection FTI_Inje;


#endif // __KERNEL_H__
