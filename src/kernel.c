
#include "interface.h"


/** General configuration information used by FTI.                         */
FTIT_configuration FTI_Conf;

/** Checkpoint information for all levels of checkpoint.                   */
FTIT_checkpoint FTI_Ckpt[5];

/** Dynamic information for this execution.                                */
FTIT_execution FTI_Exec;

/** Topology of the system.                                                */
FTIT_topology FTI_Topo;

/** Array of datasets and all their internal information.                  */
FTIT_dataset FTI_Data[FTI_BUFS];

/** SDC injection model and all the required information.                  */
FTIT_injection FTI_Inje;
