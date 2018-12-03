/**
 * @file kernel_interrupt.c
 * @brief Interface functions to protect long-running kernels.
 *
 * @author Max M. Baird (maxbaird.gy@gmail.com)
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <cuda_runtime.h>
#include <stdbool.h>
#include "fti.h"
#include "interface.h"
#include "api_cuda.h"

/**
 * @brief Maximum value quantum should be increased to.
 *
 * The quantum will be increased up to this limit if it has been
 * set to less than 5 seconds and a kernel launch does not progress
 * between interrupts. Not progressing means that the number of executed
 * blocks has not increased before the kernel was interrupted. When this
 * happens it means that the quantum expired before the kernel was able
 * to launch any new blocks which means the quantum needs to be increased.
 *
 * The MAX_QUANTUM is the ceiling to how much the quantum needs to be increased.
 * This is a VERY generous value as the launching of all blocks should take
 * less than 1 second.
 */
#define MAX_QUANTUM 5000000 /* 5 seconds */

/**
 * @brief              Initialized to reference to FTI_Topo.
 */
static FTIT_topology *FTI_Topo = NULL;

/**
 * @brief              Initialized to reference to FTI_Exec.
 */
static FTIT_execution *FTI_Exec = NULL;

/*
 * @brief List of references to protected kernel handles to be freed.
 *
 * This list is necessary so that allocated handles
 * can be freed during the finalization of FTI. While FTI_execution holds a
 * reference to protected kernels, the type of reference can change if the
 * application has been restarted. When initializing a kernel for protection,
 * memory for its handle is allocated and a reference to this handle is stored
 * in FTI_exec->kernelInfo. At restart, the reference is restored from
 * FTI_exec->meta[].kernelInfo which is a single allocation of size FTI_BUFS.
 * Because of the two different allocations that can be made to protect kernels,
 * two different frees need to be made. This list stores references to allocations
 * made when initializing protected kernels, so that they can be freed later.
*/
static FTIT_kernelInfo *kernelMacroHandle[FTI_BUFS];

/*
 * @brief Count of newly initialized kernels.
 *
 * This variable keeps track of the number of initialized kernels that
 * have not been restored. This is so that the correct number of frees
 * can be made on their handles if the application does not need to be
 * restarted again.
*/
static int protectedInitCount;

/**
 * @brief             Updates the number of executed blocks.
 * @param             FTI_KernelInfo kernel metadata.
 *
 * This function updates the previous count of executed blocks
 * as well as the current count. This function is called each time
 * the kernel is interrupted.
 */
static void update_executed_block_count(FTIT_kernelInfo *FTI_KernelInfo){
  size_t i = 0;
  size_t cnt = 0;

 /* Store total number of blocks executed at the last kernel interrupt */
 FTI_KernelInfo->lastExecutedBlockCnt = FTI_KernelInfo->executedBlockCnt;

 /* Count total blocks executed thus far by kernel */
  for(i = 0; i < *FTI_KernelInfo->block_amt; i++){
    if(FTI_KernelInfo->h_is_block_executed[i]){
      cnt++;
    }
  }

 FTI_KernelInfo->executedBlockCnt = cnt;
 //fprintf(stdout, "%d block count = %zu\n", FTI_Topo->myRank, FTI_KernelInfo->executedBlockCnt);
 //fflush(stdout);
}

/**
 * @brief              Recovers kernel information of protected kernels at restart.
 *
 * @return             FTI_SCES on completion
 */
static int recover_kernelInfo(){
  int i = 0;
  char str[FTI_BUFS];

  sprintf(str, "Restoring kernel info from level %d", FTI_Exec->ckptLvel);
  FTI_Print(str, FTI_DBUG);

  /* Iterate through all protected kernels to restore their data from the checkpoint level */
  for(i = 0; i < FTI_Exec->nbKernels; i++){
    FTI_Exec->kernelInfo[i] = &FTI_Exec->meta[FTI_Exec->ckptLvel].kernelInfo[i];
  }

  return FTI_SCES;
}

/**
 * @brief              Gets a reference to the topology and execution structures.
 * @param[in]          topo The current FTI_Topo object.
 * @param[in]          exec The current FTI_Exec object.
 *
 * @return             integer FTI_SCES if successful.
 */
int FTI_kernel_protect_init(FTIT_topology *topo, FTIT_execution *exec)
{
  FTI_Topo = topo;
  FTI_Exec = exec;

  if(FTI_Exec->reco && FTI_Exec->ckptLvel != 0){
    /* If this is a recovery, try to recover the kernel metadata */
    int res = FTI_Try(recover_kernelInfo(), "recover kernel info");

    if(res != FTI_SCES){
      return FTI_NSCS;
    }
  }

  return FTI_SCES;
}

/**
 * @brief              Converts seconds to microseconds.
 * @param[in]          quantum  Represents the time to convert in seconds
 * @return             The time in microseconds
 *
 * This function is used to convert the quantum, which is specified in seconds,
 * to microseconds. Microseconds are needed because usleep() is used to be able to
 * give the kernel shorter timeouts than 1 second (as would be the case if sleep() was used).
 */
static inline unsigned int seconds_to_microseconds(double quantum)
{
  return fabs(quantum) * 1000000.0;
}

/*
 * @brief             Determines if a kernel has already been protected.
 * @param[in]         kernelId    The ID of the kernel to check.
 * @param[in,out]     index       The index of the protected kernel (if found).
 * @return            bool        Boolean true value if kernel has been protected.
 */
static inline bool is_kernel_protected(int kernelId, unsigned int *index){
  unsigned int i = 0;
  bool kernel_protected = false;
  for(i = 0; i < FTI_Exec->nbKernels; i++){
    if(kernelId == *FTI_Exec->kernelInfo[i]->id){
      kernel_protected = true;
      break;
    }
  }
  *index = i;
  return kernel_protected;
}

/*
 * @brief             Determines if all protected kernels have finished.
 * @return            bool    Boolean value true if all finished; false otherwise.
*/
static inline bool all_kernels_complete(){
  int i = 0;

  for(i = 0; i < FTI_Exec->nbKernels; i++){
    if(*FTI_Exec->kernelInfo[i]->complete == false){
      return false;
    }
  }

  return true;
}

/**
 * @brief              Initializes protection for a kernel.
 * @param              kernelMacroInfo   Initialized with values required by kernel launch macro.
 * @param              kernelId          The ID of the kernel to protect.
 * @param              quantum           How long the kernel should be executed before interruption.
 * @param              num_blocks        The number of blocks launched by the kernel.
 * @return             integer           FTI_SCES if successful.
 *
 * This function initializes the KernelMacroInfo parameter and saves a reference
 * to the members of this parameter whose values need to be saved to the
 * metadata files at checkpoint time. In the event that the application needs to
 * be restarted, this function is called again. If the kernel has already been
 * protected, then the values for it to resume will be restored from
 * FTI_Exec->kernelInfo.
 */
int FTI_kernel_init(FTIT_kernelInfo** KernelMacroInfo, int kernelId, double quantum, dim3 num_blocks){
  //size_t i = 0;
  unsigned int kernel_index = 0;
  char str[FTI_BUFS];
  volatile bool *quantum_expired = NULL;
  FTIT_kernelInfo *kernelInfo = NULL;

  snprintf(str, FTI_BUFS, "Entered function %s", __func__);
  FTI_Print(str, FTI_DBUG);

  bool kernel_protected = is_kernel_protected(kernelId, &kernel_index);

  CUDA_ERROR_CHECK(cudaHostAlloc((void **)&quantum_expired, sizeof(volatile bool), cudaHostAllocMapped));

  if(!kernel_protected){
    snprintf(str, FTI_BUFS, "kernelId %d not protected.", kernelId);
    FTI_Print(str, FTI_DBUG);

    kernelInfo = (FTIT_kernelInfo *)malloc(sizeof(FTIT_kernelInfo));

    if(kernelInfo == NULL){
      return FTI_NSCS;
    }

    if(FTI_Exec->nbKernels >= FTI_BUFS){
      FTI_Print("Unable to protect kernel. Too many kernels already registered.", FTI_WARN);
      return FTI_NSCS;
    }

    snprintf(str, FTI_BUFS, "Allocating memory for kernelId %d", kernelId);
    FTI_Print(str, FTI_DBUG);
    kernelInfo->id                  = (int *)malloc(sizeof(int));
    kernelInfo->block_amt           = (size_t *)malloc(sizeof(size_t));
    kernelInfo->all_done            = (bool *)calloc(FTI_Topo->nbApprocs * FTI_Topo->nbNodes, sizeof(bool));
    kernelInfo->complete            = (bool *)malloc(sizeof(bool));
    kernelInfo->quantum             = (unsigned int*)malloc(sizeof(unsigned int));
    kernelInfo->quantum_expired     = quantum_expired;

    if(kernelInfo->id                  == NULL){return FTI_NSCS;}
    if(kernelInfo->all_done            == NULL){return FTI_NSCS;}
    if(kernelInfo->complete            == NULL){return FTI_NSCS;}
    if(kernelInfo->quantum             == NULL){return FTI_NSCS;}

    snprintf(str, FTI_BUFS, "Successfully allocated memory for kernelId %d", kernelId);
    FTI_Print(str, FTI_DBUG);

    *kernelInfo->id                 = kernelId;
    *kernelInfo->complete           = false;
    *kernelInfo->quantum            = seconds_to_microseconds(quantum);
    *kernelInfo->quantum_expired    = false;
    *kernelInfo->block_amt          = num_blocks.x * num_blocks.y * num_blocks.z;
    kernelInfo->block_info_bytes    = *kernelInfo->block_amt * sizeof(bool);

    snprintf(str, FTI_BUFS, "Allocating memory for block_info_bytes for kernelId %d", kernelId);
    FTI_Print(str, FTI_DBUG);

    kernelInfo->h_is_block_executed = (bool *)calloc(kernelInfo->block_info_bytes, sizeof(bool));
    if(kernelInfo->h_is_block_executed  == NULL){return FTI_NSCS;}

    snprintf(str, FTI_BUFS, "Successfully allocated block_info_bytes for kernelId %d", kernelId);
    FTI_Print(str, FTI_DBUG);

    /* Keep track of KernelMacroInfo for checkpointing */
    FTI_Exec->kernelInfo[FTI_Exec->nbKernels] = kernelInfo;
    /* Increase index of protected kernel */
    FTI_Exec->nbKernels = FTI_Exec->nbKernels + 1;

    /* Store handle so that it can be freed later */
    kernelMacroHandle[FTI_Exec->nbKernels] = kernelInfo;
    /* Increase count of initialized kernels */
    protectedInitCount = protectedInitCount + 1;
  }
  else{
    /* Restore after restart */
    snprintf(str, FTI_BUFS, "Restoring kernel execution data for kernelId %d", kernelId);
    FTI_Print(str, FTI_DBUG);

    //FTI_Exec->kernelInfo[kernel_index]->quantum_expired = kernelInfo->quantum_expired;
    kernelInfo = FTI_Exec->kernelInfo[kernel_index];

    /* Not checkpointed, so block_info_bytes needs to be recalculated */
    kernelInfo->block_info_bytes = *kernelInfo->block_amt * sizeof(bool);

    kernelInfo->quantum_expired = quantum_expired;
    *kernelInfo->quantum_expired = true;

    bool all_protected_kernels_complete = all_kernels_complete();

    if(all_protected_kernels_complete){
      fprintf(stdout, "All other kernels complete kernel %d executing again\n", kernelId);
      fflush(stdout);
      /* Kernel needs to be executed again */
      snprintf(str, FTI_BUFS, "All other kernels complete. kernel Id %d will be re-executed", kernelId);
      FTI_Print(str, FTI_DBUG);

      /* Reset necessary values so kernel is re-executed */
      *kernelInfo->complete = false;
      *kernelInfo->quantum_expired = false;
      *kernelInfo->quantum = seconds_to_microseconds(quantum);

      /* Reset array indicating that this kernel has been completed by all processes */
      memset(kernelInfo->all_done, 0, FTI_Topo->nbApprocs * FTI_Topo->nbNodes);

      /* Reset array keeping track of executed blocks */
      memset(kernelInfo->h_is_block_executed, 0, *kernelInfo->block_amt);
    }
  }

  kernelInfo->executedBlockCnt = 0;
  kernelInfo->lastExecutedBlockCnt = 0;

  update_executed_block_count(kernelInfo); //Update the count of executed blocks

  snprintf(str, FTI_BUFS, "Kernel Id %d allocating memory on GPU", kernelId);
  FTI_Print(str, FTI_DBUG);
  CUDA_ERROR_CHECK(cudaMalloc((void **)&kernelInfo->d_is_block_executed, kernelInfo->block_info_bytes));
  CUDA_ERROR_CHECK(cudaMemcpy(kernelInfo->d_is_block_executed, kernelInfo->h_is_block_executed, kernelInfo->block_info_bytes, cudaMemcpyHostToDevice));

  /* Every kernel being initialized has its initial quantum reset */
  kernelInfo->initial_quantum  = seconds_to_microseconds(quantum);

  *KernelMacroInfo = kernelInfo;

  return FTI_SCES;
}

/*
 * @brief              Frees allocations made on the GPU to protect kernel.
 * @param[in]          FTI_KernelInfo  kernel metadata.
 * @return             integer FTI_SCES if successful.
 */
int FTI_FreeDeviceAlloc(FTIT_kernelInfo* FTI_KernelInfo){
  char str[FTI_BUFS];
  snprintf(str, FTI_BUFS, "Freeing device allocations made for kernel %d", *FTI_KernelInfo->id);
  FTI_Print(str, FTI_DBUG);

  CUDA_ERROR_CHECK(cudaFree(FTI_KernelInfo->d_is_block_executed));
  CUDA_ERROR_CHECK(cudaFreeHost((void *)FTI_KernelInfo->quantum_expired));

  return FTI_SCES;
}

/**
 * @brief              Called in FTI_Finalize to release GPU resources.
 * @return             integer FTI_SCES if successful.
 *
 * Frees GPU resources for each kernel that was protected.
 */
int FTI_FreeKernelInfo()
{
  FTI_Print("Freeing memory used to protect kernel", FTI_DBUG);
  int i = 0;
  for(i = 0; i < FTI_Exec->nbKernels; i++){
     free(FTI_Exec->kernelInfo[i]->id);
     free(FTI_Exec->kernelInfo[i]->block_amt);
     free(FTI_Exec->kernelInfo[i]->all_done);
     free(FTI_Exec->kernelInfo[i]->h_is_block_executed);
     free(FTI_Exec->kernelInfo[i]->complete);
     free(FTI_Exec->kernelInfo[i]->quantum);
  }

  /* Free handles for kernels which were not restored from a checkpoint */
  for(i = 0; i < protectedInitCount; i++){
    free(kernelMacroHandle[i]);
  }

  return FTI_SCES;
}

/**
 * @brief              Determines if all processes have executed their kernels.
 * @param[in]          FTIT_kernelInfo Metadata of currently executing kernel.
 * @return             bool True or False whether all processes are complete or not.
 */
bool FTI_all_procs_complete(FTIT_kernelInfo* FTI_KernelInfo)
{
  FTI_Print("Checking if all procs complete", FTI_DBUG);

  int i = 0;

  /* Iterate over all application processes at each node */

  for(i = 0; i < FTI_Topo->nbApprocs * FTI_Topo->nbNodes; i++){
    /* Check if each process is finished with the current kernel */
    if(!FTI_KernelInfo->all_done[i]){
      return false;
    }
  }
  return true;
}

/**
 * @brief           Waits for quantum to expire.
 * @param[in]       FTI_KernelInfo Metadata of currently executing protected kernel.
 *
 * Used when initially waiting for the kernel to expire.
 * If the specified quantum is in minutes this function loops to perform a
 * cudaEventQuery every second; this is to handle the case of the kernel
 * returning before the quantum has expired so that no long period of time is
 * spent idly waiting.
 */
static inline void wait(FTIT_kernelInfo* FTI_KernelInfo)
{
  FTI_Print("In kernel wait function", FTI_DBUG);
  char str[FTI_BUFS];
  int q = *FTI_KernelInfo->quantum / 1000000.0f; /* Convert to seconds */

  /* If quantum is in seconds or minutes check kernel every second */
  if(q != 0)
  {
    sprintf(str, "Sleeping for : %d seconds\n", q);
    FTI_Print(str, FTI_DBUG);

    cudaEvent_t event;
    cudaEventCreate(&event);
    cudaEventRecord(event, 0);
    cudaError_t err;
    while(q)
    {
      err = cudaEventQuery(event);
      sprintf(str, "Event Query: %s", cudaGetErrorString(err));
      FTI_Print(str, FTI_DBUG);
      if(err == cudaSuccess)
      {
        break;
      }
      sleep(1); //TODO maybe let this sleep longer??
      q--;
    }
    cudaEventDestroy(event);
  }
  else
  {
    sprintf(str, "Sleeping for : %u microseconds\n", *FTI_KernelInfo->quantum);
    FTI_Print(str, FTI_DBUG);
    usleep(*FTI_KernelInfo->quantum);
  }
}

/**
 * @brief             Signals the kernel to return and then waits for it to return.
 * @param             FTI_KernelInfo   Metadata of currently executing protected kernel.
 * @return            integer          FTI_SCES if successful.
 *
 * Called in #FTI_kernel_monitor() to tell the kernel it should finish the currently
 * executing blocks and prevent new blocks from continuing on to their main
 * body of work.
 */
static inline int signal_kernel(FTIT_kernelInfo* FTI_KernelInfo)
{
  char str[FTI_BUFS];

  FTI_Print("Signalling kernel to return...", FTI_DBUG);
  *FTI_KernelInfo->quantum_expired = true;
  FTI_Print("Attempting to snapshot", FTI_DBUG);

  FTI_Print("Waiting on kernel...", FTI_DBUG);
  CUDA_ERROR_CHECK(cudaDeviceSynchronize());
  FTI_Print("Kernel came back...", FTI_DBUG);

  /*
    NOTE 1:
    FTI_Snapshot *MUST* be called by every process. It cannot be conditionally
    executed. This is because FTI_Snapshot performs an MPI collective operation
    which will cause the application to hang if some processes execute it and
    others don't.

    NOTE 2:
    FTI_Snapshot was specifically placed here to perform a snapshot *BEFORE* the
    block execution metadata is transferred from the device. This handles the case
    of sequentially executing kernels where a fault can occur exactly between
    kernel executions after the snapshot has been taken. If the snapshot included
    the kernel's block execution metadata, the complete status of the kernel would
    be captured here. Therefore, at restart, the entire kernel would once again be executed.
    By capturing the block execution data of the kernel after the snapshot, if a failure
    was to occur only the tail end of the kernel would be re-executed instead of the
    entire kernel.
  */
  int res = FTI_Snapshot();

  if(res == FTI_DONE)
  {
    FTI_Print("Successfully wrote snapshot at kernel interrupt", FTI_DBUG);
  }
  else
  {
    FTI_Print("No snapshot was taken", FTI_DBUG);
  }

  snprintf(str, FTI_BUFS, "Kernel id %d copying block information from GPU.", *FTI_KernelInfo->id);
  FTI_Print(str, FTI_DBUG);
  CUDA_ERROR_CHECK(cudaMemcpy(FTI_KernelInfo->h_is_block_executed, FTI_KernelInfo->d_is_block_executed, FTI_KernelInfo->block_info_bytes, cudaMemcpyDeviceToHost));

  update_executed_block_count(FTI_KernelInfo);

  *FTI_KernelInfo->complete = *FTI_KernelInfo->block_amt == FTI_KernelInfo->executedBlockCnt;
  sprintf(str, "Kernel id %d complete: %s", *FTI_KernelInfo->id, *FTI_KernelInfo->complete ? "True" : "False");
  FTI_Print(str, FTI_DBUG);

  return FTI_SCES;
}

/**
 * @brief            Increases quantum if necessary.
 * @param            FTI_KernelInfo     Metadata of currently protected kernel.
 *
 * This function will increase the quantum by comparing previous number of executed
 * blocks with the current number of executed blocks. If the current number of blocks
 * executed by the kernel thus far is less than or equal to the previous total number
 * of blocks, the quantum is increased.
 *
 * Each time a protected kernel is launched it will attempt to execute all blocks. As
 * the kernel progresses, more and more blocks will be marked as complete. However,
 * the kernel will still spend time launching already completed blocks each time it is
 * invoked. This takes time and if the quantum is very very small, it can expire before
 * any unexecuted blocks are scheduled. This function handles the case of a tiny quantum.
 */
static inline void increase_quantum(FTIT_kernelInfo* FTI_KernelInfo){
  if(FTI_KernelInfo->executedBlockCnt <= FTI_KernelInfo->lastExecutedBlockCnt &&
     *FTI_KernelInfo->quantum < MAX_QUANTUM){
    *FTI_KernelInfo->quantum = *FTI_KernelInfo->quantum + FTI_KernelInfo->initial_quantum;
  }
}

/**
 * @brief            Executed when control is returned to host after kernel is interrupted or finished.
 * @parm[in]         FTI_KernelInfo Metadata of currently executing protected kernel.
 *
 * This function handles the control returned to the host after the kernel has finished executing. It
 * also executes any callback function to be executed at kernel interrupt. The callback function is
 * only executed if the kernel is not finished. This function also increases the quantum by a default
 * or specified amount. This increase is necessary in cases where the quantum is short enough to
 * cause the kernel to return again by the time it has finished re-launching previously executed blocks.
 */
static inline void handle_kernel_suspension(FTIT_kernelInfo* FTI_KernelInfo)
{
  char str[FTI_BUFS];
  if(*FTI_KernelInfo->complete == false)
  {
    snprintf(str, FTI_BUFS, "Kernel id %d incomplete, resuming", *FTI_KernelInfo->id);
    FTI_Print(str, FTI_DBUG);

    /* Kernel not complete, so reset the quantum_expired boolean to false */
    *FTI_KernelInfo->quantum_expired = false;

    /*
      If the quantum is very very short (0.00001 seconds), the kernel may
      exhaust the quantum before getting to blocks that have not been executed. For
      this reason, the quantum is increased by the initial amount specified at each
      interrupt.  This should not be necessary for a quantum in minutes and should be
      removed.
    */
    increase_quantum(FTI_KernelInfo);
  }
}

/**
 * @brief              Monitors the kernel's execution until it completes.
 * @param[in,out]      FTI_KernelInfo Metadata of currently executing protected kernel.
 * @return             integer FTI_SCES if successful.
 *
 * This function monitors the execution of a protected kernel by using FTI_KernelInfo
 * as a handle to the protected kernel.
 */
int FTI_kernel_monitor(FTIT_kernelInfo* FTI_KernelInfo)
{
  FTI_Print("Monitoring kernel execution", FTI_DBUG);
  int ret;

  /* Wait for quantum to expire */
  wait(FTI_KernelInfo);

  /* Tell kernel to finish and return */
  ret = FTI_Try(signal_kernel(FTI_KernelInfo), "signal and wait on kernel");

  if(ret != FTI_SCES)
  {
    FTI_Print("Failed to signal and wait on kernel", FTI_EROR);
    return FTI_NSCS;
  }

  /* Handle interrupted kernel */
  handle_kernel_suspension(FTI_KernelInfo);

  return FTI_SCES;
}

/**
 * @brief              Gathers the complete status of current kernel.
 * @param              FTI_KernelInfo  Metadata about currently executing protected kernel.
 * @return             integer         FTI_SCES if successful.
 *
 * FTI needs to synchronize all processes at checkpoint time. Each kernel makes
 * a call to FTI_Snapshot at interrupt time and if it is time to checkpoint, all
 * processes need to synchronize. This function is needed to ensure that no
 * process finishes its kernel early and exits the kernel launch loop. If some
 * processes finish their kernels before others, the remaining kernel executions
 * may trigger a checkpoint that will cause the application to hang. So to avoid
 * this, the kernel launch loop is only terminated when all processes have
 * finished executing their kernels. This function gathers the complete status of
 * the currently executing kernel at each kernel interrupt.
 */
int FTI_BACKUP_gather_kernel_status(FTIT_kernelInfo* FTI_KernelInfo){
  MPI_Allgather(FTI_KernelInfo->complete, 1, MPI_C_BOOL, FTI_KernelInfo->all_done, 1, MPI_C_BOOL, FTI_COMM_WORLD);
  return FTI_SCES;
}

/**
 * @brief              A wrapper around FTI_Print.
 * @param              msg        The text to print.
 * @param              priority   The priority of msg.
 *
 * Used so that kernel interrupt macros can have access
 * to the standard FTI_Print() function.
 */
void FTI_BACKUP_Print(char *msg, int priority)
{
  FTI_Print(msg, priority);
}
