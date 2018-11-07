/**
 * @file kernel_interrupt.c
 * @brief Interface functions for the library.
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
 * @brief              Initialized to reference to FTI_Topo.
 */
static FTIT_topology *FTI_Topo = NULL;

/**
 * @brief              Initialized to reference to FTI_Exec.
 */
static FTIT_execution *FTI_Exec = NULL;

/**
 * @brief              Determines if kernel is complete by checking #h_is_block_executed.
 * @param[in,out]      complete   Indicates whether kernel is complete or not.
 *
 * Sets *complete* to 1 or 0 respectively if the kernel is finished or not.
 */
static void computation_complete(FTIT_gpuInfo* FTI_GpuInfo)
{
  size_t i = 0;
  bool complete = true;
  for(i = 0; i < *FTI_GpuInfo->block_amt; i++)
  {
    if(!FTI_GpuInfo->h_is_block_executed[i])
    {
      complete = false; 
      break;
    }
  }
  *FTI_GpuInfo->complete = complete;
}

/**
 * @brief              Gets a reference to the topology and execution structures.
 * @param[in]          topo The current FTI_Topo object.
 * @param[in]          exec The current FTI_Exec object.
 *
 * @return             FTI_SCES on completion
 */
int FTI_gpu_protect_init(FTIT_topology *topo, FTIT_execution *exec)
{
  FTI_Topo = topo;
  FTI_Exec = exec;
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

static inline bool is_kernel_protected(int kernelId, unsigned int *index){
  unsigned int i = 0;
  bool kernel_protected = false;
  for(i = 0; i < FTI_Exec->nbKernels; i++){
    if(kernelId == *FTI_Exec->gpuInfo[i].id){
      kernel_protected = true;
      break;
    }
  }
  *index = i;
  return kernel_protected;
}

/**
 * @brief              Initializes protection for a kernel.
 * @param              GpuMacroInfo   Initialized with values required by kernel launch macro.
 * @param              kernelId       The ID of the kernel to protect.
 * @param              quantum        How long the kernel should be executed before interruption.
 * @param              num_blocks     The number of blocks launched by the kernel.
 * @return             integer        FTI_SCES if successful.
 *
 * This function does initialization for the kernel launch macro and
 * initializes a handle of type FTIT_kernelProtectHandle. The macro's values
 * are returned via the GpuMacroInfo parameter. The internal handle is specific
 * to the kernel ID and is used to initialize FTI_GpuInfo within FTI_Exec. This
 * is so that the kernel's information can be captured at checkpoint time.
 */
int FTI_BACKUP_init(FTIT_gpuInfo* GpuMacroInfo, int kernelId, double quantum, dim3 num_blocks){
  size_t i = 0;
  unsigned int kernel_index = 0;

  bool kernel_protected = is_kernel_protected(kernelId, &kernel_index);

  CUDA_ERROR_CHECK(cudaHostAlloc((void **)&GpuMacroInfo->quantum_expired, sizeof(volatile bool), cudaHostAllocMapped));

  if(!kernel_protected){
    if(FTI_Exec->nbKernels >= FTI_BUFS){
      FTI_Print("Unable to protect kernel. Too many kernels already registered.", FTI_WARN);
      return FTI_NSCS;
    }

    //TODO ensure all of this is freed
    GpuMacroInfo->id                  = (int *)malloc(sizeof(int));
    GpuMacroInfo->block_amt           = (size_t *)malloc(sizeof(size_t));
    GpuMacroInfo->all_done            = (bool *)malloc(sizeof(bool) * FTI_Topo->nbProc);
    GpuMacroInfo->complete            = (bool *)malloc(sizeof(bool));
    GpuMacroInfo->quantum             = (unsigned int*)malloc(sizeof(unsigned int));

    if(GpuMacroInfo->id                   == NULL){return FTI_NSCS;}
    if(GpuMacroInfo->all_done             == NULL){return FTI_NSCS;}
    if(GpuMacroInfo->complete             == NULL){return FTI_NSCS;}
    if(GpuMacroInfo->quantum              == NULL){return FTI_NSCS;}

    *GpuMacroInfo->id                 = kernelId;
    *GpuMacroInfo->complete           = false;
    *GpuMacroInfo->quantum            = seconds_to_microseconds(quantum);
    *GpuMacroInfo->quantum_expired    = false;
    *GpuMacroInfo->block_amt          = num_blocks.x * num_blocks.y * num_blocks.z;
    GpuMacroInfo->block_info_bytes    = *GpuMacroInfo->block_amt * sizeof(bool);

    GpuMacroInfo->h_is_block_executed = (bool *)malloc(GpuMacroInfo->block_info_bytes);
    if(GpuMacroInfo->h_is_block_executed  == NULL){return FTI_NSCS;}

    for(i = 0; i < FTI_Topo->nbProc; i++){
      GpuMacroInfo->all_done[i] = false;
    }

    for(i = 0; i < *GpuMacroInfo->block_amt; i++){
      GpuMacroInfo->h_is_block_executed[i] = false;
    }

    /* Save for checkpointing */
    FTI_Exec->gpuInfo[FTI_Exec->nbKernels].id                   = GpuMacroInfo->id;
    FTI_Exec->gpuInfo[FTI_Exec->nbKernels].complete             = GpuMacroInfo->complete;
    FTI_Exec->gpuInfo[FTI_Exec->nbKernels].block_amt            = GpuMacroInfo->block_amt;
    FTI_Exec->gpuInfo[FTI_Exec->nbKernels].all_done             = GpuMacroInfo->all_done;
    FTI_Exec->gpuInfo[FTI_Exec->nbKernels].h_is_block_executed  = GpuMacroInfo->h_is_block_executed;
    FTI_Exec->gpuInfo[FTI_Exec->nbKernels].quantum              = GpuMacroInfo->quantum;
    FTI_Exec->nbKernels                                         = FTI_Exec->nbKernels + 1;
  }
  else{
    /* Restore after restart */
    GpuMacroInfo->id                    =  FTI_Exec->gpuInfo[kernel_index].id;
    GpuMacroInfo->complete              =  FTI_Exec->gpuInfo[kernel_index].complete;
    GpuMacroInfo->all_done              =  FTI_Exec->gpuInfo[kernel_index].all_done;
    GpuMacroInfo->block_info_bytes      = *FTI_Exec->gpuInfo[kernel_index].block_amt * sizeof(bool);
    GpuMacroInfo->block_amt             =  FTI_Exec->gpuInfo[kernel_index].block_amt;
    GpuMacroInfo->h_is_block_executed   =  FTI_Exec->gpuInfo[kernel_index].h_is_block_executed;
    GpuMacroInfo->quantum               =  FTI_Exec->gpuInfo[kernel_index].quantum;
    *GpuMacroInfo->quantum_expired      =  true;
  }

  //TODO check if removing the cast to void** affects anything
  CUDA_ERROR_CHECK(cudaMalloc((void **)&GpuMacroInfo->d_is_block_executed, GpuMacroInfo->block_info_bytes));
  CUDA_ERROR_CHECK(cudaMemcpy(GpuMacroInfo->d_is_block_executed, GpuMacroInfo->h_is_block_executed, GpuMacroInfo->block_info_bytes, cudaMemcpyHostToDevice));

  GpuMacroInfo->initial_quantum     = seconds_to_microseconds(quantum);

  return FTI_SCES;
}

/**
 * @brief              Calls #cleanup().
 * @param[in]          kernel_name    The name of the kernel.
 *
 * An interface function to clean up allocated resources and print
 * how often *kernel_name* was suspended.
 */
int FTI_FreeGpuInfo()
{
  FTI_Print("Freeing memory used for GPU Info", FTI_DBUG);
  int i = 0;
  for(i = 0; i < FTI_Exec->nbKernels; i++){
     free(FTI_Exec->gpuInfo[i].id);
     free(FTI_Exec->gpuInfo[i].block_amt);
     free(FTI_Exec->gpuInfo[i].all_done);
     free(FTI_Exec->gpuInfo[i].h_is_block_executed);
     free(FTI_Exec->gpuInfo[i].complete);
     free(FTI_Exec->gpuInfo[i].quantum);
     CUDA_ERROR_CHECK(cudaFree(FTI_Exec->gpuInfo[i].d_is_block_executed));
     CUDA_ERROR_CHECK(cudaFreeHost((void *)FTI_Exec->gpuInfo[i].quantum_expired));
  }
  return FTI_SCES;
}

/**
 * @brief              Determines if all processes have executed their kernels.
 * @param[in]          kernelId The ID of the kernel to check.
 *
 * @return             True or False whether all processes are complete or not.
 *
 * This function iterates over the procs array. If all values in the array have
 * a true value then all processes have completed executing the current kernel.
 */
bool FTI_all_procs_complete(FTIT_gpuInfo* FTI_GpuInfo)
{
  FTI_Print("Checking if all procs complete", FTI_DBUG);

  int i = 0;
  for(i = 0; i < FTI_Topo->nbProc; i++){
    if(!FTI_GpuInfo->all_done[i]){
      return false;
    }
  }

  return true;
}

/**
 * @brief           Waits for quantum to expire.
 *
 * Used in #FTI_BACKUP_monitor when initially waiting for the kernel to expire.
 * If the specified quantum is in minutes this function loops to perform a
 * cudaEventQuery every second; this is to handle the case of the kernel
 * returning before the quantum has expired so that no long period of time is
 * spent idly waiting.
 */
static inline void wait(FTIT_gpuInfo* FTI_GpuInfo)
{
  FTI_Print("IN kernel wait function", FTI_DBUG);
  char str[FTI_BUFS];
  int q = *FTI_GpuInfo->quantum / 1000000.0f; /* Convert to seconds */

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
    sprintf(str, "Sleeping for : %u microseconds\n", *FTI_GpuInfo->quantum);
    FTI_Print(str, FTI_DBUG);
    usleep(*FTI_GpuInfo->quantum);
  }
}

/**
 * @brief             Signals the GPU to return and then waits for it to return.
 * @param[in,out]     complete      Initialized to *true* or *false* if all blocks
 *                                  in kernel have finished or not
 * @return             #FTI_SCES or #FTI_NSCS for success or failure respectively.
 *
 * Called in #FTI_BACKUP_monitor() so tell the GPU it should finish the currently
 * executing blocks and prevent new blocks from continuing on to their main
 * body of work.
 */
static inline int signal_gpu_then_wait(FTIT_gpuInfo* FTI_GpuInfo)
{
  char str[FTI_BUFS];

  FTI_Print("Signalling kernel to return...", FTI_DBUG);
  *FTI_GpuInfo->quantum_expired = true;
  FTI_Print("Attempting to snapshot", FTI_DBUG);

  FTI_Print("Waiting on kernel...", FTI_DBUG);
  CUDA_ERROR_CHECK(cudaDeviceSynchronize());
  FTI_Print("Kernel came back...", FTI_DBUG);

  int res = FTI_Snapshot();

  if(res == FTI_DONE)
  {
    FTI_Print("Successfully wrote snapshot at kernel interrupt", FTI_DBUG);
  }
  else
  {
    FTI_Print("No snapshot was taken", FTI_DBUG);
  }

  CUDA_ERROR_CHECK(cudaMemcpy(FTI_GpuInfo->h_is_block_executed, FTI_GpuInfo->d_is_block_executed, FTI_GpuInfo->block_info_bytes, cudaMemcpyDeviceToHost));

  FTI_Print("Checking if complete", FTI_DBUG);
  computation_complete(FTI_GpuInfo);

  sprintf(str, "Done checking: %s", *FTI_GpuInfo->complete ? "True" : "False");
  FTI_Print(str, FTI_DBUG);

  return FTI_SCES;
}

/**
 * @brief            Executed when control is returned to host after GPU is interrupted or finished.
 * @parm[in]         complete          Used to check if kernel finished executing all blocks or not. 
 *
 * This function handles the control returned to the host after the GPU has finished executing. It
 * also executes any callback function to be executed at GPU interrupt. The callback function is
 * only executed if the GPU is not finished. This function also increases the quantum by a default
 * or specified amount. This increase is necessary in cases where the quantum is short enough to 
 * cause the GPU to return again by the time it has finished re-launching previously executed blocks.
 */
static inline void handle_gpu_suspension(FTIT_gpuInfo* FTI_GpuInfo)
{
  char str[FTI_BUFS];
  if(*FTI_GpuInfo->complete == false)
  {
    FTI_Print("Incomplete, resuming", FTI_DBUG);
    *FTI_GpuInfo->quantum_expired = false;

    *FTI_GpuInfo->quantum = *FTI_GpuInfo->quantum + FTI_GpuInfo->initial_quantum;

    sprintf(str, "Quantum is now: %u", *FTI_GpuInfo->quantum);
    FTI_Print(str, FTI_DBUG);
  }
}

/**
 * @brief              Monitors the kernel's execution until it completes.
 * @param[in,out]      complete           Initialized to *true* or *false* if the kernel has finished or not. 
 * @return             #FTI_SCES or #FTI_NSCS for success or failure respectively.
 *
 * Sleeps until the quantum has expired and then sets the quantum_expired
 * variable to *true* and waits for the kernel to return. The value in
 * quantum_expired is immediately available to the kernel which then does not
 * allow any new blocks to execute.  After the kernel returns, the boolean
 * array is copied to the host and checked to determine if all blocks executed.
 * If all blocks are finished *complete* is set to *true* and no further kernel
 * launches are made.  Otherwise this process repeats iteratively.
 */
int FTI_BACKUP_monitor(FTIT_gpuInfo* FTI_GpuInfo)
{
  FTI_Print("Monitoring kernel execution", FTI_DBUG);
  int ret;

  /* Wait for quantum to expire */
  wait(FTI_GpuInfo);

  /* Tell GPU to finish and come back now */
  ret = FTI_Try(signal_gpu_then_wait(FTI_GpuInfo), "signal and wait on kernel");

  if(ret != FTI_SCES)
  {
    FTI_Print("Failed to signal and wait on kernel", FTI_EROR);
    return FTI_NSCS;
  }

  /* Handle interrupted GPU */
  handle_gpu_suspension(FTI_GpuInfo);

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
