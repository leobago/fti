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
 * @brief              Initialized to reference to FTI_Exec->FTI_GpuInfo.
 */
static FTIT_gpuInfo *FTI_GpuInfo = NULL;

/**
 * @brief              Array of handles for protected kernels.
 *
 * A handle at index FTI_Exec->nbKernels is initialized for each new kernel.
 * This handle is then later searched for by the protected kernel's ID. The
 * functions in this file use a kernel's handle for monitoring and interruption.
 */
static FTIT_kernelProtectHandle FTI_KernelProtectHandle[FTI_BUFS];

/** 
 * @brief              Determines if kernel is complete by checking #h_is_block_executed. 
 * @param[in,out]      complete   Indicates whether kernel is complete or not.
 *
 * Sets *complete* to 1 or 0 respectively if the kernel is finished or not.
 */
static void computation_complete(FTIT_kernelProtectHandle *handle, bool *complete)
{
  size_t i = 0;
  for(i = 0; i < handle->block_amt; i++)
  {
    if(handle->h_is_block_executed[i])
    {
      continue;
    }
    break;
  }
  *complete = (i == handle->block_amt);
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
  FTI_GpuInfo = exec->gpuInfo;
  return FTI_SCES;
}

/** 
 * @brief              Determines if all processes have executed their kernels. 
 * @param[in]          procs The current FTI_Exec object.
 *
 * @return             True or False whether all processes are complete or not.
 *
 * This function iterates over the procs array. If all values in the array have
 * a true value then all processes have completed executing the current kernel.
 */
bool FTI_all_procs_complete(bool *procs)
{
  int i = 0;
  for(i = 0; i < FTI_Topo->nbProc; i++)
  {
    if(procs[i] == false)break;
  }
  return (i == FTI_Topo->nbProc);
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
static inline useconds_t seconds_to_microseconds(double quantum)
{
  return fabs(quantum) * 1000000.0; 
}

///** 
// * @brief              Frees host and device allocated memory
// * @return             FTI_SCES or FTI_NSCS on success or failure respectively.
// *
// * Frees any memory that was allocated to help execute and suspend the kernel.
// */
//static int free_memory(FTIT_kernelProtectHandle *handle)
//{
//  //safeFree((void **)&h_is_block_executed);
//  free(handle->h_is_block_executed);
//  free(handle->all_done_array);
//
//  /* MACROS surrounding CUDA functions return FTI_NSCS on error. */
//  CUDA_ERROR_CHECK(cudaFree(handle->d_is_block_executed));
//  CUDA_ERROR_CHECK(cudaFreeHost((void *)handle->quantum_expired));
//
//  return FTI_SCES;
//} 

///**
// * @brief              Resets all global variables to their default value.
// * @return             FTI_SCES or FTI_NSCS on success or failure respectively.
// */
//static int reset_globals(FTIT_kernelProtectHandle *handle)
//{
//  /* Reset all globals */
//  //TODO is this even needed??
//  //block_amt = 0;
//  //quantum = 0.0;
//  //initial_quantum = 0.0;
//  //block_info_bytes = 0;
//  //quantum_expired = NULL;
//  //h_is_block_executed = NULL;
//  //d_is_block_executed = NULL;
//  //suspension_count = 0;
//  //all_done_array = NULL;
//
//  return FTI_SCES;
//}

static FTIT_kernelProtectHandle* getKernelProtectHandle(int kernelId){
  int i = 0;
  FTIT_kernelProtectHandle *handle = NULL;
  for(i = 0; i < FTI_Exec->nbKernels; i++){
    if(FTI_KernelProtectHandle[i].id == kernelId){
      handle = &FTI_KernelProtectHandle[i];
      break;
    }
  }
  //Return pointer to handle for current kernel
  return handle;
}

/**
 * @brief              Initializes the backup library.
 * @param[in,out]      timeout      Pinned variable used by host to communicate with kernel.
 * @param[in,out]      b_info       The boolean array to be passed to kernel. 
 * @param[in]          q            The quantum for the kernel execution time.
 * @param[in,out]      complete     Track reference to tell kernel execution loop when to quit.
 * @param[in]          num_blocks   The number of blocks used in the kernel launch.
 * @return             #FTI_SCES or #FTI_NSCS for success or failure respectively. 
 *
 * This function sets up the library to interrupt the kernel. Variables to track the kernel
 * execution are initialized and kept track of. Of particular importance is the *timeout* and 
 * boolean array (*b_info*). The timeout is allocated using cudaHostAlloc() so that the host
 * may communicate with the device directly without an explicit memory copy. The boolean array
 * has a size of *num_blocks* and has a record of which blocks have executed at each interrupt.
 */
int FTI_BACKUP_init(int kernelId, volatile bool **timeout, bool **b_info, double q, bool *complete, bool **all_processes_done, dim3 num_blocks)
{
  char str[FTI_BUFS];
  size_t i = 0;
  bool kernel_already_protected = false;
  FTIT_kernelProtectHandle *handle = NULL;

  for(i = 0; i < FTI_Exec->nbKernels; i++){
    if(kernelId == *FTI_GpuInfo[i].id){
      kernel_already_protected = true;
      break;
    }
  }

  CUDA_ERROR_CHECK(cudaHostAlloc((void **)&(*timeout), sizeof(volatile bool), cudaHostAllocMapped));

  if(kernel_already_protected){
    /* Restore kernel information */
    handle = getKernelProtectHandle(kernelId);

    *complete = *FTI_GpuInfo[i].complete;
    *all_processes_done = FTI_GpuInfo[i].all_done;
    **timeout = *FTI_GpuInfo[i].quantum_expired;

    handle->block_info_bytes = *FTI_GpuInfo[i].block_amt * sizeof(bool);
    handle->block_amt = *FTI_GpuInfo[i].block_amt;
    handle->h_is_block_executed = FTI_GpuInfo[i].h_is_block_executed;
    handle->quantum = *FTI_GpuInfo[i].quantum;
  }
  else{
    /* Initialize new kernel */
    handle =  &FTI_KernelProtectHandle[FTI_Exec->nbKernels];
    handle->id = kernelId;

    if(FTI_Exec->nbKernels >= FTI_BUFS){
      FTI_Print("Unable to protect kernel. Too many kernels already registered.", FTI_WARN);
      return FTI_NSCS;
    }

    **timeout = false;
    *all_processes_done = (bool *)malloc(sizeof(bool) * FTI_Topo->nbProc);

    if(*all_processes_done == NULL){
      sprintf(str, "Cannot allocate memory for all_done");
      FTI_Print(str, FTI_EROR);
      return FTI_NSCS;
    }

    handle->block_amt = num_blocks.x * num_blocks.y * num_blocks.z;

    /* Block info host setup */
    handle->block_info_bytes = handle->block_amt * sizeof(bool);
    handle->h_is_block_executed = (bool *)malloc(handle->block_info_bytes);

    if(handle->h_is_block_executed == NULL){
      sprintf(str, "Cannot allocate %zu bytes for block info!", handle->block_info_bytes);
      FTI_Print(str, FTI_EROR);
      return FTI_NSCS;
    }

    for(i = 0; i < handle->block_amt; i++){
      handle->h_is_block_executed[i] = false;
    }

    for(i = 0; i < FTI_Topo->nbProc; i++){
      (*all_processes_done)[i] = false;
    }

    handle->quantum = seconds_to_microseconds(q);
    *complete = false;

    /* Add information necessary to protect interrupt info */
    FTI_GpuInfo[FTI_Exec->nbKernels].id = &kernelId; 
    FTI_GpuInfo[FTI_Exec->nbKernels].complete = complete;
    FTI_GpuInfo[FTI_Exec->nbKernels].block_amt = &handle->block_amt;
    FTI_GpuInfo[FTI_Exec->nbKernels].all_done = *all_processes_done;
    FTI_GpuInfo[FTI_Exec->nbKernels].h_is_block_executed = handle->h_is_block_executed;
    FTI_GpuInfo[FTI_Exec->nbKernels].quantum = &handle->quantum;
    FTI_GpuInfo[FTI_Exec->nbKernels].quantum_expired = *timeout;
    FTI_Exec->nbKernels = FTI_Exec->nbKernels + 1;
  }

  //Do things that both protected and non-protected kernels need

  CUDA_ERROR_CHECK(cudaMalloc((void **)&(*b_info), handle->block_info_bytes));
  CUDA_ERROR_CHECK(cudaMemcpy(*b_info, handle->h_is_block_executed, handle->block_info_bytes, cudaMemcpyHostToDevice));

  handle->suspension_count = 0;
  handle->d_is_block_executed = *b_info;
  handle->all_done_array = *all_processes_done; 
  handle->quantum_expired = *timeout;
  handle->initial_quantum = seconds_to_microseconds(q); 

  return FTI_SCES;
}

/**
 * @brief              Prints how many times the kernel was suspended. 
 * @param[in]          kernel_name  The name of the kernel.
 */
//TODO Should I still use this?
//static void print_stats(FTIT_kernelProtectHandle *handle, const char *kernel_name)
//{
//  char str[FTI_BUFS];
//  sprintf(str,"%s suspensions = %zu", kernel_name, handle->suspension_count);
//  FTI_Print(str, FTI_DBUG);
//}

/**
 * @brief              Prints kernel suspension count and frees memory.
 * @param[in]          kernel_name    The name of the kernel.
 * @return             #FTI_SCES or #FTI_NSCS if successful or not.
 *  
 * This function calls #print_stats() and #free_memory().
 */
//static int cleanup(FTIT_kernelProtectHandle *handle, const char *kernel_name)
//{
//  //TODO have this function iterate over the array of GpuInfo
//  //structures and free all memory
//  //TODO come back to this after making library "thread-safe"
//
//  int res;
//  print_stats(handle, kernel_name);
//  res = FTI_Try(free_memory(handle), "free memory");
//
//  /* Reset global variables to their defaults */
//  FTI_Try(reset_globals(handle), "reset global variables");
//
//  return res;
//}

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
     fprintf(stdout, "Cleaning up kernel information\n");
     fflush(stdout);
     free(FTI_Exec->gpuInfo[i].id);
     free(FTI_Exec->gpuInfo[i].block_amt);
     free(FTI_Exec->gpuInfo[i].all_done);
     free(FTI_Exec->gpuInfo[i].complete);
     free(FTI_Exec->gpuInfo[i].h_is_block_executed);
     free(FTI_Exec->gpuInfo[i].quantum);
     free((void *)FTI_Exec->gpuInfo[i].quantum_expired);

     free(FTI_KernelProtectHandle[i].h_is_block_executed);
     free(FTI_KernelProtectHandle[i].all_done_array);

     /* MACROS surrounding CUDA functions return FTI_NSCS on error. */
     CUDA_ERROR_CHECK(cudaFree(FTI_KernelProtectHandle[i].d_is_block_executed));
     CUDA_ERROR_CHECK(cudaFreeHost((void *)FTI_KernelProtectHandle[i].quantum_expired));
  }

  fprintf(stdout, "Finished cleaning kernel information\n");
  fflush(stdout);
  return FTI_SCES;
  //int ret = FTI_Try(cleanup(handle, kernel_name), "cleaning up resources");
  //
  //if(ret != FTI_SCES)
  //{
  //  FTI_Print("Failed to properly clean up resources", FTI_EROR);
  //}
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
static inline void wait(FTIT_kernelProtectHandle *handle)
{
  char str[FTI_BUFS];
  int q = handle->quantum / 1000000.0f; /* Convert to seconds */

  /* If quantum is in seconds or minutes check kernel every second */
  if(q != 0)
  {
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
    usleep(handle->quantum);
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
static inline int signal_gpu_then_wait(FTIT_kernelProtectHandle *handle, bool *complete)
{
  char str[FTI_BUFS];

  FTI_Print("Signalling kernel to return...", FTI_DBUG);
  *handle->quantum_expired = true;
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

  CUDA_ERROR_CHECK(cudaMemcpy(handle->h_is_block_executed, handle->d_is_block_executed, handle->block_info_bytes, cudaMemcpyDeviceToHost)); 
  FTI_Print("Checking if complete", FTI_DBUG);
  computation_complete(handle, complete);

  sprintf(str, "Done checking: %s", *complete ? "True" : "False");
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
static inline void handle_gpu_suspension(FTIT_kernelProtectHandle *handle, bool *complete)
{
  char str[FTI_BUFS];
  if(*complete == false)
  {
    FTI_Print("Incomplete, resuming", FTI_DBUG);
    *handle->quantum_expired = false;

    handle->quantum = handle->quantum + handle->initial_quantum;

    sprintf(str, "Quantum is now: %u", handle->quantum);
    FTI_Print(str, FTI_DBUG);
    handle->suspension_count++; 
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
int FTI_BACKUP_monitor(int kernelId, bool *complete)
{
  FTIT_kernelProtectHandle *handle = getKernelProtectHandle(kernelId);
  int ret;

  /* Wait for quantum to expire */
  wait(handle);
  
  /* Tell GPU to finish and come back now */
  ret = FTI_Try(signal_gpu_then_wait(handle, complete), "signal and wait on kernel");

  if(ret != FTI_SCES)
  {
    FTI_Print("Failed to signal and wait on kernel", FTI_EROR);
    return FTI_NSCS;
  }

  /* Handle interrupted GPU */
  handle_gpu_suspension(handle, complete);

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
