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
 * @brief              Used to determine the size of boolean array.
 *
 * The boolean array is used for book-keeping to keep track of executed
 * blocks.
 */
static size_t block_amt  = 0;

/**
 * @brief              Determines how frequently the kernel is interrupted.
 */
static useconds_t quantum = 0.0;

/**
 * @brief              Keeps track of the value initially passed to the quantum.
 *
 * During configuration (see BACKUP_config()), if the quantum is set to increase
 * but no increase value is specified, the quantum is increased per interrupt by
 * its initial amount. I.e the amount assigned to #initial_quantum.
 */
static useconds_t initial_quantum = 0.0; 

/**
 * @brief              Represents the size in bytes of the boolean array.
 *
 * Initialized in BACKUP_init and is used later in BACKUP_monitor when
 * performing a device to host transfer of the boolean array (#h_is_block_executed). 
 */
static size_t block_info_bytes = 0;

/**
 * @brief              Holds a reference to the timeout variable.
 *
 * Assigned to 1 in BACKUP_monitor() when the quantum has expired so that 
 * the kernel can know it should not launch any new blocks.  
 */
static volatile bool *quantum_expired = NULL;

/**
 * @brief              Host-side boolean array.
 *
 * An array of unsigned short int type with size #block_amt. Each value
 * in the array represents a thread block and is set to 1 when the block
 * has finished executing. Is copied to host at each interrupt and checked
 * to determine if all blocks have executed.
 */
bool *h_is_block_executed = NULL;

/**
 * @brief              Device-side boolean array. See #h_is_block_executed.
 */
bool *d_is_block_executed = NULL;

/**
 * @brief              Used to specify the amount quantum should be increased by.
 *
 * If not set to zero and #increase_quantum is set to true, the quantum is increased
 * by this amount at each interrupt. If set to zero and #increase_quantum is true, then
 * the quantum is increased by #initial_quantum at each kernel interrupt.
 */
static double quantum_inc = 0.0;

/**
 * @brief              Keeps track of how many times the kernel was interrupted.
 *
 * This value is printed when the kernel has completed.
 */
static size_t suspension_count;

/** 
 * @brief              Determines if kernel is complete by checking #h_is_block_executed. 
 * @param[in,out]      complete   Indicates whether kernel is complete or not.
 *
 * Sets *complete* to 1 or 0 respectively if the kernel is finished or not.
 */
static void computation_complete(bool *complete)
{
  size_t i = 0;
  for(i = 0; i < block_amt; i++)
  {
    if(h_is_block_executed[i] == true)
    {
      continue;
    }
    break;
  }
  *complete = (i == block_amt); //? true : false;
}

static FTIT_topology *FTI_Topo = NULL;
static FTIT_execution *FTI_Exec = NULL;

int FTI_get_topo_and_exec(FTIT_topology *topo, FTIT_execution *exec)
{
  FTI_Topo = topo; 
  FTI_Exec = exec;
  return FTI_SCES;
}

bool *all_done_array = NULL;
bool FTI_all_procs_complete(bool *procs)
{
  int i = 0;
  for(i = 0; i < FTI_Topo->nbProc; i++)
  {
    if(procs[i] == false)break;
  }
  return (i == FTI_Topo->nbProc); //? true : false;
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

/** 
 * @brief              Frees host and device allocated memory
 * @return             FTI_SCES or FTI_NSCS on success or failure respectively.
 *
 * Frees any memory that was allocated to help execute and suspend the kernel.
 */
static int free_memory()
{
  //safeFree((void **)&h_is_block_executed);
  free(h_is_block_executed);
  free(all_done_array);

  /* MACROS surrounding CUDA functions return FTI_NSCS on error. */
  CUDA_ERROR_CHECK(cudaFree(d_is_block_executed));
  CUDA_ERROR_CHECK(cudaFreeHost((void *)quantum_expired));

  return FTI_SCES;
} 

/**
 * @brief              Resets all global variables to their default value.
 * @return             FTI_SCES or FTI_NSCS on success or failure respectively.
 */
static int reset_globals()
{
  /* Reset all globals */
  block_amt = 0;
  quantum = 0.0;
  initial_quantum = 0.0;
  block_info_bytes = 0;
  quantum_expired = NULL;
  h_is_block_executed = NULL;
  d_is_block_executed = NULL;
  suspension_count = 0;
  quantum_inc = 0.0;
  all_done_array = NULL;

  /* Globals from cuda_backup.h */

  return FTI_SCES;
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
int FTI_BACKUP_init(volatile bool **timeout, bool **b_info, double q, bool *complete, bool **all_processes_done, dim3 num_blocks)
{
  char str[FTI_BUFS];

  *all_processes_done = malloc(sizeof(bool) * FTI_Topo->nbProc);
  
  if(*all_processes_done == NULL)
  {
    sprintf(str, "Cannot allocate memory for all_done");
    FTI_Print(str, FTI_EROR);
    return FTI_NSCS;
  }

  FTI_Print("Initialized backup", FTI_DBUG);

  /* Set default values */
  suspension_count = 0;
  size_t i = 0;
  *complete = false;

  block_amt = num_blocks.x * num_blocks.y * num_blocks.z; //TODO remove from global scope in file

  /* Block info host setup */
  block_info_bytes = block_amt * sizeof(bool);
  h_is_block_executed = (bool *)malloc(block_info_bytes);

  if(h_is_block_executed == NULL)
  {
    sprintf(str, "Cannot allocate %zu bytes for block info!", block_info_bytes);
    FTI_Print(str, FTI_EROR);
    return FTI_NSCS;
  }

  for(i = 0; i < block_amt; i++)
  {
    h_is_block_executed[i] = false;
  }

  for(i = 0; i < FTI_Topo->nbProc; i++)
  {
    (*all_processes_done)[i] = false;
  }

  quantum = seconds_to_microseconds(q);
  initial_quantum = quantum;

  CUDA_ERROR_CHECK(cudaHostAlloc((void **)&(*timeout), sizeof(volatile bool), cudaHostAllocMapped));

  /* Block info device setup */
  CUDA_ERROR_CHECK(cudaMalloc((void **)&(*b_info), block_info_bytes));
  CUDA_ERROR_CHECK(cudaMemcpy(*b_info, h_is_block_executed, block_info_bytes, cudaMemcpyHostToDevice));

  **timeout = 0;

  /* Keep track of some things locally */
  quantum_expired = *timeout;
  d_is_block_executed = *b_info;
  all_done_array = *all_processes_done; 

  /* Now protect all necessary variables */
  //FTIT_type C_BOOL;
  //FTI_InitType(&C_BOOL, sizeof(bool));
  //FTI_Protect(22, (void *)all_done_array, FTI_Topo->nbProc, C_BOOL);
  //FTI_Protect(23, (void *)complete, 1, C_BOOL);
  //FTI_Protect(24, (void *)h_is_block_executed, block_amt, C_BOOL);
  //FTI_Protect(25, (void *)&quantum, 1, FTI_DBLE);
  //FTI_Protect(26, (void *)quantum_expired, 1, C_BOOL);

  if(FTI_Exec->gpuInfo.exists){
    //TODO restore values here
    block_amt = FTI_Exec->gpuInfo.block_amt;
    *complete = FTI_Exec->gpuInfo.complete;
    all_done_array = FTI_Exec->gpuInfo.all_done;
    h_is_block_executed = FTI_Exec->gpuInfo.h_is_block_executed;
    quantum = FTI_Exec->gpuInfo.quantum;
    *quantum_expired = FTI_Exec->gpuInfo.quantum_expired;
  }
  else{
    //Keep track of things here
    FTI_Exec->gpuInfo.block_amt = block_amt; 
    FTI_Exec->gpuInfo.complete = *complete;
    FTI_Exec->gpuInfo.all_done = all_done_array;
    FTI_Exec->gpuInfo.h_is_block_executed = h_is_block_executed;
    FTI_Exec->gpuInfo.quantum = quantum;
    FTI_Exec->gpuInfo.quantum_expired = *quantum_expired;
    FTI_Exec->gpuInfo.exists = true;
  }

  //if(FTI_Status() != 0)
  //{
  //  FTI_Recover();
  //
  //  for(i = 0; i < block_amt; i++)
  //  {
  //    sprintf(str, "block_executed[%zu]=%d\n", i, h_is_block_executed[i]);
  //    FTI_Print(str, FTI_DBUG);
  //  }
  //}

  return FTI_SCES;
}

/**
 * @brief              Prints how many times the kernel was suspended. 
 * @param[in]          kernel_name  The name of the kernel.
 */
static void print_stats(const char *kernel_name)
{
  char str[FTI_BUFS];
  sprintf(str,"%s suspensions = %zu", kernel_name, suspension_count);
  FTI_Print(str, FTI_DBUG);
}

/**
 * @brief              Prints kernel suspension count and frees memory.
 * @param[in]          kernel_name    The name of the kernel.
 * @return             #FTI_SCES or #FTI_NSCS if successful or not.
 *  
 * This function calls #print_stats() and #free_memory().
 */
static int cleanup(const char *kernel_name)
{
  int res;
  print_stats(kernel_name);
  res = FTI_Try(free_memory(), "free memory");

  /* Reset global variables to their defaults */
  FTI_Try(reset_globals(), "reset global variables");

  return res;
}

/**
 * @brief              Calls #cleanup().
 * @param[in]          kernel_name    The name of the kernel.
 *
 * An interface function to clean up allocated resources and print
 * how often *kernel_name* was suspended.
 */
void FTI_BACKUP_cleanup(const char *kernel_name)
{
  int ret = FTI_Try(cleanup(kernel_name), "cleaning up resources");
  
  if(ret != FTI_SCES)
  {
    FTI_Print("Failed to properly clean up resources", FTI_EROR);
  }
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
static inline void wait()
{
  char str[FTI_BUFS];
  int q = quantum / 1000000.0f; /* Convert to seconds */

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
      sleep(1);
      q--;
    }
    cudaEventDestroy(event);
  }
  else
  {
    usleep(quantum);
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
static inline int signal_gpu_then_wait(bool *complete)
{
  char str[FTI_BUFS];

  FTI_Print("Signalling kernel to return...", FTI_DBUG);
  *quantum_expired = true;
  FTI_Print("Attempting to snapshot", FTI_DBUG);

  int res = FTI_Snapshot();
  
  if(res == FTI_DONE)
  {
    FTI_Print("Successfully wrote snapshot at kernel interrupt", FTI_DBUG);
  }
  else
  {
    FTI_Print("No snapshot was taken", FTI_DBUG);
  }


  FTI_Print("Waiting on kernel...", FTI_DBUG);
  CUDA_ERROR_CHECK(cudaDeviceSynchronize());
  FTI_Print("Kernel came back...", FTI_DBUG);

  CUDA_ERROR_CHECK(cudaMemcpy(h_is_block_executed, d_is_block_executed, block_info_bytes, cudaMemcpyDeviceToHost)); 
  FTI_Print("Checking if complete", FTI_DBUG);
  computation_complete(complete);

  sprintf(str, "Done checking: %d", *complete);
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
static inline void handle_gpu_suspension(bool *complete)
{
  char str[FTI_BUFS];
  if(*complete == false)
  {

    FTI_Print("Incomplete, resuming", FTI_DBUG);
    *quantum_expired = false;

    quantum = quantum + initial_quantum;

    sprintf(str, "Quantum is now: %u", quantum);
    FTI_Print(str, FTI_DBUG);
    suspension_count++; 
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
int FTI_BACKUP_monitor(bool *complete)
{
  int ret;

  /* Wait for quantum to expire */
  wait();
  
  /* Tell GPU to finish and come back now */
  ret = FTI_Try(signal_gpu_then_wait(complete), "signal and wait on kernel");

  if(ret != FTI_SCES)
  {
    FTI_Print("Failed to signal and wait on kernel", FTI_EROR);
    return FTI_NSCS;
  }

  /* Handle interrupted GPU */
  handle_gpu_suspension(complete);

  return FTI_SCES;
}

/**
 * @brief              A wrapper around FTI_Print.
 * @param              msg        The text to print.
 * @param              priority   The priority of msg.
 *
 * Used so that kernel interrupt macros can have access
 * to the standard FTI_Print() function.
 *
 * TODO: check if this is still needed.
 */
void FTI_BACKUP_Print(char *msg, int priority)
{
  FTI_Print(msg, priority);
}
