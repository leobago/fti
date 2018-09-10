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
static volatile unsigned int *t = NULL;

/**
 * @brief              Host-side boolean array.
 *
 * An array of unsigned short int type with size #block_amt. Each value
 * in the array represents a thread block and is set to 1 when the block
 * has finished executing. Is copied to host at each interrupt and checked
 * to determine if all blocks have executed.
 */
backup_t *h_is_block_executed = NULL;

/**
 * @brief              Device-side boolean array. See #h_is_block_executed.
 */
backup_t *d_is_block_executed = NULL;

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
    if(h_is_block_executed[i] == 1)
    {
      continue;
    }
    break;
  }
  *complete = (i == block_amt) ? true : false;
}

static FTIT_topology *FTI_Topo = NULL;

int FTI_get_topo(FTIT_topology *topology)
{
  FTI_Topo = topology; 
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
  return (i == FTI_Topo->nbProc) ? true : false;
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
  CUDA_ERROR_CHECK(cudaFreeHost((void *)t));

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
  t = NULL;
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
int FTI_BACKUP_init(volatile unsigned int **timeout, backup_t **b_info, double q, bool *complete, bool **all_done, dim3 num_blocks)
{
  char str[FTI_BUFS];

  *all_done = malloc(sizeof(bool) * FTI_Topo->nbProc);
  
  if(*all_done == NULL)
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

  block_amt = num_blocks.x * num_blocks.y * num_blocks.z; 

  /* Block info host setup */
  block_info_bytes = block_amt * sizeof(backup_t);
  h_is_block_executed = (backup_t *)malloc(block_info_bytes);

  if(h_is_block_executed == NULL)
  {
    sprintf(str, "Cannot allocate %zu bytes for block info!", block_info_bytes);
    FTI_Print(str, FTI_EROR);
    return FTI_NSCS;
  }

  for(i = 0; i < block_amt; i++)
  {
    h_is_block_executed[i] = 0;
  }

  for(i = 0; i < FTI_Topo->nbProc; i++)
  {
    (*all_done)[i] = false;
  }

  quantum = seconds_to_microseconds(q);
  initial_quantum = quantum;

  CUDA_ERROR_CHECK(cudaHostAlloc((void **)&(*timeout), sizeof(volatile unsigned int), cudaHostAllocMapped));

  /* Block info device setup */
  CUDA_ERROR_CHECK(cudaMalloc((void **)&(*b_info), block_info_bytes));
  CUDA_ERROR_CHECK(cudaMemcpy(*b_info, h_is_block_executed, block_info_bytes, cudaMemcpyHostToDevice));

  **timeout = 0;

  /* Keep track of some things locally */
  t = *timeout;
  d_is_block_executed = *b_info;
  all_done_array = *all_done;

  /* Now protect all necessary variables */
  FTIT_type C_BOOL;
  FTI_InitType(&C_BOOL, sizeof(bool));
  FTI_Protect(22, (void *)all_done_array, FTI_Topo->nbProc, C_BOOL);
  FTI_Protect(23, (void *)complete, 1, C_BOOL);
  FTI_Protect(24, (void *)h_is_block_executed, block_amt, FTI_USHT);
  FTI_Protect(25, (void *)&quantum, 1, FTI_DBLE);
  FTI_Protect(26, (void *)t, 1, FTI_UINT);

  if(FTI_Status() != 0)
  {
    FTI_Recover();
  
    for(i = 0; i < block_amt; i++)
    {
      sprintf(str, "block_executed[%zu]=%d\n", i, h_is_block_executed[i]);
      FTI_Print(str, FTI_DBUG);
    }
  }

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
 * @brief              Monitors the kernel's execution until it completes.
 * @param[out]         complete           Initialized to 0 or 1 if the kernel has finished or not. 
 * @param[in]          do_memory_backup   Whether to perform device to host transfers at each interrupt. 
 * @return             #FTI_SCES or #FTI_NSCS for success or failure respectively.
 *
 * Sleeps until the quantum has expired and then sets the timeout variable to 1
 * and waits for the kernel to return. The value in timeout is immediately
 * available to the kernel which then does not allow any new blocks to execute.
 * After the kernel returns, the boolean array is copied to the host and
 * checked to determine if all blocks executed. If all blocks are finished
 * complete is set to 1 and no further kernel launches are made. Otherwise this
 * process repeats iteratively.
 *
 * If the quantum is in seconds or larger the kernel's execution is checked
 * every second for completion. This is avoids the case of the host sleeping
 * unnecessarily when a large quantum is set and the kernel completes very
 * early.
 */
int FTI_BACKUP_monitor(bool *complete)
{
  //int ret;
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
      FTI_Print("sleep()", FTI_DBUG);
      sleep(10);
      FTI_Print("sleep() Waking up", FTI_DBUG);
      q--;
    }
    cudaEventDestroy(event);
  }
  else
  {
    usleep(quantum);
  }

  FTI_Print("Signalling kernel to return...", FTI_DBUG);
  *t = 1;
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

  if(*complete == false)
  {
    FTI_Print("Waiting on kernel...", FTI_DBUG);
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    FTI_Print("Kernel came back...", FTI_DBUG);

    CUDA_ERROR_CHECK(cudaMemcpy(h_is_block_executed, d_is_block_executed, block_info_bytes, cudaMemcpyDeviceToHost)); 
    FTI_Print("Checking if complete", FTI_DBUG);
    computation_complete(complete);

    sprintf(str, "Done checking: %s", *complete ? "True" : "False");
    FTI_Print(str, FTI_DBUG);

    if(*complete == false)
    {
      FTI_Print("Incomplete, resuming", FTI_DBUG);
      *t = 0;

      CUDA_ERROR_CHECK(cudaMemcpy(d_is_block_executed, h_is_block_executed, block_info_bytes, cudaMemcpyHostToDevice)); 
      FTI_Print("Increasing quantum", FTI_DBUG);
      /* Automatically increase quantum, may not be necessary! */
      quantum = quantum + initial_quantum;

      sprintf(str, "Quantum is now: %d", quantum);
      FTI_Print(str, FTI_DBUG);
      suspension_count++; 
    }
  }

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
