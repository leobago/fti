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
 * @param[in,out]      FTI_GpuInfo Metadata of the currently executing protected kernel.
 *
 * Sets FTI_GpuInfo->complete to true or false respectively if the kernel is finished or not.
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
 * @brief              Recovers GPU information of protected kernels at restart.
 *
 * @return             FTI_SCES on completion
 */
static int recover_gpuInfo(){
  int i = 0;
  char str[FTI_BUFS];

  sprintf(str, "Restoring GPU info from level %d", FTI_Exec->ckptLvel);
  FTI_Print(str, FTI_DBUG);

  /* Iterate through all protected kernels to restore their data from the checkpoint level */
  for(i = 0; i < FTI_Exec->nbKernels; i++){
    FTI_Exec->gpuInfo[i].id                   = FTI_Exec->meta[FTI_Exec->ckptLvel].gpuInfo[i].id;
    FTI_Exec->gpuInfo[i].complete             = FTI_Exec->meta[FTI_Exec->ckptLvel].gpuInfo[i].complete;
    FTI_Exec->gpuInfo[i].all_done             = FTI_Exec->meta[FTI_Exec->ckptLvel].gpuInfo[i].all_done;
    FTI_Exec->gpuInfo[i].block_info_bytes     = FTI_Exec->meta[FTI_Exec->ckptLvel].gpuInfo[i].block_info_bytes;
    FTI_Exec->gpuInfo[i].block_amt            = FTI_Exec->meta[FTI_Exec->ckptLvel].gpuInfo[i].block_amt;
    FTI_Exec->gpuInfo[i].h_is_block_executed  = FTI_Exec->meta[FTI_Exec->ckptLvel].gpuInfo[i].h_is_block_executed;
    FTI_Exec->gpuInfo[i].quantum              = FTI_Exec->meta[FTI_Exec->ckptLvel].gpuInfo[i].quantum;
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
int FTI_gpu_protect_init(FTIT_topology *topo, FTIT_execution *exec)
{
  FTI_Topo = topo;
  FTI_Exec = exec;

  if(FTI_Exec->reco){
    /* If this is a recovery, try to recover the GPU metadata */
    int res = FTI_Try(recover_gpuInfo(), "recover GPU info");
    
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
    if(kernelId == *FTI_Exec->gpuInfo[i].id){
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
    if(*FTI_Exec->gpuInfo[i].complete == false){
      fprintf(stdout, "Kernel %d not complete!\n", *FTI_Exec->gpuInfo[i].id);
      fflush(stdout);
      return false;
    }
  }

  return true;
}

/**
 * @brief              Initializes protection for a kernel.
 * @param              GpuMacroInfo   Initialized with values required by kernel launch macro.
 * @param              kernelId       The ID of the kernel to protect.
 * @param              quantum        How long the kernel should be executed before interruption.
 * @param              num_blocks     The number of blocks launched by the kernel.
 * @return             integer        FTI_SCES if successful.
 *
 * This function initializes the GpumacroInfo parameter and saves a reference
 * to the members of this parameter whose values need to be saved to the
 * metadata files at checkpoint time. In the event that the application needs to
 * be restarted, this function is called again. If the kernel has already been
 * protected, then the values for it to resume will be restored from
 * FTI_Exec->gpuInfo.
 */
int FTI_kernel_init(FTIT_gpuInfo* GpuMacroInfo, int kernelId, double quantum, dim3 num_blocks){
  size_t i = 0;
  unsigned int kernel_index = 0;
  char str[FTI_BUFS];

  snprintf(str, FTI_BUFS, "Entered function %s", __func__); 
  FTI_Print(str, FTI_DBUG);

  bool kernel_protected = is_kernel_protected(kernelId, &kernel_index);

  CUDA_ERROR_CHECK(cudaHostAlloc((void **)&GpuMacroInfo->quantum_expired, sizeof(volatile bool), cudaHostAllocMapped));

  if(!kernel_protected){
    snprintf(str, FTI_BUFS, "kernelId %d not protected.", kernelId);
    FTI_Print(str, FTI_DBUG);

    if(FTI_Exec->nbKernels >= FTI_BUFS){
      FTI_Print("Unable to protect kernel. Too many kernels already registered.", FTI_WARN);
      return FTI_NSCS;
    }

    snprintf(str, FTI_BUFS, "Allocating memory for kernelId %d", kernelId);
    FTI_Print(str, FTI_DBUG);
    GpuMacroInfo->id                  = (int *)malloc(sizeof(int));
    GpuMacroInfo->block_amt           = (size_t *)malloc(sizeof(size_t));
    GpuMacroInfo->all_done            = (bool *)malloc(sizeof(bool) * FTI_Topo->nbApprocs * FTI_Topo->nbNodes);
    GpuMacroInfo->complete            = (bool *)malloc(sizeof(bool));
    GpuMacroInfo->quantum             = (unsigned int*)malloc(sizeof(unsigned int));

    if(GpuMacroInfo->id              == NULL){return FTI_NSCS;}
    if(GpuMacroInfo->all_done        == NULL){return FTI_NSCS;}
    if(GpuMacroInfo->complete        == NULL){return FTI_NSCS;}
    if(GpuMacroInfo->quantum         == NULL){return FTI_NSCS;}

    snprintf(str, FTI_BUFS, "Successfully allocated memory for kernelId %d", kernelId);
    FTI_Print(str, FTI_DBUG);

    *GpuMacroInfo->id                 = kernelId;
    *GpuMacroInfo->complete           = false;
    *GpuMacroInfo->quantum            = seconds_to_microseconds(quantum);
    *GpuMacroInfo->quantum_expired    = false;
    *GpuMacroInfo->block_amt          = num_blocks.x * num_blocks.y * num_blocks.z;
    GpuMacroInfo->block_info_bytes    = *GpuMacroInfo->block_amt * sizeof(bool);

    snprintf(str, FTI_BUFS, "Allocating memory for block_info_bytes for kernelId %d", kernelId);
    FTI_Print(str, FTI_DBUG);

    GpuMacroInfo->h_is_block_executed = (bool *)malloc(GpuMacroInfo->block_info_bytes);
    if(GpuMacroInfo->h_is_block_executed  == NULL){return FTI_NSCS;}

    snprintf(str, FTI_BUFS, "Successfully allocated block_info_bytes for kernelId %d", kernelId);
    FTI_Print(str, FTI_DBUG);

    //TODO should I use calloc instead to initialize? - yes
    for(i = 0; i < FTI_Topo->nbApprocs * FTI_Topo->nbNodes; i++){
      GpuMacroInfo->all_done[i] = false;
    }

    //TODO should I use calloc instead to initialize? - yes
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
    snprintf(str, FTI_BUFS, "Restoring kernel execution data for kernelId %d", kernelId);
    FTI_Print(str, FTI_DBUG);

    GpuMacroInfo->id                    =  FTI_Exec->gpuInfo[kernel_index].id;
    GpuMacroInfo->complete              =  FTI_Exec->gpuInfo[kernel_index].complete;
    GpuMacroInfo->all_done              =  FTI_Exec->gpuInfo[kernel_index].all_done;
    GpuMacroInfo->block_info_bytes      = *FTI_Exec->gpuInfo[kernel_index].block_amt * sizeof(bool);
    GpuMacroInfo->block_amt             =  FTI_Exec->gpuInfo[kernel_index].block_amt;
    GpuMacroInfo->h_is_block_executed   =  FTI_Exec->gpuInfo[kernel_index].h_is_block_executed;
    GpuMacroInfo->quantum               =  FTI_Exec->gpuInfo[kernel_index].quantum;
    *GpuMacroInfo->quantum_expired      =  true;

    bool all_protected_kernels_complete = all_kernels_complete();

    if(all_protected_kernels_complete){
      /* Kernel needs to be executed again */
      snprintf(str, FTI_BUFS, "All other kernels complete. kernel Id %d will be re-executed", kernelId);
      FTI_Print(str, FTI_DBUG);

      *GpuMacroInfo->complete = false;
      *GpuMacroInfo->quantum_expired = false;
      *GpuMacroInfo->quantum = seconds_to_microseconds(quantum);

      //TODO should I use memset instead to initialize? - yes
      for(i = 0; i < FTI_Topo->nbApprocs * FTI_Topo->nbNodes; i++){
        GpuMacroInfo->all_done[i] = false;
      }

      //TODO should I use memset instead to initialize? - yes
      for(i = 0; i < *GpuMacroInfo->block_amt; i++){
        GpuMacroInfo->h_is_block_executed[i] = false;
      }
    }
  }

  //TODO check if removing the cast to void** affects anything
  snprintf(str, FTI_BUFS, "Kernel Id %d allocating memory on GPU", kernelId);
  FTI_Print(str, FTI_DBUG);
  CUDA_ERROR_CHECK(cudaMalloc((void **)&GpuMacroInfo->d_is_block_executed, GpuMacroInfo->block_info_bytes));
  CUDA_ERROR_CHECK(cudaMemcpy(GpuMacroInfo->d_is_block_executed, GpuMacroInfo->h_is_block_executed, GpuMacroInfo->block_info_bytes, cudaMemcpyHostToDevice));
  
  /* Every kernel being initialized has its initial quantum reset */
  GpuMacroInfo->initial_quantum  = seconds_to_microseconds(quantum);

  return FTI_SCES;
}

/* 
 * @brief              Frees allocations made on the GPU to protect kernel.
 * @param[in]          FTI_GpuInfo  GPU metadata.
 * @return             integer FTI_SCES if successful.
 */
int FTI_FreeDeviceAlloc(FTIT_gpuInfo* FTI_GpuInfo){
  char str[FTI_BUFS];
  snprintf(str, FTI_BUFS, "Freeing device allocations made for kernel %d", *FTI_GpuInfo->id);
  FTI_Print(str, FTI_DBUG);

  CUDA_ERROR_CHECK(cudaFree(FTI_GpuInfo->d_is_block_executed));
  CUDA_ERROR_CHECK(cudaFreeHost((void *)FTI_GpuInfo->quantum_expired));

  return FTI_SCES;
}

/**
 * @brief              Called in FTI_Finalize to release GPU resources.
 * @return             integer FTI_SCES if successful.
 *
 * Frees GPU resources for each kernel that was protected.
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
  }
  return FTI_SCES;
}

/**
 * @brief              Determines if all processes have executed their kernels.
 * @param[in]          FTIT_gpuInfo Metadata of currently executing kernel.
 * @return             bool True or False whether all processes are complete or not.
 */
bool FTI_all_procs_complete(FTIT_gpuInfo* FTI_GpuInfo)
{
  FTI_Print("Checking if all procs complete", FTI_DBUG);

  int i = 0;
  
  /* Iterate over all application processes at each node */  

  for(i = 0; i < FTI_Topo->nbApprocs * FTI_Topo->nbNodes; i++){
    /* Check if each process is finished with the current kernel */
    if(!FTI_GpuInfo->all_done[i]){
      return false;
    }
  }

  return true;
}

/**
 * @brief           Waits for quantum to expire.
 * @param[in]       FTI_GpuInfo Metadata of currently executing protected kernel.
 *
 * Used when initially waiting for the kernel to expire.
 * If the specified quantum is in minutes this function loops to perform a
 * cudaEventQuery every second; this is to handle the case of the kernel
 * returning before the quantum has expired so that no long period of time is
 * spent idly waiting.
 */
static inline void wait(FTIT_gpuInfo* FTI_GpuInfo)
{
  FTI_Print("In kernel wait function", FTI_DBUG);
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
 * @param             FTI_GpuInfo   Metadata of currently executing protected kernel. 
 * @return            integer       FTI_SCES if successful.
 *
 * Called in #FTI_kernel_monitor() to tell the GPU it should finish the currently
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

  snprintf(str, FTI_BUFS, "Kernel id %d copying block information from GPU.", *FTI_GpuInfo->id);
  FTI_Print(str, FTI_DBUG);
  CUDA_ERROR_CHECK(cudaMemcpy(FTI_GpuInfo->h_is_block_executed, FTI_GpuInfo->d_is_block_executed, FTI_GpuInfo->block_info_bytes, cudaMemcpyDeviceToHost));

  computation_complete(FTI_GpuInfo);

  sprintf(str, "Kernel id %d complete: %s", *FTI_GpuInfo->id, *FTI_GpuInfo->complete ? "True" : "False");
  FTI_Print(str, FTI_DBUG);

  return FTI_SCES;
}

/**
 * @brief            Executed when control is returned to host after GPU is interrupted or finished.
 * @parm[in]         FTI_GpuInfo Metadata of currently executing protected kernel. 
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
    snprintf(str, FTI_BUFS, "Kernel id %d incomplete, resuming", *FTI_GpuInfo->id);
    FTI_Print(str, FTI_DBUG);

    /* Kernel not complete, so reset the quantum_expired boolean to false */
    *FTI_GpuInfo->quantum_expired = false;

    /* 
      If the quantum is very very short (0.00001 seconds), the kernel may
      exhaust the quantum before getting to blocks that have not been executed. For
      this reason, the quantum is increased by the initial amount specified at each
      interrupt.  This should not be necessary for a quantum in minutes and should be
      removed.
    */
    //TODO consider if this is still necessary.
    *FTI_GpuInfo->quantum = *FTI_GpuInfo->quantum + FTI_GpuInfo->initial_quantum;
  }
}

/**
 * @brief              Monitors the kernel's execution until it completes.
 * @param[in,out]      FTI_GpuInfo Metadata of currently executing protected kernel. 
 * @return             integer FTI_SCES if successful.
 * 
 * This function monitors the execution of a protected kernel by using FTI_GpuInfo
 * as a handle to the protected kernel.
 */
int FTI_kernel_monitor(FTIT_gpuInfo* FTI_GpuInfo)
{
  FTI_Print("Monitoring kernel execution", FTI_DBUG);
  int ret;

  /* Wait for quantum to expire */
  wait(FTI_GpuInfo);

  /* Tell GPU to finish and return */
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
 * @brief              Gathers the complete status of current kernel.
 * @param              FTI_GpuInfo  Metadata about currently executing protected kernel. 
 * @return             integer  FTI_SCES if successful.
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
int FTI_BACKUP_gather_kernel_status(FTIT_gpuInfo* FTI_GpuInfo){
  MPI_Allgather(FTI_GpuInfo->complete, 1, MPI_C_BOOL, FTI_GpuInfo->all_done, 1, MPI_C_BOOL, FTI_COMM_WORLD);
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
