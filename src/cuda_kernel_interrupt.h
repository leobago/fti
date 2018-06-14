#ifndef _KERNEL_INTERRUPT_H
#define _KERNEL_INTERRUPT_H

#define FTI_KERNEL_LAUNCH(quantum, kernel_name, grid_dim, block_dim, ns, s, ...)                          \
do{                                                                                                       \
    int ret;                                                                                              \
    ret = FTI_Try(FTI_BACKUP_init(&BACKUP_timeout, &BACKUP_block_info, quantum,                           \
                     &BACKUP_complete, grid_dim), "initialize");                                          \
    if(ret != FTI_SCES)                                                                                   \
    {                                                                                                     \
      FTI_Print("Running kernel without interrupts", FTI_WARN);                                           \
      kernel_name<<<grid_dim, block_dim, ns, s>>>(NULL, NULL, ## __VA_ARGS__);                            \
    }                                                                                                     \
    else                                                                                                  \
    {                                                                                                     \
      size_t count = 0;                                                                                   \
      char str[FTI_BUFS];                                                                                 \
      while(!BACKUP_complete)                                                                             \
      {                                                                                                   \
        /*BACKUP_dbug_println(); */                                                                       \
        sprintf(str, "%s interrupts = %d", #kernel_name, count);                                          \
        FTI_Print(str, BACKUP_DBUG);                                                                      \
        kernel_name<<<grid_dim, block_dim, ns, s>>>(BACKUP_timeout, BACKUP_block_info, ## __VA_ARGS__);   \
        ret = FTI_BACKUP_monitor(&BACKUP_complete);                                                       \
        if(ret != FTI_SCES)                                                                               \
        {                                                                                                 \
          BACKUP_print("Monitoring of kernel execution failed", FTI_EROR);                                \
        }                                                                                                 \
        count = count + 1;                                                                                \
      }                                                                                                   \
      FTI_BACKUP_cleanup(#kernel_name);                                                                   \
    }                                                                                                     \
}while(0)

#define FTI_KERNEL_DEF(kernel_name, ...)                                                                  \
  kernel_name(volatile unsigned int *timeout, backup_t *is_block_executed, ## __VA_ARGS__)

#define FTI_CONTINUE()                                                                                    \
do{                                                                                                       \
  /* These can be NULL if BACKUP_config is not called prior to kernel launch */                           \
  if(timeout != NULL && is_block_executed != NULL)                                                        \
  {                                                                                                       \
    __shared__ unsigned int block_time_out;                                                               \
    unsigned long long int bid = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.z * blockIdx.z);          \
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)                                         \
    {                                                                                                     \
       block_time_out = *timeout;                                                                         \
    }                                                                                                     \
                                                                                                          \
    if(is_block_executed[bid] == 1)                                                                       \
    {                                                                                                     \
      return;                                                                                             \
    }                                                                                                     \
    __syncthreads();                                                                                      \
                                                                                                          \
    if(block_time_out == 1 && is_block_executed[bid] == 0)                                                \
    {                                                                                                     \
      return;                                                                                             \
    }                                                                                                     \
    is_block_executed[bid] = 1;                                                                           \
  }                                                                                                       \
}while(0)

int BACKUP_complete;
volatile unsigned int *BACKUP_timeout;
typedef unsigned short int backup_t;
backup_t *BACKUP_block_info; /* Initialized and then passed at kernel launch */
int FTI_BACKUP_init(volatile unsigned int **timeout, backup_t **b_info, double q, int *complete, dim3 num_blocks);
int FTI_BACKUP_monitor(int *complete);
void FTI_BACKUP_cleanup(const char *kernel_name);

#endif /* _KERNEL_INTERRUPT_H */
