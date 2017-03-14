/**
 *  @file   diffSizes.c
 *  @author Karol Sierocinski (ksiero@man.poznan.pl)
 *  @date   March, 2017
 *  @brief  FTI testing program.
 *
 *  Testing FTI_Init, FTI_Checkpoint, FTI_Status, FTI_Recover, FTI_Finalize,
 *  saving last checkpoint to PFS
 *
 *  Program adds number in array, does MPI_Allgather each iteration and checkpoint
 *  every ITER_CHECK interations with level passed in argv, but recovery is always
 *  from L4, because of FTI_Finalize() call.
 *
 *  First execution this program should be with fail flag = 1, because
 *  then FTI saves checkpoint and program stops after ITER_STOP iteration.
 *  Second execution must be with the same #defines and flag = 0 to
 *  properly recover data. It is important that FTI config file got
 *  keep_last_ckpt = 1.
 */

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <fti.h>
#include <string.h>

#define ITERATIONS 97          //iterations for every level
#define ITER_CHECK 10           //every ITER_CHECK iterations make checkpoint
#define ITER_STOP 54            //stop work after ITER_STOP iterations

#define WORK_DONE 0
#define VERIFY_FAILED 1
#define WORK_STOPED 2
#define CHECKPOINT_FAILED 3
#define RECOVERY_FAILED 4

#define VERIFY_SUCCESS 0


#define INIT_SIZE 100   //multiplied by world_size gives origin array size

int verify(long* array, int world_size);
/*-------------------------------------------------------------------------*/
/**
    @brief      Do work to makes checkpoints
    @param      array               Pointer to array, length == app. proc.
    @param      world_rank          FTI_COMM rank
    @param      world_size          FTI_COMM size
    @param      checkpoint_level    Checkpont level to all checkpoints
    @param      fail                True if stop after ITER_STOP, false if resuming work
    @return     integer             WORK_DONE if successful.
 **/
/*-------------------------------------------------------------------------*/
int do_work(int world_rank, int world_size, int checkpoint_level, int fail) {
    int res;
    int i = 0, j;
    int size = world_size*INIT_SIZE;
    int part = size/world_size;
    int offset = world_rank*part;
    long* array = malloc (sizeof(long) * size);
    FTI_Protect(0, &i, 1, FTI_INTG);
    FTI_Protect(1, array, size, FTI_LONG);
    FTI_Protect(2, &size, 1, FTI_INTG);

    //checking if this is recovery run
    if (FTI_Status() != 0 && fail == 0)
    {
        res = FTI_Recover();
        if (res != 0) {
            printf("%d: FTI_Recover returned %d.\n", world_rank, res);
            return RECOVERY_FAILED;
        }
    }
    //if recovery, but recover values don't match
    if (fail == 0 && i != (ITER_STOP - ITER_STOP%ITER_CHECK)) {
        return RECOVERY_FAILED;
    }
    printf("%d: Starting work at i = %d.\n", world_rank, i);
    for (; i < ITERATIONS; i++) {
        //printf("%d: i = %d\n", world_rank, i);
        //checkpoints after every ITER_CHECK iterations
        if (i%ITER_CHECK == 0) {
            res = FTI_Checkpoint(i/ITER_CHECK + 1, checkpoint_level);
            if (res != FTI_DONE) {
                printf("%d: FTI_Checkpoint returned %d.\n", world_rank, res);
                return CHECKPOINT_FAILED;
            }
        }
        part = size/world_size;
        array = realloc (array, sizeof(long) * size);
        long* buf = malloc (sizeof(long) * part);
        for (j = 0; j < part; j++) {
                buf[j] = size;
        }
        MPI_Allgather(buf, part, MPI_LONG, array, part, MPI_LONG, FTI_COMM_WORLD);
        FTI_Protect(1, array, size, FTI_LONG);

        free(buf);
        //stoping after ITER_STOP iterations
        if(fail && i >= ITER_STOP){
            return WORK_STOPED;
        }
        size += world_size;
    }
    int rtn = verify(array, world_size);
    free(array);
    return rtn;
}


int init(char** argv, int* checkpoint_level, int* fail) {
    int rtn = 0;    //return value
    if (argv[1] == NULL) {
        printf("Missing first parameter (config file).\n");
        rtn = 1;
    }
    if (argv[2] == NULL) {
        printf("Missing second parameter (checkpoint level).\n");
        rtn = 1;
    } else {
        *checkpoint_level = atoi(argv[2]);
    }
    if (argv[3] == NULL) {
        printf("Missing third parameter (if fail).\n");
        rtn = 1;
    } else {
        *fail = atoi(argv[3]);
    }
    return rtn;
}

int verify(long* array, int world_size) {
    int i;
    int size = world_size * (INIT_SIZE + ITERATIONS-2);
    for (i = 0; i < size; i++) {
        if (array[i] != size) {
            printf("array[%d] = %ld, should be %d.\n", i, array[i], size);
            return VERIFY_FAILED;
        }
    }
    return VERIFY_SUCCESS;
}


/*-------------------------------------------------------------------------*/
/**
    @return     integer     0 if successful, 1 otherwise
 **/
/*-------------------------------------------------------------------------*/
int main(int argc, char** argv){
    int checkpoint_level, fail;
    if (init(argv, &checkpoint_level, &fail)) return 0;   //verify args
    MPI_Init(&argc, &argv);
    FTI_Init(argv[1], MPI_COMM_WORLD);
    int world_rank, world_size; //FTI_COMM rank and size
    MPI_Comm_rank(FTI_COMM_WORLD, &world_rank);
    MPI_Comm_size(FTI_COMM_WORLD, &world_size);

    int rtn = do_work(world_rank, world_size, checkpoint_level, fail);
    switch(rtn){
        case WORK_DONE:
            if (world_rank == 0) {
            	printf("Success.\n");
            }
            break;
        case WORK_STOPED:
            if (world_rank == 0) {
            	printf("Work stopped at i = %d.\n", ITER_STOP);
            }
            rtn = 0;
            break;
        case CHECKPOINT_FAILED:
            printf("%d: Checkpoint failed!\n", world_rank);
            if (world_rank == 0) rtn = 1;
            break;
        case RECOVERY_FAILED:
            printf("%d: Recovery failed!\n", world_rank);
            if (world_rank == 0) rtn = 1;
            break;
    }
    FTI_Finalize();
    MPI_Finalize();
    return rtn;
}
