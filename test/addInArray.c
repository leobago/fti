/**
 *  @file   addInArray.c
 *  @author Karol Sierocinski (ksiero@man.poznan.pl)
 *  @date   Feburary, 2017
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

#define ITERATIONS 120          //iterations for every level
#define ITER_CHECK 10           //every ITER_CHECK iterations make checkpoint
#define ITER_STOP 63            //stop work after ITER_STOP iterations

#define WORK_DONE 0
#define CHECKPOINT_FAILED 1
#define RECOVERY_FAILED 2

#define VERIFY_SUCCESS 0
#define VERIFY_FAILED 1

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
int do_work(int* array, int world_rank, int world_size, int checkpoint_level, int fail)
{
    int res, number = world_rank;
    int i = 0;
    //adding variables to protect
    FTI_Protect(1, &i, 1, FTI_INTG);
    FTI_Protect(2, &number, 1, FTI_INTG);

    //checking if this is recovery run
    if (FTI_Status() != 0 && fail == 0) {
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
    if (world_rank == 0) {
        printf("Starting work at i = %d.\n", i);
    }
    for (; i < ITERATIONS; i++) {
        //checkpoints after every ITER_CHECK iterations
        if (i%ITER_CHECK == 0) {
            res = FTI_Checkpoint(i/ITER_CHECK + 1, checkpoint_level);
            if (res != FTI_DONE) {
                printf("%d: FTI_Checkpoint returned %d.\n", world_rank, res);
                return CHECKPOINT_FAILED;
            }
        }
        MPI_Allgather(&number, 1, MPI_INT, array, 1, MPI_INT, FTI_COMM_WORLD);

        //stoping after ITER_STOP iterations
        if(fail && i >= ITER_STOP) {
            if (world_rank == 0) {
                printf("Work stopped at i = %d.\n", ITER_STOP);
            }
            break;
        }
        number += 1;
    }
    return WORK_DONE;
}


int init(char** argv, int* checkpoint_level, int* fail)
{
    int rtn = 0;    //return value
    if (argv[1] == NULL) {
        printf("Missing first parameter (config file).\n");
        rtn = 1;
    }
    if (argv[2] == NULL) {
        printf("Missing second parameter (checkpoint level).\n");
        rtn = 1;
    }
    else {
        *checkpoint_level = atoi(argv[2]);
    }
    if (argv[3] == NULL) {
        printf("Missing third parameter (if fail).\n");
        rtn = 1;
    }
    else {
        *fail = atoi(argv[3]);
    }
    return rtn;
}

int verify(int* array, int world_size) {
    int i;
    for (i = 0; i < world_size; i++) {
        if (array[i] != ITERATIONS + (i-1)) {
            printf("array[%d] = %d, should be %d.\n", i, array[i], ITERATIONS + (i-1));
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
int main(int argc, char** argv)
{
    int checkpoint_level, fail;
    if (init(argv, &checkpoint_level, &fail)) return 0;   //verify args
    MPI_Init(&argc, &argv);
    FTI_Init(argv[1], MPI_COMM_WORLD);
    int world_rank, world_size; //FTI_COMM rank and size
    MPI_Comm_rank(FTI_COMM_WORLD, &world_rank);
    MPI_Comm_size(FTI_COMM_WORLD, &world_size);

    int *array = (int*) malloc (sizeof(int)*world_size);

    int rtn = do_work(array, world_rank, world_size, checkpoint_level, fail);

    if (world_rank == 0 && rtn == 0 && !fail) {               //verify result
        rtn = verify(array, world_size);
        if (rtn != VERIFY_SUCCESS) {
            printf("Failure.\n");
        }
        else {
            printf("Success.\n");
        }
    }
    free(array);
    FTI_Finalize();
    MPI_Finalize();
    return rtn;
}
