/**
 *  @file   diffSizes.c
 *  @author Karol Sierocinski (ksiero@man.poznan.pl)
 *  @date   March, 2017
 *  @brief  FTI testing program.
 *
 *  Testing FTI_Init, FTI_Checkpoint, FTI_Status, FTI_Recover, FTI_Finalize,
 *  saving last checkpoint to PFS
 *
 *  Every process in every iteration expand their array and set value to each
 *  index and make checkpoint. Even ranks have 3 times longer array than odd ranks.
 *  After ITERATIONS iterations every process send their array to rank 0 process.
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

#define ITERATIONS 101          //iterations
#define ITER_CHECK 10           //every ITER_CHECK iterations make checkpoint
#define ITER_STOP 54            //stop work after ITER_STOP iterations

#define WORK_DONE 0
#define VERIFY_FAILED 1
#define WORK_STOPED 2
#define CHECKPOINT_FAILED 3
#define RECOVERY_FAILED 4

#define VERIFY_SUCCESS 0


#define INIT_SIZE 50   //multiplied by world_size*2 gives origin array size


int verify(long* array, int world_size) {
    int i;
    int size = world_size * (ITERATIONS + INIT_SIZE * 2);
    for (i = 0; i < size - world_size; i++) {
        if (array[i] != size) {
            printf("array[%d] = %ld, should be %d.\n", i, array[i], size);
            return VERIFY_FAILED;
        }
    }
    return VERIFY_SUCCESS;
}

void getPart(int* myPart, int* offset, int size, int world_rank, int world_size) {
    int part = size / world_size / 2;
    *offset = world_rank * part * 2;
    //even rank processes get 3 part of work; odd get 1 part
    if (world_rank % 2 == 0) {
        *myPart = part * 3;
    } else {
        *myPart = part;
        *offset += part;
    }
}

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
    int size = world_size * INIT_SIZE * 2;
    int offset, myPart;
    getPart(&myPart, &offset, size, world_rank, world_size);

    FTI_Protect(0, &i, 1, FTI_INTG);
    FTI_Protect(1, &size, 1, FTI_INTG);
    long* buf = malloc (sizeof(long) * myPart);
    for (j = 0; j < myPart; j++) {
            buf[j] = size;
    }
    FTI_Protect(2, buf, myPart, FTI_LONG);
    //checking if this is recovery run
    if (FTI_Status() != 0 && fail == 0)
    {
        res = FTI_Recover();
        if (res != 0) {
            printf("%d: Recovery failed! FTI_Recover returned %d.\n", world_rank, res);
            return RECOVERY_FAILED;
        } else {
            getPart(&myPart, &offset, size, world_rank, world_size);
            buf = realloc (buf, sizeof(long) * myPart);
            for (j = 0; j < myPart; j++) {
                if (j < myPart/2 - 1)
                    buf[myPart - j - 1] = buf[j];
            }
        }
    }
    //if recovery, but recover values don't match
    if (fail == 0) {
        if (i != (ITER_STOP - ITER_STOP % ITER_CHECK)){
            printf("%d: i = %d, should be %d\n", world_rank, i, (ITER_STOP - ITER_STOP % ITER_CHECK));
            return RECOVERY_FAILED;
        }
        if (size != world_size * (i + INIT_SIZE * 2)) {
            printf("%d: size = %d, should be %d\n", world_rank, size, world_size * (i + INIT_SIZE * 2));
            return RECOVERY_FAILED;
        }
        getPart(&myPart, &offset, size, world_rank, world_size);
        buf = realloc (buf, sizeof(long) * myPart);
        for (j = 0; j < myPart; j++) {
            if (buf[j] != size) {
                printf("%d: buf[%d] = %ld, should be %d\n", world_rank, j, buf[j], size);
                return RECOVERY_FAILED;
            }
        }
    }
    if(world_rank == 0)
        printf("Starting work at i = %d.\n", i);
    for (; i < ITERATIONS; i++) {
        //checkpoints after every ITER_CHECK iterations
        if (i%ITER_CHECK == 0) {
            res = FTI_Checkpoint(i/ITER_CHECK + 1, checkpoint_level);
            if (res != FTI_DONE) {
                printf("%d: Checkpoint failed! FTI_Checkpoint returned %d.\n", world_rank, res);
                return CHECKPOINT_FAILED;
            }
        }
        long tempValue = buf[myPart - 1];
        size += world_size;
        getPart(&myPart, &offset, size, world_rank, world_size);
        buf = realloc (buf, sizeof(long) * myPart);
        for (j = 0; j < myPart; j++) {
                buf[j] = tempValue + world_size;
        }
        FTI_Protect(2, buf, myPart, FTI_LONG);
        //stoping after ITER_STOP iterations
        if(fail && i >= ITER_STOP){
            if (world_rank == 0) {
            	printf("Work stopped at i = %d.\n", ITER_STOP);
            }
            return WORK_DONE;
        }
    }
    long* array = malloc (sizeof(long) * size);
    if (world_rank == 0) {
        int part = size / world_size / 2;
        for (j = 1; j < world_size; j++) {
            if (j%2 == 0) {
                MPI_Recv(array + (j * part * 2), part * 3, MPI_LONG, j, 0, FTI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else {
                MPI_Recv(array + (j * part * 2) + part, part, MPI_LONG, j, 0, FTI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        memcpy(array, buf, sizeof(long) * myPart);
        return verify(array, world_size);
    } else {
        MPI_Send(buf, myPart, MPI_LONG, 0, 0, FTI_COMM_WORLD);
    }

    free(buf);
    return WORK_DONE;
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
    if (world_rank == 0 && !fail && !rtn) {
        printf("Success.\n");
    }
    FTI_Finalize();
    MPI_Finalize();
    return rtn;
}
