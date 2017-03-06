/**
 *  @file   addInArray.c
 *  @author Karol Sierocinski (ksiero@man.poznan.pl)
 *  @date   Feburary, 2017
 *  @brief  FTI testing program.
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

#define ITERATIONS 100  //iterations for every level
#define ITER_CHECK 10   //every ITER_CHECK iterations make checkpoint
#define ITER_STOP 54    //simulate failure after ITER_STOP iterations

#define WORK_DONE 0
#define WORK_STOPED 1
#define CHECKPOINT_FAILED 2
#define RECOVERY_FAILED 3

#define VERIFY_SUCCESS 0
#define VERIFY_FAILED 1

/*-------------------------------------------------------------------------*/
/**
    @brief      Do work to makes checkpoints
    @param      array       Pointer to array, length == app. proc.
    @param      fail        True if stop after ITER_STOP, false if resuming work
    @return     integer     WORK_DONE if successful.
 **/
/*-------------------------------------------------------------------------*/
int do_work(int* array, int world_rank, int world_size, int checkpoint_level, int fail) {
    int res;
    int number = array[world_rank];
    int i = 0;
    FTI_Protect(1, &i, 1, FTI_INTG);
    FTI_Protect(2, &number, 1, FTI_INTG);
    if (FTI_Status() != 0 && fail == 0)
    {
        res = FTI_Recover();
        if (res != 0) {
            printf("Res from FTI_Recover = %d\n", res);
            return RECOVERY_FAILED;
        }
    }
    if (fail == 0 && i != (ITER_STOP - ITER_STOP%ITER_CHECK)) {
        return RECOVERY_FAILED;
    }
    //printf("Starting work (i = %d).\n", i);
    for (; i < ITERATIONS; i++) {
        if (i%ITER_CHECK == 0) {
            res = FTI_Checkpoint(i/ITER_CHECK, checkpoint_level);
            if (res != FTI_DONE) {
                printf("Res from FTI_Checkpoint = %d\n", res);
                return CHECKPOINT_FAILED;
            }
            //else printf("Checkpoint made (L%d, i = %d)\n", checkpoint_level, i);
        }
        if(fail && i >= ITER_STOP) return WORK_STOPED;
        MPI_Allgather(&number, 1, MPI_INT, array, 1, MPI_INT, FTI_COMM_WORLD);
        number += 1;
    }
    return WORK_DONE;
}


int init(char** argv, char* cnfgFile, int* checkpoint_level, int* fail) {
    int rtn = 0;    //return value
    if (argv[1] == NULL) {
        printf("Missing first parameter (config file).\n");
        rtn = 1;
    } else {
        cnfgFile = argv[1];
        //printf("Config file: %s\n", cnfgFile);
    }
    if (argv[2] == NULL) {
        printf("Missing second parameter (checkpoint level).\n");
        rtn = 1;
    } else {
        *checkpoint_level = atoi(argv[2]);
        //printf("Checkpoint level: %d\n", *checkpoint_level);
    }
    if (argv[3] == NULL) {
        printf("Missing third parameter (if fail).\n");
        rtn = 1;
    } else {
        *fail = atoi(argv[3]);
        //printf("Fail: %d\n", *fail);
    }
    return rtn;
}

int verify(int* array, int world_size) {
    int i;
    for (i = 0; i < world_size; i++) {
        //printf("%d = %d\n", array[i], ITERATIONS + (i-1));
        if (array[i] != ITERATIONS + (i-1)) return VERIFY_FAILED;
    }
    return VERIFY_SUCCESS;
}

/*
    Prints:
        0 if everything is OK
        1 if calculation error
        2 if checkpoint failed
        3 if recovery failed
*/
int main(int argc, char** argv){

    char *cnfgFile;
    int checkpoint_level, fail;
    if (init(argv, cnfgFile, &checkpoint_level, &fail)) return 0;   //verify args

    MPI_Init(&argc, &argv);
    FTI_Init(argv[1], MPI_COMM_WORLD);
    int world_rank, world_size;
    MPI_Comm_rank(FTI_COMM_WORLD, &world_rank);
    MPI_Comm_size(FTI_COMM_WORLD, &world_size);

    int *array = (int*) malloc (sizeof(int)*world_size);
    array[world_rank] = world_rank;

    MPI_Barrier(FTI_COMM_WORLD);
    int res = do_work(array, world_rank, world_size, checkpoint_level, fail);
    int rtn = 0; //return value
    switch(res){
        case WORK_DONE:
            if (world_rank == 0) {               //verify result
		printf("All work done. Verifying result...\n");
                res = verify(array, world_size);
                if (res != VERIFY_SUCCESS) {
		    rtn = 1;
		    printf("FAILURE.\n");
                } else {
		    printf("SUCCESS.\n");
		}
            }
            break;
        case WORK_STOPED:
            if (world_rank == 0) {
            	printf("Work stopped at i = %d\n", ITER_STOP);              
            }
            break;
        case CHECKPOINT_FAILED:
            printf("Checkpoint failed!\n");
            if (world_rank == 0) rtn = 1;
            break;
        case RECOVERY_FAILED:
            printf("Recovery failed!\n");
            if (world_rank == 0) rtn = 1;
            break;
    }
    free(array);
    int allRtn = 0;
    MPI_Allreduce(&rtn, &allRtn, 1, MPI_INT, MPI_SUM, FTI_COMM_WORLD);
    FTI_Finalize();
    MPI_Finalize();
    return allRtn;
}