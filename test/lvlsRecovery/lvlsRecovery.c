/**
 *  @file   lvlsRecovery.c
 *  @author Karol Sierocinski (ksiero@man.poznan.pl)
 *  @date   March, 2017
 *  @brief  FTI testing program.
 *
 *  Testing recovery from all checkpoints levels.
 *
 *  First execution this program should be with fail flag = 1, because
 *  then FTI saves checkpoint and program stops after ITER_STOP iteration.
 *  Second execution must be with the same #defines and flag = 0 to
 *  properly recover data.
 */

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <fti.h>

#include "../../deps/iniparser/iniparser.h"
#include "../../deps/iniparser/dictionary.h"

#define ITERATIONS 100  //iterations for every level
#define ITER_CHECK 10   //every ITER_CHECK iterations make checkpoint
#define ITER_STOP 54    //simulate failure after ITER_STOP iterations

#define WORK_DONE 0
#define WORK_STOPED 1
#define CHECKPOINT_FAILED 2
#define RECOVERY_FAILED 3

#define VERIFY_SUCCESS 0
#define VERIFY_FAILED 1

#define MATRIX_SIZE 250

/*-------------------------------------------------------------------------*/
/**
    @brief      Do work to makes checkpoints
    @param      matrix              Pointer to array, length == app. proc.
    @param      world_rank          FTI_COMM rank
    @param      world_size          FTI_COMM size
    @param      checkpoint_level    Checkpont level to all checkpoints
    @param      fail                True if stop after ITER_STOP, false if resuming work
    @return     integer             WORK_DONE if successful.
 **/
/*-------------------------------------------------------------------------*/
int do_work(double** matrix, int world_rank, int world_size, int checkpoint_level, int fail) {
    int res;
    int i, j, k;
    for (i = 0; i < MATRIX_SIZE; i++) {
        for (j = 0; j < MATRIX_SIZE; j++) {
            matrix[i][j] = 0.0;
        }
    }
    i = 0;
    FTI_Protect(1, &i, 1, FTI_INTG);
    for (j = 0; j < MATRIX_SIZE; j++) {
        FTI_Protect(j+2, matrix[j], MATRIX_SIZE, FTI_DBLE);
    }
    //checking if this is recovery run
    if (FTI_Status() != 0 && fail == 0)
    {
        res = FTI_Recover();
        if (res != 0) {
            printf("%d: Recovery failed! FTI_Recover returned %d.\n", world_rank, res);
            return RECOVERY_FAILED;
        }
    }
    //if recovery, but recover values don't match
    if (!fail) {
        if (i != ITER_STOP - ITER_STOP % ITER_CHECK)
            return RECOVERY_FAILED;
        for (j = 0; j < MATRIX_SIZE; j++) {
            for (k = 0; k < MATRIX_SIZE; k++) {
                if (j == k && matrix[j][k] != i) {
                    printf("%d: Did not recovered properly.\n", world_rank);
                    return RECOVERY_FAILED;
                }
                else if (j != k && matrix[j][k] != 0.0) {
                    printf("%d: Did not recovered properly.\n", world_rank);
                    return RECOVERY_FAILED;
                }
            }
        }
    }
    if (world_rank == 0)
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

        //stoping after ITER_STOP iterations
	    if(fail && i >= ITER_STOP) {
            if (world_rank == 0)
                printf("Work stopped at i = %d.\n", ITER_STOP);
            break;
        }
        for (j = 0; j < MATRIX_SIZE; j++) {
            matrix[j][j] += 1.0;
        }
    }
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

int verify(double** matrix, int world_rank) {
    int i, j;
    for (i = 0; i < MATRIX_SIZE; i++) {
        for (j = 0; j < MATRIX_SIZE; j++) {
            if (i == j && matrix[i][j] != ITERATIONS) {
                printf("%d: matrix[%d][%d] = %f, should be %d\n", world_rank, i, j, matrix[i][j], ITERATIONS);
                return VERIFY_FAILED;
            }
            else if (i != j && matrix[i][j] != 0.0) {
                printf("%d: matrix[%d][%d] = %f, should be 0.0\n", world_rank, i, j, matrix[i][j]);
                return VERIFY_FAILED;
            }
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
    if (init(argv, &checkpoint_level, &fail)) return 0;     //verify args

    MPI_Init(&argc, &argv);
    int global_world_rank;                                  //MPI_COMM rank
    MPI_Comm_rank(MPI_COMM_WORLD, &global_world_rank);

    FTI_Init(argv[1], MPI_COMM_WORLD);
    int world_rank, world_size;                             //FTI_COMM rank & size
    MPI_Comm_rank(FTI_COMM_WORLD, &world_rank);
    MPI_Comm_size(FTI_COMM_WORLD, &world_size);

    double** matrix = (double**) malloc (sizeof(double*) * MATRIX_SIZE);
    int i;
    for (i = 0; i < MATRIX_SIZE; i++) {
        matrix[i] = (double*) malloc (sizeof(double) * MATRIX_SIZE);
    }

    int rtn = do_work(matrix, world_rank, world_size, checkpoint_level, fail);

    if (!fail) {
        if (world_rank == 0)
            printf("All work done. Verifying result...\t");
        rtn = verify(matrix, world_rank);
        if (rtn != VERIFY_SUCCESS) {
            printf("%d: Failure.\n", world_rank);
        } else {
            if (world_rank == 0)
                printf("Success.\n");
        }
    }
    MPI_Barrier(FTI_COMM_WORLD);
    for (i = 0; i < MATRIX_SIZE; i++) {
        free(matrix[i]);
    }
    free(matrix);
    dictionary* ini = iniparser_load("config.fti");
    int heads = (int)iniparser_getint(ini, "Basic:head", -1);
    int nodeSize = (int)iniparser_getint(ini, "Basic:node_size", -1);
    int res;
    if (checkpoint_level != 1) {
        int isInline = -1;
        int heads = (int)iniparser_getint(ini, "Basic:head", -1);
        switch (checkpoint_level) {
            case 2:
                isInline = (int)iniparser_getint(ini, "Basic:inline_l2", 1);
            break;
            case 3:
                isInline = (int)iniparser_getint(ini, "Basic:inline_l3", 1);
                break;
            case 4:
                isInline = (int)iniparser_getint(ini, "Basic:inline_l4", 1);
                break;
        }
        if (isInline == 0) {
            //waiting untill head do Post-checkpointing
            MPI_Recv(&res, 1, MPI_INT, global_world_rank - (global_world_rank%nodeSize) , 2612, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

    }
    if (heads > 0) {
        res = FTI_ENDW;
        //sending END WORK to head to stop listening
        MPI_Send(&res, 1, MPI_INT, global_world_rank - (global_world_rank%nodeSize), 2612, MPI_COMM_WORLD);
        //Barrier needed for heads (look FTI_Finalize() in api.c)
        MPI_Barrier(MPI_COMM_WORLD);
    }
    //There is no FTI_Finalize(), because want to recover also from L1, L2, L3
    MPI_Finalize();
    return rtn;
}
