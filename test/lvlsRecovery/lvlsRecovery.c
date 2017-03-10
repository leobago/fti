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
    @param      array       Pointer to array, length == app. proc.
    @param      fail        True if stop after ITER_STOP, false if resuming work
    @return     integer     WORK_DONE if successful.
 **/
/*-------------------------------------------------------------------------*/
int do_work(double** matrix, int world_rank, int world_size, int checkpoint_level, int fail) {
    int res;
    int i, j, k;
    for (i = 0; i < MATRIX_SIZE; i++) {
        for (j = 0; j < MATRIX_SIZE; j++) {
            if (i == j) matrix[i][j] = 1.0;
            else matrix[i][j] = 0.0;
        }
    }
    i = 0;
    FTI_Protect(1, &i, 1, FTI_INTG);
    for (j = 0; j < MATRIX_SIZE; j++) {
        FTI_Protect(j+2, matrix[j], MATRIX_SIZE, FTI_DBLE);
    }

    if (FTI_Status() != 0 && fail == 0)
    {
        res = FTI_Recover();
        if (res != 0) {
            printf("%d: FTI_Recover returned %d.\n", world_rank, res);
            return RECOVERY_FAILED;
        }
    }
    if (fail == 0 && i != (ITER_STOP - ITER_STOP%ITER_CHECK)) {
        return RECOVERY_FAILED;
    }
    printf("%d: Starting work at i = %d.\n", world_rank, i);
    for (; i < ITERATIONS; i++) {
        if (i%ITER_CHECK == 0) {
            res = FTI_Checkpoint(i/ITER_CHECK + 1, checkpoint_level);
            if (res != FTI_DONE) {
                printf("%d: FTI_Checkpoint returned %d.\n", world_rank, res);
                return CHECKPOINT_FAILED;
            }
            //else printf("Checkpoint made (L%d, i = %d)\n", checkpoint_level, i);
        }

	    if(fail && i >= ITER_STOP) {
            return WORK_STOPED;
         }
        for (j = 0; j < MATRIX_SIZE; j++) {
            matrix[j][j] += 1.0;
        }
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

int verify(double** matrix) {
    int i, j;
    for (i = 0; i < MATRIX_SIZE; i++) {
        for (j = 0; j < MATRIX_SIZE; j++) {
            if (i == j && matrix[i][j] != (double)(ITERATIONS + 1)) return VERIFY_FAILED;
            else if (i != j && matrix[i][j] != 0.0) return VERIFY_FAILED;
        }
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
    int global_world_rank, global_world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &global_world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &global_world_size);

    FTI_Init(argv[1], MPI_COMM_WORLD);
    int world_rank, world_size;
    MPI_Comm_rank(FTI_COMM_WORLD, &world_rank);
    MPI_Comm_size(FTI_COMM_WORLD, &world_size);

    double** matrix = (double**) malloc (sizeof(double*) * MATRIX_SIZE);
    int i;
    for (i = 0; i < MATRIX_SIZE; i++) {
        matrix[i] = (double*) malloc (sizeof(double) * MATRIX_SIZE);
    }

    MPI_Barrier(FTI_COMM_WORLD);
    int res = do_work(matrix, world_rank, world_size, checkpoint_level, fail);
    int rtn = 0; //return value
    switch(res){
        case WORK_DONE:
            if (world_rank == 0) {               //verify result
                printf("All work done. Verifying result...\n");
                res = verify(matrix);
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
            	printf("Work stopped at i = %d.\n", ITER_STOP);
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
    MPI_Barrier(FTI_COMM_WORLD);
    for (i = 0; i < MATRIX_SIZE; i++) {
        free(matrix[i]);
    }
    free(matrix);
    dictionary* ini = iniparser_load("config.fti");

    if (checkpoint_level != 1) {
        int isInline = -1;
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
        dictionary* ini = iniparser_load("config.fti");
        int nodeSize = (int)iniparser_getint(ini, "Basic:node_size", -1);
        if (isInline == 0)     MPI_Recv(&res, 1, MPI_INT, global_world_rank - (global_world_rank%nodeSize) , 2612, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(FTI_COMM_WORLD);
    if (world_rank == 0) {
        int mpi_res = MPI_Abort(MPI_COMM_WORLD, 0);
    }
    //FTI_Finalize();
    MPI_Finalize();
    return rtn;
}
