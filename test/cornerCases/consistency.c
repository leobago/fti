#include <stdio.h>
#include <stdlib.h>
#include <fti.h>
#include "../../deps/iniparser/iniparser.h"
#include "../../deps/iniparser/dictionary.h"

#define FTI_ENDW 995

#define ARRAY_SIZE 1024 * 1024
#define DATASET_SIZE (ARRAY_SIZE/4)
#define FIRST array
#define SECOND (array + DATASET_SIZE)
#define THIRD (array + DATASET_SIZE*2)
#define FOURTH (array + DATASET_SIZE*3)

int* array;
int world_rank;
int world_size;
int global_world_rank;
int global_world_size;
int checkpoint_level;
int fail;
int initStatus;

void simulateCrash() {
    MPI_Barrier(FTI_COMM_WORLD);
    dictionary* ini = iniparser_load("config.fti");
    int heads = (int)iniparser_getint(ini, "Basic:head", -1);
    int nodeSize = (int)iniparser_getint(ini, "Basic:node_size", -1);
    int general_tag = (int)iniparser_getint(ini, "Advanced:general_tag", 2612);
    int final_tag = (int)iniparser_getint(ini, "Advanced:final_tag", 3107);
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
            printf("%d: Receiving.\n", world_rank);
            MPI_Recv(&res, 1, MPI_INT, global_world_rank - (global_world_rank%nodeSize) , general_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("%d: Received.\n", world_rank);
        }
    }
    iniparser_freedict(ini);
    printf("%d: Heads = %d.\n", global_world_rank, heads);
    if (heads > 0) {
        printf("%d: Sending end WORK to %d.\n", global_world_rank, global_world_rank - (global_world_rank%nodeSize));
        res = FTI_ENDW;
        //sending END WORK to head to stop listening
        MPI_Send(&res, 1, MPI_INT, global_world_rank - (global_world_rank%nodeSize), final_tag, MPI_COMM_WORLD);
        //Barrier needed for heads (look FTI_Finalize() in api.c)
        printf("%d: END_WORK sent.\n", global_world_rank);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    printf("%d: After 1st barrier.\n", global_world_rank);
    MPI_Barrier(FTI_COMM_WORLD);
    //There is no FTI_Finalize(), because want to recover also from L1, L2, L3
    MPI_Finalize();
    free(array);
    exit(0);
}

void simulateCrashWithoutCkpt() {
    MPI_Barrier(FTI_COMM_WORLD);
    dictionary* ini = iniparser_load("config.fti");
    int heads = (int)iniparser_getint(ini, "Basic:head", -1);
    int nodeSize = (int)iniparser_getint(ini, "Basic:node_size", -1);
    int general_tag = (int)iniparser_getint(ini, "Advanced:general_tag", 2612);
    int final_tag = (int)iniparser_getint(ini, "Advanced:final_tag", 3107);
    int res;
    iniparser_freedict(ini);

    if (heads > 0) {
        printf("%d: Sending end WORK without ckpt.\n", world_rank);
        res = FTI_ENDW;
        //sending END WORK to head to stop listening
        MPI_Send(&res, 1, MPI_INT, global_world_rank - (global_world_rank%nodeSize), final_tag, MPI_COMM_WORLD);
        //Barrier needed for heads (look FTI_Finalize() in api.c)
        printf("%d: END_WORK sent  without ckpt.\n", world_rank);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    printf("%d: After 1st barrier without ckpt.\n", world_rank);
    MPI_Barrier(FTI_COMM_WORLD);
    //There is no FTI_Finalize(), because want to recover also from L1, L2, L3
    MPI_Finalize();
    free(array);
    exit(0);
}

void initArray(int* tab) {
    int i;
    for (i = 0; i < ARRAY_SIZE; i++) {
        tab[i] = (i + world_rank);
    }
}

int checkArray(int* tab) {
    int i;
    for (i = 0; i < ARRAY_SIZE; i++) {
        if (tab[i] != (i + world_rank)) {
            printf("%d: array[%d]: %d != %d\n", world_rank,  i, tab[i], (i + world_rank));
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    MPI_Barrier(FTI_COMM_WORLD);
    if (world_rank == 0) printf("Array values correct.\n");
    return 0;
}

/*  After the restart from any level, we claim to have the same state
    as after the successful checkpoint in the preceeding execution.   */
void initAfterSuccessfulRecovery() {
    if (initStatus != FTI_SCES && fail != 2) {
            printf("Cannot FTI_Init: initStatus = %d\n", initStatus);
            MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (fail == 2) {
        if (checkpoint_level == 1 || checkpoint_level == 4) {
            if (initStatus == FTI_SCES) {
                printf("Shouldn't init on level 1 and 4, but initStatus = %d\n", initStatus);
                MPI_Abort(MPI_COMM_WORLD, 1);
            } else {
                printf("Succes, fail == 2 and cannot recover.\n");
                FTI_Finalize();
                MPI_Finalize();
                exit(0);
            }
        } else {
            if (initStatus != FTI_SCES) {
                printf("Should init on level 2 and 3, but initStatus = %d\n", initStatus);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
    }
}

int afterSuccessfulRecovery() {
    if (fail == 0) {
        FTI_Checkpoint(1, checkpoint_level);
        if (world_rank == 0) printf("Checkpoint done. Simulating crash.\n");
        simulateCrash();
    }

    if (fail == 1) {
        if (world_rank == 0) printf("Trying to recover.\n");
        int res = FTI_Recover();
        if (res != FTI_SCES) {
            printf("%d: Should recover, but res = %d!\n", world_rank, res);
            return 1;
        }
        if (world_rank == 0) printf("Recover done. Simulating crash.\n");
        simulateCrashWithoutCkpt();
    }

    if (fail == 2) {
        if (world_rank == 0) printf("Trying to recover.\n");
        int res = FTI_Recover();
        if (res != FTI_SCES) {
            printf("%d: Should recover, but res = %d!\n", world_rank, res);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (world_rank == 0) printf("Recover done.\n");
        return 0;
    }
}

void initKeep_last_ckpt() {
    if (initStatus != FTI_SCES) {
            printf("Error FTI_Init returned %d in keep_last_ckpt case.\n", initStatus);
            MPI_Finalize();
            MPI_Abort(MPI_COMM_WORLD, 1);
    }
}


/* After the successful restart, we claim to have the situation as after
   the last checkpoint in the preceeding execution.                     */
int keep_last_ckpt() {
    if (fail == 0) {
        FTI_Checkpoint(1, checkpoint_level);
        return 0;
    }
    int res = FTI_Recover();
    if (fail == 1) {
        if (res == FTI_SCES) {
            if (world_rank == 0) printf("Recover done. Simulating crash.\n");
            simulateCrashWithoutCkpt();
        } else {
            if (world_rank == 0) printf("Error: recover failed when fail == 1.\n");
            return 1;
        }
    }
    if (fail == 2) {
        if (res == FTI_SCES) {
            if (world_rank == 0) printf("Recover done.\n");
            return 0;
        } else {
            if (world_rank == 0) printf("Error: recover failed when fail == 2.\n");
            return 1;
        }
    }
}

int initReInit() {
        FTI_Checkpoint(1, checkpoint_level);
        memset(array, 0, sizeof(int) * ARRAY_SIZE);
        return FTI_Recover();
}

/* check for a correct restart without the crash of the application. */
int reInit() {
        int initres = FTI_Init("config2.fti", MPI_COMM_WORLD);
        int* array2 = malloc(sizeof(int)* ARRAY_SIZE);
        
        initArray(array2);
        FTI_Protect(1, array2, ARRAY_SIZE, FTI_INTG);
        FTI_Checkpoint(1, checkpoint_level);

        memset(array2, 0, sizeof(int) * ARRAY_SIZE);

        int rtn = FTI_Recover();

        rtn += checkArray(array2);

        free(array2);
        FTI_Finalize();
        return rtn;
}


int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &global_world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &global_world_rank);

    int testCase;
    if (argc != 4) {
        if (global_world_rank == 0) printf("Argc == %d. Use: testCase(1/2/3/4), level(1/2/3/4), fail(0/1/2)\n", argc);
        MPI_Barrier(MPI_COMM_WORLD);
        return 1;
    } else {
        testCase = atoi(argv[1]);
        checkpoint_level = atoi(argv[2]);
        fail = atoi(argv[3]);
        if (global_world_rank == 0) printf("testCase = %d, ckpt_lvl = %d, fail = %d\n", testCase, checkpoint_level, fail);
    }

    initStatus = FTI_Init("config.fti", MPI_COMM_WORLD);

    //check initStatus
    if (testCase == 1) {
        initAfterSuccessfulRecovery();
    } else if (testCase == 2) {
        initKeep_last_ckpt();
    }
    

    MPI_Comm_size(FTI_COMM_WORLD, &world_size);
    MPI_Comm_rank(FTI_COMM_WORLD, &world_rank);
    array = malloc(sizeof(int) * ARRAY_SIZE);
    if (fail == 0) {
        initArray(array);
    }

    FTI_Protect(1, FIRST, DATASET_SIZE, FTI_INTG);
    FTI_Protect(2, SECOND, DATASET_SIZE, FTI_INTG);
    FTI_Protect(3, THIRD, DATASET_SIZE, FTI_INTG);
    FTI_Protect(4, FOURTH, DATASET_SIZE, FTI_INTG);

    int rtn = 0;
    if (testCase == 1) {
        rtn += afterSuccessfulRecovery();
    } else if (testCase == 2) {
        rtn += keep_last_ckpt();
    } else if (testCase == 3) {
        rtn += initReInit();
    } else if (testCase == 4) {
        rtn += FTI_Init("config.fti", MPI_COMM_WORLD);
    }

    rtn += checkArray(array);
    FTI_Finalize();

    if (testCase == 3) {
        rtn += reInit();
    }

    MPI_Finalize();
    free(array);
    return rtn;
}
