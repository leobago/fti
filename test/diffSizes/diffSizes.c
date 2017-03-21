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
 *  index and make checkpoint. Even ranks have 3 times larger array than odd ranks.
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
#include <dirent.h>
#include <sys/stat.h>

#include "../../deps/iniparser/iniparser.h"
#include "../../deps/iniparser/dictionary.h"

#define ITERATIONS 111          //iterations
#define ITER_CHECK 10           //every ITER_CHECK iterations make checkpoint
#define ITER_STOP 63            //stop work after ITER_STOP iterations

#define WORK_DONE 0
#define VERIFY_FAILED 1
#define CHECKPOINT_FAILED 2
#define RECOVERY_FAILED 3

#define VERIFY_SUCCESS 0

#define INIT_SIZE 53   //multiplied by world_size*2 gives origin array size

/*-------------------------------------------------------------------------*/
/**
    Verifies final result.
 **/
/*-------------------------------------------------------------------------*/
int verify(long* array, int world_size) {
    int i;
    int size = world_size * ((ITERATIONS * 2) + INIT_SIZE * 2);
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
    Updates myPart. Even ranks get 3/4 of work, odd 1/4 of work.
 **/
/*-------------------------------------------------------------------------*/
void getPart(int* myPart, int size, int world_rank, int world_size) {
    int part = size / world_size / 2;
    //even rank processes get 3 part of work; odd get 1 part
    if (world_rank % 2 == 0) {
        *myPart = part * 3;
    } else {
        *myPart = part;
    }
}

/*-------------------------------------------------------------------------*/
/**
    @brief      Do work to makes checkpoints
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
    int size = (world_size * INIT_SIZE * 2);
    int originSize = size;
    int addToSize = (world_size * 2);
    int myPart;
    getPart(&myPart, size, world_rank, world_size);

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
        /*
        when we add FTI_Realloc();
        buf = FTI_Realloc(2, buf);
        if (buf == NULL) {
            printf("%d: Reallocation failed!\n", world_rank);
            return RECOVERY_FAILED;
        }*/

        //need to call FTI_Recover twice to get all data
        res = FTI_Recover();
        getPart(&myPart, size, world_rank, world_size);
        buf = realloc (buf, sizeof(long) * myPart);

        FTI_Protect(2, buf, myPart, FTI_LONG);
        res = FTI_Recover();
        if (res != 0) {
            printf("%d: Recovery failed! FTI_Recover returned %d.\n", world_rank, res);
            return RECOVERY_FAILED;
        }
    }
    //if recovery, but recover values don't match
    if (fail == 0) {
        int expectedI = ITER_STOP - ITER_STOP % ITER_CHECK;
        if (i != expectedI){
            printf("%d: i = %d, should be %d\n", world_rank, i, expectedI);
            return RECOVERY_FAILED;
        }
        int expectedSize = originSize + (i * addToSize);
        if (size != expectedSize) {
            printf("%d: size = %d, should be %d\n", world_rank, size, expectedSize);
            return RECOVERY_FAILED;
        }
        getPart(&myPart, size, world_rank, world_size);
        int recoverySize = 2 * sizeof(int); //i and size
        /* when we add FTI_Realloc();
        recoverySize += 3 * sizeof(long); //counts
        */
        for (j = 0; j < myPart; j++) {
            if (buf[j] != size) {
                printf("%d: Recovery size = %d\n", world_rank, recoverySize);
                printf("%d: buf[%d] = %ld, should be %d\n", world_rank, j, buf[j], size);
                return RECOVERY_FAILED;
            }
            recoverySize += sizeof(long);
        }
        printf("%d: Recovery size = %d\n", world_rank, recoverySize);
    }
    if(world_rank == 0)
        printf("Starting work at i = %d.\n", i);

    for (; i < ITERATIONS; i++) {
        //checkpoint after every ITER_CHECK iterations
        if (i%ITER_CHECK == 0) {
            FTI_Protect(2, buf, myPart, FTI_LONG);
            res = FTI_Checkpoint(i/ITER_CHECK + 1, checkpoint_level);
            if (res != FTI_DONE) {
                printf("%d: Checkpoint failed! FTI_Checkpoint returned %d.\n", world_rank, res);
                return CHECKPOINT_FAILED;
            }
        }

        size += addToSize;                      //enlarge size
        long tempValue = buf[myPart - 1];
        getPart(&myPart, size, world_rank, world_size);    //update myPart
        buf = realloc (buf, sizeof(long) * myPart);
        for (j = 0; j < myPart; j++) {
                buf[j] = tempValue + addToSize;
        }

        //stoping after ITER_STOP iterations
        if(fail && i >= ITER_STOP){
            if (world_rank == 0)
            	printf("Work stopped at i = %d.\n", ITER_STOP);
            return WORK_DONE;
        }
    }

    if (world_rank == 0) {
        //gather all part and verify result
        long* array = malloc (sizeof(long) * size);
        int part = size / world_size / 2;
        for (j = 1; j < world_size; j++) {
            if (j%2 == 0) {
                MPI_Recv(array + (j * part * 2), part * 3, MPI_LONG, j, 0, FTI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else {
                MPI_Recv(array + (j * part * 2) + part, part, MPI_LONG, j, 0, FTI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        memcpy(array, buf, sizeof(long) * myPart);
        int rtn = verify(array, world_size);
        free(array);
        return rtn;
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


int checkFileSizes(int* mpi_ranks, int world_size, int fail){
    dictionary* ini = iniparser_load("config.fti");
    char* exec_id = malloc (sizeof(char) * 256);
    exec_id = iniparser_getstring(ini, "Restart:exec_id", NULL);
    char str[256];
    char path[256];

    DIR *dir;
    struct dirent *ent;
    sprintf(path, "./Global/%s/l4", exec_id);
    free(exec_id);

    if ((dir = opendir (path)) != NULL) {
      while ((ent = readdir (dir)) != NULL) {
        if (strcmp(ent->d_name , ".") && strcmp(ent->d_name, "..")) {
            sprintf(str, "%s/%s", path, ent->d_name);

            FILE* f = fopen(str, "rb");
            fseek(f, 0L, SEEK_END);
            int fileSize = ftell(f);

            //get rank from file name
            int i, id, rank;
            sscanf(ent->d_name, "Ckpt%d-Rank%d.fti", &id, &rank);
            for (i = 0; i < world_size; i++) {
                if (rank == mpi_ranks[i]) {
                    rank = i;
                    break;
                }
            }

            int expectedSize = 0;
            /* when we add FTI_Realloc();
            expectedSize += sizeof(long) * 2; //(i and size) length
            */
            expectedSize += sizeof(int) * 2; //i and size

            int lastCheckpointIter;
            if (fail) {
                lastCheckpointIter = ITER_STOP - ITER_STOP % ITER_CHECK;
            } else {
                lastCheckpointIter = (ITERATIONS - 1) - (ITERATIONS - 1) % ITER_CHECK;
            }
            int arrayExpectedLength = world_size * ((lastCheckpointIter  * 2) + INIT_SIZE * 2);

            int myPart;
            getPart(&myPart, arrayExpectedLength, rank, world_size);
            /* when we add FTI_Realloc();
            expectedSize += sizeof(long);               //myPart length
            */
            expectedSize += sizeof(long) * myPart;      //myPart size
            printf("%d: Last checkpoint file size = %d\n", rank, fileSize);
            if (fileSize != expectedSize) {
                printf("%d: Last checkpoint file size = %d, should be %d.\n", rank, fileSize, expectedSize);
                fclose(f);
                closedir (dir);
                return 1;
            }
            fclose(f);
        }
      }
      closedir (dir);
    } else {
      //could not open directory
      perror ("");
      return 1;
    }
    return 0;
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
    int global_world_rank;                          //MPI_COMM rank
    MPI_Comm_rank(MPI_COMM_WORLD, &global_world_rank);

    FTI_Init(argv[1], MPI_COMM_WORLD);
    int world_rank, world_size;                     //FTI_COMM rank and size
    MPI_Comm_rank(FTI_COMM_WORLD, &world_rank);
    MPI_Comm_size(FTI_COMM_WORLD, &world_size);

    int rtn = do_work(world_rank, world_size, checkpoint_level, fail);

    //need MPI ranks to know checkpoint files names
    int* mpi_ranks = malloc (sizeof(int) * world_size);
    MPI_Gather(&global_world_rank, 1, MPI_INT, mpi_ranks, 1, MPI_INT, 0, FTI_COMM_WORLD);

    FTI_Finalize();
    MPI_Finalize();

    if (world_rank == 0 && !rtn) {
        rtn = checkFileSizes(mpi_ranks, world_size, fail);
        if (!rtn && !fail)
            printf("Success.\n");
    }
    free(mpi_ranks);
    return rtn;
}
