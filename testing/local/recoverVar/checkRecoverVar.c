/**
 *  @file   check.c
 *  @author Kai Keller (kellekai@gmx.de)
 *  @date   June, 2017
 *  @brief  FTI testing program.
 *
 *	The program may test the correct behaviour for checkpoint
 *	and restart for all configurations. The recovered data is also
 *	tested upon correct data fields.
 *
 *	The program takes four arguments:
 *	  - arg1: FTI configuration file
 *	  - arg2: Interrupt yes/no (1/0)
 *	  - arg3: Checkpoint level (1, 2, 3, 4)
 *	  - arg4: different ckpt. sizes yes/no (1/0)
 *
 * If arg2 = 0, the program simulates a clean run of FTI:
 *    FTI_Init
 *    FTI_Protect
 *    if FTI_Status = 0
 *      FTI_Checkpoint
 *    else
 *      FTI_Recover
 *    FTI_Finalize
 *
 * If arg2 = 1, the program simulates an execution failure:
 *    FTI_Init
 *    FTI_Protect
 *    if FTI_Status = 0
 *      exit(10)
 *    else
 *      FTI_Recover
 *    FTI_Finalize
 *
 */

#include "mpi.h"
#include "fti.h"
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <unistd.h>
#include "../../../src/deps/iniparser/iniparser.h"
#include "../../../src/deps/iniparser/dictionary.h"

#define CNTRLD_EXIT 10
#define RECOVERY_FAILED 20
#define DATA_CORRUPT 30
#define WRONG_ENVIRONMENT 50
#define KEEP 2
#define RESTART 1
#define INIT 0


void printArray(int *array[], int *sizes, int *order ,int N){
    for (int k = 0; k < N; k++){
        int i=k;
        printf("%d) ", i);
        for ( int j = 0 ; j < sizes[i]; j++){
            printf ("%d ",array[i][j]);
        }
        printf("\n");
    }


}

void initArray(int *array[], int *sizes, int N){
    int i;
    for ( i = 0; i < N; i++){
        for ( int j = 0 ; j < sizes[i]; j++)
            array[i][j] = i;
    }
}

void allocArray(int *array[], int *sizes, int N){
    int i;
    for ( i = 0; i < N; i++)
        array[i]= (int *) malloc (sizeof(int)*sizes[i]);
}


void shuffle(int *array, size_t n)
{
    if (n > 1) 
    {
        size_t i;
        for (i = 0; i < n - 1; i++) 
        {
            size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
            int t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
}

int checkCorrectness( int *array, int size, int val){
    for ( int i = 0 ; i < size; i++)
        if (array[i]!= val)
            return 0;
    return 1;            
}


int main(int argc, char* argv[]) {
    unsigned char parity, crash, level, state, diff_sizes, enable_icp = -1;
    int FTI_APP_RANK, result, tmp, success = 1;
    int *array[10];
    int sizes[10] = {42, 85, 8, 19, 95, 26, 66, 33, 65, 83};
    int order[10] = {0,1,2,3,4,5,6,7,8,9};


    srand(time(NULL));
    size_t asize, asize_chk;
    allocArray(array, sizes, 10);

    MPI_Init(&argc, &argv);
    result = FTI_Init(argv[1], MPI_COMM_WORLD);
    if (result == FTI_NREC) {
        exit(RECOVERY_FAILED);
    }

    crash = atoi(argv[2]);
    level = atoi(argv[3]);
    int executionTimes = atoi(argv[4]);

    MPI_Comm_rank(FTI_COMM_WORLD,&FTI_APP_RANK);
    dictionary *ini = iniparser_load( argv[1] );
    int grank;    
    MPI_Comm_rank(MPI_COMM_WORLD,&grank);
    int nbHeads = (int)iniparser_getint(ini, "Basic:head", -1); 
    int finalTag = (int)iniparser_getint(ini, "Advanced:final_tag", 3107);
    int nodeSize = (int)iniparser_getint(ini, "Basic:node_size", -1);
    int headRank = grank - grank%nodeSize;

    if ( (nbHeads<0) || (nodeSize<0) ) {
        printf("wrong configuration (for head or node-size settings)!\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    int correct = 1;

    state = FTI_Status();
    if (state == INIT) {
        initArray(array,sizes,10);
        for ( int i = 0 ; i < 10; i++){
            FTI_Protect(i,array[i], sizes[i], FTI_INTG); 
        }
        FTI_Checkpoint(executionTimes,level);
        if ( crash ) {
            if( nbHeads > 0 ) { 
                int value = FTI_ENDW;
                MPI_Send(&value, 1, MPI_INT, headRank, finalTag, MPI_COMM_WORLD);
                MPI_Barrier(MPI_COMM_WORLD);
            }
            MPI_Finalize();
            exit(0);
        }
    }
    else if ( state == RESTART || state == KEEP ) {
        shuffle(order,10);
        int res = FTI_RecoverVarInit();
        for ( int i = 0; i < 10 ; i++){
            int index = order[i];
            FTI_Protect(index,array[index], sizes[index], FTI_INTG); 
            res += FTI_RecoverVar(index);
        }
        res += FTI_RecoverVarFinalize();
        if (res != FTI_SCES ){
            if (result != FTI_SCES) {
                exit(RECOVERY_FAILED);
            }
        }
        for ( int i = 0; i < 10 ; i++){
            correct &= checkCorrectness(array[i],sizes[i],i);
        }
        MPI_Barrier(FTI_COMM_WORLD);
        if (correct != 1) {
            exit(DATA_CORRUPT);
        }
    }

    if (FTI_APP_RANK == 0 && (state == RESTART || state == KEEP)) {
        if (correct == 1) {
            printf("[SUCCESSFUL]\n");
        } else {
            printf("[NOT SUCCESSFUL]\n");
            correct =0;
        }
    }

    MPI_Barrier(FTI_COMM_WORLD);
    FTI_Finalize();
    MPI_Finalize();

    if (correct == 1)
        return 0;
    else
        exit(DATA_CORRUPT);
}

