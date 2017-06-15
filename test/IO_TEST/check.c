#include "mpi.h"
#include "fti.h"
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <unistd.h>

#define N 100000

#define ZERO 0
#define ONE 1
#define TWO 2
#define SAVE 2
#define RESTART 1
#define INIT 0

/** 
 * function prototypes
 */

int init_arrays(double* A, double* B, size_t asize);

int vecmulc(double* A, double* B, size_t asize);

int validify(double* A, double* B, size_t asize);

int write_data(double* B, size_t* asize, int rank);

int read_data(double* B_chk, size_t* asize_chk, int rank, size_t asize);

/**
 * main
 */ 

int main(int argc, char* argv[]) {

    unsigned char parity, crash, level, state;
    int wrank, wsize, result, tmp, success = 1;
    double *A, *B, *B_chk;

    size_t asize, asize_chk;

    srand(time(NULL));

    MPI_Init(&argc, &argv);
    FTI_Init(argv[1], MPI_COMM_WORLD);
    
    crash = atoi(argv[2]);
    level = atoi(argv[3]);

    MPI_Comm_rank(FTI_COMM_WORLD,&wrank);
    
    asize = N;
  
    parity = wrank%7;
  
    switch (parity) {
        
        case 0:
            asize = N;
            break;
            
        case 1:
            asize = 2*N;
            break;

        case 2:
            asize = 3*N;
            break;
            
        case 3:
            asize = 4*N;
            break;

        case 4:
            asize = 5*N;
            break;

        case 5:
            asize = 6*N;
            break;

        case 6:
            asize = 7*N;
            break;

    }

    A = (double*) malloc(asize*sizeof(double));
    B = (double*) malloc(asize*sizeof(double));

    FTI_Protect(0, A, asize, FTI_DBLE);
    FTI_Protect(1, B, asize, FTI_DBLE);
    FTI_Protect(2, &asize, 1, FTI_INTG);
    
    state = FTI_Status();

    if (state == INIT) {
        init_arrays(A, B, asize);
        write_data(B, &asize, wrank);
    } else {
        result = FTI_Recover();
        if (result != FTI_SCES) {
            sleep(1);
            exit(1);
        }
        B_chk = (double*) malloc(asize*sizeof(double));
        result = read_data(B_chk, &asize_chk, wrank, asize);
        MPI_Barrier(FTI_COMM_WORLD);
        if (result != 0) {
            sleep(1);
            exit(1);
        }
    }
    
    if (state == INIT) {
        FTI_Checkpoint(1,level);
        sleep(2);
    }

    if (crash && wrank == 0) { 
        exit(1);
    }

    vecmulc(A, B, asize);

    if (state == RESTART || state == SAVE) {
        result = validify(A, B_chk, asize);
        result += (asize_chk == asize) ? 0 : -1;
        MPI_Allreduce(&result, &tmp, 1, MPI_INT, MPI_SUM, FTI_COMM_WORLD);
        result = tmp;
        free(B_chk);
    }
    
    free(A);
    free(B);
    
    if (wrank == 0 && (state == RESTART || state == SAVE)) {
        if (result == 0) {
            printf("[SUCCESSFULL]\n");
        } else {
            printf("[NOT SUCCESSFULL]\n");
            success=0;
        }
    }
    
    MPI_Barrier(FTI_COMM_WORLD);
    FTI_Finalize();
    MPI_Finalize();

    if (success == 1) 
        return 0;
    else 
        exit(2612);

}

/**
 * function definitions
 */

int init_arrays(double* A, double* B, size_t asize) {
    int i;
    double r;
    for (i = 0; i< asize; i++) {
        A[i] = 1.0;
        B[i] = ((double)rand()/RAND_MAX)*5.0; 
    }
    return 0;
}

int vecmulc(double* A, double* B, size_t asize) {
    int i;
    for (i=0; i<asize; i++) {
        A[i] = A[i]*B[i];
    }
    return 0;
}

int validify(double* A, double* B, size_t asize) {
    int i;
    for (i=0; i<asize; i++) {
        if (A[i] != B[i]){
            return -1;
        }
    }
    return 0;
}

int write_data(double* B, size_t *asize, int rank) {
    char str[256];
    sprintf(str, "chk/check-%i.tst", rank);
    FILE* f = fopen(str, "wb");
    size_t written = 0;

    fwrite( (void*) asize, sizeof(size_t), 1, f);

    while ( written < (*asize) ) {
        written += fwrite( (void*) B, sizeof(double), (*asize), f);
    }

    fclose(f);

    return 0;
}

int read_data(double* B_chk, size_t *asize_chk, int rank, size_t asize) {
    char str[256];
    sprintf(str, "chk/check-%i.tst", rank);
    FILE* f = fopen(str, "rb");
    size_t read = 0;

    fread( (void*) asize_chk, sizeof(size_t), 1, f);
    if ((*asize_chk) != asize) {
        printf("[ERROR -%i] : wrong dimension 'asize' -- asize: %zd, asize_chk: %zd\n", rank, asize, *asize_chk);
        fflush(stdout);
        return -1;
    }
    while ( read < (*asize_chk) ) {
        read += fread( (void*) B_chk, sizeof(double), (*asize_chk), f);
    }

    fclose(f);

    return 0;
}

