/**
 *  Copyright (c) 2017 Leonardo A. Bautista-Gomez
 *  All rights reserved
 *
 *  @file   checkDCPPosix.c
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
 *	  - arg3: different ckpt. sizes yes/no (1/0)
 *	  - arg4: recover strategy FTI_Recover / FTI_RecoverVar (0/1)
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

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#include "../../../../src/deps/iniparser/dictionary.h"
#include "../../../../src/deps/iniparser/iniparser.h"
#include "fti.h"
#include "mpi.h"

#define N 100000
#define CNTRLD_EXIT 10
#define RECOVERY_FAILED 20
#define DATA_CORRUPT 30
#define WRONG_ENVIRONMENT 50
#define KEEP 2
#define RESTART 1
#define INIT 0

/**
 * function prototypes
 */

void shuffle(int* array, size_t n) {
  if (n > 1) {
    size_t i;
    for (i = 0; i < n - 1; i++) {
      size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
      int t = array[j];
      array[j] = array[i];
      array[i] = t;
    }
  }
}

/*-------------------------------------------------------------------------*/
/**
  @brief      Initialize test data
  @param      [out] A				Unit vector (1, 1, ....., 1)
  @param      [out] B				Random vector
  @param      [in] asize			Dimension

  Initializes A with 1's and B with random numbers r,  0 <= r <= 5.
  Dimension of both vectors is 'asize'
 **/
/*-------------------------------------------------------------------------*/
void init_arrays(double* A, double* B, size_t asize);

/*-------------------------------------------------------------------------*/
/**
  @brief      Multiplies components of A and B and stores result into A
  @param      [in/out] A			Unit vector (1, 1, ....., 1)
  @param      [in] B				Random vector
  @param      [in] asize			Dimension

  After function call, A equals B.
 **/
/*-------------------------------------------------------------------------*/
void vecmult(double* A, double* B, size_t start, size_t end);

/*-------------------------------------------------------------------------*/
/**
  @brief      Validifies the recovered data
  @param      [in] A			    A returned from vecmult
  @param      [in] B_chk			POSIX Backup of B
  @param      [in] asize			Dimension
  @return     integer             0 if successful, -1 else.

  Checks entry for entry if A equals the POSIX Backup of B, B_chk, from
  the preceding execution. This function must be called after the call to
  vecmult(A, B, asize).
 **/
/*-------------------------------------------------------------------------*/
int validify(double* A, double* B_chk, size_t asize);

/*-------------------------------------------------------------------------*/
/**
  @brief      Writes 'B' and 'asize' to file, using POSIX fwrite.
  @param      [in] B              Random array B from init_array call
  @param      [in] asize			Dimension
  @param      [in] rank           FTI application rank
 **/
/*-------------------------------------------------------------------------*/
int write_data(double* B, size_t* asize, int rank);

/*-------------------------------------------------------------------------*/
/**
  @brief      Recovers 'B' and 'asize' to 'B_chk' and 'asize_chk' from file,
  using POSIX fread.
  @param      [out] B_chk         B backup
  @param      [out] asize_chk     Dimension backup
  @param      [in] rank           FTI application rank
  @param      [in] asize			Dimension
  @return     integer             0 if successful, -1 else.

  Before recovering B, the function checks if 'asize_chk' equals 'asize',
  to prevent SIGSEGV. If not 'asize_chk' = 'asize' it returns -1.
 **/
/*-------------------------------------------------------------------------*/
int read_data(double* B_chk, size_t* asize_chk, int rank, size_t asize,
              size_t stop);

/**
 * main
 */

int main(int argc, char* argv[]) {
  unsigned char parity, crash, state, diff_sizes, enable_icp = -1;
  int FTI_APP_RANK, result, tmp, success = 1;
  double *A, *B, *B_chk;
  size_t i;

  size_t asize, asize_chk;

  srand(time(NULL));

  MPI_Init(&argc, &argv);
  result = FTI_Init(argv[1], MPI_COMM_WORLD);
  if (result == FTI_NREC) {
    exit(RECOVERY_FAILED);
  }

  crash = atoi(argv[2]);
  diff_sizes = atoi(argv[3]);
  int recoveryType = atoi(argv[4]);

  MPI_Comm_rank(FTI_COMM_WORLD, &FTI_APP_RANK);

  dictionary* ini = iniparser_load(argv[1]);
  int grank;
  int lrank;
  MPI_Comm_rank(MPI_COMM_WORLD, &grank);
  MPI_Comm_rank(FTI_COMM_WORLD, &lrank);
  if (lrank == 0) printf("The recovery type is %d\n", recoveryType);

  int nbHeads = (int)iniparser_getint(ini, "Basic:head", -1);
  int finalTag = (int)iniparser_getint(ini, "Advanced:final_tag", 3107);
  int nodeSize = (int)iniparser_getint(ini, "Basic:node_size", -1);
  int headRank = grank - grank % nodeSize;
  int numberIter = 0;

  if ((nbHeads < 0) || (nodeSize < 0)) {
    printf("wrong configuration (for head or node-size settings)!\n");
    MPI_Abort(MPI_COMM_WORLD, -1);
  }

  asize = N;

  if (diff_sizes) {
    parity = FTI_APP_RANK % 7;

    switch (parity) {
      case 0:
        asize = N;
        break;

      case 1:
        asize = 2 * N;
        break;

      case 2:
        asize = 3 * N;
        break;

      case 3:
        asize = 4 * N;
        break;

      case 4:
        asize = 5 * N;
        break;

      case 5:
        asize = 6 * N;
        break;

      case 6:
        asize = 7 * N;
        break;
    }
  }

  A = (double*)malloc(asize * sizeof(double));
  B = (double*)malloc(asize * sizeof(double));

  FTI_Protect(0, A, asize, FTI_DBLE);
  FTI_Protect(1, B, asize, FTI_DBLE);
  FTI_Protect(2, &asize, sizeof(size_t), FTI_CHAR);
  FTI_Protect(3, &numberIter, sizeof(size_t), FTI_CHAR);

  state = FTI_Status();

  if (state == INIT) {
    init_arrays(A, B, asize);
    write_data(B, &asize, FTI_APP_RANK);
    FTI_Checkpoint(numberIter, 8);
    MPI_Barrier(FTI_COMM_WORLD);
    if (crash) {
      if (nbHeads > 0) {
        int value = FTI_ENDW;
        MPI_Send(&value, 1, MPI_INT, headRank, finalTag, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
      }
      MPI_Finalize();
      exit(0);
    }
  }

  if (state == RESTART || state == KEEP) {
    if (recoveryType == 0) {
      printf("START RECOVER\n");
      result = FTI_Recover();
    } else {
      printf("START RECOVER---VAR\n");
      int order[4] = {0, 1, 2, 3};
      shuffle(order, 4);
      result = FTI_RecoverVarInit();
      for (i = 0; i < 4; i++) {
        if (lrank == 0) {
          printf("Recovering Var %d\n", order[i]);
        }
        result += FTI_RecoverVar(order[i]);
      }
      result += FTI_RecoverVarFinalize();
      if (result != FTI_SCES) {
        exit(RECOVERY_FAILED);
      }
    }

    if (result != FTI_SCES) {
      exit(RECOVERY_FAILED);
    }

    B_chk = (double*)malloc(asize * sizeof(double));

    result = read_data(B_chk, &asize_chk, FTI_APP_RANK, asize,
                       (numberIter) * (asize / 5));
    MPI_Barrier(FTI_COMM_WORLD);
    if (result != 0) {
      exit(DATA_CORRUPT);
    }
  }

  for (i = numberIter; i < 5; i++) {
    size_t start = i * (asize / 5);
    size_t end = (i + 1) * (asize / 5);
    double ChangedBytes = (end - start) * sizeof(double);
    double totalBytes = 2 * asize * sizeof(double) + 2 * sizeof(size_t);
    if (lrank == 0) {
      printf(
          "I am executing iteration %ld I am computing %.2f Mb, Changed Size "
          "(%%) %.2f \n",
          i, (end - start) / (1024.0 * 1024.0),
          (ChangedBytes / totalBytes) * 100.0);
    }
    vecmult(A, B, start, end);
    numberIter += 1;
    FTI_Checkpoint(numberIter, 8);
    if (numberIter == crash) {
      if (nbHeads > 0) {
        int value = FTI_ENDW;
        MPI_Send(&value, 1, MPI_INT, headRank, finalTag, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
      }
      MPI_Finalize();
      exit(0);
    }
  }

  if (state == RESTART || state == KEEP) {
    result = validify(A, B_chk, asize);
    result += (asize_chk == asize) ? 0 : -1;
    MPI_Allreduce(&result, &tmp, 1, MPI_INT, MPI_SUM, FTI_COMM_WORLD);
    result = tmp;
    free(B_chk);
  }

  free(A);
  free(B);

  if (FTI_APP_RANK == 0 && (state == RESTART || state == KEEP)) {
    if (result == 0) {
      printf("[SUCCESSFUL]\n");
    } else {
      printf("[NOT SUCCESSFUL]\n");
      success = 0;
    }
  }

  MPI_Barrier(FTI_COMM_WORLD);
  FTI_Finalize();
  MPI_Finalize();

  if (success == 1)
    return 0;
  else
    exit(DATA_CORRUPT);
}

/**
 * function definitions
 */

void init_arrays(double* A, double* B, size_t asize) {
  int i;
  double r;
  for (i = 0; i < asize; i++) {
    A[i] = 1.0;
    B[i] = ((double)rand() / RAND_MAX) * 5.0;
  }
}

void vecmult(double* A, double* B, size_t start, size_t end) {
  int i;
  for (i = start; i < end; i++) {
    A[i] = A[i] * B[i];
  }
}

int validify(double* A, double* B_chk, size_t asize) {
  int i;
  for (i = 0; i < asize; i++) {
    if (A[i] != B_chk[i]) {
      return -1;
    }
  }
  return 0;
}

int write_data(double* B, size_t* asize, int rank) {
  char str[256];
  sprintf(str, "check-%i.tst", rank);
  FILE* f = fopen(str, "wb");
  size_t written = 0;

  fwrite((void*)asize, sizeof(size_t), 1, f);

  while (written < (*asize)) {
    written += fwrite((void*)B, sizeof(double), (*asize), f);
  }

  fclose(f);

  return 0;
}

int read_data(double* B_chk, size_t* asize_chk, int rank, size_t asize,
              size_t stop) {
  char str[256];
  sprintf(str, "check-%i.tst", rank);
  FILE* f = fopen(str, "rb");
  size_t read = 0;

  fread((void*)asize_chk, sizeof(size_t), 1, f);
  if ((*asize_chk) != asize) {
    printf(
        "[ERROR -%i] : wrong dimension 'asize' -- asize: %zd, asize_chk: %zd\n",
        rank, asize, *asize_chk);
    fflush(stdout);
    return -1;
  }
  while (read < stop) {
    read += fread((void*)B_chk, sizeof(double), (*asize_chk), f);
  }

  fclose(f);

  return 0;
}
