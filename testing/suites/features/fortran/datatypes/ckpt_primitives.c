/**
 * Copyright (c) 2017 Leonardo A. Bautista-Gomez
 * All rights reserved
 *
 * @file   ckpt_primitives.c
 * @author Alexandre de Limas Santana (alexandre.delimassantana@bsc.es)
 * @date   August, 2020
**/
#include <fti.h>
#include <stdio.h>
#include <stdlib.h>

#define SIZE 5

// Patterns to initialize primitive data for testing
// Any pattern will do, these were selected empirically
#define C1_INIT 'A'             // Stored as its ASCII value, 65
#define LX_INIT 1               // Fortran 'true' maps to 1 in C
#define I1_INIT 127 - i         // Greater 8-bit number minus i
#define I2_INIT 32767 - i       // Greater 16-bit number minus i
#define I4_INIT 2147483647 - i  // Greater 32-bit number minus i
#define I8_INIT 2147483647 + i  // Greater 32-bit number plus i
#define R4_INIT 10 + i
#define R8_INIT 20 + i
#define R16_INIT 30 + i

typedef struct Complex4 {
    float r, i;
} Complex4;

typedef struct Complex8 {
    double r, i;
} Complex8;

typedef struct Complex16 {
    long double r, i;
} Complex16;

int main(int argc, char * argv[]) {
  char *configfile;
  int doInitData, rank, ret, i;

  /**
   * These are the variables for testing.
   * Their types should be verified in the meta-data files.
  **/
  char c1[SIZE];

  // Logicals are stored as integers in ANSI C, there are no booleans
  char i1[SIZE], l1[SIZE];
  short i2[SIZE], l2[SIZE];
  int i4[SIZE], l4[SIZE];
  long i8[SIZE], l8[SIZE];

  float r4[SIZE];
  double r8[SIZE];
  long double r16[SIZE];

  Complex4 cp4[SIZE];
  Complex8 cp8[SIZE];
  Complex16 cp16[SIZE];

  fti_id_t complex_4, complex_8, complex_16;

  if (argc < 2) {
    printf("Expected two arguments: configfile (str) and doInitData (bool)");
    return 1;
  }

  configfile = argv[1];
  doInitData = atoi(argv[2]);

  MPI_Init(&argc, &argv);
  FTI_Init(configfile, MPI_COMM_WORLD);
  MPI_Comm_rank(FTI_COMM_WORLD, &rank);

  /**
   * Conditionally initialize the data based on command-line arguments.
   * Any initialization will do as long as the verification repeats it.
   **/
  if (doInitData) {
    if (rank == 0) {
      printf("Application initializes its own data");
    }
    for (i = 0; i < SIZE; ++i) {
      c1[i] = C1_INIT;
      l1[i] = LX_INIT;
      l2[i] = LX_INIT;
      l4[i] = LX_INIT;
      l8[i] = LX_INIT;
      i1[i] = I1_INIT;
      i2[i] = I2_INIT;
      i4[i] = I4_INIT;
      i8[i] = I8_INIT;
      r4[i] = R4_INIT;
      r8[i] = R8_INIT;
      r16[i] = R16_INIT;
      cp4[i].i = i;
      cp4[i].r = i;
      cp8[i].i = i;
      cp8[i].r = i;
      cp16[i].i = i;
      cp16[i].r = i;
    }
  }
  // Initializing custom types
  complex_4 = FTI_InitComplexType("Complex4", sizeof(Complex4), NULL);
  FTI_AddSimpleField(complex_4, "r", FTI_SFLT, offsetof(Complex4, r));
  FTI_AddSimpleField(complex_4, "i", FTI_SFLT, offsetof(Complex4, i));
  complex_8 = FTI_InitComplexType("Complex8", sizeof(Complex8), NULL);
  FTI_AddSimpleField(complex_8, "r", FTI_SFLT, offsetof(Complex8, r));
  FTI_AddSimpleField(complex_8, "i", FTI_SFLT, offsetof(Complex8, i));
  complex_16 = FTI_InitComplexType("Complex16", sizeof(Complex16), NULL);
  FTI_AddSimpleField(complex_16, "r", FTI_SFLT, offsetof(Complex16, r));
  FTI_AddSimpleField(complex_16, "i", FTI_SFLT, offsetof(Complex16, i));

  FTI_Protect(0, c1, SIZE, FTI_CHAR);

  FTI_Protect(10, l1, SIZE, FTI_CHAR);
  FTI_Protect(11, l2, SIZE, FTI_SHRT);
  FTI_Protect(12, l4, SIZE, FTI_INTG);
  FTI_Protect(13, l8, SIZE, FTI_LONG);

  FTI_Protect(20, i1, SIZE, FTI_CHAR);
  FTI_Protect(21, i2, SIZE, FTI_SHRT);
  FTI_Protect(22, i4, SIZE, FTI_INTG);
  FTI_Protect(23, i8, SIZE, FTI_LONG);

  FTI_Protect(30, r4, SIZE, FTI_SFLT);
  FTI_Protect(31, r8, SIZE, FTI_DBLE);
  FTI_Protect(32, r16, SIZE, FTI_LDBE);

  FTI_Protect(40, cp4, SIZE, complex_4);
  FTI_Protect(41, cp8, SIZE, complex_8);
  FTI_Protect(42, cp16, SIZE, complex_16);

  /**
   * If the application initialized the data, do the following:
   * 1) Checkpoint the data using FTI_Checkpoint()
   * 2) Simulate a crash by not calling FTI_Finalize()
   * 3) Return 0 because everything went as planned
  **/
  if (doInitData != 0) {
    FTI_Checkpoint(1, 1);
    MPI_Finalize();
    return 0;
  } else {
    // If the application did not initialize the data, FTI does so.
    // In this case, we move on to check if the values are correct.
    FTI_Recover();
  }

  // Test if data was initialized following the same patterns
  // Checks if the IO library stored data in accordance to their binary format.
  ret = 1;
  for (i = 0; i < SIZE; ++i) {
    if (c1[i] != C1_INIT) {
      if (rank == 0)
        printf("character(1) was corrupted %c\n", c1[i]);
      goto end;
    }

    if (l1[i] != LX_INIT) {
      if (rank == 0)
        printf("logical(1) was corrupted %d\n", l1[i]);
      goto end;
    }
    if (l2[i] != LX_INIT) {
      if (rank == 0)
        printf("logical(2) was corrupted %d\n", l2[i]);
      goto end;
    }
    if (l4[i] != LX_INIT) {
      if (rank == 0)
        printf("logical(4) was corrupted %d\n", l4[i]);
      goto end;
    }
    if (l8[i] != LX_INIT) {
      if (rank == 0)
        printf("logical(8) was corrupted %ld\n", l8[i]);
      goto end;
    }

    if (i1[i] != I1_INIT) {
      if (rank == 0)
        printf("integer(1) was corrupted %d\n", i1[i]);
      goto end;
    }
    if (i2[i] != I2_INIT) {
      if (rank == 0)
        printf("integer(2) was corrupted %hd\n", i2[i]);
      goto end;
    }
    if (i4[i] != I4_INIT) {
      if (rank == 0)
        printf("integer(4) was corrupted %d\n", i4[i]);
      goto end;
    }
    if (i8[i] != I8_INIT) {
      if (rank == 0)
        printf("integer(8) was corrupted %ld\n", i8[i]);
      goto end;
    }

    if (r4[i] != R4_INIT) {
      if (rank == 0)
        printf("real(4) was corrupted %f\n", r4[i]);
      goto end;
    }
    if (r8[i] != R8_INIT) {
      if (rank == 0)
        printf("real(8) was corrupted %f\n", r8[i]);
      goto end;
    }
    if (r16[i] != R16_INIT) {
      if (rank == 0)
        printf("real(16) was corrupted %Lf\n", r16[i]);
      goto end;
    }
    if (cp4[i].i != i || cp4[i].r != i) {
      if (rank == 0)
        printf("complex(4) corrupted r=%f, i=%f\n", cp4[i].r, cp4[i].i);
      goto end;
    }
    if (cp8[i].i != i || cp8[i].r != i) {
      if (rank == 0)
        printf("complex(8) corrupted r=%f, i=%f\n", cp8[i].r, cp8[i].i);
      goto end;
    }
    if (cp16[i].i != i || cp16[i].r != i) {
      if (rank == 0)
        printf("complex(16) corrupted r=%Lf, i=%Lf\n", cp16[i].r, cp16[i].i);
      goto end;
    }
  }
  ret = 0;
  end:
  FTI_Finalize();
  MPI_Finalize();
  return ret;
}
