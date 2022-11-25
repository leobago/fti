/**
 *  Copyright (c) 2017 Leonardo A. Bautista-Gomez
 *  All rights reserved
 * 
 *  @file   hdf5Test.c
 *  @author Tomasz Paluszkiewicz (tomaszp@man.poznan.pl)
 *  @date   November, 2017
 *  @brief  FTI testing program.
 *
 *  Testing FTI_InitType, FTI_InitSimpleTypeWithNames,
 *  FTI_InitCompositeTypeWithNames, FTI_ProtectWithName, FTI_Checkpoint,
 *  FTI_Recover, saving last checkpoint to PFS
 *
 *  Program creates complex data structures, then adds it to protect list
 *  and makes a checkpoint. Second run recovers the files and checks if
 *  values match.
 *
 *  First execution this program should be with fail flag = 1, because
 *  then FTI saves checkpoint and program stops. Second execution
 *  must be with the flag = 0 to properly recover data.
 */

#include <fcntl.h>
#include <fti.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>

#include "../../../../src/deps/iniparser/dictionary.h"
#include "../../../../src/deps/iniparser/iniparser.h"

#define VERIFY_SUCCESS 0
#define VERIFY_FAILED 1

typedef struct AsByteArray {
  char character;
  long longs[1024];
} AsByteArray;

typedef struct Chars {
  char chars[10];
  unsigned char uChars[2][3][4][5];
  AsByteArray bytes[2];
} Chars;

typedef struct Integers {
  short shortInteger;
  int integer;
  long longInteger;
} Integers;

typedef struct UIntegers {
  unsigned short shortInteger;
  unsigned int integer;
  unsigned long longInteger;
} UIntegers;

typedef struct Floats {
  float singlePrec;
  double doublePrec;
  // makes trouble    long double longDoublePrec;
} Floats;

typedef struct AllInts {
  Integers integers[5];
  UIntegers uIntegers[4];
} AllInts;

typedef struct FloatsChars {
  Floats floats[3];
  Chars chars[2];
} FloatsChars;

typedef struct AllTypes {
  AllInts allInts;
  FloatsChars floatsChars;
} AllTypes;

void defaultChars(Chars* in, int shift) {
  snprintf(in->chars, sizeof(in->chars), "%s", "abcdefg");
  int i, j, k, l;

  for (i = 0; i < 7; i++) in->chars[i] += shift % (25 - in->chars[i] - 'a');

  for (i = 0; i < 2; i++) {
    for (j = 0; j < 3; j++) {
      for (k = 0; k < 4; k++) {
        for (l = 0; l < 5; l++) {
          in->uChars[i][j][k][l] = 'a' + ((i + j + k + l) + shift) % 25;
        }
      }
    }
    in->bytes[i].character = 'C' + shift % 23;
    for (j = 0; j < 1024; j++) {
      in->bytes[i].longs[j] = (j + 1) * 2 + shift;
    }
  }
}

void defaultInts(Integers* in, int shift) {
  in->shortInteger = -12 - shift;
  in->integer = -123 - shift;
  in->longInteger = -1234 - shift;
}

void defaultUInts(UIntegers* in, int shift) {
  in->shortInteger = 12 + shift;
  in->integer = 123 + shift;
  in->longInteger = 1234 + shift;
}

void defaultFloats(Floats* in, int shift) {
  in->singlePrec = 12.5f + shift;
  in->doublePrec = 123.25 + shift;
  //    in->longDoublePrec = 1234.125 + shift;
}

void defaultAllInts(AllInts* in, int shift) {
  int i;
  for (i = 0; i < 5; i++) {
    defaultInts(in->integers + i, shift + i);
  }

  for (i = 0; i < 4; i++) {
    defaultUInts(in->uIntegers + i, shift + i);
  }
}

void defaultFloatsChars(FloatsChars* in, int shift) {
  int i;
  for (i = 0; i < 3; i++) {
    defaultFloats(in->floats + i, shift + i);
  }

  for (i = 0; i < 2; i++) {
    defaultChars(in->chars + i, shift + i);
  }
}

void defaultAllTypes(AllTypes* in, int shift) {
  defaultAllInts(&in->allInts, shift);
  defaultFloatsChars(&in->floatsChars, shift);
}

int verifyChars(Chars* in, int shift, int rank, char* name) {
  char buff[] = "abcdefg";
  int i;
  for (i = 0; i < 7; i++) buff[i] += shift % (25 - buff[i] - 'a');
  if (strcmp(in->chars, buff)) {
    printf("[ %06d ] : %s.chars = %s should be %s \n", rank, name, in->chars,
           buff);
    return VERIFY_FAILED;
  }
  int j, k, l;
  for (i = 0; i < 2; i++) {
    for (j = 0; j < 3; j++) {
      for (k = 0; k < 4; k++) {
        for (l = 0; l < 5; l++) {
          if (in->uChars[i][j][k][l] != 'a' + ((i + j + k + l) + shift) % 25) {
            printf("[ %06d ] : %s.uChars[%d][%d][%d][%d] = %u should be %u \n",
                   rank, name, i, j, k, l, in->uChars[i][j][k][l],
                   'a' + ((i + j + k + l) + shift) % 25);
            return VERIFY_FAILED;
          }
        }
      }
    }
    if (in->bytes[i].character != 'C' + shift % 23) {
      printf("[ %06d ] : %s.bytes[%d].character = %c should be %c \n", rank,
             name, i, in->bytes[i].character, ('C' + shift % 23));
      return VERIFY_FAILED;
    }
    for (j = 0; j < 1024; j++) {
      if (in->bytes[i].longs[j] != (j + 1) * 2 + shift) {
        printf("[ %06d ] : %s.bytes[%d].longs[%d] = %ld should be %d \n", rank,
               name, i, j, in->bytes[i].longs[j], (j + 1) * 2 + shift);
        return VERIFY_FAILED;
      }
    }
  }
  return VERIFY_SUCCESS;
}

int verifyInts(Integers* in, int shift, int rank, char* name) {
  if (in->shortInteger != -12 - shift) {
    printf("[ %06d ] : %s.shortInteger = %hd should be %hd \n", rank, name,
           in->shortInteger, -12 - shift);
    return VERIFY_FAILED;
  }
  if (in->integer != -123 - shift) {
    printf("[ %06d ] : %s.integer = %d should be %d \n", rank, name,
           in->integer, -123 - shift);
    return VERIFY_FAILED;
  }
  if (in->longInteger != -1234 - shift) {
    printf("[ %06d ] : %s.longtInteger = %ld should be %d \n", rank, name,
           in->longInteger, -1234 - shift);
    return VERIFY_FAILED;
  }
  return VERIFY_SUCCESS;
}

int verifyUInts(UIntegers* in, int shift, int rank, char* name) {
  if (in->shortInteger != 12 + shift || in->integer != 123 + shift ||
      in->longInteger != 1234 + shift)
    return VERIFY_FAILED;

  if (in->shortInteger != 12 + shift) {
    printf("[ %06d ] : %s.shortInteger = %hu should be %hu \n", rank, name,
           in->shortInteger, 12 + shift);
    return VERIFY_FAILED;
  }
  if (in->integer != 123 + shift) {
    printf("[ %06d ] : %s.integer = %u should be %u \n", rank, name,
           in->integer, 123 + shift);
    return VERIFY_FAILED;
  }
  if (in->longInteger != 1234 + shift) {
    printf("[ %06d ] : %s.longInteger = %lu should be %u \n", rank, name,
           in->longInteger, 1234 + shift);
    return VERIFY_FAILED;
  }
  return VERIFY_SUCCESS;
}

int verifyFloats(Floats* in, int shift, int rank, char* name) {
  if (in->singlePrec != 12.5f + shift) {
    printf("[ %06d ] : %s.singlePrec = %f should be %f \n", rank, name,
           in->singlePrec, 12.5f + shift);
    return VERIFY_FAILED;
  }
  if (in->doublePrec != 123.25 + shift) {
    printf("[ %06d ] : %s.doublePrec = %f should be %f \n", rank, name,
           in->doublePrec, 123.25 + shift);
    return VERIFY_FAILED;
  }
  // if (in->longDoublePrec != 1234.125 + shift) {
  //    printf("[ %06d ] : %s.longDoublePrec = %Lf should be %f \n", rank, name,
  //    in->longDoublePrec, 1234.125 + shift); return VERIFY_FAILED;
  //}
  return VERIFY_SUCCESS;
}

int verifyAllInts(AllInts* in, int shift, int rank, char* name) {
  int i, res = 0;
  char buff[256];
  for (i = 0; i < 5; i++) {
    snprintf(buff, sizeof(buff), "%s.integers[%d]", name, i);
    res += verifyInts(&(in->integers[i]), shift + i, rank, buff);
  }

  for (i = 0; i < 4; i++) {
    snprintf(buff, sizeof(buff), "%s.uIntegers[%d]", name, i);
    res += verifyUInts(&(in->uIntegers[i]), shift + i, rank, buff);
  }
  return res;
}

int verifyFloatsChars(FloatsChars* in, int shift, int rank, char* name) {
  int i, res = 0;
  char buff[256];
  for (i = 0; i < 3; i++) {
    snprintf(buff, sizeof(buff), "%s.floats[%d]", name, i);
    res += verifyFloats(&(in->floats[i]), shift + i, rank, buff);
  }

  for (i = 0; i < 2; i++) {
    snprintf(buff, sizeof(buff), "%s.chars[%d]", name, i);
    res += verifyChars(&(in->chars[i]), shift + i, rank, buff);
  }
  return res;
}

int verifyAllTypes(AllTypes* in, int shift, int rank, char* name) {
  char buff[256];
  snprintf(buff, sizeof(buff), "%s.allInts", name);
  int res = verifyAllInts(&(in->allInts), shift, rank, buff);
  snprintf(buff, sizeof(buff), "%s.floatsChars", name);
  res += verifyFloatsChars(&(in->floatsChars), shift, rank, buff);
  return res;
}

int init(char** argv, int* checkpoint_level, int* fail) {
  int rtn = 0;  // return value
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
int main(int argc, char** argv) {
  int checkpoint_level, fail;
  init(argv, &checkpoint_level, &fail);
  MPI_Init(&argc, &argv);
  int global_world_rank, global_world_size;  // MPI_COMM rank and size
  MPI_Comm_rank(MPI_COMM_WORLD, &global_world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &global_world_size);
  FTI_Init(argv[1], MPI_COMM_WORLD);
  int world_rank, world_size;  // FTI_COMM rank and size
  MPI_Comm_rank(FTI_COMM_WORLD, &world_rank);
  MPI_Comm_size(FTI_COMM_WORLD, &world_size);

  fti_id_t bytesType;
  FTI_InitType(&bytesType, sizeof(AsByteArray));

  // Create groups for types
  // AllIntegers datatype
  FTIT_H5Group allIntsGroup;
  FTI_InitGroup(&allIntsGroup, "All Integers", NULL);

  // Integers datatype
  FTIT_H5Group intsGroup;
  FTI_InitGroup(&intsGroup, "Integers", &allIntsGroup);

  // UIntegers datatype
  FTIT_H5Group uIntsGroup;
  FTI_InitGroup(&uIntsGroup, "Unsigned Integers", &allIntsGroup);

  // Chars And Floats datatype
  FTIT_H5Group charsAndFloatsGroup;
  FTI_InitGroup(&charsAndFloatsGroup, "Chars and Floats", NULL);

  // Chars and array of bytes
  fti_id_t CharsType = FTI_InitCompositeType("Chars", sizeof(Chars), NULL);

  int64_t dimLength[4];
  dimLength[0] = 10;
  FTI_AddVectorField(CharsType, "char array", FTI_CHAR,
      offsetof(Chars, chars), 1, dimLength);

  dimLength[0] = 2;
  dimLength[1] = 3;
  dimLength[2] = 4;
  dimLength[3] = 5;
  FTI_AddVectorField(CharsType, "unsigned char multi-array", FTI_UCHR,
      offsetof(Chars, uChars), 4, dimLength);

  dimLength[0] = 2;
  FTI_AddVectorField(CharsType, "byte array", bytesType,
      offsetof(Chars, bytes), 1, dimLength);

  // Integers
  fti_id_t IntegersType = FTI_InitCompositeType("struct Integers",
      sizeof(Integers), &intsGroup);

  FTI_AddScalarField(IntegersType, "short int", FTI_SHRT,
      offsetof(Integers, shortInteger));
  FTI_AddScalarField(IntegersType, "int", FTI_INTG,
      offsetof(Integers, integer));
  FTI_AddScalarField(IntegersType, "long int", FTI_LONG,
      offsetof(Integers, longInteger));

  // Unsigned integers
  fti_id_t UIntegersType = FTI_InitCompositeType("struct UIntegers",
      sizeof(UIntegers), &uIntsGroup);

  FTI_AddScalarField(UIntegersType, "unsigned short int", FTI_USHT,
      offsetof(UIntegers, shortInteger));
  FTI_AddScalarField(UIntegersType, "unsigned int", FTI_UINT,
      offsetof(UIntegers, integer));
  FTI_AddScalarField(UIntegersType, "unsigned long int", FTI_ULNG,
      offsetof(UIntegers, longInteger));

  // Floats
  fti_id_t FloatsType = FTI_InitCompositeType("struct Floats",
      sizeof(Floats), &charsAndFloatsGroup);

  FTI_AddScalarField(FloatsType, "float", FTI_SFLT,
      offsetof(Floats, singlePrec));
  FTI_AddScalarField(FloatsType, "double", FTI_DBLE,
      offsetof(Floats, doublePrec));

  // Integers aggregated
  fti_id_t AllIntsType = FTI_InitCompositeType("struct AllInts",
      sizeof(AllInts), &allIntsGroup);

  dimLength[0] = 5;
  FTI_AddVectorField(AllIntsType, "struct Integers array", IntegersType,
      offsetof(AllInts, integers), 1, dimLength);

  dimLength[0] = 4;
  FTI_AddVectorField(AllIntsType, "struct UIntegers array",
      UIntegersType, offsetof(AllInts, uIntegers), 1, dimLength);

  // Floats and chars aggregated
  fti_id_t FloatsCharsType = FTI_InitCompositeType("struct FloatsChars",
      sizeof(FloatsChars), &charsAndFloatsGroup);

  dimLength[0] = 3;
  FTI_AddVectorField(FloatsCharsType, "struct Floats array",
      FloatsType, offsetof(FloatsChars, floats), 1, dimLength);

  dimLength[0] = 2;
  FTI_AddVectorField(FloatsCharsType, "struct Chars array",
      CharsType, offsetof(FloatsChars, chars), 1, dimLength);

  // All types aggregated
  fti_id_t AllTypesType = FTI_InitCompositeType("struct AllTypes",
      sizeof(AllTypes), NULL);

  FTI_AddScalarField(AllTypesType, "struct AllInts", AllIntsType,
      offsetof(AllTypes, allInts));
  FTI_AddScalarField(AllTypesType, "struct FloatsChars", FloatsCharsType,
      offsetof(AllTypes, floatsChars));

  Chars charVars[2];
  Integers intVars[2];
  Integers intVars2;
  UIntegers uintVars;
  Floats floatVars;
  AllInts allIntVars;
  FloatsChars floatCharVars;
  AllTypes allTypesVar[2][2];

  // Create groups for datasets
  // AllIntegers datatype
  FTIT_H5Group allIntsGroup2;
  FTI_InitGroup(&allIntsGroup2, "All Integers for Dataset", NULL);

  FTI_Protect(1, charVars, 2, CharsType);
  FTI_DefineDataset(1, 0, NULL, "chars", &charsAndFloatsGroup);

  FTI_Protect(2, intVars, 2, IntegersType);
  FTI_DefineDataset(2, 0, NULL, "ints", &allIntsGroup2);

  FTI_Protect(3, &uintVars, 1, UIntegersType);
  FTI_DefineDataset(3, 0, NULL, "unsigned ints", &allIntsGroup2);

  FTI_Protect(4, &floatVars, 1, FloatsType);
  FTI_DefineDataset(4, 0, NULL, "floats", &charsAndFloatsGroup);

  if (fail == 1) FTI_Checkpoint(1, checkpoint_level);
  FTI_Protect(5, &allIntVars, 1, AllIntsType);
  FTI_DefineDataset(5, 0, NULL, "all ints", &allIntsGroup);

  FTI_Protect(6, &floatCharVars, 1, FloatsCharsType);
  FTI_DefineDataset(6, 0, NULL, "floats and chars", &charsAndFloatsGroup);

  dimLength[0] = 2;
  dimLength[1] = 2;
  FTI_Protect(7, allTypesVar, 4, AllTypesType);
  FTI_DefineDataset(7, 2, dimLength, "all types2D", NULL);

  FTI_Protect(8, &intVars2, 1, IntegersType);
  FTI_DefineDataset(8, 0, NULL, "ints2", &allIntsGroup2);

  if (fail == 1) {
    defaultChars(charVars, 0);
    defaultChars((charVars + 1), 1);
    defaultInts(intVars, 0);
    defaultInts((intVars + 1), 1);
    defaultInts(&intVars2, 5);
    defaultUInts(&uintVars, 0);
    defaultFloats(&floatVars, 0);
    defaultAllInts(&allIntVars, 2);
    defaultFloatsChars(&floatCharVars, 2);
    defaultAllTypes(allTypesVar[0], 3);
    defaultAllTypes((allTypesVar[0] + 1), 4);
    defaultAllTypes(allTypesVar[1], 3);
    defaultAllTypes((allTypesVar[1] + 1), 4);

    FTI_Checkpoint(2, checkpoint_level);

    dictionary* ini = iniparser_load(argv[1]);
    int heads = (int)iniparser_getint(ini, "Basic:head", -1);
    int nodeSize = (int)iniparser_getint(ini, "Basic:node_size", -1);
    int final_tag = (int)iniparser_getint(ini, "Advanced:final_tag", 3107);
    int general_tag = (int)iniparser_getint(ini, "Advanced:general_tag", 2612);
    int res;
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
      if (isInline == 0) {
        // waiting untill head do Post-checkpointing
        MPI_Recv(&res, 1, MPI_INT,
                 global_world_rank - (global_world_rank % nodeSize),
                 general_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
    }
    iniparser_freedict(ini);
    if (heads > 0) {
      res = FTI_ENDW;
      // sending END WORK to head to stop listening
      MPI_Send(&res, 1, MPI_INT,
               global_world_rank - (global_world_rank % nodeSize), final_tag,
               MPI_COMM_WORLD);
      // Barrier needed for heads (look FTI_Finalize() in api.c)
      MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Barrier(FTI_COMM_WORLD);
    // There is no FTI_Finalize(), because want to recover also from L1, L2, L3
    MPI_Finalize();
  } else {
    FTI_Recover();
    FTI_Checkpoint(3, checkpoint_level);
    int res = 0;
    res += verifyChars(charVars, 0, world_rank, "charVars[0]");
    res += verifyChars((charVars + 1), 1, world_rank, "charVars[1]");
    res += verifyInts(intVars, 0, world_rank, "intVars[0]");
    res += verifyInts((intVars + 1), 1, world_rank, "intVars[1]");
    res += verifyInts(&intVars2, 5, world_rank, "intVars2");
    res += verifyUInts(&uintVars, 0, world_rank, "uIntVars");
    res += verifyFloats(&floatVars, 0, world_rank, "floatVars");
    res += verifyAllInts(&allIntVars, 2, world_rank, "allIntVars");
    res += verifyFloatsChars(&floatCharVars, 2, world_rank, "floatCharVars");
    res += verifyAllTypes(allTypesVar[0], 3, world_rank, "allTypesVar[0][0]");
    res += verifyAllTypes((allTypesVar[0] + 1), 4, world_rank,
                          "allTypesVar[0][1]");
    res += verifyAllTypes(allTypesVar[1], 3, world_rank, "allTypesVar[1][0]");
    res += verifyAllTypes((allTypesVar[1] + 1), 4, world_rank,
                          "allTypesVar[1][1]");

    FTI_Finalize();
    MPI_Finalize();
    if (res != VERIFY_SUCCESS && world_rank == 0) {
      printf("Test failed!\n");
      return 1;
    }
    if (world_rank == 0) printf("Success.\n");
  }
  return 0;
}

