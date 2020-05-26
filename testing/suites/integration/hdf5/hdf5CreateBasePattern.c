/**
 *  Copyright (c) 2017 Leonardo A. Bautista-Gomez
 *  All rights reserved
 *
 *  @file   hdf5Test.c
 *  @author Tomasz Paluszkiewicz (tomaszp@man.poznan.pl)
 *  @author Kai Keller (kellekai@gmx.de)
 *  @date   November, 2017
 *  @brief  FTI testing program.
 *
 *  Program creates HDF5 file hiearchy with groups,
 *  datatypes and datasets identical to the FTI HDF5
 *  test program hdf5Test.c
 */

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "hdf5.h"
#include "hdf5_hl.h"

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
  short int shortInteger;
  int integer;
  long int longInteger;
} Integers;

typedef struct UIntegers {
  unsigned short int shortInteger;
  unsigned int integer;
  unsigned long int longInteger;
} UIntegers;

typedef struct Floats {
  float singlePrec;
  double doublePrec;
  // makes trouble  ->  long double longDoublePrec;
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

int main(int argc, char** argv) {
  if (argc < 2)
    return 1;

  // create hdf5 file
  hid_t file_id = H5Fcreate(argv[1], H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  // root group id is file id
  hid_t root_group_id = file_id;

  // Create groups for types
  // AllIntegers datatype
  hid_t allIntsGroup = H5Gcreate2(root_group_id, "All Integers", H5P_DEFAULT,
                                  H5P_DEFAULT, H5P_DEFAULT);
  hid_t allIntsGroup2 = H5Gcreate2(root_group_id, "All Integers for Dataset",
                                   H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  // Integers datatype
  hid_t intsGroup = H5Gcreate2(allIntsGroup, "Integers", H5P_DEFAULT,
                               H5P_DEFAULT, H5P_DEFAULT);

  // UIntegers datatype
  hid_t uIntsGroup = H5Gcreate2(allIntsGroup, "Unsigned Integers", H5P_DEFAULT,
                                H5P_DEFAULT, H5P_DEFAULT);

  // Chars And Floats datatype
  hid_t charsAndFloatsGroup = H5Gcreate2(root_group_id, "Chars and Floats",
                                         H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  /*
   * Chars -> Complex Type
   * --------------------------------------
   *
   * > 3 fields
   *
   * |       Type         |       rank        |       dim         | name |
   * |========================================================================================|
   * |    FTI_CHAR        |       1           |       {10}        |       char
   * array          | |    FTI_UCHR        |       4           |     {2,3,4,5}
   * | unsigned char multiarray  | |    bytesType       |       1           |
   * {2}         |       byte array          |
   * |========================================================================================|
   *
   * > belongs to 'charsAndFloatsGroup' group
   *
   */

  Chars charVars[2];
  defaultChars(charVars, 0);
  defaultChars((charVars + 1), 1);
  hid_t charsType = H5Tcreate(H5T_COMPOUND, sizeof(Chars));

  hid_t bytesType = H5Tcopy(H5T_NATIVE_CHAR);
  H5Tset_size(bytesType, sizeof(AsByteArray));

  hsize_t dimChar = 10;
  hsize_t dimUChar[4] = {2, 3, 4, 5};
  hsize_t dimBytesType = 2;

  hid_t charArray = H5Tarray_create(H5T_NATIVE_CHAR, 1, &dimChar);
  hid_t uCharArray = H5Tarray_create(H5T_NATIVE_UCHAR, 4, dimUChar);
  hid_t bytesTypeArray = H5Tarray_create(bytesType, 1, &dimBytesType);

  H5Tinsert(charsType, "char array", offsetof(Chars, chars), charArray);
  H5Tinsert(charsType, "unsigned char multi-array", offsetof(Chars, uChars),
            uCharArray);
  H5Tinsert(charsType, "byte array", offsetof(Chars, bytes), bytesTypeArray);

  H5Tcommit(file_id, "Chars", charsType, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  hsize_t dimChars = 2;

  H5LTmake_dataset(charsAndFloatsGroup, "chars", 1, &dimChars, charsType,
                   charVars);

  /*
   * Integers -> Simple Fields
   * --------------------------------------
   *
   * > 3 fields
   *
   * |       Type         |       rank        |        dim        | name |
   * |========================================================================================|
   * |    FTI_SHRT        |        0          |         -         |        short
   * int          | |    FTI_INTG        |        0          |         - | int |
   * |    FTI_LONG        |        0          |         -         |         long
   * int          |
   * |========================================================================================|
   *
   * > belongs to 'intsGroup' group
   *
   */

  Integers intVars[2];
  Integers intVars2;
  defaultInts(intVars, 0);
  defaultInts((intVars + 1), 1);
  defaultInts(&intVars2, 5);
  hid_t intType = H5Tcreate(H5T_COMPOUND, sizeof(Integers));

  H5Tinsert(intType, "short int", offsetof(Integers, shortInteger),
            H5T_NATIVE_SHORT);
  H5Tinsert(intType, "int", offsetof(Integers, integer), H5T_NATIVE_INT);
  H5Tinsert(intType, "long int", offsetof(Integers, longInteger),
            H5T_NATIVE_LONG);

  H5Tcommit(intsGroup, "struct Integers", intType, H5P_DEFAULT, H5P_DEFAULT,
            H5P_DEFAULT);

  hsize_t dimIntegers = 2;
  hsize_t dimIntegers2 = 1;

  H5LTmake_dataset(allIntsGroup2, "ints", 1, &dimIntegers, intType, intVars);
  H5LTmake_dataset(allIntsGroup2, "ints2", 1, &dimIntegers2, intType,
                   &intVars2);

  /*
   * Unsigned Integers -> Simple Fields
   * --------------------------------------
   *
   * > 3 fields
   *
   * |       Type         |       rank        |        dim        | name |
   * |========================================================================================|
   * |    FTI_USHT        |        0          |         -         |    unsigned
   * short int     | |    FTI_UINT        |        0          |         - |
   * unsigned int        | |    FTI_ULNG        |        0          |         -
   * |     unsigned long int     |
   * |========================================================================================|
   *
   * > belongs to 'uIntsGroup' group
   *
   */

  UIntegers uintVars;
  defaultUInts(&uintVars, 0);

  hid_t uIntType = H5Tcreate(H5T_COMPOUND, sizeof(UIntegers));

  H5Tinsert(uIntType, "unsigned short int", offsetof(UIntegers, shortInteger),
            H5T_NATIVE_USHORT);
  H5Tinsert(uIntType, "unsigned int", offsetof(UIntegers, integer),
            H5T_NATIVE_UINT);
  H5Tinsert(uIntType, "unsigned long int", offsetof(UIntegers, longInteger),
            H5T_NATIVE_ULONG);

  H5Tcommit(uIntsGroup, "struct UIntegers", uIntType, H5P_DEFAULT, H5P_DEFAULT,
            H5P_DEFAULT);

  hsize_t dimUIntegers = 1;

  H5LTmake_dataset(allIntsGroup2, "unsigned ints", 1, &dimUIntegers, uIntType,
                   &uintVars);

  /*
   * Floats -> Simple Fields
   * --------------------------------------
   *
   * > 3 fields
   *
   * |       Type         |       rank        |        dim        | name |
   * |========================================================================================|
   * |    FTI_SFLT        |        0          |         -         | float | |
   * FTI_DBLE        |        0          |         -         |          double |
   * |    FTI_LDBE        |        0          |         -         |        long
   * double        |
   * |========================================================================================|
   *
   * > belongs to 'charsAndFloatsGroup' group
   *
   */

  Floats floatVars;
  defaultFloats(&floatVars, 0);

  hid_t floatsType = H5Tcreate(H5T_COMPOUND, sizeof(Floats));

  H5Tinsert(floatsType, "float", offsetof(Floats, singlePrec),
            H5T_NATIVE_FLOAT);
  H5Tinsert(floatsType, "double", offsetof(Floats, doublePrec),
            H5T_NATIVE_DOUBLE);
  // H5Tinsert( floatsType, "long double", offsetof(Floats, longDoublePrec),
  // H5T_NATIVE_LDOUBLE);

  H5Tcommit(charsAndFloatsGroup, "struct Floats", floatsType, H5P_DEFAULT,
            H5P_DEFAULT, H5P_DEFAULT);

  hsize_t dimFloats = 1;

  H5LTmake_dataset(charsAndFloatsGroup, "floats", 1, &dimFloats, floatsType,
                   &floatVars);

  /*
   * Integers Aggregated -> Complex Fields
   * --------------------------------------
   *
   * > 2 fields
   *
   * |       Type         |       rank        |        dim        | name |
   * |========================================================================================|
   * |    Integers        |        1          |         5         |  struct
   * Integers array    | |    UIntegers       |        1          |         4 |
   * struct UIntegers array   |
   * |========================================================================================|
   *
   * > belongs to 'allIntsGroup' group
   *
   */

  AllInts allIntVars;
  defaultAllInts(&allIntVars, 2);

  hid_t allIntsType = H5Tcreate(H5T_COMPOUND, sizeof(AllInts));

  hsize_t dimInteger = 5;
  hsize_t dimUInteger = 4;

  hid_t IntegersArray = H5Tarray_create(intType, 1, &dimInteger);
  hid_t UIntegersArray = H5Tarray_create(uIntType, 1, &dimUInteger);

  H5Tinsert(allIntsType, "struct Integers array", offsetof(AllInts, integers),
            IntegersArray);
  H5Tinsert(allIntsType, "struct UIntegers array", offsetof(AllInts, uIntegers),
            UIntegersArray);

  H5Tcommit(allIntsGroup, "struct AllInts", allIntsType, H5P_DEFAULT,
            H5P_DEFAULT, H5P_DEFAULT);

  hsize_t dimAllInts = 1;

  H5LTmake_dataset(allIntsGroup, "all ints", 1, &dimAllInts, allIntsType,
                   &allIntVars);

  /*
   * Chars and Floats Aggregated -> Complex Fields
   * --------------------------------------
   *
   * > 3 fields
   *
   * |       Type         |       rank        |        dim        | name |
   * |========================================================================================|
   * |    Floats          |        1          |         3         |    struct
   * Floats array    | |    Chars           |        1          |         2 |
   * struct Chars array     |
   * |========================================================================================|
   *
   * > belongs to 'charsAndFloatsGroup' group
   *
   */

  FloatsChars floatCharVars;
  defaultFloatsChars(&floatCharVars, 2);

  hid_t floatsCharsType = H5Tcreate(H5T_COMPOUND, sizeof(FloatsChars));

  hsize_t dimAggFloats = 3;
  hsize_t dimAggChars = 2;

  hid_t FloatsArray = H5Tarray_create(floatsType, 1, &dimAggFloats);
  hid_t CharsArray = H5Tarray_create(charsType, 1, &dimAggChars);

  H5Tinsert(floatsCharsType, "struct Floats array",
            offsetof(FloatsChars, floats), FloatsArray);
  H5Tinsert(floatsCharsType, "struct Chars array", offsetof(FloatsChars, chars),
            CharsArray);

  H5Tcommit(charsAndFloatsGroup, "struct FloatsChars", floatsCharsType,
            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  hsize_t dimFloatsChars = 1;

  H5LTmake_dataset(charsAndFloatsGroup, "floats and chars", 1, &dimFloatsChars,
                   floatsCharsType, &floatCharVars);

  /*
   * AllTypes -> Simple Fields
   * --------------------------------------
   *
   * > 2 fields
   *
   * |       Type         |       rank        |        dim        | name |
   * |========================================================================================|
   * |     AllInts        |        0          |         -         |       struct
   * AllInts      | |   FloatsChars      |        0          |         - |
   * struct FloatsChars    |
   * |========================================================================================|
   *
   * > belongs to 'root_group_id' group
   *
   */

  AllTypes allTypesVar[2][2];
  defaultAllTypes(allTypesVar[0], 3);
  defaultAllTypes((allTypesVar[0] + 1), 4);
  defaultAllTypes(allTypesVar[1], 3);
  defaultAllTypes((allTypesVar[1] + 1), 4);

  hid_t allTypesType = H5Tcreate(H5T_COMPOUND, sizeof(AllTypes));

  H5Tinsert(allTypesType, "struct AllInts", offsetof(AllTypes, allInts),
            allIntsType);
  H5Tinsert(allTypesType, "struct FloatsChars", offsetof(AllTypes, floatsChars),
            floatsCharsType);

  H5Tcommit(root_group_id, "struct AllTypes", allTypesType, H5P_DEFAULT,
            H5P_DEFAULT, H5P_DEFAULT);

  hsize_t dimAllTypes[2] = {2, 2};

  H5LTmake_dataset(root_group_id, "all types2D", 2, dimAllTypes, allTypesType,
                   allTypesVar);

  H5Fclose(file_id);

  return 0;
}
