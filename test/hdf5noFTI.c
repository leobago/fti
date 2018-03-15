/**
 *  @file   hdf5noFTI.c
 *  @author Karol Sierocinski (ksiero@man.poznan.pl)
 *  @author Tomasz Paluszkiewicz (tomaszp@man.poznan.pl)
 *  @date   January, 2018
 *  @brief  FTI testing program.
 *
 *  Testing correctness of HDF5 checkpoint file.
 *
 *  Program loads datatypes saved in hdf5 file by name and verifies values.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <hdf5.h>
#include <hdf5_hl.h>

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
    short int shortInteger;
    int integer;
    long int longInteger;
} Integers;

typedef struct UIntegers {
    unsigned short int shortInteger;
    unsigned int integer;
    unsigned long int longInteger;
} UIntegers;

typedef struct Floats{
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

int verifyChars(Chars* in, int shift, int rank, char* name) {
    char buff[] = "abcdefg";
    int i;
    for (i = 0; i < 7; i++)
        buff[i] += shift % (25 - buff[i]- 'a');
    if (strcmp(in->chars, buff)) {
        printf("[ %06d ] : %s.chars = %s should be %s \n", rank, name, in->chars, buff);
        return VERIFY_FAILED;
    }
    int j, k, l;
    for (i = 0; i < 2; i++) {
        for (j = 0; j < 3; j++) {
            for (k = 0; k < 4; k++) {
                for (l = 0; l < 5; l++) {
                    if (in->uChars[i][j][k][l] != 'a' + ((i + j + k + l) + shift) % 25) {
                        printf("[ %06d ] : %s.uChars[%d][%d][%d][%d] = %u should be %u \n", rank, name,
                                i, j, k, l, in->uChars[i][j][k][l], 'a' + ((i + j + k + l) + shift) % 25);
                        return VERIFY_FAILED;
                    }
                }
            }
        }
        if (in->bytes[i].character != 'C' + shift % 23) {
            printf("[ %06d ] : %s.bytes[%d].character = %c should be %c \n",
                    rank, name, i, in->bytes[i].character, ('C' + shift % 23));
            return VERIFY_FAILED;
        }
        for (j = 0; j < 1024; j++) {
            if (in->bytes[i].longs[j] != (j + 1) * 2 + shift) {
                printf("[ %06d ] : %s.bytes[%d].longs[%d] = %ld should be %d \n",
                        rank, name, i, j, in->bytes[i].longs[j], (j + 1) * 2 + shift);
                return VERIFY_FAILED;
            }
        }
    }
    return VERIFY_SUCCESS;
}

int verifyInts(Integers* in, int shift, int rank, char* name) {
    if (in->shortInteger != -12 - shift) {
        printf("[ %06d ] : %s.shortInteger = %hd should be %hd \n", rank, name, in->shortInteger, -12 - shift);
        return VERIFY_FAILED;
    }
    if (in->integer != -123 - shift) {
        printf("[ %06d ] : %s.shortInteger = %d should be %d \n", rank, name, in->integer, -123 - shift);
        return VERIFY_FAILED;
    }
    if (in->longInteger != -1234 - shift) {
        printf("[ %06d ] : %s.shortInteger = %ld should be %d \n", rank, name, in->longInteger, -1234 - shift);
        return VERIFY_FAILED;
    }
    return VERIFY_SUCCESS;
}

int verifyUInts(UIntegers* in, int shift, int rank, char* name) {
    if (in->shortInteger != 12 + shift || in->integer != 123 + shift || in->longInteger != 1234 + shift)
        return VERIFY_FAILED;

    if (in->shortInteger != 12 + shift) {
        printf("[ %06d ] : %s.shortInteger = %hu should be %hu \n", rank, name, in->shortInteger, 12 + shift);
        return VERIFY_FAILED;
    }
    if (in->integer != 123 + shift) {
        printf("[ %06d ] : %s.shortInteger = %u should be %u \n", rank, name, in->integer, 123 + shift);
        return VERIFY_FAILED;
    }
    if (in->longInteger != 1234 + shift) {
        printf("[ %06d ] : %s.shortInteger = %lu should be %u \n", rank, name, in->longInteger, 1234 + shift);
        return VERIFY_FAILED;
    }
    return VERIFY_SUCCESS;
}

int verifyFloats(Floats * in, int shift, int rank, char* name) {
    if (in->singlePrec != 12.5f + shift) {
        printf("[ %06d ] : %s.singlePrec = %f should be %f \n", rank, name, in->singlePrec, 12.5f + shift);
        return VERIFY_FAILED;
    }
    if (in->doublePrec != 123.25 + shift) {
        printf("[ %06d ] : %s.doublePrec = %f should be %f \n", rank, name, in->doublePrec, 123.25 + shift);
        return VERIFY_FAILED;
    }
    //if (in->longDoublePrec != 1234.125 + shift) {
    //    printf("[ %06d ] : %s.longDoublePrec = %Lf should be %f \n", rank, name, in->longDoublePrec, 1234.125 + shift);
    //    return VERIFY_FAILED;
    //}
    return VERIFY_SUCCESS;
}

int verifyAllInts(AllInts * in, int shift, int rank, char* name) {
    int i, res = 0;
    char buff[256];
    for(i = 0; i < 5; i++) {
        sprintf(buff, "%s.integers[%d]", name, i);
        res += verifyInts(&(in->integers[i]), shift + i, rank, buff);
    }

    for(i = 0; i < 4; i++) {
        sprintf(buff, "%s.uIntegers[%d]", name, i);
        res += verifyUInts(&(in->uIntegers[i]), shift + i, rank, buff);
    }
    return res;
}

int verifyFloatsChars(FloatsChars * in, int shift, int rank, char* name) {
    int i, res = 0;
    char buff[256];
    for(i = 0; i < 3; i++) {
        sprintf(buff, "%s.floats[%d]", name, i);
        res += verifyFloats(&(in->floats[i]), shift + i, rank, buff);
    }

    for(i = 0; i < 2; i++) {
        sprintf(buff, "%s.chars[%d]", name, i);
        res += verifyChars(&(in->chars[i]), shift + i, rank, buff);
    }
    return res;
}

int verifyAllTypes(AllTypes * in, int shift, int rank, char* name) {
    char buff[256];
    sprintf(buff, "%s.allInts", name);
    int res = verifyAllInts(&(in->allInts), shift, rank, buff);
    sprintf(buff, "%s.floatsChars", name);
    res += verifyFloatsChars(&(in->floatsChars), shift, rank, buff);
    return res;
}

int main(int argc, char** argv)
{
    //open file
    hid_t file_id = H5Fopen("offlineVerify.h5", H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) {
        printf("Could not open checkpoint file.");
        return 1;
    }
    
    //Create groups for types
        //AllIntegers datatype
        hid_t allIntsGroup = H5Gopen(file_id, "All Integers", H5P_DEFAULT);

        //Integers datatype
        hid_t intsGroup = H5Gopen(allIntsGroup, "Integers", H5P_DEFAULT);

        //UIntegers datatype
        hid_t uIntsGroup = H5Gopen(allIntsGroup, "Unsigned Integers", H5P_DEFAULT);

        //Chars And Floats datatype
        hid_t charsAndFloatsGroup = H5Gopen(file_id, "Chars and Floats", H5P_DEFAULT);

        //AllIntegers datatype
        hid_t allIntsGroup2 = H5Gopen(file_id, "All Integers for Dataset", H5P_DEFAULT);

    //open types
    hid_t chars_id = H5Topen(file_id, "Chars", H5P_DEFAULT);
    hid_t floats_id = H5Topen(charsAndFloatsGroup, "struct Floats", H5P_DEFAULT);
    hid_t floatChars_id = H5Topen(charsAndFloatsGroup, "struct FloatsChars", H5P_DEFAULT);

    hid_t integers_id = H5Topen(intsGroup, "struct Integers", H5P_DEFAULT);
    hid_t uintegers_id = H5Topen(uIntsGroup, "struct UIntegers", H5P_DEFAULT);
    hid_t allInts_id = H5Topen(allIntsGroup, "struct AllInts", H5P_DEFAULT);

    hid_t allTypes_id = H5Topen(file_id, "struct AllTypes", H5P_DEFAULT);

    //read datasets
    Chars charVars[2];
    herr_t res = H5LTread_dataset(charsAndFloatsGroup, "chars", chars_id, &charVars);
    if (res != 0) printf("Cannot read dataset!\n");

    Floats floatVars;
    res = H5LTread_dataset(charsAndFloatsGroup, "floats", floats_id , &floatVars);
    if (res != 0) printf("Cannot read dataset!\n");

    FloatsChars floatCharVars;
    res = H5LTread_dataset(charsAndFloatsGroup, "floats and chars", floatChars_id, &floatCharVars);
    if (res != 0) printf("Cannot read dataset!\n");

    Integers intVars[2];
    res = H5LTread_dataset(allIntsGroup2, "ints", integers_id, &intVars);
    if (res != 0) printf("Cannot read dataset!\n");

    UIntegers uintVars;
    res = H5LTread_dataset(allIntsGroup2, "unsigned ints", uintegers_id, &uintVars);
    if (res != 0) printf("Cannot read dataset!\n");

    AllInts allIntVars;
    res = H5LTread_dataset(allIntsGroup, "all ints", allInts_id, &allIntVars);
    if (res != 0) printf("Cannot read dataset!\n");

    AllTypes allTypesVar[2][2];
    res = H5LTread_dataset(file_id, "all types2D", allTypes_id, &allTypesVar);
    if (res != 0) printf("Cannot read dataset!\n");

    Integers intVars2;
    res = H5LTread_dataset(allIntsGroup2, "ints2", integers_id, &intVars2);
    if (res != 0) printf("Cannot read dataset!\n");

    //close types
    H5Tclose(chars_id);
    H5Tclose(floats_id);
    H5Tclose(floatChars_id);
    H5Tclose(integers_id);
    H5Tclose(uintegers_id);
    H5Tclose(allInts_id);
    H5Tclose(allTypes_id);

    //close groups
    H5Gclose(allIntsGroup);
    H5Gclose(intsGroup);
    H5Gclose(uIntsGroup);
    H5Gclose(charsAndFloatsGroup);
    H5Gclose(allIntsGroup2);

    //close file
    H5Fclose(file_id);

    int res1 = 0;
    int world_rank = 0;
    res1 += verifyChars(charVars, 0, world_rank, "charVars[0]");
    res1 += verifyChars((charVars + 1),  1, world_rank, "charVars[1]");
    res1 += verifyInts(intVars, 0, world_rank, "intVars[0]");
    res1 += verifyInts((intVars + 1), 1, world_rank, "intVars[1]");
    res1 += verifyInts(&intVars2, 5, world_rank, "intVars2");
    res1 += verifyUInts(&uintVars,  0, world_rank, "uIntVars");
    res1 += verifyFloats(&floatVars, 0, world_rank, "floatVars");
    res1 += verifyAllInts(&allIntVars, 2, world_rank, "allIntVars");
    res1 += verifyFloatsChars(&floatCharVars, 2, world_rank, "floatCharVars");
    res1 += verifyAllTypes(allTypesVar[0], 3, world_rank, "allTypesVar[0][0]");
    res1 += verifyAllTypes((allTypesVar[0] + 1), 4, world_rank, "allTypesVar[0][1]");
    res1 += verifyAllTypes(allTypesVar[1], 3, world_rank, "allTypesVar[1][0]");
    res1 += verifyAllTypes((allTypesVar[1] + 1), 4, world_rank, "allTypesVar[1][1]");

    if (res1 != 0) {
        printf("Failed!");
        return 1;
    }
    printf("Success!\n");
    return 0;
}
