/**
 *  @file   hdf5Test.c
 *  @author Tomasz Paluszkiewicz (tomaszp@man.poznan.pl)
 *  @date   November, 2017
 *  @brief  FTI testing program.
 *
 *  Testing FTI_InitType, FTI_InitSimpleTypeWithNames,
 *  FTI_InitComplexTypeWithNames, FTI_ProtectWithName, FTI_Checkpoint, FTI_Recover,
 *  saving last checkpoint to PFS
 *
 *  Program creates complex data structures, then adds it to protect list
 *  and makes a checkpoint. Second run recovers the files and checks if
 *  values match.
 *
 *  First execution this program should be with fail flag = 1, because
 *  then FTI saves checkpoint and program stops. Second execution
 *  must be with the flag = 0 to properly recover data.
 */

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <fti.h>
#include <string.h>

#include "../deps/iniparser/iniparser.h"
#include "../deps/iniparser/dictionary.h"

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
    long double longDoublePrec;
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
    sprintf(in->chars, "%s", "abcdefg");
    int i, j, k, l;

    for (i = 0; i < 7; i++)
        in->chars[i] += shift % (25 - in->chars[i] - 'a');

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
    in->longDoublePrec = 1234.125 + shift;
}

void defaultAllInts(AllInts* in, int shift) {
    int i;
    for(i = 0; i < 5; i++) {
        defaultInts(in->integers + i, shift + i);
    }

    for(i = 0; i < 4; i++) {
        defaultUInts(in->uIntegers + i, shift + i);
    }
}

void defaultFloatsChars(FloatsChars* in, int shift) {
    int i;
    for(i = 0; i < 3; i++) {
        defaultFloats(in->floats + i, shift + i);
    }

    for(i = 0; i < 2; i++) {
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
    if (in->longDoublePrec != 1234.125 + shift) {
        printf("[ %06d ] : %s.longDoublePrec = %Lf should be %f \n", rank, name, in->longDoublePrec, 1234.125 + shift);
        return VERIFY_FAILED;
    }
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


int init(char** argv, int* checkpoint_level, int* fail) {
    int rtn = 0;    //return value
    if (argv[1] == NULL) {
        printf("Missing first parameter (config file).\n");
        rtn = 1;
    }
    if (argv[2] == NULL) {
        printf("Missing second parameter (checkpoint level).\n");
        rtn = 1;
    }
    else {
        *checkpoint_level = atoi(argv[2]);
    }
    if (argv[3] == NULL) {
        printf("Missing third parameter (if fail).\n");
        rtn = 1;
    }
    else {
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
    int global_world_rank, global_world_size; //MPI_COMM rank and size
    MPI_Comm_rank(MPI_COMM_WORLD, &global_world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &global_world_size);
    FTI_Init(argv[1], MPI_COMM_WORLD);
    int world_rank, world_size; //FTI_COMM rank and size
    MPI_Comm_rank(FTI_COMM_WORLD, &world_rank);
    MPI_Comm_size(FTI_COMM_WORLD, &world_size);

    FTIT_type bytesType;
    FTI_InitType(&bytesType, sizeof(AsByteArray));
    FTIT_complexType CharsDef;
    FTIT_type CharsType;

    CharsDef.length = 3;
    CharsDef.size = sizeof(Chars);
    sprintf(CharsDef.name, "Chars");

    int dimLength[4];
    dimLength[0] = 10;
    FTI_AddComplexFieldWithName(&CharsDef, &FTI_CHAR, F_OFFSET(Chars, chars), 1, dimLength, 0, "char array");

    dimLength[0] = 2;
    dimLength[1] = 3;
    dimLength[2] = 4;
    dimLength[3] = 5;
    FTI_AddComplexFieldWithName(&CharsDef, &FTI_UCHR, F_OFFSET(Chars, uChars), 4, dimLength, 1, "unsigned char multi-array");

    dimLength[0] = 2;
    FTI_AddComplexFieldWithName(&CharsDef, &bytesType, F_OFFSET(Chars, bytes), 1, dimLength, 2, "byte array");

    FTI_InitComplexTypeWithNames(&CharsType, &CharsDef);

    FTIT_complexType IntegersDef;
    FTIT_type IntegersType;
    IntegersDef.length = 3;
    IntegersDef.size = sizeof(Integers);
    sprintf(IntegersDef.name, "struct Integers");

    FTI_AddSimpleFieldWithName(&IntegersDef, &FTI_SHRT, F_OFFSET(Integers, shortInteger), 0, "short int");
    FTI_AddSimpleFieldWithName(&IntegersDef, &FTI_INTG, F_OFFSET(Integers, integer), 1, "int");
    FTI_AddSimpleFieldWithName(&IntegersDef, &FTI_LONG, F_OFFSET(Integers, longInteger), 2, "long int");

    FTI_InitSimpleTypeWithNames(&IntegersType, &IntegersDef);

    FTIT_complexType UIntegersDef;
    FTIT_type UIntegersType;
    UIntegersDef.length = 3;
    UIntegersDef.size = sizeof(UIntegers);
    sprintf(UIntegersDef.name, "struct UIntegers");
    FTI_AddSimpleFieldWithName(&UIntegersDef, &FTI_USHT, F_OFFSET(UIntegers, shortInteger), 0, "unsigned short int");
    FTI_AddSimpleFieldWithName(&UIntegersDef, &FTI_UINT, F_OFFSET(UIntegers, integer), 1, "unsigned int");
    FTI_AddSimpleFieldWithName(&UIntegersDef, &FTI_ULNG, F_OFFSET(UIntegers, longInteger), 2, "unsigned long int");

    FTI_InitSimpleTypeWithNames(&UIntegersType, &UIntegersDef);

    FTIT_complexType FloatsDef;
    FTIT_type FloatsType;
    FloatsDef.field[0].type = &FTI_SFLT;
    FloatsDef.field[1].type = &FTI_DBLE;
    FloatsDef.field[2].type = &FTI_LDBE;
    FloatsDef.field[0].offset = F_OFFSET(Floats, singlePrec);
    FloatsDef.field[1].offset = F_OFFSET(Floats, doublePrec);
    FloatsDef.field[2].offset = F_OFFSET(Floats, longDoublePrec);
    sprintf(FloatsDef.field[0].name, "float");
    sprintf(FloatsDef.field[1].name, "double");
    sprintf(FloatsDef.field[2].name, "long double");
    sprintf(FloatsDef.name, "struct Floats");
    FloatsDef.length = 3;
    FloatsDef.size = sizeof(Floats);
    FTI_InitSimpleTypeWithNames(&FloatsType, &FloatsDef);

    FTIT_complexType AllIntsDef;
    FTIT_type AllIntsType;
    AllIntsDef.field[0].type = &IntegersType;
    AllIntsDef.field[1].type = &UIntegersType;
    AllIntsDef.field[0].offset = F_OFFSET(AllInts, integers);
    AllIntsDef.field[1].offset = F_OFFSET(AllInts, uIntegers);
    AllIntsDef.field[0].rank = 1;
    AllIntsDef.field[1].rank = 1;
    AllIntsDef.field[0].dimLength[0] = 5;
    AllIntsDef.field[1].dimLength[0] = 4;
    sprintf(AllIntsDef.field[0].name, "struct Integers array");
    sprintf(AllIntsDef.field[1].name, "struct UIntegers array");
    sprintf(AllIntsDef.name, "sturct AllInts");
    AllIntsDef.length = 2;
    AllIntsDef.size = sizeof(AllInts);
    FTI_InitComplexTypeWithNames(&AllIntsType, &AllIntsDef);

    FTIT_complexType FloatsCharsDef;
    FTIT_type FloatsCharsType;
    FloatsCharsDef.field[0].type = &FloatsType;
    FloatsCharsDef.field[1].type = &CharsType;
    FloatsCharsDef.field[0].offset = F_OFFSET(FloatsChars, floats);
    FloatsCharsDef.field[1].offset = F_OFFSET(FloatsChars, chars);
    FloatsCharsDef.field[0].rank = 1;
    FloatsCharsDef.field[1].rank = 1;
    FloatsCharsDef.field[0].dimLength[0] = 3;
    FloatsCharsDef.field[1].dimLength[0] = 2;
    sprintf(FloatsCharsDef.field[0].name, "struct Floats array");
    sprintf(FloatsCharsDef.field[1].name, "struct Chars array");
    sprintf(FloatsCharsDef.name, "sturct FloatsChars");
    FloatsCharsDef.length = 2;
    FloatsCharsDef.size = sizeof(FloatsChars);
    FTI_InitComplexTypeWithNames(&FloatsCharsType, &FloatsCharsDef);

    FTIT_complexType AllTypesDef;
    FTIT_type AllTypesType;
    AllTypesDef.field[0].type = &AllIntsType;
    AllTypesDef.field[1].type = &FloatsCharsType;
    AllTypesDef.field[0].offset = F_OFFSET(AllTypes, allInts);
    AllTypesDef.field[1].offset = F_OFFSET(AllTypes, floatsChars);
    sprintf(AllTypesDef.field[0].name, "sturct AllInts");
    sprintf(AllTypesDef.field[1].name, "sturct FloatsChars");
    sprintf(AllTypesDef.name, "struct AllTypes");
    AllTypesDef.length = 2;
    AllTypesDef.size = sizeof(AllTypes);
    FTI_InitSimpleTypeWithNames(&AllTypesType, &AllTypesDef);

    Chars charVars[2];
    Integers intVars[2];
    Integers intVars2;
    UIntegers uintVars;
    Floats floatVars;
    AllInts allIntVars;
    FloatsChars floatCharVars;
    AllTypes allTypesVar[2];

    FTI_ProtectWithName(1, charVars, 2, CharsType, "chars");
    FTI_ProtectWithName(2, intVars, 2, IntegersType, "ints");
    FTI_ProtectWithName(3, &uintVars, 1, UIntegersType, "unsigned ints");
    FTI_ProtectWithName(4, &floatVars, 1, FloatsType, "floats");
    if (fail == 1) FTI_Checkpoint(1, checkpoint_level);
    FTI_ProtectWithName(5, &allIntVars, 1, AllIntsType, "all ints");
    FTI_ProtectWithName(6, &floatCharVars, 1, FloatsCharsType, "floats and chars");
    FTI_ProtectWithName(7, allTypesVar, 2, AllTypesType, "all types");
    FTI_ProtectWithName(8, &intVars2, 1, IntegersType, "ints2");

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
        defaultAllTypes(allTypesVar, 3);
        defaultAllTypes((allTypesVar + 1), 4);

        FTI_Checkpoint(2, checkpoint_level);

        dictionary* ini = iniparser_load(argv[1]);
        int heads = (int)iniparser_getint(ini, "Basic:head", -1);
        int nodeSize = (int)iniparser_getint(ini, "Basic:node_size", -1);
        int tag = (int)iniparser_getint(ini, "Advanced:mpi_tag", -1);
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
                MPI_Recv(&res, 1, MPI_INT, global_world_rank - (global_world_rank % nodeSize) , tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        iniparser_freedict(ini);
        if (heads > 0) {
            res = FTI_ENDW;
            //sending END WORK to head to stop listening
            MPI_Send(&res, 1, MPI_INT, global_world_rank - (global_world_rank % nodeSize), tag, MPI_COMM_WORLD);
            //Barrier needed for heads (look FTI_Finalize() in api.c)
            MPI_Barrier(MPI_COMM_WORLD);
        }

        MPI_Barrier(FTI_COMM_WORLD);
        //There is no FTI_Finalize(), because want to recover also from L1, L2, L3
        MPI_Finalize();
    }
    else {
        FTI_Recover();
        FTI_Checkpoint(3, checkpoint_level);
        int res = 0;
        res += verifyChars(charVars, 0, world_rank, "charVars[0]");
        res += verifyChars((charVars + 1),  1, world_rank, "charVars[1]");
        res += verifyInts(intVars, 0, world_rank, "intVars[0]");
        res += verifyInts((intVars + 1), 1, world_rank, "intVars[1]");
        res += verifyInts(&intVars2, 5, world_rank, "intVars2");
        res += verifyUInts(&uintVars,  0, world_rank, "uIntVars");
        res += verifyFloats(&floatVars, 0, world_rank, "floatVars");
        res += verifyAllInts(&allIntVars, 2, world_rank, "allIntVars");
        res += verifyFloatsChars(&floatCharVars, 2, world_rank, "floatCharVars");
        res += verifyAllTypes(allTypesVar, 3, world_rank, "allTypesVar[0]");
        res += verifyAllTypes((allTypesVar + 1), 4, world_rank, "allTypesVar[1]");

        FTI_Finalize();
        MPI_Finalize();
        if (res != VERIFY_SUCCESS && world_rank == 0) {
            printf("Test failed!\n");
            return 1;
        }
        if (world_rank == 0)
            printf("Success.\n");
    }
    return 0;
}
