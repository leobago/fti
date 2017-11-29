/**
 *  @file   addInArray.c
 *  @author Karol Sierocinski (ksiero@man.poznan.pl)
 *  @date   Feburary, 2017
 *  @brief  FTI testing program.
 *
 *  Testing FTI_Init, FTI_Checkpoint, FTI_Status, FTI_Recover, FTI_Finalize,
 *  saving last checkpoint to PFS
 *
 *  Program adds number in array, does MPI_Allgather each iteration and checkpoint
 *  every ITER_CHECK interations with level passed in argv, but recovery is always
 *  from L4, because of FTI_Finalize() call.
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

#include "../deps/iniparser/iniparser.h"
#include "../deps/iniparser/dictionary.h"

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

void defaultChars(Chars * in) {
    sprintf(in->chars, "%s", "abcdefg");
    int i, j, k, l;
    for (i = 0; i < 2; i++) {
        for (j = 0; j < 3; j++) {
            for (k = 0; k < 4; k++) {
                for (l = 0; l < 5; l++) {
                    in->uChars[i][j][k][l] = 'a' + (i + j + k + l) % 25;
                }
            }
        }
    }
}

void defaultInts(Integers * in) {
    in->shortInteger = -12;
    in->integer = -123;
    in->longInteger = -1234;
}

void defaultUInts(UIntegers * in) {
    in->shortInteger = 12;
    in->integer = 123;
    in->longInteger = 1234;
}

void defaultFloats(Floats * in) {
    in->singlePrec = 12.5f;
    in->doublePrec = 123.25;
    in->longDoublePrec = 1234.125;
}

void defaultAllInts(AllInts * in) {
    int i;
    for(i = 0; i < 5; i++) {
        defaultInts(in->integers + i);
    }

    for(i = 0; i < 4; i++) {
        defaultUInts(in->uIntegers + i);
    }
}

void defaultFloatsChars(FloatsChars * in) {
    int i;
    for(i = 0; i < 3; i++) {
        defaultFloats(in->floats + i);
    }

    for(i = 0; i < 2; i++) {
        defaultChars(in->chars + i);
    }
}

void defaultAllTypes(AllTypes * in) {
    defaultAllInts(&in->allInts);
    defaultFloatsChars(&in->floatsChars);
}


int verifyChars(Chars * in) {
    if (strcmp(in->chars, "abcdefg"))
        return 1;
    int i, j, k, l;
    for (i = 0; i < 2; i++) {
        for (j = 0; j < 3; j++) {
            for (k = 0; k < 4; k++) {
                for (l = 0; l < 5; l++) {
                    if (in->uChars[i][j][k][l] != 'a' + (i + j + k + l) % 25)
                        return 1;
                }
            }
        }
    }
    return 0;
}

int verifyInts(Integers * in) {
    if (in->shortInteger != -12 || in->integer != -123 || in->longInteger != -1234)
        return 1;
    return 0;
}

int verifyUInts(UIntegers * in) {
    if (in->shortInteger != 12 || in->integer != 123 || in->longInteger != 1234)
        return 1;
    return 0;
}

int verifyFloats(Floats * in) {
    if (in->singlePrec != 12.5f || in->doublePrec != 123.25 || in->longDoublePrec != 1234.125)
        return 1;
    return 0;
}

int verifyAllInts(AllInts * in) {
    int i, res = 0;
    for(i = 0; i < 5; i++) {
        res += verifyInts(&(in->integers[i]));
    }

    for(i = 0; i < 4; i++) {
        res += verifyUInts(&(in->uIntegers[i]));
    }
    return res;
}

int verifyFloatsChars(FloatsChars * in) {
    int i, res = 0;
    for(i = 0; i < 3; i++) {
        res += verifyFloats(&(in->floats[i]));
    }

    for(i = 0; i < 2; i++) {
        res += verifyChars(&(in->chars[i]));
    }
    return res;
}

int verifyAllTypes(AllTypes * in) {
    return verifyAllInts(&(in->allInts)) + verifyFloatsChars(&(in->floatsChars));
}


int init(char** argv, int* checkpoint_level, int* fail)
{
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
int main(int argc, char** argv)
{
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
    FTI_InitType(&bytesType, sizeof(bytesType));

    FTIT_complexType CharsDef;
    FTIT_type CharsType;
    CharsDef.field[0].type = &FTI_CHAR;
    CharsDef.field[1].type = &FTI_UCHR;
    CharsDef.field[2].type = &bytesType;
    CharsDef.field[0].offset = F_OFFSET(Chars, chars);
    CharsDef.field[1].offset = F_OFFSET(Chars, uChars);
    CharsDef.field[2].offset = F_OFFSET(Chars, bytes);
    CharsDef.field[0].rank = 1;
    CharsDef.field[1].rank = 4;
    CharsDef.field[2].rank = 1;
    CharsDef.field[0].dimLength[0] = 10;
    CharsDef.field[1].dimLength[0] = 2;
    CharsDef.field[1].dimLength[1] = 3;
    CharsDef.field[1].dimLength[2] = 4;
    CharsDef.field[1].dimLength[3] = 5;
    CharsDef.field[2].dimLength[0] = 2;
    sprintf(CharsDef.field[0].name, "char array");
    sprintf(CharsDef.field[1].name, "unsigned char multi-array");
    sprintf(CharsDef.field[2].name, "byte array");
    sprintf(CharsDef.name, "Chars");
    CharsDef.length = 3;
    CharsDef.size = sizeof(Chars);
    FTI_InitComplexTypeWithNames(&CharsType, &CharsDef);

    FTIT_complexType IntegersDef;
    FTIT_type IntegersType;
    IntegersDef.field[0].type = &FTI_SHRT;
    IntegersDef.field[1].type = &FTI_INTG;
    IntegersDef.field[2].type = &FTI_LONG;
    IntegersDef.field[0].offset = F_OFFSET(Integers, shortInteger);
    IntegersDef.field[1].offset = F_OFFSET(Integers, integer);
    IntegersDef.field[2].offset = F_OFFSET(Integers, longInteger);
    sprintf(IntegersDef.field[0].name, "short int");
    sprintf(IntegersDef.field[1].name, "int");
    sprintf(IntegersDef.field[2].name, "long int");
    sprintf(IntegersDef.name, "struct Integers");
    IntegersDef.length = 3;
    IntegersDef.size = sizeof(Integers);
    FTI_InitSimpleTypeWithNames(&IntegersType, &IntegersDef);

    FTIT_complexType UIntegersDef;
    FTIT_type UIntegersType;
    UIntegersDef.field[0].type = &FTI_USHT;
    UIntegersDef.field[1].type = &FTI_UINT;
    UIntegersDef.field[2].type = &FTI_ULNG;
    UIntegersDef.field[0].offset = F_OFFSET(UIntegers, shortInteger);
    UIntegersDef.field[1].offset = F_OFFSET(UIntegers, integer);
    UIntegersDef.field[2].offset = F_OFFSET(UIntegers, longInteger);
    sprintf(UIntegersDef.field[0].name, "unsigned short int");
    sprintf(UIntegersDef.field[1].name, "unsigned int");
    sprintf(UIntegersDef.field[2].name, "unsigned long int");
    sprintf(UIntegersDef.name, "struct UIntegers");
    UIntegersDef.length = 3;
    UIntegersDef.size = sizeof(UIntegers);
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
    UIntegers uintVars;
    Floats floatVars;
    AllInts allIntVars;
    FloatsChars floatCharVars;
    AllTypes allTypesVar[2];

    FTI_ProtectWithName(1, &charVars, 2, CharsType, "chars");
    FTI_ProtectWithName(2, &intVars, 2, IntegersType, "ints");
    FTI_ProtectWithName(3, &uintVars, 1, UIntegersType, "unsigned ints");
    FTI_ProtectWithName(4, &floatVars, 1, FloatsType, "floats");
    if (fail == 1) FTI_Checkpoint(1, checkpoint_level);
    FTI_ProtectWithName(5, &allIntVars, 1, AllIntsType, "all ints");
    FTI_ProtectWithName(6, &floatCharVars, 1, FloatsCharsType, "floats and chars");
    FTI_ProtectWithName(7, &allTypesVar, 2, AllTypesType, "all types");
    FTI_ProtectWithName(8, &intVars, 1, IntegersType, "ints2");


    if (fail == 1) {
        defaultChars(charVars);
        defaultChars((charVars + 1));
        defaultInts(intVars);
        defaultInts((intVars + 1));
        defaultUInts(&uintVars);
        defaultFloats(&floatVars);
        defaultAllInts(&allIntVars);
        defaultFloatsChars(&floatCharVars);
        defaultAllTypes(allTypesVar);
        defaultAllTypes((allTypesVar + 1));

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
        //FTI_Finalize();
        //MPI_Finalize();
    }
    else {
        FTI_Recover();
        FTI_Checkpoint(3, checkpoint_level);
        int res = 0;
        res += verifyChars(charVars);
        res += verifyChars((charVars + 1));
        res += verifyInts(intVars);
        res += verifyInts((intVars + 1));
        res += verifyUInts(&uintVars);
        res += verifyFloats(&floatVars);
        res += verifyAllInts(&allIntVars);
        res += verifyFloatsChars(&floatCharVars);
        res += verifyAllTypes(allTypesVar);
        res += verifyAllTypes((allTypesVar + 1));

        FTI_Finalize();
        MPI_Finalize();
        if (res != 0) {
            printf("Test failed!\n");
            return 1;
        }
    }
    return 0;
}
