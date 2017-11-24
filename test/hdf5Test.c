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

typedef struct srubaT {
    int dlugosc;
    int szerokosc;
} srubaT;

typedef struct koloT {
   srubaT sruba;
   char   opona[360];
} koloT;

typedef struct pozycjaT {
    int x[4][2];
    int y[8];
    int z[2][2][2];
} pozycjaT;

typedef struct silnikT {
    int liczbaTlokow;
    int pozycjaTloka[8][8][8][8];
    pozycjaT pozycjaSilnika;
} silnikT;

typedef struct samochodT {
    koloT kolo[5];
    silnikT silnik;
    float calareszta[256];
} samochodT;

/*-------------------------------------------------------------------------*/
/**
    @return     integer     0 if successful, 1 otherwise
 **/
/*-------------------------------------------------------------------------*/
int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    FTI_Init(argv[1], MPI_COMM_WORLD);
    int world_rank, world_size; //FTI_COMM rank and size
    MPI_Comm_rank(FTI_COMM_WORLD, &world_rank);
    MPI_Comm_size(FTI_COMM_WORLD, &world_size);

    printf("Sruba:\n");

    FTIT_complexType srubaDef;
    srubaDef.field[0].type = &FTI_INTG;
    srubaDef.field[1].type = &FTI_INTG;
    sprintf(srubaDef.field[0].name, "dlugosc");
    sprintf(srubaDef.field[1].name, "szerokosc");
    sprintf(srubaDef.name, "sruba");
    srubaDef.length = 2;
    FTIT_type srubaType;
    FTI_InitSimpleTypeWithNames(&srubaType, &srubaDef);
    srubaT sruba = {10, 20};

    printf("Koła:\n");

    FTIT_complexType koloDef;
    koloDef.field[0].type = &srubaType;
    koloDef.field[1].type = &FTI_CHAR;
    koloDef.field[0].rank = 1;
    koloDef.field[1].rank = 1;
    koloDef.field[0].dimLength[0] = 1;
    koloDef.field[1].dimLength[0] = 360;
    sprintf(koloDef.field[0].name, "sruba");
    sprintf(koloDef.field[1].name, "opona");
    sprintf(koloDef.name, "kolo");
    koloDef.length = 2;
    FTIT_type koloType;
    FTI_InitComplexTypeWithNames(&koloType, &koloDef);
    koloT kolo;
    kolo.sruba = sruba;
    int i;
    for (i = 0; i < 360; i++) {
        char x = 'a' + (i % 25);
        kolo.opona[i] = x;
    }

    printf("Pozycja:\n");

    FTIT_complexType pozycjaDef;
    pozycjaDef.field[0].type = &FTI_INTG;
    pozycjaDef.field[1].type = &FTI_INTG;
    pozycjaDef.field[2].type = &FTI_INTG;
    pozycjaDef.field[0].rank = 2;
    pozycjaDef.field[1].rank = 1;
    pozycjaDef.field[2].rank = 3;
    pozycjaDef.field[0].dimLength[0] = 4;
    pozycjaDef.field[0].dimLength[1] = 2;
    pozycjaDef.field[1].dimLength[0] = 8;
    pozycjaDef.field[2].dimLength[0] = 2;
    pozycjaDef.field[2].dimLength[1] = 2;
    pozycjaDef.field[2].dimLength[2] = 2;
    sprintf(pozycjaDef.field[0].name, "x");
    sprintf(pozycjaDef.field[1].name, "y");
    sprintf(pozycjaDef.field[2].name, "z");
    sprintf(pozycjaDef.name, "pozycja");
    pozycjaDef.length = 3;
    FTIT_type pozycjaType;
    FTI_InitComplexTypeWithNames(&pozycjaType, &pozycjaDef);
    pozycjaT pozycja;
    int j;
    for (i = 0; i < 4; i++) {
        for (j = 0; j < 2; j++) {
            pozycja.x[i][j] = (i * 2) + j;
        }
    }
    for (i = 0; i < 8; i++) {
        pozycja.y[i] = i;
    }
    int k;
    for (i = 0; i < 2; i++) {
        for (j = 0; j < 2; j++) {
            for (k = 0; k < 2; k++) {
                pozycja.z[i][j][k] = (i * 4) + (j * 2) + k;
            }
        }
    }

    printf("Silnik:\n");

    FTIT_complexType silnikDef;
    silnikDef.field[0].type = &FTI_INTG;
    silnikDef.field[1].type = &FTI_INTG;
    silnikDef.field[2].type = &pozycjaType;
    silnikDef.field[0].rank = 1;
    silnikDef.field[1].rank = 4;
    silnikDef.field[2].rank = 1;
    silnikDef.field[0].dimLength[0] = 1;
    silnikDef.field[1].dimLength[0] = 8;
    silnikDef.field[1].dimLength[1] = 8;
    silnikDef.field[1].dimLength[2] = 8;
    silnikDef.field[1].dimLength[3] = 8;
    silnikDef.field[2].dimLength[0] = 1;
    sprintf(silnikDef.field[0].name, "liczbaTlokow");
    sprintf(silnikDef.field[1].name, "pozycjaTlokow");
    sprintf(silnikDef.field[2].name, "pozycjaSilnika");
    sprintf(silnikDef.name, "silnik");
    silnikDef.length = 3;
    FTIT_type silnikType;
    FTI_InitComplexTypeWithNames(&silnikType, &silnikDef);
    silnikT silnik;
    silnik.liczbaTlokow = 8;
    int l;
    for (i = 0; i < 8; i++) {
        for (j = 0; j < 8; j++) {
            for (k = 0; k < 8; k++) {
                for (l = 0; l < 8; l++) {
                    silnik.pozycjaTloka[i][j][k][l] = (i * 8*8*8) + (j * 8*8) + (k * 8) + l;
                }
            }
        }
    }
    silnik.pozycjaSilnika = pozycja;

    printf("Samochod:\n");

    FTIT_complexType samochodDef;
    samochodDef.field[0].type = &koloType;
    samochodDef.field[1].type = &silnikType;
    samochodDef.field[2].type = &FTI_SFLT;
    samochodDef.field[0].rank = 1;
    samochodDef.field[1].rank = 1;
    samochodDef.field[2].rank = 1;
    samochodDef.field[0].dimLength[0] = 5;
    samochodDef.field[1].dimLength[0] = 1;
    samochodDef.field[2].dimLength[0] = 256;
    sprintf(samochodDef.field[0].name, "kolo");
    sprintf(samochodDef.field[1].name, "silnik");
    sprintf(samochodDef.field[2].name, "calareszta");
    sprintf(samochodDef.name, "samochod");
    samochodDef.length = 3;
    FTIT_type samochodType;
    FTI_InitComplexTypeWithNames(&samochodType, &samochodDef);
    samochodT samochod[3];
    for (i = 0; i < 3; i++) {
        samochod[i].kolo[0] = kolo;
        samochod[i].kolo[1] = kolo;
        samochod[i].kolo[2] = kolo;
        samochod[i].kolo[3] = kolo;
        samochod[i].kolo[4] = kolo;
        samochod[i].silnik = silnik;
        for (j = 0; j < 256; j++) {
            samochod[i].calareszta[j] = 0.01 + (float) j;
        }
    }

    FTI_ProtectWithName(1, &sruba, 1, srubaType, "sruba");
    FTI_ProtectWithName(2, &kolo, 1, koloType, "kolo");
    FTI_ProtectWithName(3, &silnik, 1, silnikType, "silnik");
    FTI_ProtectWithName(4, &pozycja, 1, pozycjaType, "pozycja");
    FTI_ProtectWithName(5, &samochod, 3, samochodType, "3Samochody");

    FTI_Recover();
    FTI_Checkpoint(2, 4);

    FTI_Finalize();
    MPI_Finalize();
    return 0;
}
