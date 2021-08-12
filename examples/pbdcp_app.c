#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fti.h>

int initBuffer(double * buffer){
    int i=0;
    for(;i<128;i++){
        buffer[i]=i;
    }
    return 0;
}
int main(int argc, char *argv[]){
    int rank,nbProcs;
    MPI_Init(&argc, &argv);
    if (FTI_Init(argv[1], MPI_COMM_WORLD) !=  0) {
        printf("FTI could not initialize properly!\n");
        return 1;
    };

    int iter=atoi(argv[2]);
    int CKPT_INTERVALL=atoi(argv[3]);
    int power=atoi(argv[4]);
    MPI_Comm_size(FTI_COMM_WORLD, &nbProcs);
    MPI_Comm_rank(FTI_COMM_WORLD, &rank);

    double *buffer;
    buffer = (double *) malloc(sizeof(double *) * 128);
    int i=0;
    initBuffer(buffer);
    FTI_Protect(1, buffer, 128, FTI_DBLE);

    if (FTI_Status() != 0) {
        FTI_Recover();
    }
    for (; i < iter; i++) {
        if (i % CKPT_INTERVALL == 0) {
            FTI_Checkpoint(i / CKPT_INTERVALL + 1, FTI_L4_PBDCP);
        }
        buffer[i%128] += 0.1*pow(10,-power);
    }
    
    free(buffer);

    FTI_Finalize();
    MPI_Finalize();
    return 0;

}
