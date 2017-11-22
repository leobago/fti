#include "fti.h"
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    
    int i, rank;
    MPI_Init(NULL, NULL);
    FTI_Init("config.fti", MPI_COMM_WORLD);

    MPI_Comm_rank(FTI_COMM_WORLD, &rank);
    
    int arr1[100];
    for(i=0;i<100;i++){arr1[i]=i;}

    int *arr2 = (int*) malloc( 200 * sizeof(int) );
    for(i=0;i<200;i++){arr2[i]=i;}
    
    int *arr3 = (int*) malloc( 300 * sizeof(int) );
    for(i=0;i<300;i++){arr3[i]=i;}
    
    int arr4[400];
    for(i=0;i<400;i++){arr4[i]=i;}
    
    int arr5[700];
    for(i=0;i<700;i++){arr5[i]=i;}

    FTI_Protect(1, arr1, 100, FTI_INTG);  
    FTI_Protect(2, arr2, 200, FTI_INTG);  
    FTI_Protect(3, arr3, 300, FTI_INTG); 
    
    FTI_Checkpoint(1,1);
    
    FTI_Protect(4, arr4, 400, FTI_INTG); 
    
    //FTI_Checkpoint(2,1);

    //arr2 = (int*) realloc(arr2, sizeof(int) * 500);
    //for(i=200;i<500;i++){arr2[i]=i;}
    //
    //arr3 = (int*) realloc(arr3, sizeof(int) * 600);
    //for(i=300;i<600;i++){arr3[i]=i;}
    //
    //FTI_Protect(2, arr2, 500, FTI_INTG);
    //FTI_Protect(3, arr3, 600, FTI_INTG);
    //
    //FTI_Checkpoint(3,1);
    
    FTI_Protect(5, arr5, 700, FTI_INTG); 
    
    FTI_Checkpoint(4,1);

    printf("expected value arr1[67] is 67. we have -> %i\n",arr1[67]);
    
    MPI_Abort(FTI_COMM_WORLD, -1);
    //finalize();
    //
    //for(i=0;i<100;i++){arr1[i]=0;}

    //for(i=0;i<500;i++){arr2[i]=0;}
    //
    //for(i=0;i<600;i++){arr3[i]=0;}
    //
    //for(i=0;i<400;i++){arr4[i]=0;}
    //
    //for(i=0;i<700;i++){arr5[i]=0;}

    //recover();

    //printf("expected value arr1[67] is 67. we have -> %i\n",arr1[67]);

    //if (pvars) {
    //    free(pvars);
    //}

    free(arr2);
    free(arr3);

    MPI_Barrier(FTI_COMM_WORLD);
    FTI_Finalize();
    MPI_Finalize();
    return 0;
}
