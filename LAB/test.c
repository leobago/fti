#include "fti.h"
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    
    long size1=10000;
    long size2=20000;
    long size3=30000;
    long size4=40000;
    long size5=50000;
    long size22=60000;
    long size32=70000;
    
    int i, rank;
    MPI_Init(NULL, NULL);
    FTI_Init("config.fti", MPI_COMM_WORLD);

    MPI_Comm_rank(FTI_COMM_WORLD, &rank);
    
    //int arr1[100];
    int *arr1 = (int*) malloc( size1 * sizeof(int) );
    int *arr2 = (int*) malloc( size2 * sizeof(int) );
    int *arr3 = (int*) malloc( size3 * sizeof(int) );
    int *arr4 = (int*) malloc( size4 * sizeof(int) );
    int *arr5 = (int*) malloc( size5 * sizeof(int) );
    //int arr4[400];
    //int arr5[700];

    if ( FTI_Status() == 0 ) {
        
        for(i=0;i<size1;i++){arr1[i]=i+1;}
        for(i=0;i<size2;i++){arr2[i]=i+1;}
        for(i=0;i<size3;i++){arr3[i]=i+1;}
        for(i=0;i<size4;i++){arr4[i]=i+1;}
        for(i=0;i<size5;i++){arr5[i]=i+1;}

        FTI_Protect(1, arr1, size1, FTI_INTG);  
        FTI_Protect(2, arr2, size2, FTI_INTG);  
        FTI_Protect(3, arr3, size3, FTI_INTG); 

        FTI_Checkpoint(1,3);

        FTI_Protect(4, arr4, size4, FTI_INTG); 

        FTI_Checkpoint(2,3);

        arr2 = (int*) realloc(arr2, sizeof(int) * size22);
        for(i=size2;i<size22;i++){arr2[i]=i+1;}
        
        arr3 = (int*) realloc(arr3, sizeof(int) * size32);
        for(i=size3;i<size32;i++){arr3[i]=i+1;}
        
        FTI_Protect(2, arr2, size22, FTI_INTG);
        FTI_Protect(3, arr3, size32, FTI_INTG);
        
        FTI_Checkpoint(3,3);

        FTI_Protect(5, arr5, size5, FTI_INTG); 

        FTI_Checkpoint(4,3);

        for(i=0;i<size1;i++){arr1[i]=0;}
        for(i=0;i<size22;i++){arr2[i]=0;}
        for(i=0;i<size32;i++){arr3[i]=0;}
        for(i=0;i<size4;i++){arr4[i]=0;}
        for(i=0;i<size5;i++){arr5[i]=0;}

        free(arr2);
        free(arr3);
        
        MPI_Barrier(FTI_COMM_WORLD);
        MPI_Abort(FTI_COMM_WORLD,-1);
        //MPI_Barrier(FTI_COMM_WORLD);
        //FTI_Finalize();
        //MPI_Finalize();
        //return 0;

    } else {

        FTI_Protect(1, arr1, size1, FTI_INTG);  
        //FTI_Protect(2, arr2, 200, FTI_INTG);
        //FTI_Protect(3, arr3, 300, FTI_INTG);
        arr2 = (int*) realloc(arr2, sizeof(int) * size22);
        arr3 = (int*) realloc(arr3, sizeof(int) * size32);
        FTI_Protect(2, arr2, size22, FTI_INTG);
        FTI_Protect(3, arr3, size32, FTI_INTG);
        FTI_Protect(4, arr4, size4, FTI_INTG); 
        FTI_Protect(5, arr5, size5, FTI_INTG); 
        
        MPI_Barrier(FTI_COMM_WORLD);
        
        FTI_Recover();
        
        MPI_Barrier(FTI_COMM_WORLD);
        
        long a1_ex = (size1/2)*(size1+1);
        long a2_ex = (size22/2)*(size22+1);
        long a3_ex = (size32/2)*(size32+1);
        long a4_ex = (size4/2)*(size4+1);
        long a5_ex = (size5/2)*(size5+1);

        long a1=0, a2=0, a3=0, a4=0, a5=0;

        for(i=0;i<size1;i++){a1+=arr1[i];}

        for(i=0;i<size22;i++){a2+=arr2[i];}
        
        for(i=0;i<size32;i++){a3+=arr3[i];}
        
        for(i=0;i<size4;i++){a4+=arr4[i];}
        
        for(i=0;i<size5;i++){a5+=arr5[i];}
   
        printf(
                "rank -> %i\n"
                "-----------------------\n"
                "a1_ex: %ld, a1: %ld\n"
                "a2_ex: %ld, a2: %ld\n"
                "a3_ex: %ld, a3: %ld\n"
                "a4_ex: %ld, a4: %ld\n"
                "a5_ex: %ld, a5: %ld\n"
                "-----------------------\n",
                rank,
                a1_ex, a1,
                a2_ex, a2,
                a3_ex, a3,
                a4_ex, a4,
                a5_ex, a5
            );

        free(arr2);
        free(arr3);

        MPI_Barrier(FTI_COMM_WORLD);
        FTI_Finalize();
        MPI_Finalize();
        return 0;
    }
}
