#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    MPI_Init_thread(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    printf("%d: Size = %d\n", rank, size);
    
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank==0) printf("Success\n");

    MPI_Finalize();
    return 0;
}
