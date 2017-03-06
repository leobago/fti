/**
 *  @file   nodeFlag.c
 *  @author Karol Sierocinski (ksiero@man.poznan.pl)
 *  @date   Feburary, 2017
 *  @brief  FTI testing program.
 *
 *	Program tests if nodeFlag == 1 is unique in node.
 */

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <fti.h>
#include <string.h>

int verify(int world_size) {
	FILE* fp;
	char str[]  = "Has nodeFlag = 1 and nodeID =";
	char temp[256];
	char strtmp[256];
    int i;
    int* nodeID = (int*) malloc (sizeof(int) * world_size);

    for (i = 0; i < world_size; i++) {
        nodeID[i] = -1;
    }
	//int counter = 0;
	//Searching in all log files
	for (i = 0; i < world_size; i++) {
		sprintf(strtmp, "./log%d.txt", i);
		//printf("\n%s\n", strtmp);
		if((fp = fopen(strtmp, "r")) == NULL) {
			return 2;
		}
		while(fgets(temp, 256, fp) != NULL) {
			if((strstr(temp, str)) != NULL) {
	            int nodeIDtmp, processIDtmp;
				//counter++;
				sscanf(temp, "[FTI Debug - %06d] : Has nodeFlag = 1 and nodeID = %d.", &processIDtmp, &nodeIDtmp);
	            printf("processID = %d, nodeID = %d\n", processIDtmp, nodeIDtmp);
	            if (nodeID[nodeIDtmp] == -1 || nodeID[nodeIDtmp] == processIDtmp) {
	                nodeID[nodeIDtmp] = processIDtmp;
	            } else {
	                return 1;
	            }
			}
		}
		fclose(fp);
	}
	//if everything is ok, deleting log files
	printf("Deleting files...\n");
	for (i = 0; i < world_size; i++) {
		sprintf(strtmp, "./log%d.txt", i);
		unlink(strtmp);
	}
	//printf("counter = %d\n", counter);
    free(nodeID);
   	return 0;
}

/*
    Prints:
        0 if everything is OK
        1 if error
		2 if cannot access file
*/
int main(int argc, char** argv){
	//Desc for unwanted messages
	int fnull = open("/dev/null", O_RDWR);
    int temp = dup(1);
    dup2(fnull, 1);
    dup2(fnull, 2);

	int world_rank, world_size, global_world_rank, global_world_size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &global_world_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &global_world_size);
	//Creating log files
	char str[256];
	sprintf(str, "./log%d.txt", global_world_rank);
	int f = open(str, O_CREAT | O_RDWR, 0666);
	dup2(f, 1);
	dup2(f, 2);

	FTI_Init(argv[1], MPI_COMM_WORLD);

	//Getting FTI ranks (only app procs)
	MPI_Comm_rank(FTI_COMM_WORLD, &world_rank);
    MPI_Comm_size(FTI_COMM_WORLD, &world_size);

	MPI_Barrier(FTI_COMM_WORLD);

	//Adding something to protect
	int* someArray = (int*) malloc (sizeof(int) * global_world_size);
	FTI_Protect(1, someArray, global_world_size, FTI_INTG);

	int i;
	for (i = 1; i < 5; i++) {
		FTI_Checkpoint(1, i);
		MPI_Barrier(FTI_COMM_WORLD);
	}

	//Backing to stdout
	dup2(temp, 1);
	dup2(temp, 2);
	close(f);
	MPI_Barrier(FTI_COMM_WORLD);
	if (world_rank == 0) {
	    int res = verify(global_world_size);
		printf("Res = %d\n", res);
		switch(res) {
			case 0:
				printf("0");
				break;
			case 1:
				printf("1");
				break;
			case 2:
				printf("2");
				break;
		}
	}
	//fflush(stdout);
	dup2(fnull, 1);
    dup2(fnull, 2);
    FTI_Finalize();
    MPI_Finalize();
	close(fnull);
	close(f);
    return 0;
}
