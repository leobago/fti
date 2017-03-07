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

int nodes = 0;

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
			    sscanf(temp, "[FTI Debug - %06d] : Has nodeFlag = 1 and nodeID = %d.", &processIDtmp, &nodeIDtmp);
			    if (nodeID[nodeIDtmp] == -1) {
				nodeID[nodeIDtmp] = processIDtmp;
				nodes++;
				printf("Node %d : Process %d\n", nodeIDtmp, processIDtmp);
			    }
			    if (nodeID[nodeIDtmp] != processIDtmp) {
				printf("Node %d : Process %d\n", nodeIDtmp, processIDtmp);
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
	int world_rank, world_size, global_world_rank, global_world_size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &global_world_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &global_world_size);
	//Creating log files
	char str[256];
	sprintf(str, "./log%d.txt", global_world_rank);
	int f = open(str, O_CREAT | O_RDWR, 0666);
	dup2(2, 1);
	int stdoutTmp = dup(1);
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
		if (world_rank == 0) {
			dup2(stdoutTmp, 1);
			printf("Making checkpoint L%d\n", i);
			dup2(f, 1);
		}
		FTI_Checkpoint(1, i);
		MPI_Barrier(FTI_COMM_WORLD);
	}
	dup2(stdoutTmp, 1); //back to stdout
	MPI_Barrier(FTI_COMM_WORLD);
	int rtn = 0; //return value
	if (world_rank == 0) {
		int res = verify(global_world_size);
		switch(res) {
			case 0:
				printf("Nodes found: %d\n", nodes);
				break;
			case 1:
				printf("There is more than 1 nodeFlag == 1 in one node.\n");
				rtn = 1;
				break;
			case 2:
				printf("Cannot read file.\n");
				rtn = 2;
				break;
		}
	}
	dup2(f, 1);
	FTI_Finalize();
	MPI_Finalize();
	close(f);
	return rtn;
}
