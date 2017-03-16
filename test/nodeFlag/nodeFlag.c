/**
 *  @file   nodeFlag.c
 *  @author Karol Sierocinski (ksiero@man.poznan.pl)
 *  @date   Feburary, 2017
 *  @brief  FTI testing program.
 *
 *	Program tests if there is only one process with nodeFlag set to 1 in node.
 *  Look FTI_PostCkpt(...) in checkpoint.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <fti.h>
#include <string.h>

#include "../../deps/iniparser/iniparser.h"
#include "../../deps/iniparser/dictionary.h"

int verify(int world_size, int nbNodes) {
	FILE* fp;
	//Shearching this string in log file
	char str[]  = "Has nodeFlag = 1 and nodeID =";
	char temp[256];
	char strtmp[256];
    int i, k;

	//Have to checking ckptLvel
	int** ckptLvel = (int**) malloc (sizeof(int*) * 5);
	for (i = 1; i < 5; i++) {
		int* nodeID = (int*) malloc (sizeof(int) * nbNodes);
		ckptLvel[i] = nodeID;
	}

    for (i = 1; i < 5; i++) {
		for (k = 0; k < nbNodes; k++) {
			ckptLvel[i][k] = -1;
		}
    }
	//Searching in all log files
	for (i = 0; i < world_size; i++) {
		sprintf(strtmp, "./log%d.txt", i);
		if((fp = fopen(strtmp, "r")) == NULL) {
			fprintf(stderr, "Cannot read file %s.\n", strtmp);
			return 2;
		}
		while(fgets(temp, 256, fp) != NULL) {
		    if((strstr(temp, str)) != NULL) {
			    int nodeIDtmp, processIDtmp, ckptLveltmp;
			    sscanf(temp, "[FTI Debug - %06d] : Has nodeFlag = 1 and nodeID = %d. CkptLvel = %d.", &processIDtmp, &nodeIDtmp, &ckptLveltmp);
				//if found first nodeFlag process for this ckptLvel
				if (ckptLvel[ckptLveltmp][nodeIDtmp] == -1) {
					ckptLvel[ckptLveltmp][nodeIDtmp] = processIDtmp;
					fprintf(stderr, "Lv%d: Node %d; Process %d\n", ckptLveltmp, nodeIDtmp, processIDtmp);
			    }
				//if found this nodeFlag process and it's not the same as before
			    if (ckptLvel[ckptLveltmp][nodeIDtmp] != processIDtmp) {
					fprintf(stderr, "Lv%d: Node %d; Process %d\n", ckptLveltmp, nodeIDtmp, processIDtmp);
					fprintf(stderr, "The node %d on checkpoint level %d has more than 1 nodeFlag process: %d and %d.\n", nodeIDtmp, ckptLveltmp, processIDtmp, ckptLvel[ckptLveltmp][nodeIDtmp]);
					return 1;
			    }
		    }
		}
		fclose(fp);
	}

	for (i = 1; i < 5; i++) {
		for (k = 0; k < nbNodes; k++) {
			if (ckptLvel[i][k] == -1) {
				fprintf(stderr, "Node %d on checkpoint level %d don't have signed process!\n", k, i);
				return 1;
			}
		}
    }

	//if everything is ok, deleting log files
	fprintf(stderr, "All log files checked. Deleting files...\n");
	for (i = 0; i < world_size; i++) {
		sprintf(strtmp, "./log%d.txt", i);
		unlink(strtmp);
	}
	for (i = 1; i < 5; i++) {
		free(ckptLvel[i]);
	}
    free(ckptLvel);
   	return 0;
}

/*-------------------------------------------------------------------------*/
/**
    @return     integer     0 if successful, 1 if error, 2 if can open file
 **/
/*-------------------------------------------------------------------------*/
int main(int argc, char** argv){
	//disable buffering
	setbuf(stdout, NULL);

	int world_rank, world_size, global_world_rank, global_world_size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &global_world_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &global_world_size);

	if (global_world_rank == 0) {
		fprintf(stderr, "Creating log files...\n");
	}
	//Creating log files
	char str[256];
	sprintf(str, "./log%d.txt", global_world_rank);
	int f = open(str, O_CREAT | O_RDWR, 0666);
	if (f < 0) {
		fprintf(stderr, "Cannot open %s file.\n", str);
		return 1;
	}
	//change stdout to file
	dup2(f, 1);

	FTI_Init(argv[1], MPI_COMM_WORLD);

	//Getting FTI ranks (only app procs)
	MPI_Comm_rank(FTI_COMM_WORLD, &world_rank);
    MPI_Comm_size(FTI_COMM_WORLD, &world_size);

	//Adding something to protect
	int* someArray = (int*) malloc (sizeof(int) * global_world_size);
	FTI_Protect(1, someArray, global_world_size, FTI_INTG);

	//making some checkpoints
	int i;
	for (i = 1; i < 5; i++) {
		if (world_rank == 0) {
			fprintf(stderr, "Making checkpoint L%d\n", i);
		}
		FTI_Checkpoint(i, i);
	}

	//only app processes can call close fd (cannot make heads to do this)
	close(f);

	dictionary* ini = iniparser_load("config.fti");
	int nodeSize = (int)iniparser_getint(ini, "Basic:node_size", -1);
	int nbNodes = global_world_size/nodeSize;
	int heads = (int)iniparser_getint(ini, "Basic:head", -1);
	int isInlineL4 = (int)iniparser_getint(ini, "Basic:inline_l4", 1);
	if (heads > 0 && !isInlineL4) {
		//waiting untill head do Post-checkpointing
		int res;
		MPI_Recv(&res, 1, MPI_INT, global_world_rank - (global_world_rank%nodeSize) , 2612, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		res = FTI_ENDW;
		//sending end of work
		MPI_Send(&res, 1, MPI_INT, global_world_rank - (global_world_rank%nodeSize), 2612, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD); //calling Barrier to let heads end work
	}

	int rtn = 0; //return value
	if (world_rank == 0) {
		fprintf(stderr, "Verifying logs...\n");
		rtn = verify(global_world_size, nbNodes);
	}

	if (!(heads > 0 && !isInlineL4)) {
   		FTI_Finalize();
    }

	MPI_Finalize();

	return rtn;
}
