#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <mpi.h>
#include <string.h>

#include "FTI_UtilAPI.h"


#define CHUNK (128*1024*1024)

int numRanks  = 0;
int myRank = 0;

int FTI_MPIOWrite(void *src, size_t size, MPI_File *fd, MPI_Offset *offset){
    size_t pos = 0;
    size_t bSize = 32*1024*1024;
    while (pos < size) {
        if ((size - pos) < CHUNK) {
            bSize = size - pos;
        }

        MPI_Datatype dType;
        MPI_Type_contiguous(bSize, MPI_BYTE, &dType);
        MPI_Type_commit(&dType);

        int err = MPI_File_write_at(*fd, *offset, src, 1, dType, MPI_STATUS_IGNORE);
        if ( err != 0) {
            char str[100], mpi_err[100];
            int reslen;
            MPI_Error_string(err, mpi_err, &reslen);
            snprintf(str, 100 , "unable to create file [MPI ERROR - %i] %s", err, mpi_err);
            fprintf(stderr, "Error: %s\n",str);
            exit(-1);
        }
        MPI_Type_free(&dType);
        src += bSize;
        *offset += bSize;
        pos = pos + bSize;
    }
    return SUCCESS;
}


void processCkpts(int start, int end, int *ckptId, int numCkpts){
    int i;
    int ret = 0;
    MPI_Info info = 0;
    MPI_File pfh = 0;

    MPI_Offset* rankSizes = (MPI_Offset*) malloc(sizeof(MPI_Offset)*numRanks);

    int *numVars = (int*) malloc (sizeof(int)*numCkpts);

    for ( i = 0; i < numCkpts; i++){
        for ( int j = start; j <  end; j++){
            ret = FTI_VerifyCkpt(ckptId[i],j);
            if (ret == ERROR) {fprintf(stderr, "Failed to verify ckpt ckptid:%d rank:%d\n", ckptId[i], j); exit (-1); }
        }
        ret = FTI_GetNumVars(ckptId[i], 0);
        if (ret == ERROR ) {fprintf(stderr, "Failed to Get number of variables ckpt ckptid:%d rank:%d\n", ckptId[i], 0); exit (-1); }
        numVars[i] = ret;
    }


    for ( i = 0; i < numCkpts; i++){
        unsigned char **buf;
        buf = NULL;
        char **name;
        size_t *size;

        int *varIds = (int *) malloc (sizeof(int)*numVars[i]);

        for (int k = 0; k < numVars[i]; k++){
            buf = (unsigned char**) malloc(sizeof(unsigned char*)*(end-start));
            for ( int r = 0; r < end - start ; r++ ){
                buf[r] = NULL;
            }
            name = (char **) malloc (sizeof(char*) *(end-start)); 
            size = (size_t *) malloc (sizeof(size_t)*(end-start));
            char fn[1000];

            int cnt = 0;
            int elements = end -start;

            for ( int r = start; r < end; r++){
                ret =  FTI_readVariableByIndex(k, ckptId[i], r, &name[cnt], &varIds[i], &buf[cnt], &size[cnt]);
                if (ret == ERROR ) {fprintf(stderr, "Failed to Read Var %d: in ckpt %d rank:%d\n",k, ckptId[i], r); exit (-1); }
                cnt++;
            }
            
            if ( strlen(name[0]) != 0 ){
                sprintf(fn,"./output/Ckpt_%d_%s.mpio",ckptId[i],name[0]);
            }
            else{
                sprintf(fn,"./output/Ckpt_%d_%d.mpio",ckptId[i], varIds[i]);
            }
            MPI_Offset mySize = 0;
            MPI_Offset offset = 0;

            for ( int s = 0; s < elements; s++)
                mySize+= size[s];

            MPI_Allgather(&mySize, 1, MPI_OFFSET, rankSizes, 1, MPI_OFFSET, MPI_COMM_WORLD);

            for ( int s = 0; s < myRank; s++){
                offset += rankSizes[s];
            }

            MPI_Info_create(&info);
            MPI_Info_set(info, "romio_cb_write", "enable");
            ret = MPI_File_open(MPI_COMM_WORLD, fn, MPI_MODE_WRONLY|MPI_MODE_CREATE, info, &pfh);
            if (ret != 0 ){
                char str[1000], mpi_err[100];
                MPI_Info_free(&info);
                int reslen;
                MPI_Error_string(ret , mpi_err, &reslen);
                fprintf(stderr, "unable to create file [MPI ERROR - %i] %s:%s aaa\n", ret, mpi_err,fn);
                exit(-1);
            }

            for ( int s = 0; s < elements; s++){
                FTI_MPIOWrite(buf[s], size[s], &pfh, &offset);
            }


            MPI_Info_free(&info);
            MPI_File_close(&pfh);

            free(name);
            free(buf);
            free(size);
            name = NULL;
            buf = NULL;
            size = NULL;

        }
        free(varIds);
    }
    free(numVars);

}

int main(int argc, char *argv[]){
    MPI_Init(&argc, &argv);
    int numCkpts = 0;
    int *checkpointIds = NULL;
    int FTI_ranks;
    char *configFile = argv[1];
    int start, end , responsibleRanks;

    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    int ret = FTI_InitUtil(configFile);         
    if (ret != SUCCESS ) {fprintf(stderr, "Failed to initialize fti util\n"); return (-1); }

    ret = FTI_GetNumberOfCkptIds(&numCkpts);
    if (ret != SUCCESS ) {fprintf(stderr, "Failed to Get Number Ckpt Ids fti util\n"); return (-1); }

    checkpointIds = (int *) malloc (sizeof(int)*numCkpts);
    assert(checkpointIds);

    ret = FTI_GetCkptIds(checkpointIds);
    if (ret != SUCCESS ) {fprintf(stderr, "Failed to get ckpt ids\n"); return (-1); }

    ret = FTI_GetUserRanks(&FTI_ranks);
    if (ret != SUCCESS ) {fprintf(stderr, "Failed to get user ranks ids\n"); return (-1); }


    if (FTI_ranks % numRanks == 0){
        responsibleRanks = FTI_ranks/numRanks;
        start = myRank * responsibleRanks ;
        end = start + responsibleRanks;
    }else{
        int tmp = FTI_ranks%numRanks; 
        if (myRank < tmp ){
            responsibleRanks = FTI_ranks/numRanks+1;
            start = myRank * responsibleRanks;
            end = start+responsibleRanks;
        }
        else{
            responsibleRanks = FTI_ranks/numRanks;
            start = myRank * responsibleRanks + tmp;
            end = start+responsibleRanks;
        }
    }

    processCkpts(start, end, checkpointIds, numCkpts); 

    free(checkpointIds);
    checkpointIds = NULL;
    FTI_FinalizeUtil();
    MPI_Finalize();
}

