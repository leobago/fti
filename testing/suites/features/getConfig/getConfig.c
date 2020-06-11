/**
 *  @file   getConftest.c
 *  @author Sawsane Ouchtal
 *  @date   May, 2020
 *  @brief  Tests GetConfig function
 *          and validates its output
 */

#include <fti.h>
#include "../../../../src/deps/iniparser/iniparser.h"
#include "../../../../src/deps/iniparser/dictionary.h"

#define CONFIG_EROR 30

int main(int argc, char *argv[]){
    int nrank, nbProcs, i;
    
    char *path = argv[1];
    MPI_Init(&argc, &argv);
    FTI_Init(argv[1], MPI_COMM_WORLD);

    MPI_Comm_size(FTI_COMM_WORLD, &nbProcs);
    MPI_Comm_rank(FTI_COMM_WORLD, &nrank);

    FTIT_allConfiguration configStruct = FTI_GetConfig(path, FTI_COMM_WORLD);
    
    //the test should parse all members in the configuration file
    //and compare them to what the structure returns
    dictionary* ini = iniparser_load(path);

    //basic
    int ckpt_io = (int)iniparser_getint(ini, "basic:ckpt_io", -1);//conf.ioMode
    bool compare = false;

    if (ckpt_io == 1){
        if (configStruct.configuration.ioMode == 1001){
            compare = true;
        }
    } else if (ckpt_io == 2){
        if (configStruct.configuration.ioMode == 1002){
            compare = true;
        }
    } else if (ckpt_io == 3){
        if (configStruct.configuration.ioMode == 1003){
            compare = true;
        }
    } else if (ckpt_io == 4){
        if (configStruct.configuration.ioMode == 1004){
            compare = true;
        }
    } else if (ckpt_io == 5){
        if (configStruct.configuration.ioMode == 1005){
            compare = true;
        }
    }
    
    if(nrank==0){
        if (compare){
            printf("SUCCESS\n");
        }else
            printf("FAILURE\n");
    }

    MPI_Barrier(FTI_COMM_WORLD);
    FTI_Finalize();
    MPI_Finalize();

    if (compare)
        return 0;
    else
        exit(CONFIG_EROR);
}
