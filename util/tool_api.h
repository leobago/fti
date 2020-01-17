#include <stdio.h>
#include <stdlib.h>

#define BUFF_SIZE 1000
#define ERROR -1
#define SUCCESS 1

typedef struct FTI_Info{
    char *execDir; // This is the path in which I started executing my application
    char *configFileDir; // Where my configuration path is
    char *configName; // Configuration file name

    char *metaDir; // Where the metadata dir is
    char *globalDir; // Where the global dir is
    char *localDir; // Local directory, this should not actually exist.

    char *execId;

    int localTest;
    int groupSize;
    int head;
    int nodeSize;
    int userRanks;
}FTI_Info;
