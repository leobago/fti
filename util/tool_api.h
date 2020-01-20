#ifndef __TOOL_API__
#define __TOOL_API__

#define BUFF_SIZE 1000
#define MAX_BUFF (16*1024*1024)
#define MD5_DIGEST_STRING_LENGTH 33
#define MD5_DIGEST_LENGTH 16

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

typedef struct FTI_DataVar{
    char *name;
    int id;
    size_t size;
    size_t pos;
    unsigned char *buf;
}FTI_DataVar;

typedef struct FTI_ckptFile{
    char *name;
    char *md5hash;
    FTI_DataVar *variables;
    int numVars;
    int globalRank;
    int applicationRank;
    int verified;
    char *pathToFile;
}FTI_CkptFile;

typedef struct FTI_collection{
    FTI_CkptFile *files;
    int numCkpts;
    int ckptId;
}FTI_Collection;

#endif
