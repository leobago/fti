/**
 *  @file   corrupt.c
 *  @author Tomasz Paluszkiewicz (tomaszp@man.poznan.pl)
 *  @date   May, 2017
 *  @brief  File corruption program.
 *
 *  Corrupting fti checkpoint files.
 *
 */
#include <dirent.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "../deps/iniparser/iniparser.h"
#include "../deps/iniparser/dictionary.h"

#define CORRUPT_SIZE 16

#define CORRUPT_FAIL 1
#define CORRUPT_SCES 0

#define TARGET_GROUP 0 //target group id

/*-------------------------------------------------------------------------*/
/**
    @brief      Corrupt a file with given file path.
    @param      file_path           Path to the file to corrupt.
    @return     integer             CORRUPT_SCES if successful,
                                    CORRUPT_FAIL if fails.
 **/
/*-------------------------------------------------------------------------*/
int corruptFile(char* file_path) {
    if (access(file_path, F_OK) != 0) {
        printf("File: %s doesnt exists!\n", file_path);
        return CORRUPT_FAIL;
    }
    FILE* fd = fopen(file_path, "r+");
    if (fd == NULL) {
        printf("Failed to open file %s to corrupt.\n", file_path);
        return CORRUPT_FAIL;
    }

    char buff[CORRUPT_SIZE];
    fread(buff, sizeof(char), CORRUPT_SIZE, fd);
    int i;
    for (i = 0; i < CORRUPT_SIZE; i++) {
        buff[i] = buff[i] ^ 0xFF;
    }
    fseek(fd, 0, SEEK_SET);
    fwrite(buff, sizeof(char), CORRUPT_SIZE, fd);
    fclose(fd);
    printf("File %s corrupted.\n", file_path);
    return CORRUPT_SCES;
}
/*-------------------------------------------------------------------------*/
/**
    @brief      Calculate path and corrupt files.
    @param      exec_id             Exec_id from config.fti.
    @param      target_node         Target node id.
    @param      target_rank           Target FTI_COMM_WORLD rank.
    @param      flag                Flag == CORRUPT_CKPT corrupts ckpt files.
                                    Flag == CORRUPT_PCOF corrupts partner files.
    @return     integer             CORRUPT_SCES if successful,
                                    CORRUPT_FAIL if fails.
 **/
/*-------------------------------------------------------------------------*/
int corruptTargetFile(char* exec_id, int target_node, int target_rank,
                        int ckptORPtner, int corrORErase, int level) {
    DIR *dir;
    struct dirent *ent;
    char folder_path[256];
    char file_path[256];
    int ckpt_id = -1;
    char buff[5];
    int res;

    if (ckptORPtner == 0) {
        sprintf(buff, "Rank");
    }
    else {
        if (level == 2) {
            sprintf(buff, "Pcof");
        }
        else {
            sprintf(buff, "RSed");
        }
    }

    if (level == 4) {
        sprintf(folder_path, "./Global/%s/l4", exec_id);
    }
    else {
        sprintf(folder_path, "./Local/node%d/%s/l%d", target_node, exec_id, level);
    }

    if ((dir = opendir(folder_path)) == NULL) {
        printf("Could not open directory: %s \n", folder_path);
        return CORRUPT_FAIL;
    }
    while ((ent = readdir(dir)) != NULL) {
        if (strstr(ent->d_name, "Ckpt") != NULL) {
            sscanf(ent->d_name, "Ckpt%d", &ckpt_id);
            break;
        }
    }
    closedir (dir);
    if (ckpt_id == -1) {
        printf("Could not find checkpoint files");
        return CORRUPT_FAIL;
    }
    sprintf(file_path, "%s/Ckpt%d-%s%d.fti", folder_path, ckpt_id, buff, target_rank);

    if (corrORErase == 0) {
        res = corruptFile(file_path);
    }
    else {
        res = unlink(file_path);
        if (res == 0) {
            printf("File %s erased.\n", file_path);
        } else {
            printf("Could not erase %s.\n", file_path);
        }
    }
    return res;
}

int init(char** argv) {
    int rtn = 0;    //return value
    if (argv[1] == NULL) {
        printf("Missing first parameter (config file).\n");
        rtn = 1;
    }
    if (argv[2] == NULL) {
        printf("Missing second parameter (level).\n");
        rtn = 1;
    } else if (atoi(argv[2]) < 1 || atoi(argv[2]) > 4) {
        printf("Second parameter (level) must be 1,2,3 or 4.\n");
        rtn = 1;
    }
    if (argv[3] == NULL) {
        printf("Missing third parameter (number of processes).\n");
        rtn = 1;
    } else if (atoi(argv[3]) < 1) {
        printf("Third parameter  (number of processes) must be greater than 0.\n");
        rtn = 1;
    }
    if (argv[4] == NULL) {
        printf("Missing fourth parameter (ckpt(0) or Pcof/Rsed(1) ).\n");
        rtn = 1;
    } else if (atoi(argv[4]) != 0 && atoi(argv[4]) != 1) {
        printf("Fourth parameter (ckpt(0) or Pcof/Rsed(1) ) must be 0 or 1.\n");
        rtn = 1;
    }
    if (argv[5] == NULL) {
        printf("Missing fifth parameter (corrupt (0) or erase (1) ).\n");
        rtn = 1;
    } else if (atoi(argv[5]) != 0 && atoi(argv[5]) != 1) {
        printf("Fifth parameter (corrupt (0) or erase (1)) must be 0 or 1.\n");
        rtn = 1;
    }
    if (argv[6] == NULL) {
        printf("Missing sixth parameter (one (0), two non adjacent\
                    Nodes (1), two adjacent Nodes (2) or all (3) ).\n");
        rtn = 1;
    } else if (atoi(argv[6]) < 0 || atoi(argv[6]) > 3) {
        printf("Sixth parameter (one (0), two non adjacent\
                Nodes (1), two adjacent Nodes (2) or all (3) ) must be 0, 1, 2 or 3.\n");
        rtn = 1;
    }
    return rtn;
}

int main(int argc, char **argv) {
    //------- set vars -------
    if (init(argv)) return 0;
    int level = atoi(argv[2]);
    int nbProcs = atoi(argv[3]);
    int ckptORPtner = atoi(argv[4]);
    int corrORErase = atoi(argv[5]);
    int corruptionLevel = atoi(argv[6]);
    int res;
    //------- read config -------
    dictionary* ini = iniparser_load(argv[1]);
    if (ini == NULL) {
        printf("Could not open FTI config file.\n");
        return 1;
    }
    char* exec_id = malloc(sizeof(char) * 256);
    exec_id = iniparser_getstring(ini, "Restart:exec_id", NULL);
    int node_size = (int)iniparser_getint(ini, "Basic:node_size", -1);
    int group_size = (int)iniparser_getint(ini, "Basic:group_size", -1);
    int head = (int)iniparser_getint(ini, "Basic:head", -1);

    int target_node, target_rank; //node id and target process rank
    target_node = (TARGET_GROUP % node_size) * group_size; //calculate first node in group
    target_rank = target_node * node_size + (TARGET_GROUP % group_size) + head; //calcualte procsess rank in node

    if (corruptionLevel == 0) { //one
        res = corruptTargetFile(exec_id, target_node, target_rank, ckptORPtner, corrORErase, level);
    } else if (corruptionLevel == 1) { //two non adjacent Nodes
        res = corruptTargetFile(exec_id, target_node, target_rank, 0, corrORErase, level);
        res += corruptTargetFile(exec_id, target_node, target_rank, 1, corrORErase, level);
        if (res) {
            printf("Failed to corrupt a file.\n");
        }
        target_node += (group_size - 2); //second target is last node - 1 in group
        target_rank = (target_node * node_size) + (TARGET_GROUP % group_size) + head; //calculate rank
        res += corruptTargetFile(exec_id, target_node , target_rank, 0, corrORErase, level);
        res += corruptTargetFile(exec_id, target_node , target_rank, 1, corrORErase, level);
    } else if (corruptionLevel == 2) { //two adjacent Nodes
        res = corruptTargetFile(exec_id, target_node, target_rank, 0, corrORErase, level);
        res += corruptTargetFile(exec_id, target_node, target_rank, 1, corrORErase, level);
        target_node += 1; //second target is next node in group
        target_rank = (target_node * node_size) + (TARGET_GROUP % group_size) + head; //calculate rank
        res += corruptTargetFile(exec_id, target_node , target_rank, 0, corrORErase, level);
        res += corruptTargetFile(exec_id, target_node , target_rank, 1, corrORErase, level);
    } else if (corruptionLevel == 3) { //all
        int i;
        for (i = 0; i < nbProcs; i++) {
            if (head && (i % node_size == 0)) continue;
            target_node = i / node_size;
            res = corruptTargetFile(exec_id, target_node , i, ckptORPtner, corrORErase, level);
            if (res) {
                printf("Faild to corrupt rank = %d, node = %d\n", i, target_node);
                break;
            }
        }
    }
    if (res) {
        printf("Failed to corrupt a file.\n");
    }
    free(exec_id);
    return res;
}
