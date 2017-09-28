/**
 *  @file   topo.c
 *  @author Leonardo A. Bautista Gomez (leobago@gmail.com)
 *  @date   July, 2013
 *  @brief  Topology functions for the FTI library.
 */

#include "interface.h"

/*-------------------------------------------------------------------------*/
/**
    @brief      It writes the topology in a file for recovery.
    @param      FTI_Conf        Configuration metadata.
    @param      FTI_Topo        Topology metadata.
    @param      nameList        The list of the node names.
    @return     integer         FTI_SCES if successful.

    This function writes the topology of the system (List of nodes and their
    ID) in a topology file that will be read during recovery to detect which
    nodes (and therefore checkpoit files) are missing in the new topology.

 **/
/*-------------------------------------------------------------------------*/
int FTI_SaveTopo(FTIT_configuration* FTI_Conf, FTIT_topology* FTI_Topo, char* nameList)
{
    char str[FTI_BUFS];
    sprintf(str, "Trying to load configuration file (%s) to create topology.", FTI_Conf->cfgFile);
    FTI_Print(str, FTI_DBUG);

    dictionary* ini = iniparser_load(FTI_Conf->cfgFile);
    if (ini == NULL) {
        FTI_Print("Iniparser cannot parse the configuration file.", FTI_WARN);
        return FTI_NSCS;
    }

    // Set topology section
    iniparser_set(ini, "topology", NULL);

    // Write list of nodes
    int i;
    for (i = 0; i < FTI_Topo->nbNodes; i++) {
        char mfn[FTI_BUFS];
        strncpy(mfn, nameList + (i * FTI_BUFS), FTI_BUFS - 1);
        sprintf(str, "topology:%d", i);
        iniparser_set(ini, str, mfn);
    }

    // Unset sections of the configuration file
    iniparser_unset(ini, "basic");
    iniparser_unset(ini, "restart");
    iniparser_unset(ini, "advanced");

    char mfn[FTI_BUFS];
    sprintf(mfn, "%s/Topology.fti", FTI_Conf->metadDir);
    sprintf(str, "Creating topology file (%s)...", mfn);
    FTI_Print(str, FTI_DBUG);

    FILE* fd = fopen(mfn, "w");
    if (fd == NULL) {
        FTI_Print("Topology file could NOT be opened", FTI_WARN);

        iniparser_freedict(ini);

        return FTI_NSCS;
    }

    // Write new topology
    iniparser_dump_ini(ini, fd);

    if (fclose(fd) != 0) {
        FTI_Print("Topology file could NOT be closed.", FTI_WARN);

        iniparser_freedict(ini);

        return FTI_NSCS;
    }

    iniparser_freedict(ini);

    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
    @brief      It reorders the nodes following the previous topology.
    @param      FTI_Conf        Configuration metadata.
    @param      FTI_Topo        Topology metadata.
    @param      nodeList        The list of the nodes.
    @param      nameList        The list of the node names.
    @return     integer         FTI_SCES if successful.

    This function writes the topology of the system (List of nodes and their
    ID) in a topology file that will be read during recovery to detect which
    nodes (and therefore checkpoit files) are missing in the new topology.

 **/
/*-------------------------------------------------------------------------*/
int FTI_ReorderNodes(FTIT_configuration* FTI_Conf, FTIT_topology* FTI_Topo,
                     int* nodeList, char* nameList)
{
    int* nl = talloc(int, FTI_Topo->nbProc);
    int* old = talloc(int, FTI_Topo->nbNodes);
    int* new = talloc(int, FTI_Topo->nbNodes);
    int i;
    for (i = 0; i < FTI_Topo->nbNodes; i++) {
        old[i] = -1;
        new[i] = -1;
    }

    char mfn[FTI_BUFS], str[FTI_BUFS];
    sprintf(mfn, "%s/Topology.fti", FTI_Conf->metadDir);
    sprintf(str, "Loading FTI topology file (%s) to reorder nodes...", mfn);
    FTI_Print(str, FTI_DBUG);

    // Checking that the topology file exist
    if (access(mfn, F_OK) != 0) {
        FTI_Print("The topology file is NOT accessible.", FTI_WARN);

        free(nl);
        free(old);
        free(new);

        return FTI_NSCS;
    }

    dictionary* ini;
    ini = iniparser_load(mfn);
    if (ini == NULL) {
        FTI_Print("Iniparser could NOT parse the topology file.", FTI_WARN);

        free(nl);
        free(old);
        free(new);

        return FTI_NSCS;
    }

    // Get the old order of nodes
    for (i = 0; i < FTI_Topo->nbNodes; i++) {
        sprintf(str, "Topology:%d", i);
        char* tmp = iniparser_getstring(ini, str, NULL);
        snprintf(str, FTI_BUFS, "%s", tmp);

        // Search for same node in current nameList
        int j;
        for (j = 0; j < FTI_Topo->nbNodes; j++) {
            // If found...
            if (strncmp(str, nameList + (j * FTI_BUFS), FTI_BUFS) == 0) {
                old[j] = i;
                new[i] = j;
                break;
            } // ...set matching IDs and break out of the searching loop
        }
    }

    iniparser_freedict(ini);

    int j = 0;
    // Introducing missing nodes
    for (i = 0; i < FTI_Topo->nbNodes; i++) {
        // For each new node..
        if (new[i] == -1) {
            // ..search for an old node not present in the new list..
            while (old[j] != -1) {
                j++;
            }
            // .. and set matching IDs
            old[j] = i;
            new[i] = j;
            j++;
        }
    }
    // Copying nodeList in nl
    for (i = 0; i < FTI_Topo->nbProc; i++) {
        nl[i] = nodeList[i];
    }
    // Creating the new nodeList with the old order
    for (i = 0; i < FTI_Topo->nbNodes; i++) {
        for (j = 0; j < FTI_Topo->nodeSize; j++) {
            nodeList[(i * FTI_Topo->nodeSize) + j] = nl[(new[i] * FTI_Topo->nodeSize) + j];
        }
    }

    // Free memory
    free(nl);
    free(old);
    free(new);

    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
    @brief      It builds the list of nodes in the current execution.
    @param      FTI_Conf        Configuration metadata.
    @param      FTI_Exec        Execution metadata.
    @param      FTI_Topo        Topology metadata.
    @param      nodeList        The list of the nodes to fill.
    @param      nameList        The list of the node names to fill.
    @return     integer         FTI_SCES if successful.

    This function makes all the processes to detect in which node are they
    located and distributes the information globally to create an uniform
    mapping structure between processes and nodes.

 **/
/*-------------------------------------------------------------------------*/
int FTI_BuildNodeList(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
                      FTIT_topology* FTI_Topo, int* nodeList, char* nameList)
{
    char* lhn = talloc(char, FTI_BUFS* FTI_Topo->nbProc);
    memset(lhn + (FTI_Topo->myRank * FTI_BUFS), 0, FTI_BUFS); // To get local hostname
    if (!FTI_Conf->test) {
        gethostname(lhn + (FTI_Topo->myRank * FTI_BUFS), FTI_BUFS); // NOT local test
    }
    else {
        snprintf(lhn + (FTI_Topo->myRank * FTI_BUFS), FTI_BUFS, "node%d", FTI_Topo->myRank / FTI_Topo->nodeSize); // Local
    }
    char hname[FTI_BUFS];
    strncpy(hname, lhn + (FTI_Topo->myRank * FTI_BUFS), FTI_BUFS - 1); // Distributing host names
    MPI_Allgather(hname, FTI_BUFS, MPI_CHAR, lhn, FTI_BUFS, MPI_CHAR, FTI_Exec->globalComm);

    int nbNodes = 0;
    int i;
    for (i = 0; i < FTI_Topo->nbProc; i++) { // Creating the node list: For each process
        int found = 0;
        int pos = 0;
        strncpy(hname, lhn + (i * FTI_BUFS), FTI_BUFS - 1); // Get node name of process i
        while ((pos < nbNodes) && (found == 0)) { // Search the node name in the current list of node names
            if (strncmp(&(nameList[pos * FTI_BUFS]), hname, FTI_BUFS) == 0) { // If we find it break out
                found = 1;
            }
            else { // Else move to the next name in the list
                pos++;
            }
        }
        if (found) { // If we found the node name in the current list...
            int p = pos * FTI_Topo->nodeSize;
            while (p < pos * FTI_Topo->nodeSize + FTI_Topo->nodeSize) { // ... we look for empty spot in this node
                if (nodeList[p] == -1) {
                    nodeList[p] = i;
                    break;
                }
                else {
                    p++;
                }
            }
        }
        else { // ... else, we add the new node to the end of the current list of nodes
            strncpy(&(nameList[pos * FTI_BUFS]), hname, FTI_BUFS - 1);
            nodeList[pos * FTI_Topo->nodeSize] = i;
            nbNodes++;
        }
    }
    for (i = 0; i < FTI_Topo->nbProc; i++) { // Checking that all nodes have nodeSize processes
        if (nodeList[i] == -1) {
            char str[FTI_BUFS];
            sprintf(str, "Node %d has no %d processes", i / FTI_Topo->nodeSize, FTI_Topo->nodeSize);
            FTI_Print(str, FTI_WARN);
            free(lhn);
            return FTI_NSCS;
        }
    }

    free(lhn);

    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
    @brief      It builds the list of nodes in the current execution.
    @param      FTI_Conf        Configuration metadata.
    @param      FTI_Exec        Execution metadata.
    @param      FTI_Topo        Topology metadata.
    @param      userProcList    The list of the app. processess.
    @param      distProcList    The list of the distributed processes.
    @param      nodeList        The list of the nodes to fill.
    @return     integer         FTI_SCES if successful.

    This function makes all the processes to detect in which node are they
    located and distributes the information globally to create an uniform
    mapping structure between processes and nodes.

 **/
/*-------------------------------------------------------------------------*/
int FTI_CreateComms(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
                    FTIT_topology* FTI_Topo, int* userProcList,
                    int* distProcList, int* nodeList)
{
    MPI_Group newGroup, origGroup;
    MPI_Comm_group(FTI_Exec->globalComm, &origGroup);
    if (FTI_Topo->amIaHead) {
        MPI_Group_incl(origGroup, FTI_Topo->nbNodes * FTI_Topo->nbHeads, distProcList, &newGroup);
        MPI_Comm_create(FTI_Exec->globalComm, newGroup, &FTI_COMM_WORLD);
        int i;
        for (i = FTI_Topo->nbHeads; i < FTI_Topo->nodeSize; i++) {
            int src = nodeList[(FTI_Topo->nodeID * FTI_Topo->nodeSize) + i];
            int buf;
            MPI_Recv(&buf, 1, MPI_INT, src, FTI_Conf->tag, FTI_Exec->globalComm, MPI_STATUS_IGNORE);
            if (buf == src) {
                FTI_Topo->body[i - FTI_Topo->nbHeads] = src;
            }
        }
    }
    else {
        MPI_Group_incl(origGroup, FTI_Topo->nbProc - (FTI_Topo->nbNodes * FTI_Topo->nbHeads), userProcList, &newGroup);
        MPI_Comm_create(FTI_Exec->globalComm, newGroup, &FTI_COMM_WORLD);
        if (FTI_Topo->nbHeads == 1) {
            MPI_Send(&(FTI_Topo->myRank), 1, MPI_INT, FTI_Topo->headRank, FTI_Conf->tag, FTI_Exec->globalComm);
        }
    }
    MPI_Comm_rank(FTI_COMM_WORLD, &FTI_Topo->splitRank);
    int buf = FTI_Topo->sectorID * FTI_Topo->groupSize;
    int group[FTI_BUFS]; // FTI_BUFS > Max. group size
    int i;
    for (i = 0; i < FTI_Topo->groupSize; i++) { // Group of node-distributed processes (Topology-aware).
        group[i] = distProcList[buf + i];
    }
    MPI_Comm_group(FTI_Exec->globalComm, &origGroup);
    MPI_Group_incl(origGroup, FTI_Topo->groupSize, group, &newGroup);
    MPI_Comm_create(FTI_Exec->globalComm, newGroup, &FTI_Exec->groupComm);
    MPI_Group_rank(newGroup, &(FTI_Topo->groupRank));
    FTI_Topo->right = (FTI_Topo->groupRank + 1) % FTI_Topo->groupSize;
    FTI_Topo->left = (FTI_Topo->groupRank + FTI_Topo->groupSize - 1) % FTI_Topo->groupSize;
    MPI_Group_free(&origGroup);
    MPI_Group_free(&newGroup);
    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
    @brief      It builds and saves the topology of the current execution.
    @param      FTI_Conf        Configuration metadata.
    @param      FTI_Exec        Execution metadata.
    @param      FTI_Topo        Topology metadata.
    @return     integer         FTI_SCES if successful.

    This function builds the topology of the system, detects and replaces
    missing nodes in case of recovery and creates the communicators required
    for FTI to work. It stores all required information in FTI_Topo.

 **/
/*-------------------------------------------------------------------------*/
int FTI_Topology(FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec,
                 FTIT_topology* FTI_Topo)
{
    char *nameList = talloc(char, FTI_Topo->nbNodes *FTI_BUFS);

    int* nodeList = talloc(int, FTI_Topo->nbNodes* FTI_Topo->nodeSize);
    int i;
    for (i = 0; i < FTI_Topo->nbProc; i++) {
        nodeList[i] = -1;
    }

    int res = FTI_Try(FTI_BuildNodeList(FTI_Conf, FTI_Exec, FTI_Topo, nodeList, nameList), "create node list.");
    if (res == FTI_NSCS) {
        free(nameList);
        free(nodeList);

        return FTI_NSCS;
    }

    if (FTI_Exec->reco > 0) {
        res = FTI_Try(FTI_ReorderNodes(FTI_Conf, FTI_Topo, nodeList, nameList), "reorder nodes.");
        if (res == FTI_NSCS) {
            free(nameList);
            free(nodeList);

            return FTI_NSCS;
        }
    }

    // Need to synchronize before editing topology file
    MPI_Barrier(FTI_Exec->globalComm);
    if (FTI_Topo->myRank == 0 && FTI_Exec->reco == 0) {
        res = FTI_Try(FTI_SaveTopo(FTI_Conf, FTI_Topo, nameList), "save topology.");
        if (res == FTI_NSCS) {
            free(nameList);
            free(nodeList);

            return FTI_NSCS;
        }
    }

    int *distProcList = talloc(int, FTI_Topo->nbNodes);
    int *userProcList = talloc(int, FTI_Topo->nbProc - (FTI_Topo->nbNodes * FTI_Topo->nbHeads));

    int mypos = -1, c = 0;
    for (i = 0; i < FTI_Topo->nbProc; i++) {
        if (FTI_Topo->myRank == nodeList[i]) {
            mypos = i;
        }
        if ((i % FTI_Topo->nodeSize != 0) || (FTI_Topo->nbHeads == 0)) {
            userProcList[c] = nodeList[i];
            c++;
        }
    }
    if (mypos == -1) {
        free(userProcList);
        free(distProcList);
        free(nameList);
        free(nodeList);

        return FTI_NSCS;
    }

    FTI_Topo->nodeRank = mypos % FTI_Topo->nodeSize;
    if (FTI_Topo->nodeRank == 0 && FTI_Topo->nbHeads == 1) {
        FTI_Topo->amIaHead = 1;
    }
    else {
        FTI_Topo->amIaHead = 0;
    }
    FTI_Topo->nodeID = mypos / FTI_Topo->nodeSize;
    FTI_Topo->headRank = nodeList[(mypos / FTI_Topo->nodeSize) * FTI_Topo->nodeSize];
    FTI_Topo->sectorID = FTI_Topo->nodeID / FTI_Topo->groupSize;
    int posInNode = mypos % FTI_Topo->nodeSize;
    FTI_Topo->groupID = posInNode;
    for (i = 0; i < FTI_Topo->nbNodes; i++) {
        distProcList[i] = nodeList[(FTI_Topo->nodeSize * i) + posInNode];
    }

    res = FTI_Try(FTI_CreateComms(FTI_Conf, FTI_Exec, FTI_Topo, userProcList, distProcList, nodeList), "create communicators.");
    if (res == FTI_NSCS) {
        free(userProcList);
        free(distProcList);
        free(nameList);
        free(nodeList);

        return FTI_NSCS;
    }

    free(userProcList);
    free(distProcList);
    free(nameList);
    free(nodeList);

    return FTI_SCES;
}
