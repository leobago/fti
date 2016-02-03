/**
 *  @file   topo.c
 *  @author Leonardo A. Bautista Gomez (leobago@gmail.com)
 *  @date   July, 2013
 *  @brief  Topology functions for the FTI library.
 */

#include "fti.h"

/*-------------------------------------------------------------------------*/
/**
    @brief      Writes the topology in a file for recovery.
    @param      nameList        The list of the node names.
    @return     integer         FTI_SCES if successful.

    This function writes the topology of the system (List of nodes and their
    ID) in a topology file that will be read during recovery to detect which
    nodes (and therefore checkpoit files) are missing in the new topology.

 **/
/*-------------------------------------------------------------------------*/
int FTI_SaveTopo(char* nameList)
{
    char mfn[FTI_BUFS], str[FTI_BUFS];
    dictionary* ini;
    int i;
    sprintf(str, "Trying to load configuration file (%s) to create topology.", FTI_Conf.cfgFile);
    FTI_Print(str, FTI_DBUG);
    ini = iniparser_load(FTI_Conf.cfgFile);
    if (ini == NULL) {
        FTI_Print("Iniparser cannot parse the configuration file.", FTI_WARN);
        return FTI_NSCS;
    }
    iniparser_set(ini, "topology", NULL); // Set topology section
    for (i = 0; i < FTI_Topo.nbNodes; i++) { // Write list of nodes
        strncpy(mfn, nameList + (i * FTI_BUFS), FTI_BUFS);
        sprintf(str, "topology:%d", i);
        iniparser_set(ini, str, mfn);
    } // Unset sections of the configuration file
    iniparser_unset(ini, "basic");
    iniparser_unset(ini, "restart");
    iniparser_unset(ini, "advanced");
    sprintf(mfn, "%s/Topology.fti", FTI_Conf.metadDir);
    sprintf(str, "Creating topology file (%s)...", mfn);
    FTI_Print(str, FTI_DBUG);
    FILE* fd = fopen(mfn, "w");
    if (fd == NULL) {
        FTI_Print("Topology file could NOT be opened", FTI_WARN);
        return FTI_NSCS;
    }
    iniparser_dump_ini(ini, fd); // Write new topology
    if (fflush(fd) != 0) {
        FTI_Print("Topology file could NOT be flushed.", FTI_WARN);
        return FTI_NSCS;
    }
    if (fclose(fd) != 0) {
        FTI_Print("Topology file could NOT be closed.", FTI_WARN);
        return FTI_NSCS;
    }
    iniparser_freedict(ini);
    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
    @brief      Reorder the nodes following the previous topology.
    @param      nodeList        The list of the nodes.
    @param      nameList        The list of the node names.
    @return     integer         FTI_SCES if successful.

    This function writes the topology of the system (List of nodes and their
    ID) in a topology file that will be read during recovery to detect which
    nodes (and therefore checkpoit files) are missing in the new topology.

 **/
/*-------------------------------------------------------------------------*/
int FTI_ReorderNodes(int* nodeList, char* nameList)
{
    char mfn[FTI_BUFS], str[FTI_BUFS], *tmp;
    int i, j, *nl, *old, *new;
    nl = talloc(int, FTI_Topo.nbProc);
    old = talloc(int, FTI_Topo.nbNodes);
    new = talloc(int, FTI_Topo.nbNodes);
    for (i = 0; i < FTI_Topo.nbNodes; i++) {
        old[i] = -1;
        new[i] = -1;
    }
    sprintf(mfn, "%s/Topology.fti", FTI_Conf.metadDir);
    sprintf(str, "Loading FTI topology file (%s) to reorder nodes...", mfn);
    FTI_Print(str, FTI_DBUG);
    if (access(mfn, F_OK) != 0) { // Checking that the topology file exist
        FTI_Print("The topology file is NOT accessible.", FTI_WARN);
        return FTI_NSCS;
    }
    dictionary* ini;
    ini = iniparser_load(mfn);
    if (ini == NULL) {
        FTI_Print("Iniparser could NOT parse the topology file.", FTI_WARN);
        return FTI_NSCS;
    }
    for (i = 0; i < FTI_Topo.nbNodes; i++) { // Get the old order of nodes
        sprintf(str, "Topology:%d", i);
        tmp = iniparser_getstring(ini, str, NULL);
        snprintf(str, FTI_BUFS, "%s", tmp);
        for (j = 0; j < FTI_Topo.nbNodes; j++) { // Search for same node in current nameList
            if (strncmp(str, nameList + (j * FTI_BUFS), FTI_BUFS) == 0) // If found...
            {
                old[j] = i;
                new[i] = j;
                break;
            } // ...set matching IDs and break out of the searching loop
        }
    }
    iniparser_freedict(ini);
    j = 0;
    for (i = 0; i < FTI_Topo.nbNodes; i++) { // Introducing missing nodes
        if (new[i] == -1) { // For each new node..
            while (old[j] != -1) { // ..search for an old node not present in the new list...
                j++;
            } // .. and set matching IDs
            old[j] = i;
            new[i] = j;
            j++;
        }
    }
    for (i = 0; i < FTI_Topo.nbProc; i++) { // Copying nodeList in nl
        nl[i] = nodeList[i];
    }
    for (i = 0; i < FTI_Topo.nbNodes; i++) { // Creating the new nodeList with the old order
        for (j = 0; j < FTI_Topo.nodeSize; j++) {
            nodeList[(i * FTI_Topo.nodeSize) + j] = nl[(new[i] * FTI_Topo.nodeSize) + j];
        }
    } // Free memory
    free(old);
    free(new);
    free(nl);
    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
    @brief      Build the list of nodes in the current execution.
    @param      nodeList        The list of the nodes to fill.
    @param      nameList        The list of the node names to fill.
    @return     integer         FTI_SCES if successful.

    This function makes all the processes to detect in which node are they
    located and distributes the information globally to create an uniform
    mapping structure between processes and nodes.

 **/
/*-------------------------------------------------------------------------*/
int FTI_BuildNodeList(int* nodeList, char* nameList)
{
    int i, found, pos, p, nbNodes = 0;
    char hname[FTI_BUFS], str[FTI_BUFS], *lhn;
    lhn = talloc(char, FTI_BUFS* FTI_Topo.nbProc);
    memset(lhn + (FTI_Topo.myRank * FTI_BUFS), 0, FTI_BUFS); // To get local hostname
    if (!FTI_Conf.test) {
        gethostname(lhn + (FTI_Topo.myRank * FTI_BUFS), FTI_BUFS); // NOT local test
    }
    else {
        snprintf(lhn + (FTI_Topo.myRank * FTI_BUFS), FTI_BUFS, "node%d", FTI_Topo.myRank / FTI_Topo.nodeSize); // Local
    }
    strncpy(hname, lhn + (FTI_Topo.myRank * FTI_BUFS), FTI_BUFS); // Distributing host names
    MPI_Allgather(hname, FTI_BUFS, MPI_CHAR, lhn, FTI_BUFS, MPI_CHAR, FTI_Exec.globalComm);

    for (i = 0; i < FTI_Topo.nbProc; i++) { // Creating the node list: For each process
        found = 0;
        pos = 0;
        strncpy(hname, lhn + (i * FTI_BUFS), FTI_BUFS); // Get node name of process i
        while ((pos < nbNodes) && (found == 0)) { // Search the node name in the current list of node names
            if (strncmp(&(nameList[pos * FTI_BUFS]), hname, FTI_BUFS) == 0) { // If we find it break out
                found = 1;
            }
            else { // Else move to the next name in the list
                pos++;
            }
        }
        if (found) { // If we found the node name in the current list...
            p = pos * FTI_Topo.nodeSize;
            while (p < pos * FTI_Topo.nodeSize + FTI_Topo.nodeSize) { // ... we look for empty spot in this node
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
            strncpy(&(nameList[pos * FTI_BUFS]), hname, FTI_BUFS);
            nodeList[pos * FTI_Topo.nodeSize] = i;
            nbNodes++;
        }
    }
    for (i = 0; i < FTI_Topo.nbProc; i++) { // Checking that all nodes have nodeSize processes
        if (nodeList[i] == -1) {
            sprintf(str, "Node %d has no %d processes", i / FTI_Topo.nodeSize, FTI_Topo.nodeSize);
            FTI_Print(str, FTI_WARN);
            return FTI_NSCS;
        }
    }
    free(lhn);
    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
    @brief      Build the list of nodes in the current execution.
    @param      userProcList    The list of the app. processess.
    @param      distProcList    The list of the distributed processes.
    @param      nodeList        The list of the nodes to fill.
    @return     integer         FTI_SCES if successful.

    This function makes all the processes to detect in which node are they
    located and distributes the information globally to create an uniform
    mapping structure between processes and nodes.

 **/
/*-------------------------------------------------------------------------*/
int FTI_CreateComms(int* userProcList, int* distProcList, int* nodeList)
{
    MPI_Status status;
    char str[FTI_BUFS];
    MPI_Group newGroup, origGroup;
    MPI_Comm_group(FTI_Exec.globalComm, &origGroup);
    int i, src, buf, group[FTI_BUFS]; // FTI_BUFS > Max. group size
    if (FTI_Topo.amIaHead) {
        MPI_Group_incl(origGroup, FTI_Topo.nbNodes * FTI_Topo.nbHeads, distProcList, &newGroup);
        MPI_Comm_create(FTI_Exec.globalComm, newGroup, &FTI_COMM_WORLD);
        for (i = FTI_Topo.nbHeads; i < FTI_Topo.nodeSize; i++) {
            src = nodeList[(FTI_Topo.nodeID * FTI_Topo.nodeSize) + i];
            MPI_Recv(&buf, 1, MPI_INT, src, FTI_Conf.tag, FTI_Exec.globalComm, &status);
            if (buf == src) {
                FTI_Topo.body[i - FTI_Topo.nbHeads] = src;
            }
        }
    }
    else {
        MPI_Group_incl(origGroup, FTI_Topo.nbProc - (FTI_Topo.nbNodes * FTI_Topo.nbHeads), userProcList, &newGroup);
        MPI_Comm_create(FTI_Exec.globalComm, newGroup, &FTI_COMM_WORLD);
        if (FTI_Topo.nbHeads == 1) {
            MPI_Send(&(FTI_Topo.myRank), 1, MPI_INT, FTI_Topo.headRank, FTI_Conf.tag, FTI_Exec.globalComm);
        }
    }
    MPI_Comm_rank(FTI_COMM_WORLD, &FTI_Topo.splitRank);
    buf = FTI_Topo.sectorID * FTI_Topo.groupSize;
    for (i = 0; i < FTI_Topo.groupSize; i++) { // Group of node-distributed processes (Topology-aware).
        group[i] = distProcList[buf + i];
    }
    MPI_Comm_group(FTI_Exec.globalComm, &origGroup);
    MPI_Group_incl(origGroup, FTI_Topo.groupSize, group, &newGroup);
    MPI_Comm_create(FTI_Exec.globalComm, newGroup, &FTI_Exec.groupComm);
    MPI_Group_rank(newGroup, &(FTI_Topo.groupRank));
    FTI_Topo.right = (FTI_Topo.groupRank + 1) % FTI_Topo.groupSize;
    FTI_Topo.left = (FTI_Topo.groupRank + FTI_Topo.groupSize - 1) % FTI_Topo.groupSize;
    MPI_Group_free(&origGroup);
    MPI_Group_free(&newGroup);
    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
    @brief      Builds and saves the topology of the current execution.
    @return     integer         FTI_SCES if successful.

    This function builds the topology of the system, detects and replaces
    missing nodes in case of recovery and creates the communicators required
    for FTI to work. It stores all required information in FTI_Topo.

 **/
/*-------------------------------------------------------------------------*/
int FTI_Topology()
{
    int res, nn, found, c1 = 0, c2 = 0, p, i, mypos, posInNode;
    char str[FTI_BUFS], *nameList = talloc(char, FTI_Topo.nbNodes *FTI_BUFS);
    int* nodeList = talloc(int, FTI_Topo.nbNodes* FTI_Topo.nodeSize);
    int* distProcList = talloc(int, FTI_Topo.nbNodes);
    int* userProcList = talloc(int, FTI_Topo.nbProc - (FTI_Topo.nbNodes * FTI_Topo.nbHeads));
    for (i = 0; i < FTI_Topo.nbProc; i++) {
        nodeList[i] = -1;
    }
    res = FTI_Try(FTI_BuildNodeList(nodeList, nameList), "create node list.");
    if (res == FTI_NSCS) {
        return FTI_NSCS;
    }
    if (FTI_Exec.reco > 0) {
        res = FTI_Try(FTI_ReorderNodes(nodeList, nameList), "reorder nodes.");
        if (res == FTI_NSCS) {
            return FTI_NSCS;
        }
    }
    MPI_Barrier(FTI_Exec.globalComm); // Need to synchronize before editing topology file
    if (FTI_Topo.myRank == 0 && FTI_Exec.reco == 0) {
        res = FTI_Try(FTI_SaveTopo(nameList), "save topology.");
        if (res == FTI_NSCS) {
            return FTI_NSCS;
        }
    }
    for (i = 0; i < FTI_Topo.nbProc; i++) {
        if (FTI_Topo.myRank == nodeList[i]) {
            mypos = i;
        }
        if ((i % FTI_Topo.nodeSize != 0) || (FTI_Topo.nbHeads == 0)) {
            userProcList[c2] = nodeList[i];
            c2++;
        }
    }
    FTI_Topo.nodeRank = mypos % FTI_Topo.nodeSize;
    if (FTI_Topo.nodeRank == 0 && FTI_Topo.nbHeads == 1) {
        FTI_Topo.amIaHead = 1;
    }
    else {
        FTI_Topo.amIaHead = 0;
    }
    FTI_Topo.nodeID = mypos / FTI_Topo.nodeSize;
    FTI_Topo.headRank = nodeList[(mypos / FTI_Topo.nodeSize) * FTI_Topo.nodeSize];
    FTI_Topo.sectorID = FTI_Topo.nodeID / FTI_Topo.groupSize;
    posInNode = mypos % FTI_Topo.nodeSize;
    FTI_Topo.groupID = posInNode;
    for (i = 0; i < FTI_Topo.nbNodes; i++) {
        distProcList[i] = nodeList[(FTI_Topo.nodeSize * i) + posInNode];
    }
    res = FTI_Try(FTI_CreateComms(userProcList, distProcList, nodeList), "create communicators.");
    if (res == FTI_NSCS) {
        return FTI_NSCS;
    }
    free(userProcList);
    free(distProcList);
    free(nameList);
    free(nodeList);
    return FTI_SCES;
}
