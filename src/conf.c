/**
 *  @file   conf.c
 *  @author Leonardo A. Bautista Gomez (leobago@gmail.com)
 *  @date   December, 2013
 *  @brief  Configuration loading functions for the FTI library.
 */

#include "fti.h"

/*-------------------------------------------------------------------------*/
/**
    @brief      Set the exec. ID and failure parameters in the conf. file.
    @param      restart         Value to set in the conf. file (0 or 1).
    @return     integer         FTI_SCES if successful.

    This function sets the execution ID and failure parameters in the
    configuration file. This is to avoid forcing the user to change these
    values manually in case of recovery needed. In this way, relaunching the
    execution in the same way as the initial time will make FTI detect that
    it is a restart. It also allows to set the failure parameter back to 0
    at the end of a successful execution.

 **/
/*-------------------------------------------------------------------------*/
int FTI_UpdateConf(int restart)
{
    char str[FTI_BUFS];
    dictionary* ini;
    ini = iniparser_load(FTI_Conf.cfgFile); // Load dictionary
    sprintf(str, "Updating configuration file (%s)...", FTI_Conf.cfgFile);
    FTI_Print(str, FTI_DBUG);
    if (ini == NULL) {
        FTI_Print("Iniparser failed to parse the conf. file.", FTI_WARN);
        return FTI_NSCS;
    }
    sprintf(str, "%d", restart);
    iniparser_set(ini, "Restart:failure", str); // Set failure to 'restart'
    iniparser_set(ini, "Restart:exec_id", FTI_Exec.id); // Set the exec. ID
    FILE* fd = fopen(FTI_Conf.cfgFile, "w");
    if (fd == NULL) {
        FTI_Print("FTI failed to open the configuration file.", FTI_EROR);
        return FTI_NSCS;
    }
    iniparser_dump_ini(ini, fd); // Write new configuration
    if (fflush(fd) != 0) {
        FTI_Print("FTI failed to flush the configuration file.", FTI_EROR);
        return FTI_NSCS;
    }
    if (fclose(fd) != 0) {
        FTI_Print("FTI failed to close the configuration file.", FTI_EROR);
        return FTI_NSCS;
    }
    iniparser_freedict(ini); // Free dictionary
    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
    @brief      It reads the configuration given in the configuration file.
    @return     integer         FTI_SCES if successful.

    This function reads the configuration given in the FTI configuration
    file and sets other required parameters.

 **/
/*-------------------------------------------------------------------------*/
int FTI_ReadConf(FTIT_injection* FTI_Inje)
{
    // Check access to FTI configuration file and load dictionary
    dictionary* ini;
    char *par, str[FTI_BUFS];
    sprintf(str, "Reading FTI configuration file (%s)...", FTI_Conf.cfgFile);
    FTI_Print(str, FTI_INFO);
    if (access(FTI_Conf.cfgFile, F_OK) != 0) {
        FTI_Print("FTI configuration file NOT accessible.", FTI_WARN);
        return FTI_NSCS;
    }
    ini = iniparser_load(FTI_Conf.cfgFile);
    if (ini == NULL) {
        FTI_Print("Iniparser failed to parse the conf. file.", FTI_WARN);
        return FTI_NSCS;
    }

    // Setting/reading checkpoint configuration metadata
    par = iniparser_getstring(ini, "Basic:ckpt_dir", NULL);
    snprintf(FTI_Conf.localDir, FTI_BUFS, "%s", par);
    par = iniparser_getstring(ini, "Basic:glbl_dir", NULL);
    snprintf(FTI_Conf.glbalDir, FTI_BUFS, "%s", par);
    par = iniparser_getstring(ini, "Basic:meta_dir", NULL);
    snprintf(FTI_Conf.metadDir, FTI_BUFS, "%s", par);
    FTI_Ckpt[1].ckptIntv = (int)iniparser_getint(ini, "Basic:ckpt_l1", -1);
    FTI_Ckpt[2].ckptIntv = (int)iniparser_getint(ini, "Basic:ckpt_l2", -1);
    FTI_Ckpt[3].ckptIntv = (int)iniparser_getint(ini, "Basic:ckpt_l3", -1);
    FTI_Ckpt[4].ckptIntv = (int)iniparser_getint(ini, "Basic:ckpt_l4", -1);
    FTI_Ckpt[1].isInline = (int)1;
    FTI_Ckpt[2].isInline = (int)iniparser_getint(ini, "Basic:inline_l2", 1);
    FTI_Ckpt[3].isInline = (int)iniparser_getint(ini, "Basic:inline_l3", 1);
    FTI_Ckpt[4].isInline = (int)iniparser_getint(ini, "Basic:inline_l4", 1);

    // Reading/setting configuration metadata
    FTI_Conf.verbosity = (int)iniparser_getint(ini, "Basic:verbosity", -1);
    FTI_Conf.saveLastCkpt = (int)iniparser_getint(ini, "Basic:keep_last_ckpt", 0);
    FTI_Conf.blockSize = (int)iniparser_getint(ini, "Advanced:block_size", -1) * 1024;
    FTI_Conf.tag = (int)iniparser_getint(ini, "Advanced:mpi_tag", -1);
    FTI_Conf.test = (int)iniparser_getint(ini, "Advanced:local_test", -1);
    FTI_Conf.l3WordSize = FTI_WORD;

    // Reading/setting execution metadata
    FTI_Exec.nbVar = 0;
    FTI_Exec.nbType = 0;
    FTI_Exec.ckpt = 0;
    FTI_Exec.ckptCnt = 0;
    FTI_Exec.ckptIcnt = 0;
    FTI_Exec.ckptID = 0;
    FTI_Exec.ckptLvel = 0;
    FTI_Exec.ckptIntv = 1;
    FTI_Exec.wasLastOffline = 0;
    FTI_Exec.ckptNext = 0;
    FTI_Exec.ckptLast = 0;
    FTI_Exec.syncIter = 1;
    FTI_Exec.lastIterTime = 0;
    FTI_Exec.totalIterTime = 0;
    FTI_Exec.meanIterTime = 0;
    FTI_Exec.reco = (int)iniparser_getint(ini, "restart:failure", 0);
    if (FTI_Exec.reco == 0) {
        time_t tim = time(NULL);
        struct tm* n = localtime(&tim);
        snprintf(FTI_Exec.id, FTI_BUFS, "%d-%02d-%02d_%02d-%02d-%02d",
            n->tm_year + 1900, n->tm_mon + 1, n->tm_mday, n->tm_hour, n->tm_min, n->tm_sec);
        MPI_Bcast(FTI_Exec.id, FTI_BUFS, MPI_CHAR, 0, FTI_Exec.globalComm);
        sprintf(str, "The execution ID is: %s", FTI_Exec.id);
        FTI_Print(str, FTI_INFO);
    }
    else {
        par = iniparser_getstring(ini, "restart:exec_id", NULL);
        snprintf(FTI_Exec.id, FTI_BUFS, "%s", par);
        sprintf(str, "This is a restart. The execution ID is: %s", FTI_Exec.id);
        FTI_Print(str, FTI_INFO);
    }

    // Reading/setting topology metadata
    FTI_Topo.nbHeads = (int)iniparser_getint(ini, "Basic:head", 0);
    FTI_Topo.groupSize = (int)iniparser_getint(ini, "Basic:group_size", -1);
    FTI_Topo.nodeSize = (int)iniparser_getint(ini, "Basic:node_size", -1);
    FTI_Topo.nbApprocs = FTI_Topo.nodeSize - FTI_Topo.nbHeads;
    FTI_Topo.nbNodes = FTI_Topo.nbProc / FTI_Topo.nodeSize;

    // Reading/setting injection parameters
    FTI_Inje->rank = (int)iniparser_getint(ini, "Injection:rank", 0);
    FTI_Inje->index = (int)iniparser_getint(ini, "Injection:index", 0);
    FTI_Inje->position = (int)iniparser_getint(ini, "Injection:position", 0);
    FTI_Inje->number = (int)iniparser_getint(ini, "Injection:number", 0);
    FTI_Inje->frequency = (int)iniparser_getint(ini, "Injection:frequency", -1);

    // Synchronize after config reading and free dictionary
    MPI_Barrier(FTI_Exec.globalComm);
    iniparser_freedict(ini);
    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
    @brief      It tests that the configuration given is correct.
    @return     integer         FTI_SCES if successful.

    This function tests the FTI configuration to make sure that all
    parameter's values are correct.

 **/
/*-------------------------------------------------------------------------*/
int FTI_TestConfig()
{
    if (FTI_Topo.nbHeads != 0 && FTI_Topo.nbHeads != 1) {
        FTI_Print("The number of heads needs to be set to 0 or 1.", FTI_WARN);
        return FTI_NSCS;
    }
    if (FTI_Topo.nbProc % FTI_Topo.nodeSize != 0) {
        FTI_Print("Number of ranks is not a multiple of the node size.", FTI_WARN);
        return FTI_NSCS;
    }
    if (FTI_Topo.nbNodes % FTI_Topo.groupSize != 0) {
        FTI_Print("The group size is not multiple of the number of nodes.", FTI_WARN);
        return FTI_NSCS;
    }
    if (FTI_Topo.groupSize <= 2) {
        FTI_Print("The group size must be bigger than 2", FTI_WARN);
        return FTI_NSCS;
    }
    if (FTI_Topo.groupSize >= 32) {
        FTI_Print("The group size must be lower than 32", FTI_WARN);
        return FTI_NSCS;
    }
    if (FTI_Conf.verbosity > 3 || FTI_Conf.verbosity < 1) {
        FTI_Print("Verbosity needs to be set to 1, 2 or 3.", FTI_WARN);
        return FTI_NSCS;
    }
    if (FTI_Conf.blockSize > (2048 * 1024) || FTI_Conf.blockSize < (1 * 1024)) {
        FTI_Print("Block size needs to be set between 1 and 2048.", FTI_WARN);
        return FTI_NSCS;
    }
    if (FTI_Conf.test != 0 && FTI_Conf.test != 1) {
        FTI_Print("Local test size needs to be set to 0 or 1.", FTI_WARN);
        return FTI_NSCS;
    }
    if (FTI_Conf.saveLastCkpt != 0 && FTI_Conf.saveLastCkpt != 1) {
        FTI_Print("Keep last ckpt. needs to be set to 0 or 1.", FTI_WARN);
        return FTI_NSCS;
    }
    int l;
    for (l = 1; l < 5; l++) {
        if (FTI_Ckpt[l].ckptIntv == 0)
            FTI_Ckpt[l].ckptIntv = -1;
        if (FTI_Ckpt[l].isInline != 0 && FTI_Ckpt[l].isInline != 1)
            FTI_Ckpt[l].isInline = 1;
        if (FTI_Ckpt[l].isInline == 0 && FTI_Topo.nbHeads != 1) {
            FTI_Print("If inline is set to 0 then head should be set to 1.", FTI_WARN);
            return FTI_NSCS;
        }
    }
    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
    @brief      It tests that the directories given is correct.
    @return     integer         FTI_SCES if successful.

    This function tests that the directories given in the FTI configuration
    are correct.

 **/
/*-------------------------------------------------------------------------*/
int FTI_TestDirectories()
{
    char str[FTI_BUFS];

    // Checking local directory
    if (access(FTI_Conf.localDir, W_OK) != 0) {
        sprintf(str, "Checking the local directory (%s)...", FTI_Conf.localDir);
        FTI_Print(str, FTI_DBUG);
        FTI_Print("The local directory does not exist or has no write access.", FTI_DBUG);
        if (mkdir(FTI_Conf.localDir, 0777) != 0) {
            FTI_Print("The local directory could NOT be created.", FTI_WARN);
            return FTI_NSCS;
        }
    }

    if (FTI_Topo.myRank == 0) {
        // Checking metadata directory
        sprintf(str, "Checking the metadata directory (%s)...", FTI_Conf.metadDir);
        FTI_Print(str, FTI_DBUG);
        if (access(FTI_Conf.metadDir, W_OK) != 0) {
            FTI_Print("The metadata directory does not exist or has no write access.", FTI_DBUG);
            if (mkdir(FTI_Conf.metadDir, 0777) != 0) {
                FTI_Print("The metadata directory could NOT be created.", FTI_WARN);
                return FTI_NSCS;
            }
        }

        // Checking global directory
        sprintf(str, "Checking the global directory (%s)...", FTI_Conf.glbalDir);
        FTI_Print(str, FTI_DBUG);
        if (access(FTI_Conf.glbalDir, W_OK) != 0) {
            FTI_Print("The global directory does not exist or has no write access.", FTI_DBUG);
            if (mkdir(FTI_Conf.glbalDir, 0777) != 0) {
                FTI_Print("The global directory could NOT be created.", FTI_WARN);
                return FTI_NSCS;
            }
        }
    }

    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
    @brief      It creates the directories required for current execution.
    @return     integer         FTI_SCES if successful.

    This function creates the temporary metadata, local and global
    directories required for the current execution.

 **/
/*-------------------------------------------------------------------------*/
int FTI_CreateDirs()
{
    char fn[FTI_BUFS];

    // Create metadata timestamp directory
    snprintf(fn, FTI_BUFS, "%s/%s", FTI_Conf.metadDir, FTI_Exec.id);
    if (access(fn, F_OK) != 0) {
        mkdir(fn, 0777);
    }
    snprintf(FTI_Conf.metadDir, FTI_BUFS, "%s", fn);
    snprintf(FTI_Conf.mTmpDir, FTI_BUFS, "%s/tmp", fn);
    snprintf(FTI_Ckpt[1].metaDir, FTI_BUFS, "%s/l1", fn);
    snprintf(FTI_Ckpt[2].metaDir, FTI_BUFS, "%s/l2", fn);
    snprintf(FTI_Ckpt[3].metaDir, FTI_BUFS, "%s/l3", fn);
    snprintf(FTI_Ckpt[4].metaDir, FTI_BUFS, "%s/l4", fn);

    // Create global checkpoint timestamp directory
    snprintf(fn, FTI_BUFS, "%s", FTI_Conf.glbalDir);
    snprintf(FTI_Conf.glbalDir, FTI_BUFS, "%s/%s", fn, FTI_Exec.id);
    if (access(FTI_Conf.glbalDir, F_OK) != 0) {
        mkdir(FTI_Conf.glbalDir, 0777);
    }
    snprintf(FTI_Conf.gTmpDir, FTI_BUFS, "%s/tmp", FTI_Conf.glbalDir);
    snprintf(FTI_Ckpt[4].dir, FTI_BUFS, "%s/l4", FTI_Conf.glbalDir);

    // Create local checkpoint timestamp directory
    if (FTI_Conf.test) { // If local test generate name by topology
        snprintf(fn, FTI_BUFS, "%s/node%d", FTI_Conf.localDir, FTI_Topo.myRank / FTI_Topo.nodeSize);
        if (access(fn, F_OK) != 0) {
            mkdir(fn, 0777);
        }
    }
    else {
        snprintf(fn, FTI_BUFS, "%s", FTI_Conf.localDir);
    }
    snprintf(FTI_Conf.localDir, FTI_BUFS, "%s/%s", fn, FTI_Exec.id);
    if (access(FTI_Conf.localDir, F_OK) != 0) {
        mkdir(FTI_Conf.localDir, 0777);
    }
    snprintf(FTI_Conf.lTmpDir, FTI_BUFS, "%s/tmp", FTI_Conf.localDir);
    snprintf(FTI_Ckpt[1].dir, FTI_BUFS, "%s/l1", FTI_Conf.localDir);
    snprintf(FTI_Ckpt[2].dir, FTI_BUFS, "%s/l2", FTI_Conf.localDir);
    snprintf(FTI_Ckpt[3].dir, FTI_BUFS, "%s/l3", FTI_Conf.localDir);
    return FTI_SCES;
}

/*-------------------------------------------------------------------------*/
/**
    @brief      It reads and tests the configuration given.
    @return     integer         FTI_SCES if successful.

    This function reads the configuration file. Then test that the
    configuration parameters are correct (including directories).

 **/
/*-------------------------------------------------------------------------*/
int FTI_LoadConf(FTIT_injection* FTI_Inje)
{
    int res;
    res = FTI_Try(FTI_ReadConf(FTI_Inje), "read configuration.");
    if (res == FTI_NSCS) {
        FTI_Print("Impossible to read configuration.", FTI_WARN);
        return FTI_NSCS;
    }
    res = FTI_Try(FTI_TestConfig(), "pass the configuration test.");
    if (res == FTI_NSCS) {
        FTI_Print("Wrong configuration.", FTI_WARN);
        return FTI_NSCS;
    }
    res = FTI_Try(FTI_TestDirectories(), "pass the directories test.");
    if (res == FTI_NSCS) {
        FTI_Print("Problem with the directories.", FTI_WARN);
        return FTI_NSCS;
    }
    res = FTI_Try(FTI_CreateDirs(), "create checkpoint directories.");
    if (res == FTI_NSCS) {
        FTI_Print("Problem creating the directories.", FTI_WARN);
        return FTI_NSCS;
    }

    return FTI_SCES;
}
