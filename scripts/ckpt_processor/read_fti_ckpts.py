# This module initiates the checkpoint
# processing of FTI files. 

import os
import glob
import os.path
import time
from fnmatch import fnmatch
import configparser
import posix_read_ckpts
import subprocess
import sys

# variables used for input validation
fti_levels = (1, 2, 3, 4)
output_formats = ('CSV', 'HDF5', 'data')

# runtime variables of FTI (ckpt and meta)
config_file = ""
ckpt_dir = ""
meta_dir = ""
global_dir = ""
group_size = 0
nbHeads = 0
nodeSize = 0
totalRanks = 0
ioMode = 0
ckpt_abs_path = ""
meta_abs_path = ""
execution_id = ""
level_meta_dir = ""
level_dir = ""


# This function reads the config_file
# and sets FTI parameters
def init_config_params(config_file):
    global execution_id
    global ckpt_dir
    global meta_dir
    global global_dir
    global group_size
    global nbHeads
    global nodeSize
    global ioMode
    if os.path.isfile(config_file) is False:
        print("Configuration file not found")
        sys.exit(2001)
    else:
        config = configparser.ConfigParser()
        config.read(config_file)
        execution_id = config['restart']['exec_id']
        ckpt_dir = config['basic']['ckpt_dir']
        meta_dir = config['basic']['meta_dir']
        global_dir = config['basic']['glbl_dir']
        group_size = config['basic']['group_size']
        nbHeads = config['basic']['head']
        nodeSize = config['basic']['node_size']
        ioMode = config['basic']['ckpt_io']


# This function processes FTI's files
# given config_file and set the absolute
# paths of meta files and ckpt files
def process_fti_paths(config_file):
    global ckpt_dir
    global meta_dir
    global ckpt_abs_path
    global meta_abs_path
    # ckpt dir
    dir_path = os.path.dirname(os.path.realpath(config_file))
    # concatenate paths
    if level_dir == '/l4/':
        # switch to global_dir
        ckpt_dir = global_dir
    if ckpt_dir.startswith('./') is True:  # same directory as config
        ckpt_abs_path = dir_path + ckpt_dir.replace('.', '')
    elif "." not in ckpt_dir:  # absolute path
        # set dir
        ckpt_abs_path = ckpt_dir
    else:  # relative path
        # iterate over the number of '../' found in ckpt_path
        os.chdir(dir_path)
        dirs = ckpt_dir.count("..")
        for i in range(dirs):
            os.chdir("..")
        # concatenate the remaining part
        for i in range(dirs):
            # remove ../
            ckpt_dir = ckpt_dir.replace('../', '')
        os.chdir(ckpt_dir)
        ckpt_abs_path = os.getcwd()
    print("ckpt_abs_path ", ckpt_abs_path)

    # meta dir
    dir_path = os.path.dirname(os.path.realpath(config_file))
    print(dir_path)
    # concatenate paths
    if meta_dir.startswith('./') is True:  # same directory as config
        # omit dot + concatenate the rest of the path
        meta_abs_path = dir_path + meta_dir.replace('.', '')
    elif "." not in meta_dir:  # absolute path
        # set dir
        meta_abs_path = meta_dir
    else:  # relative path
        # iterate over the number of '../' found in ckpt_path
        os.chdir(dir_path)
        dirs = meta_dir.count("..")
        for i in range(dirs):
            os.chdir("..")
        # concatenate the remaining part
        for i in range(dirs):
            # remove ../
            meta_dir = meta_dir.replace('../', '')
        os.chdir(meta_dir)
        meta_abs_path = os.getcwd()
    print("meta_abs_path ", meta_abs_path)


# This function returns the path of the
# ckpt corresponding to rank_id
def find_ckpt_file(rank_id):
    pattern_ckpt_file = ""
    pattern_ckpt_path = execution_id+level_dir
    if level_dir == '/l1/' or level_dir == '/l4/':  # local
        pattern_ckpt_file = "*-Rank"+str(rank_id)+".fti"
    if level_dir == '/l4/' and ioMode == "2":  # global
        pattern_ckpt_file = "-mpiio.fti"#Ckpt1-mpiio.fti

    ckpt_file = ""
    for root, dirs, files in os.walk(os.path.abspath(ckpt_abs_path)):
        for file in files:
            file = os.path.join(root, file)
            if pattern_ckpt_path in file and pattern_ckpt_file in file:
                ckpt_file = file
    if level_dir == '/l4/' and ioMode == "2":  # global
        PFSfile = ckpt_file
        # recover from L4 to tmp/
        ckpt_file = recover_mpiio_l4(rank_id, PFSfile)

    if ckpt_file == "":
        print("Checkpoint file not found")
        sys.exit(2002)
    return ckpt_file


# This function is called if io=2 and level=4
# it recovers the file from l4 directory in mpiio format
# to tmp/file in posix format
def recover_mpiio_l4(rank_id, PFSfile):
    # preparing input for mpiio recovery
    global nodeSize
    global nbApprocs
    global nbNodes
    global nbHeads

    nodeSize = int(nodeSize)
    nbHeads = int(nbHeads)
    nbApprocs = nodeSize - nbHeads
    nbNodes = totalRanks / nodeSize if nodeSize else 0
    nbNodes = int(nbNodes)

    executable_path = "./mpiio/"
    # get fileSize from metafile
    # read ckpt_file_size entry of second section
    fileSize = 0
    meta_pattern = "sector"
    meta_file = ""
    for root, dirs, files in os.walk(os.path.abspath(meta_abs_path)):
        for file in files:
            if file.startswith(meta_pattern) is True:
                file = os.path.join(root, file)
                print(file)
                meta_file = file
                break

    # processing the meta file for the size
    config = configparser.ConfigParser()
    config.read(meta_file)
    fileSize = config['0']['ckpt_file_size']

    os.chdir(executable_path)
    cmd = "./mpiio_main "+str(rank_id)+" "+str(PFSfile)+" "+str(fileSize)+" "+str(nbApprocs)+" "+str(nbNodes)
    subprocess.check_call(cmd, shell=True)
    print("Rank ", str(rank_id), " is done copying...")
    print(
    "MPI-IO recovery finished successfully. "
    "Now current dir is",
    os.getcwd())

    # look for what has been stored under /tmp
    ckpt_path = os.getcwd()+"/tmp"  # Ckpt1-mpiio.fti
    pattern_ckpt_file = "*.fti"
    ckpt_file = ""
    # find file in this directory
    for root, dirs, files in os.walk(os.path.abspath(ckpt_path)):
        for file in files:
            file = os.path.join(root, file)
            if fnmatch(file, pattern_ckpt_file):
                ckpt_file = file
    if ckpt_path == "":
        print("Could not recover from MPI-IO")
        sys.exit()
    return ckpt_file


# This function returns the path of the
# meta corresponding to the ckpt_file
# note: for now it works with level 1
def find_meta_file(ckpt_file):
    meta_file = ""
    if level_dir == '/l4/' and ioMode == "2":
        print("should take any sector file")
        for path, subdirs, files in os.walk(meta_abs_path):
            for file in files:
                file = meta_abs_path+'/'+execution_id+level_dir+file
                
                meta_file = file
                break

    # traverse all meta files in the directory
    else:  # levels (1,2,3)
        for path, subdirs, files in os.walk(meta_abs_path):
            for file in files:
                file = meta_abs_path+'/'+execution_id+level_dir+file
                
                if os.path.isfile(file) is True:
                    config = configparser.ConfigParser()
                    config.read(file)

                    ckpt = ckpt_file.rsplit('/', 1)[1]
                    for section in config.sections():
                        if section.isdigit() is True:
                            if config[section]['ckpt_file_name'] == ckpt:
                                meta_file = file
                                break
    if meta_file == "":
        print("Metadata file not found")
        sys.exit(2004)

    return meta_file


# This function sets FTI's files paths
# depending on the level where the ckpt is stored
def process_level(level):
    global level_dir
    level_dir = '/l'+str(level)+'/'
    # print("level dir : ", level_dir)


# This function compares ckpt directories
# and returns the level to which the last ckpt was stored
def get_latest_ckpt():
    latest = max(glob.glob(
        os.path.join(ckpt_abs_path, '*/')), key=os.path.getmtime)
    latest = latest.rsplit('/', 1)[0]
    latest = latest.rsplit('/', 1)[1]
    level = latest[1]
    return level


# API to read the checkpoints given config and rank
# def read_checkpoints(config_file, rank_id, level=None, output=None):
def read_checkpoints(config_file, rank_id, ranks=None,
                     level=None, output=None):
    init_config_params(config_file)
    if level in fti_levels:
        process_level(level)
    elif level is None:
        # check for latest ckpt
        last_level = get_latest_ckpt()
        process_level(level)
    else:
        # invalid fti level
        print("Invalid FTI level")
        sys.exit(1001)
    if output is not None and output not in output_formats:
        print("Wrong output format. Choose one")
        print("CSV (default)::  Comma Separated Values file")
        print("HDF5         ::  Hierarchical Data Format file")
        print("data         ::  numpy array")
        sys.exit(1002)
    elif output is None:
        # default output format (CSV)
        output = 'CSV'

    if level == 4 and ioMode == 2 and ranks is None:
        print("Total # of ranks is required when reading MPI-IO"
        " chekpoints from level 4")
        sys.exit(1003)

    global totalRanks
    totalRanks = ranks

    process_fti_paths(config_file)

    ckpt_file = find_ckpt_file(rank_id)
    meta_file = find_meta_file(ckpt_file)
    print("Processing ", ckpt_file, " using meta ", meta_file)
    # posix_read_ckpts.read_checkpoint(
    #     ckpt_file, meta_file, config_file, group_size, level, output)
    if output == "data":
        return posix_read_ckpts.read_checkpoint(
        ckpt_file, meta_file, config_file, group_size, level, output)
    else:
        posix_read_ckpts.read_checkpoint(
        ckpt_file, meta_file, config_file, group_size, level, output)

