#!/bin/sh
# This script contains three examples of 
# using the checkpoint_processor package.

# In all scenarios, a config file is set up
# to run FTI with the HeatDistribution app,
# The configuration file is kept by activating 
# "keep_last_ckpt" parameter. Later, a python 
# script that uses the package is executed 
# with the corresponding parameters to 
# imitate the behavior in the following cases:
# 1) POSIX, Level 1, return data (numpy array).
# 2) POSIX, Level 1, return CSV file.
# 3) MPI-IO, Level 4, return CSV file.


APP_PATH="../../build/tutorial/L1/"
ROOT_DIR=$(pwd)

########Test 1 :: Return data ##################

#prepare config file 
printf '%s\n' '[basic]' 'head                           = 0' \
    'node_size                      = 2' \
    'ckpt_dir                       = ./local' \
    "glbl_dir                       = ./global" \
    "meta_dir                       = ./meta" \
    "ckpt_l1                       = 1" \
    "ckpt_l2                       = 0" \
    "ckpt_l3                       = 0" \
    'ckpt_l4                        = 0'\
	'inline_l2                      = 1'\
	'inline_l3                      = 1'\
	'inline_l4                      = 1'\
	'keep_last_ckpt                 = 1'\
	'group_size                     = 4'\
	'verbosity                      = 2'\
	'max_sync_intv                  = 512'\
	'ckpt_io                        = 1'\
	'transfer_size                  = 16'\
	'[restart]'\
	'failure                        = 1'\
	'exec_id                        = 2020-09-30_17-15-31'\
	'[injection]'\
	'rank                           = 0'\
	'number                         = 0'\
	'position                       = 0'\
	'frequency                      = 0'\
	'[advanced]'\
	'block_size                     = 1024'\
	'mpi_tag                        = 2612'\
	'local_test                     = 1'\
    'transfer_size                  = 16' >$APP_PATH/config.fti

#run FTI app
cd $APP_PATH
mpirun --oversubscribe -n 8 ./hdl1.exe 16 config.fti

#run Python script
cd $ROOT_DIR
python3.6 process_checkpoint_ret_data.py $APP_PATH/config.fti

########Test 2 :: Write to CSV - POSIX ##########

#prepare config file 
printf '%s\n' '[basic]' 'head                           = 0' \
    'node_size                      = 2' \
    'ckpt_dir                       = ./local' \
    "glbl_dir                       = ./global" \
    "meta_dir                       = ./meta" \
    "ckpt_l1                       = 1" \
    "ckpt_l2                       = 0" \
    "ckpt_l3                       = 0" \
    'ckpt_l4                        = 0'\
	'inline_l2                      = 1'\
	'inline_l3                      = 1'\
	'inline_l4                      = 1'\
	'keep_last_ckpt                 = 1'\
	'group_size                     = 4'\
	'verbosity                      = 2'\
	'max_sync_intv                  = 512'\
	'ckpt_io                        = 1'\
	'transfer_size                  = 16'\
	'[restart]'\
	'failure                        = 1'\
	'exec_id                        = 2020-09-30_17-15-31'\
	'[injection]'\
	'rank                           = 0'\
	'number                         = 0'\
	'position                       = 0'\
	'frequency                      = 0'\
	'[advanced]'\
	'block_size                     = 1024'\
	'mpi_tag                        = 2612'\
	'local_test                     = 1'\
    'transfer_size                  = 16' >$APP_PATH/config.fti

#run FTI app
cd $APP_PATH
mpirun --oversubscribe -n 8 ./hdl1.exe 16 config.fti

#run Python script
cd $ROOT_DIR
python3.6 process_checkpoint_ret_csv.py $APP_PATH/config.fti


########Test 3::  Write to CSV - MPI-IO #########

#prepare config file 
printf '%s\n' '[basic]' 'head                           = 0' \
    'node_size                      = 2' \
    'ckpt_dir                       = ./local' \
    "glbl_dir                       = ./global" \
    "meta_dir                       = ./meta" \
    "ckpt_l1                       = 0" \
    "ckpt_l2                       = 0" \
    "ckpt_l3                       = 0" \
    'ckpt_l4                        = 1'\
	'inline_l2                      = 1'\
	'inline_l3                      = 1'\
	'inline_l4                      = 1'\
	'keep_last_ckpt                 = 1'\
	'group_size                     = 4'\
	'verbosity                      = 2'\
	'max_sync_intv                  = 512'\
	'ckpt_io                        = 2'\
	'transfer_size                  = 16'\
	'[restart]'\
	'failure                        = 1'\
	'exec_id                        = 2020-09-30_17-15-31'\
	'[injection]'\
	'rank                           = 0'\
	'number                         = 0'\
	'position                       = 0'\
	'frequency                      = 0'\
	'[advanced]'\
	'block_size                     = 1024'\
	'mpi_tag                        = 2612'\
	'local_test                     = 1'\
    'transfer_size                  = 16' >$APP_PATH/config.fti

#run FTI app
cd $APP_PATH
mpirun --oversubscribe -n 8 ./hdl1.exe 16 config.fti

#run Python script
cd $ROOT_DIR
python3.6 process_checkpoint_mpiio.py $APP_PATH/config.fti
