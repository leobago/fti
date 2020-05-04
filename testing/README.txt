Environmental variables to set before running tests.sh script:
-minimum:
    CONFIG - defines which config from configs folder to use.
    LEVEL - defines on which level to make checkpoints.
-optional:
    CKPT_IO - defines checkpoint IO mode (default 1 (POSIX), other values: 2 (MPI), 3 (FTI_FF), 4 (SIONLIB), 5 (HDF5))
    TEST - defines the test to run, by default all basic tests are run.
    NOTCORRUPT - set to make clean run (without any corruption or missing files).
-options below are mandatory if TEST is set and NOTCORRUPT is not set:
    CKPTORPTNER - defines target of corruption 0 - ckpt files 1 - partner or L3 encoded files.
    CORRORERASE - defines type of error, 0 - corrupted file, 1 - erased file.
    CORRUPTIONLEVEL - defines level of corruption, 0 - one file, 1 - two non adjacent nodes, 2- two adjacent nodes, 3 - all files.

Example 1:
#export necessary variables
export CONFIG=configH1I1.fti LEVEL=4 CKPT_IO=2 NOTCORRUPT=true
#run test
./testing/tests.sh

This example will check if FTI uses SIONlib to perform checkpoint and is able to
restart from it.

Example 2:
#make sure variable NOTCORRUPT is not set
unset NOTCORRUPT
#export necessary variables
export CONFIG=configH0I1.fti LEVEL=2 CKPT_IO=5 CKPTORPTNER=1 CORRORERASE=1 CORRUPTIONLEVEL=2
#run test
./testing/tests.sh

This example will run a FTI test program without heads, checkpointing on level 2,
using HDF5-IO, which will check if FTI handles properly missing ckpt file and corresponding partner file.
