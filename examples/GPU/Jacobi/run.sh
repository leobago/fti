#!/bin/bash

demoCase=$1
enable_fti=0
configFile="NONE"
lvl=1
name="NOTSET"
inMemory=0

if [ "$demoCase" == "ORIGINAL" ]; then
    name="ORIGINAL"
    export FTI_CKPT_L4=1
    export IN_MEMORY=0
    export ENABLE_FTI=0
elif [ "$demoCase" == "NON_OPT" ]; then
    name="NON_OPT"
    configFile="H0POSIX.fti"
    export FTI_CKPT_L4=4
    export IN_MEMORY=0
    export ENABLE_FTI=1
elif [ "$demoCase" == "DCP" ]; then
    name="DCP"
    configFile="H0FTIFFDCP.fti"
    export FTI_CKPT_L4=8
    export IN_MEMORY=0
    export ENABLE_FTI=1
elif [ "$demoCase" == "DCP_OPT" ]; then
    name="DCP_OPT"
    configFile="H0POSIXDCP.fti"
    export FTI_CKPT_L4=8
    export IN_MEMORY=0
    export ENABLE_FTI=1
elif [ "$demoCase" == "DCP_INMEMORY" ]; then
    name="DCP_INMEMORY"
    configFile="H0POSIXDCP.fti"
    export FTI_CKPT_L4=8
    export IN_MEMORY=1
    export ENABLE_FTI=1
else
    echo "Options are : ORIGINAL, NON_OPT, DCP, DCP_OPT, DCP_INMEMORY"
    exit
fi


export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/kparasyris/usr/lib/

mpirun -n 4 ./Jacobi.exe -fticonfig $configFile -fs -t 2 2 -d 8192 8192 | tee tmpFile
execTime=$(cat tmpFile | grep "Total Jacobi run time:"  | cut -d ':' -f 2)
gflops=$(cat tmpFile | grep "Measured FLOPS:"  | cut -d ':' -f 2 | cut -d ',' -f 1 | cut -d '(' -f 1)
echo "$name:$execTime:$gflops" >> results
rm -f tmpFile

