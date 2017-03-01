#!/bin/bash
function printError {
    case $1 in
        "1") echo "Error: Result not correct!" ;;
        "2") echo "Error: Checkpoint failed!" ;;
        "3") echo "Error: Recovery failed!"
    esac
}

function test {
    cp configBkp.fti config.fti
    FLAG=$(sudo mpirun -n 8 ./addInArray config.fti $1 1)
    if [ $FLAG != 0 ]
    then
        printError $FLAG
        exit 1
    fi
    FLAG=$(sudo mpirun -n 8 ./addInArray config.fti $1 0)
    if [ $FLAG != 0 ]
    then
        printError $FLAG
        exit 1
    fi
}


echo "Including FTI..."
C_INCLUDE_PATH=../../include/
export C_INCLUDE_PATH
echo "Making..."
make
#echo "Testing L1..."
#test 1
#echo "Testing L2..."
#test 2
#echo "Testing L3..."
#test 3
echo "Testing recovery from checkpoint taken before finalize (L4)..."
test 4

exit 0
