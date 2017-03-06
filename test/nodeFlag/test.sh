#!/bin/bash
printError () {
    case $1 in
        "1") echo "	Error: nodeFlag not unique in node!"
	     exit 1;;
	*) echo "	$1"
	     #exit 1
    esac
}

test () {
    cp ../configs/$1 ./config.fti
    FLAG=$(sudo mpirun -n 8 ./nodeFlag config.fti)
    if [ "$FLAG" != 0 ]
    then
        printError "$FLAG"
    fi
}



cd nodeFlag
echo "	Making..."
make
echo "	Testing..."
test $1
printf "	nodeFlag tests succeed.\n\n"
cd ..
exit 0
