#!/bin/bash

test () {
    cp ../configs/$1 ./config.fti
    sudo mpirun -n $2 ./nodeFlag config.fti
    if [ $? != 0 ]
    then
        exit 1
    fi
}

cd nodeFlag
echo "	Making..."
make
echo "	Testing..."
test $1 $2
printf "	nodeFlag tests succeed.\n\n"
cd ..
exit 0
