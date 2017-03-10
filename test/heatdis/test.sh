#!/bin/bash

test () {
    cp ../configs/$1 ./config.fti
    sudo mpirun -n $2 ./heatdis config.fti 1
    if [ $? != 0 ]
    then
        exit 1
    fi
    echo "Resuming..."
    sudo mpirun -n $2 ./heatdis config.fti 0
    if [ $? != 0 ]
    then
        exit 1
    fi
}

cd heatdis
echo "	Making..."
make
echo "	Testing..."
test $1 $2
printf "	heatdis tests succeed.\n\n"
cd ..
exit 0