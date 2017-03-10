#!/bin/bash

test () {
    cp ../configs/$2 ./config.fti
    sudo mpirun -n 16 ./addInArray config.fti $1 1
    if [ $? != 0 ]
    then
        exit 1
    fi
    sudo mpirun -n 16 ./addInArray config.fti $1 0
    if [ $? != 0 ]
    then
        exit 1
    fi
}



cd addInArray
echo "	Making..."
make
for i in ${@:2}
do
	echo "	Testing L"$i"..."
	test $i $1
done
printf "	addInArray tests succeed.\n\n"
cd ..
exit 0
