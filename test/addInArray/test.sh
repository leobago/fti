#!/bin/bash

test () {
    cp ../configs/$1 ./config.fti
    sudo mpirun -n $2 ./addInArray config.fti $3 1
    if [ $? != 0 ]
    then
        exit 1
    fi
    sudo mpirun -n $2 ./addInArray config.fti $3 0
    if [ $? != 0 ]
    then
        exit 1
    fi
}



cd addInArray
echo "	Making..."
make
for i in ${@:3}
do
	echo "	Testing L"$i"..."
	test $1 $2 $i
done
printf "	addInArray tests succeed.\n\n"
cd ..
exit 0
