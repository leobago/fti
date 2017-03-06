#!/bin/bash

test () {
    cp ../configs/$2 ./config.fti
    sudo mpirun -n 8 ./addInArray config.fti $1 1
    if [ $? != 0 ]
    then
        exit $?
    fi
    sudo mpirun -n 8 ./addInArray config.fti $1 0
    if [ $? != 0 ]
    then
        exit $?
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
