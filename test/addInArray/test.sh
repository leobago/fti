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
echo "***mpicc show" > output.txt
	mpicc -show -o gtest mpi-test.c >> output.txt
	echo "***mpiicc show" >> output.txt
	mpiicc -show -o itest mpi-test.c >> output.txt
	mpicc -o gtest mpi-test.c
	mpiicc -o itest mpi-test.c
	echo "***ldd gtest" >> output.txt
	ldd gtest >> output.txt
	echo "***ldd itest" >> output.txt
	ldd itest >> output.txt
	echo "***runtest gtest" >> output.txt
	mpirun -n 8 -genv I_MPI_DEBUG 5 -verbose ./gtest 2>&1 >> output.txt
	echo "***runtest itest" >> output.txt
	mpirun -n 8 -genv I_MPI_DEBUG 5 -verbose ./itest 2>&1 >> output.txt
echo "output.txt"
cat output.txt
echo "	Making..."
make
sudo mpirun -np 16 -host localhost ./testMPI
for i in ${@:3}
do
	echo "	Testing L"$i"..."
	test $1 $2 $i
done
printf "	addInArray tests succeed.\n\n"
cd ..
exit 0
