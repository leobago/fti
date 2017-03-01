#!/bin/bash
printError () {
    case $1 in
        "1") echo "	Error: Result not correct!"
	     exit 1 ;;
        "2") echo "	Error: Checkpoint failed!"
             exit 1 ;;
        "3") echo "	Error: Recovery failed!" 
             exit 1 ;;
	*) echo "	$1"
    esac
}

test () {
    cp ../configs/$2 ./config.fti
    FLAG=$(sudo mpirun -n 8 ./addInArray config.fti $1 1)
    if [ "$FLAG" != 0 ]
    then
        printError "$FLAG"
        exit 1
    fi
    FLAG=$(sudo mpirun -n 8 ./addInArray config.fti $1 0)
    if [ "$FLAG" != 0 ]
    then
        printError "$FLAG"
        exit 1
    fi
}



cd addInArray
echo "	Including FTI..."
C_INCLUDE_PATH=../../include/ #access from subfolder
export C_INCLUDE_PATH
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
