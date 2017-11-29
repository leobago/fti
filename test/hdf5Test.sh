#!/bin/bash
#
#   @file   hdf5Test.sh
#   @author Karol Sierocinski (ksiero@man.poznan.pl)
#   @date   November, 2017
#   @brief  hdf5 test for Travis
#


printRun () {
	printf "_______________________________________________________________________________________\n\n"
	echo "		Running HDF5 test... ($1) L$2"
	printf "_______________________________________________________________________________________\n"
}
printResume () {
	printf "_______________________________________________________________________________________\n\n"
	echo "		 Resuming HDF5 test... ($1) L$2"
	printf "_______________________________________________________________________________________\n"
}
printSuccess () {
	printf "_______________________________________________________________________________________\n\n"
	echo "		HDF5 test succeed. ($1) L$2"
	printf "_______________________________________________________________________________________\n\n"
}

configs=("configH1I1.h5" "configH0I1.h5" "configH1I0.h5")

for config in ${configs[*]}; do
	for level in 1 2 3 4; do
		printRun $config $level
		cp configs/$config config.fti
		mpirun -n 16 ./hdf5Test config.fti $level 1 #&> logFile1
		if [ $? != 0 ]; then
			cat logFile1
			exit 1
		fi
		printResume $config $level
		mpirun -n 16 ./hdf5Test config.fti $level 0 #&> logFile2
		if [ $? != 0 ]; then
			cat logFile2
			exit 1
		fi
		rm logFile1 logFile2
		rm -r ./Local ./Global ./Meta
		printSuccess $config $level
	done
done