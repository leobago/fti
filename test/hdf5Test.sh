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
printOffline () {
	printf "_______________________________________________________________________________________\n\n"
	echo "		 Offline HDF5 test... ($1) L$2"
	printf "_______________________________________________________________________________________\n"
}

configs=("configH0I1.h5" "configH1I1.h5" "configH1I0.h5")

for config in ${configs[*]}; do
	for level in 1 2 3 4; do
		printRun $config $level
		cp configs/$config config.fti
		mpirun -n 16 ./hdf5Test config.fti $level 1 &> logFile1
		if [ $? != 0 ]; then
			cat logFile1
			exit 1
		fi
		execid=$(grep "The execution ID is" logFile1 | tail -c 21 | head -c 19)
		if [ $level != 4 ]; then
			file="Local/node0/$execid/l$level/Ckpt2-Rank1.h5"
		else
			file="Global/$execid/l4/Ckpt2-Rank1.h5"
		fi

		h5dump $file | tail -n +2 > h5dump.log
		if [ $? != 0 ]; then
			exit 1
		fi

		diff h5dump.log patterns/h5dumpOrigin.log
		if [ $? != 0 ]; then
			echo "h5dump.log:"
			cat h5dump.log
			echo "h5dumpOrigin.log:"
			cat patterns/h5dumpOrigin.log
			exit 1
		fi

		printOffline $config $level
		cp $file offlineVerify.h5
		./hdf5noFTI &> offlineLog
		if [ $? != 0 ]; then
			cat offlineLog
			exit 1
		fi
		rm offlineLog

		printResume $config $level
		mpirun -n 16 ./hdf5Test config.fti $level 0 &> logFile2
		if [ $? != 0 ]; then
			cat logFile2
			exit 1
		fi

		h5dump Global/$execid/l4/Ckpt3-Rank1.h5 | tail -n +2 > h5dump.log
		if [ $? != 0 ]; then
			exit 1
		fi
		diff -q h5dump.log patterns/h5dumpOrigin.log
		if [ $? != 0 ]; then
			cat logFile1
			exit 1
		fi
		rm h5dump.log
		rm logFile1 logFile2
		rm -r ./Local ./Global ./Meta
		printSuccess $config $level
	done
done