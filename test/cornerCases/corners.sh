#!/bin/bash
printRun () {
	printf "_______________________________________________________________________________________\n\n"
	echo "		Running case $1 test... ($2) L$3"
	printf "_______________________________________________________________________________________\n\n"
}
printFailure () {
	printf "_______________________________________________________________________________________\n\n"
	echo "		$1 test case FAILED. ($2) L$3"
	printf "_______________________________________________________________________________________\n\n"
}
printSuccess () {
	printf "_______________________________________________________________________________________\n\n"
	echo "		$1 test case succeed. ($2) L$3"
	printf "_______________________________________________________________________________________\n\n"
}


configs=(configH0I1Silent.fti configH1I1Silent.fti configH1I0Silent.fti)

#test case 1
#<<case1
for config in ${configs[@]}; do
	for level in 1 2 3 4; do
		cp ../configs/$config config.fti
		printRun 1 $config $level
		for run in 0 1 2; do
			echo mpirun -n 16 ./consistency 1 $level $run
			mpirun -n 16 ./consistency 1 $level $run &>> logFile
			if [ $? != 0 ]; then
				printFailure 1 $config $level
				echo LOG:
				cat logFile
				exit 1
			fi
			if ! ([ $run = 2 ] && ([ $level = 1 ] || [ $level = 4 ])); then
				if grep -q "Error" logFile; then
					printFailure 1 $config $level
					echo Error was found in the log file!
					echo LOG:
					cat logFile
					exit 1
				fi
			fi
			if [ $run = 0 ] && ([ $level = 2 ] || [ $level = 3 ]); then 
				../corrupt config.fti $level 16 1 1 3 >> logFile  #args: config ckptLevel numberOfProc ckptORPtner corrORErase corruptLevel
			fi
			if [ $run = 1 ]; then
				../corrupt config.fti $level 16 0 1 3 >> logFile
			fi
		done
		rm logFile
		printSuccess 1 $config $level
	done
done
#case1

#<<case2
for config in ${configs[@]}; do
	for level in 1 2 3 4; do
		cp ../configs/$config config.fti
		printRun 2 $config $level
		for run in 0 1 2; do
			echo mpirun -n 16 ./consistency 2 $level $run
			mpirun -n 16 ./consistency 2 $level $run &>> logFile
			if [ $? != 0 ]; then
				printFailure 2 $config $level
				echo LOG:
				cat logFile
				exit 1
			fi
			if grep -q "Error" logFile; then
				printFailure 2 $config $level
				echo Error was found in the log file!
				echo LOG:
				cat logFile
				exit 1
			fi
		done
		rm logFile
		printSuccess 2 $config $level
	done
done
#case2

#<<case3
for level in 1 2 3 4; do
	cp ../configs/configH0I1Silent.fti config.fti
	cp ../configs/configH0I1Silent.fti config2.fti
	printRun 3 configH0I1Silent.fti $level
	echo mpirun -n 16 ./consistency 3 $level
	mpirun -n 16 ./consistency 3 $level 0 &>> logFile
	if [ $? != 0 ]; then
		printFailure 3 configH0I1Silent.fti $level
		echo LOG:
		cat logFile
		exit 1
	fi
	if grep -q "Error" logFile; then
		printFailure 3 configH0I1Silent.fti $level
		echo Error was found in the log file!
		echo LOG:
		cat logFile
		exit 1
	fi
	if grep -q "Warning" logFile; then
		printFailure 3 configH0I1Silent.fti $level
		echo Warning was found in the log file!
		echo LOG:
		cat logFile
		exit 1
	fi
	rm logFile
	printSuccess 3 configH0I1Silent.fti $level
done
#case3

#<<case4
for level in 1 2 3 4; do
	cp ../configs/configH0I1Silent.fti config.fti
	printRun 4 configH0I1Silent.fti $level
	echo mpirun -n 16 ./consistency 4 $level
	mpirun -n 16 ./consistency 4 $level 0 &>> logFile
	if [ $? != 0 ]; then
		printFailure 4 configH0I1Silent.fti $level
		echo LOG:
		cat logFile
		exit 1
	fi
	if grep -q "Error" logFile; then
		printFailure 4 configH0I1Silent.fti $level
		echo Error was found in the log file!
		echo LOG:
		cat logFile
		exit 1
	fi
	if grep -q "Warning" logFile; then
		printFailure 4 configH0I1Silent.fti $level
		echo Warning was found in the log file!
		echo LOG:
		cat logFile
		exit 1
	fi
	rm logFile
	printSuccess 4 configH0I1Silent.fti $level
done
#case4

