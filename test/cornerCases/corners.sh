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
<<desc
Consistency of the ckpt-files after successful recovery
After the restart from any level, we claim to have the same state as after the successful checkpoint
in the preceeding execution. We need to proof this in a unitary test. For instance:
        1. init fti
        2. perform checkpoint
        3. simulate crash
        4. delete all partner/encoded files
        5. restart
        6. after successful restart, simulate crash again
        7. delete all checkpoint files
        8. restart and check if restart successful
desc
for config in ${configs[@]}; do
	for level in 1 2 3 4; do
		cp ../configs/$config config.fti
		printRun 1 $config $level
		for run in 0 1 2; do
			echo mpirun $MPIRUN_ARGS -n 16 ./consistency.exe 'config.fti' 1 $level $run
			mpirun $MPIRUN_ARGS -n 16 ./consistency.exe 'config.fti' 1 $level $run &>> logFile
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
<<desc
Assure the consistency of checkpoint files for keep_last_ckpt is true
Same as before. After the successful restart, we claim to have the situation as after the last
checkpoint in the preceeding execution.
        1. init fti
        2. perform checkpoint
        3. finalize
        4. restart
        5. after successful restart, simulate crash
        6. restart and check if restart successful
desc
for config in ${configs[@]}; do
	for level in 1 2 3 4; do
		cp ../configs/$config config.fti
		printRun 2 $config $level
		for run in 0 1 2; do
			echo mpirun $MPIRUN_ARGS -n 16 ./consistency.exe 'config.fti' 2 $level $run
			mpirun $MPIRUN_ARGS -n 16 ./consistency.exe 'config.fti' 2 $level $run &>> logFile
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
<<desc
combine two different problems in one code. Init the first and finalize at the end. then (within
same source and execution) init fti again and perform the execution of the second problem and
finalize at the end. Check for warnings or errors.
desc
for level in 1 2 3 4; do
	cp ../configs/configH0I1Silent.fti config.fti
	cp ../configs/configH0I1Silent.fti config2.fti
	printRun 3 configH0I1Silent.fti $level
	echo mpirun $MPIRUN_ARGS -n 16 ./consistency.exe 'config.fti' 3 $level
	mpirun $MPIRUN_ARGS -n 16 ./consistency.exe 'config.fti' 3 $level 0 'config2.fti' &>> logFile
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
<<desc
check for a correct restart without the crash of the application.
    for all levels:
        1. init fti
        2. perform checkpoint
        3. init fti
        4. finalize
desc
for level in 1 2 3 4; do
	cp ../configs/configH0I1Silent.fti config.fti
	printRun 4 configH0I1Silent.fti $level
	echo mpirun $MPIRUN_ARGS -n 16 ./consistency.exe 'config.fti' 4 $level
	mpirun $MPIRUN_ARGS -n 16 ./consistency.exe 'config.fti' 4 $level 0 &>> logFile
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

