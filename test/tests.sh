#!/bin/bash

checkLog () { # $1 - log file $2 - pattern file $3 - expecting error
	echo "Checking logs... (Pattern file: $2)"
	while read pattern
	do
	  	if ! grep -q "$pattern" "$1"; then
			echo "\"$pattern\" was not found in the log file ($1)!"
			echo "LOG:"
			cat logFile1 logFile2
			echo "END OF LOG"
			exit 1
		else
			echo "	$pattern"
		fi
	done <"$2"
	if grep -q "Error" "$1" && [ "$3" = 0 ]; then
		echo "Error was found."
		grep "Error" "$1"
		exit 1
	fi
	echo "Logs correct."
}

# $1 - test name, $2 - log file name
checkFinalize () {
	if [ "$1" = "addInArray" ]; then
		echo "Checking if finalize in logs..."
		if ! grep -q "FTI has been finalized." $2; then
			echo "\"FTI has been finalized.\" was not found in the log file ($2)!"
			echo "LOG:"
			cat logFile1 logFile2
			echo "END OF LOG"
			exit 1
		fi
		echo "Finalize correct."
	fi
}

printRun () {
	printf "_______________________________________________________________________________________\n\n"	
	echo "		Running $1 test... ($2) L$3"
	printf "_______________________________________________________________________________________\n\n"
}
printResume () {
	printf "_______________________________________________________________________________________\n\n"	
	echo "		 Resuming $1 test... ($2) L$3"
	printf "_______________________________________________________________________________________\n\n"
}
printSuccess () {
	printf "_______________________________________________________________________________________\n\n"	
	echo "		$1 test succeed. ($2) L$3"
	printf "_______________________________________________________________________________________\n\n"
}

printCorrupt () {
	if [ $2 = 0 ]; then
		printf "Corrupting "
	else
		printf "Erasing "
	fi
	printf "L$4 "
	if [ $3 = 1 ]; then 
		printf "non adjacent nodes\n"
	elif [ $3 = 2 ]; then
		printf "adjacent nodes\n"
	else 
		if [ $3 = 0 ]; then
			printf "one "
		else
			printf "all "
		fi
		if [ $1 = 0 ]; then
			printf "checkpoint file(s)\n"
		else
			printf "partner file(s)\n"
		fi
	fi
}

#$1 - test name $2 - config name; $3 - number of processes;
#$4 - checkpoint level; #$5 - 0=ckpt 1=ptner; $6 - 0=corrupt 1=erase; $7 - 0=onefile 1=nonadjNodes 2=adjNodes 3=all;
startTestCorr () {
	printRun $1 $2 $4
	cp configs/$2 config.fti
	mpirun -n $3 ./$1 config.fti $4 1 &> logFile1
	rtn=$?
	if [ $rtn != 0 ]; then
		echo "Failure. Program returned $rtn code."
		cat logFile1
		exit $rtn
	fi
	checkLog logFile1 patterns/L"$4"INIT 0
	if [ $4 != "4" ] || [ $6 != "0" ]; then #corruption only for local checkpoint
		printCorrupt $5 $6 $7 $4
		./corrupt config.fti $4 $3 $5 $6 $7 #args: config ckptLevel numberOfProc ckptORPtner corrORErase corruptLevel
		rtn=$?
		if [ $rtn != 0 ]; then
			echo "Corrupt failed, returned $rtn code."
			exit $rtn
		fi
	fi
	printResume $1 $2 $4
	mpirun -n $3 ./$1 config.fti $4 0 &> logFile2
	if [ $4 = "1" ] || [ $4 = "4" ]; then 	#if L1 or L4 test should fail
		if [ $4 != "4" ] || [ $6 != "0" ]; then #corruption only for local checkpoint
			checkLog logFile2 patterns/L"$4$6" 1
		fi
	elif [ $4 = "2" ] && [ $7 = "2" ]; then		#if L2 and corruptLevel=2 test should fail
		checkLog logFile2 patterns/L2"$6"2 1 
	else						#else tests should succeed
		checkLog logFile2 patterns/L"$4$6" 0
	fi	
	printSuccess $1 $2 $4
	rm logFile1 logFile2
	rm -r ./Local ./Global ./Meta
}

#$1 - test name $2 - config name; $3 - number of processes; $4 - checkpoint level; $5 - 0=not flushed 1=flushed (L4)
startTestLogVerify () {
	if [ $5 = "1" ]; then
		level="4"
	fi
	printRun $1 $2 $4
	cp configs/$2 config.fti
	mpirun -n $3 ./$1 config.fti $4 1 &> logFile1
	if [ $? != 0 ]; then
		exit 1
	fi
	checkLog logFile1 patterns/L"$4INIT" 0
	checkFinalize $1 logFile1
	printResume $1 $2 $4
	mpirun -n $3 ./$1 config.fti $4 0 &> logFile2
	if [ $? != 0 ]; then
		exit 1
	fi
	checkLog logFile2 patterns/L"$4""Clean""$level" 0
	checkFinalize $1 logFile2
	printSuccess $1 $2 $4
	rm logFile1 logFile2
	rm -r ./Local ./Global ./Meta
}

startTest () { #$1 - test name $2 - config name; $3 - number of processes; $4 - checkpoint level
	printRun $1 $2 $4
	cp configs/$2 config.fti
	mpirun -n $3 ./$1 config.fti $4 1 &> logFile1
	rtn=$?
	if [ $rtn != 0 ]; then
		cat logFile1
		exit $rtn
	fi
	printResume $1 $2 $4
	mpirun -n $3 ./$1 config.fti $4 0 &> logFile2
	rtn=$?
	if [ $rtn != 0 ]; then
		cat logFile2
		exit $rtn
	fi
	printSuccess $1 $2 $4
	rm logFile1 logFile2
	rm -r ./Local ./Global ./Meta
}

runAllConfiguration() {
	for corrORErase in 0 1; do #for corrupt or erase
		if [ $LEVEL = 1 ] || [ $LEVEL = 4 ]; then #L1 or L4 don't have Ptners files 
			startTestCorr diffSizes $CONFIG $1 $LEVEL 0 $corrORErase 0 0
		else
			for ckptORPtner in 0 1; do #for checkpoint or partner file
				for corruptionLevel in {0..3}; do #for all corruption levels (one file, non adj nodes, adj nodes, all files)
					startTestCorr diffSizes $CONFIG $1 $LEVEL $ckptORPtner $corrORErase $corruptionLevel 0
				done
			done
		fi

	done
	startTestLogVerify diffSizes $CONFIG $1 $LEVEL 0 #recover from not flushed checkpoints without corrupting
	startTestLogVerify addInArray $CONFIG $1 $LEVEL 1 #recover from flushed checkpoints without corrupting (L4)

	#run only once for all levels
	if [ $LEVEL = 1 ]; then	
		startTest nodeFlag $CONFIG $1
		#slow test at the end
		startTest heatdis $CONFIG $1
	fi
}

cd test
#--------Pattern: startTest testName configFile procNo args(ex. checkpoint levels)-------
#----------------------------------------------------------------------------------------
#-------------------------------- Write tests here --------------------------------------

	./heartbeat.sh &
	pid=$!
	if  [ ! -z "$TEST" ]; then
		if [ "$TEST" = "diffSizes" ]; then
			if [ -z "$NOTCORRUPT" ]; then
				startTestCorr diffSizes "$CONFIG" 16 "$LEVEL" "$CKPTORPTNER" "$CORRORERASE" "$CORRUPTIONLEVEL"
			else
				startTestLogVerify diffSizes "$CONFIG" 16 "$LEVEL"
			fi
		else
			startTest "$TEST" "$CONFIG" 16 "$LEVEL"
		fi
	else
		runAllConfiguration 16
	fi
	kill -9 $pid

#----------------------------------------------------------------------------------------
cd ..
