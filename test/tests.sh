#!/bin/bash

startTest () { #$1 - test name $2 - config name; $3 - number of processes; $4 - checkpoint level
	printf "_______________________________________________________________________________________\n\n"	
	echo "		Running $1 test... ($2)"
	printf "_______________________________________________________________________________________\n\n"
	cp configs/$2 config.fti
	mpirun -n $3 ./$1 config.fti $4 1
	if [ $? != 0 ]
	then
		exit 1
	fi
	printf "_______________________________________________________________________________________\n\n"	
	echo "		 Resuming $1 test... ($2)"
	printf "_______________________________________________________________________________________\n\n"
	mpirun -n $3 ./$1 config.fti $4 0
	if [ $? != 0 ]
	then
		exit 1
	fi
	printf "_______________________________________________________________________________________\n\n"	
	echo "		$1 test succeed. ($2)"
	printf "_______________________________________________________________________________________\n\n"
	rm -r ./Local ./Global ./Meta
}

runAllConfiguration() {
	for i in {0..2}
	do
		for j in {1..4}
		do
		startTest addInArray ${silentConfigs[$i]} $1 $j
		startTest diffSizes ${silentConfigs[$i]} $1 $j
		startTest tokenRing ${silentConfigs[$i]} $1 $j
		done
		startTest nodeFlag ${configs[$i]} $1

	done
	#slow test at the end
	for i in {0..2}
	do
		startTest heatdis ${silentConfigs[$i]} $1
	done
}

cd test
#--------Pattern: startTest testName configFile procNo args(ex. checkpoint levels)-------
#----------------------------------------------------------------------------------------
#-------------------------------- Write tests here --------------------------------------

	configs=(configH0I1.fti configH1I1.fti configH1I0.fti)
	silentConfigs=(configH0I1Silent.fti configH1I1Silent.fti configH1I0Silent.fti)

	if  [ -z "$TEST" ] || [ -z "$CONFIG" ]
	then
		runAllConfiguration 16
	else 
		startTest "$TEST" "$CONFIG" 16 "$LEVEL"
	fi

#----------------------------------------------------------------------------------------
cd ..