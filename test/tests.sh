#!/bin/bash

check () {
	if [ $1 != 0 ]
	then
		echo "Exit status: $1"
		exit 1
	fi
}

startTest () {
	echo "Running $1 test... ($2)"
		bash ./$1/test.sh ${@:2}
		check $?
}
runAllConfiguration() {
	for i in {0..2}
	do
		startTest addInArray ${silentConfigs[$i]} $1 1 2 3 4
		startTest nodeFlag ${configs[$i]} $1
		startTest tokenRing ${silentConfigs[$i]} $1 1 2 3 4
		startTest diffSizes ${silentConfigs[$i]} $1 1 2 3 4
		startTest lvlsRecovery ${silentConfigs[$i]} $1 1 2 3 4
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

	#runs all configuration with 16 processes
	runAllConfiguration 16

#----------------------------------------------------------------------------------------
cd ..