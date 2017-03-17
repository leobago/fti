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
	#startTest addInArray configH0I1Silent.fti $1 1 2 3 4
	#startTest addInArray configH1I1Silent.fti $1 1 2 3 4
	#startTest addInArray configH1I0Silent.fti $1 1 2 3 4
	#startTest nodeFlag configH0I1.fti $1
	#startTest nodeFlag configH1I1.fti $1
	#startTest nodeFlag configH1I0.fti $1
	#startTest tokenRing configH0I1Silent.fti $1 1 2 3 4
	#startTest tokenRing configH1I1Silent.fti $1 1 2 3 4
	#tartTest tokenRing configH1I0Silent.fti $1 1 2 3 4
	startTest diffSizes configH1I0Silent.fti 16 1 2 3 4
	startTest diffSizes configH1I1Silent.fti 16 1 2 3 4
	startTest diffSizes configH0I1Silent.fti 16 1 2 3 4
	#startTest lvlsRecovery configH0I1Silent.fti $1 1 2 3 4
	#startTest lvlsRecovery configH1I1Silent.fti $1 1 2 3 4
	#startTest lvlsRecovery configH1I0Silent.fti $1 1 2 3 4
	#startTest heatdis configH0I1Silent.fti $1
	#startTest heatdis configH1I1Silent.fti $1
	#startTest heatdis configH1I0Silent.fti $1
}

cd test
#Pattern: startTest testName configFile procNo args(ex. checkpoint levels)
#----------------------
#-- Write tests here --

runAllConfiguration 16

#----------------------
cd ..