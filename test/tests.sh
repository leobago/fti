#!/bin/bash

check () {
	if [ $1 != 0 ]
	then
		echo "Exit status: $1"
		exit 1
	fi
}

addInArray () {
	echo "Running addInArray test... ($1)"
		bash ./addInArray/test.sh $@
		check $?
}

nodeFlag () {
	echo "Running nodeFlag test... ($1)"
		bash ./nodeFlag/test.sh $@
		check $?
}

heatdis () {
	echo "Running heatdis test... ($1)"
		bash ./heatdis/test.sh $@
		check $?
}

tokenRing () {
	echo "Running tokenRing test... ($1)"
		bash ./tokenRing/test.sh $@
		check $?
}

startTest () {
	case "$1" in
	  "addInArray") addInArray ${@:2} ;;
	  "nodeFlag") nodeFlag $2 ;;
	  "heatdis") heatdis $2 ;;
	  "tokenRing") tokenRing ${@:2} ;;
	  *) echo "Wrong test name." ;;
	esac
}
runAllConfiguration() {
	#startTest addInArray configH0I1.fti 1 2 3 4
	#startTest addInArray configH1I1.fti 1 2 3 4
	#startTest addInArray configH1I0.fti 1 2 3 4
	#startTest nodeFlag configH0I1.fti
	#startTest nodeFlag configH1I1.fti
	#startTest nodeFlag configH1I0.fti
	startTest tokenRing configH0I1.fti 1 2 3 4
	startTest tokenRing configH1I1.fti 1 2 3 4
	startTest tokenRing configH1I0.fti 1 2 3 4
	#startTest heatdis configH0I1Silent.fti
	#startTest heatdis configH1I1Silent.fti
	#startTest heatdis configH1I0Silent.fti
}

cd test
#Pattern: startTest testName configFile args(ex. checkpoint levels)
#----------------------
#-- Write tests here --

runAllConfiguration

#----------------------
cd ..