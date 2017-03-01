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

startTest () {
	case "$1" in
	  "addInArray") addInArray ${@:2} ;;
	  *) echo "Wrong test name." ;;
	esac
}
runAllConfiguration() {
	startTest addInArray configH0I1.fti 1 2 3 4
	startTest addInArray configH1I1.fti 1 2 3 4
	startTest addInArray configH1I0.fti 1 2 3 4
}

cd testing
#Pattern: startTest testName configFile args(ex. checkpoint levels)
#----------------------
#-- Write tests here --

runAllConfiguration

#----------------------
cd ..