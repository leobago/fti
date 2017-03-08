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

startTest () {
	case "$1" in
	  "addInArray") addInArray ${@:2} ;;
	  "nodeFlag") nodeFlag $2 ;;
	  *) echo "Wrong test name." ;;
	esac
}
runAllConfiguration() {
	startTest addInArray configH0I1Silent.fti 1 2 3 4
	startTest addInArray configH1I1Silent.fti 1 2 3 4
	startTest addInArray configH1I0Silent.fti 1 2 3 4
	startTest nodeFlag configH0I1.fti
	startTest nodeFlag configH1I1.fti
	startTest nodeFlag configH1I0.fti
}

cd test
#Pattern: startTest testName configFile args(ex. checkpoint levels)
#----------------------
#-- Write tests here --

#runAllConfiguration
for i in 1 2 3
do
echo "Loop $i"
	startTest addInArray configH0I1Silent.fti 1 2 3 4
	startTest addInArray configH1I1Silent.fti 1 2 3 4
	startTest addInArray configH1I0Silent.fti 1 2 3 4
done
#----------------------
cd ..