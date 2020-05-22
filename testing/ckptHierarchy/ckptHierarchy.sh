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

#<<case4.1.1.1
<<desc
If we have performed a successful checkpoint, the older checkpoint from before is kept if the
security (level) was higher then the current
desc
for config in ${configs[@]}; do
	printRun 4.1.1.1 $config
	cp ../configs/${config} config.fti
	mpirun -n 16 ./ckptHierarchy 4 3 2 1 1 0
	exec_id=$(grep "exec_id" ./config.fti | awk '{print $(NF)}')
	for level in 1 2 3; do
		for node in 0 1 2 3; do
			for sector in 0; do
				if [ "$config" == "configH0I1Silent.fti" ]; then
					groups=(0 1 2 3)
				else
					groups=(1 2 3)
				fi
				for group in ${groups[@]}; do
					rank=$(grep -A 1 "\[${node}\]" ./Meta/${exec_id}/l4/sector${sector}-group${group}.fti | grep -o 'Rank[0-9]\+' | tail -n 1 | cut -c 5-)
					lv=$((5-$level))
					ls -l ./Local/node${node}/${exec_id}/l${level}/Ckpt${lv}-Rank${rank}.fti > /dev/null
					if [ $? != 0 ]; then
						printFailure 4.1.1.1 $config
						exit 1;
					fi
				done
			done
		done
	done
	printSuccess 4.1.1.1 $config
done
#case4.1.1.1

#<<case4.1.1.2
<<desc
In the other case, if the security was lower, the
checkpoint will be removed.
desc
for config in ${configs[@]}; do
	printRun 4.1.1.2 $config 
	cp ../configs/${config} config.fti
	mpirun -n 16 ./ckptHierarchy 1 2 3 4 1 0
	exec_id=$(grep "exec_id" ./config.fti | awk '{print $(NF)}')
	for level in 1 2 3; do
		for node in 0 1 2 3; do
			for sector in 0; do
				if [ "$config" == "configH0I1Silent.fti" ]; then
					groups=(0 1 2 3)
				else
					groups=(1 2 3)
				fi
				for group in ${groups[@]}; do
					rank=$(grep -A 1 "\[${node}\]" ./Meta/${exec_id}/l4/sector${sector}-group${group}.fti | grep -o 'Rank[0-9]\+' | tail -n 1 | cut -c 5-)
					ls -l ./Local/node${node}/${exec_id}/l${level}/Ckpt${level}-Rank${rank}.fti &> /dev/null
					if [ $? == 0 ]; then
						ls -l ./Local/node${node}/${exec_id}/l${level}/Ckpt${level}-Rank${rank}.fti
						printFailure 4.1.1.2 $config 
						exit 1;
					fi
				done
			done
		done
	done
	printSuccess 4.1.1.2 $config 
done
#case4.1.1.2

#<<case4.1.2
<<desc
Also, for keep_last_ckpt=1, beside the flushed L4 checkpoint on the parallel file system, the
local checkpoint files (for the case of L1, L2, L3) aren't kept.

We need a unitary test which proofs these features.
desc
for config in ${configs[@]}; do
	for level in 1 2 3; do
		printRun 4.1.2 $config $level
		cp ../configs/${config} config.fti
		mpirun -n 16 ./ckptHierarchy $level $level $level $level 0 0
		exec_id=$(grep "exec_id" ./config.fti | awk '{print $(NF)}')
		for node in 0 1 2 3; do
			for sector in 0; do
				if [ "$config" == "configH0I1Silent.fti" ]; then
					groups=(0 1 2 3)
				else
					groups=(1 2 3)
				fi
				for group in ${groups[@]}; do
					rank=$(grep -A 1 "\[${node}\]" ./Meta/${exec_id}/l4/sector${sector}-group${group}.fti | grep -o 'Rank[0-9]\+' | tail -n 1 | cut -c 5-)
					ls -l ./Local/node${node}/${exec_id}/l${level}/Ckpt${level}-Rank${rank}.fti &> /dev/null
					if [ $? == 0 ]; then
						ls -l ./Local/node${node}/${exec_id}/l${level}/Ckpt${level}-Rank${rank}.fti
						printFailure 4.1.2 $config $level
						exit 1;
					fi
				done
			done
		done
	done
	printSuccess 4.1.2 $config $level
done
#case4.1.2

#<<case4.2
<<desc
This is related to 4.1. On a restart, FTI will restart from the lowest level. Hence if we have both,
L1 and L4, FTI should restart from L1. If it fails, it should restart from L4 (see 4.1.2).
desc
for config in ${configs[@]}; do
	for level in 1 2 3; do
		printRun 4.2 $config $level
		cp ../configs/${config} config.fti
		mpirun -n 16 ./ckptHierarchy 4 3 2 1 1 0 &> logFile
		../corrupt config.fti 1 16 0 1 3 &> logFile
		recoFrom=2
		if [ $level > 1 ]; then
			../corrupt config.fti 2 16 0 1 3 &> logFile #args: config ckptLevel numberOfProc ckptORPtner corrORErase corruptLevel
			../corrupt config.fti 2 16 1 1 3 &> logFile #args: config ckptLevel numberOfProc ckptORPtner corrORErase corruptLevel
			recoFrom=3
		fi
		if [ $level > 2 ]; then
			../corrupt config.fti 3 16 0 1 3 &> logFile #args: config ckptLevel numberOfProc ckptORPtner corrORErase corruptLevel
			../corrupt config.fti 3 16 1 1 3 &> logFile #args: config ckptLevel numberOfProc ckptORPtner corrORErase corruptLevel
			recoFrom=4
		fi
		mpirun -n 16 ./ckptHierarchy 1 1 1 1 0 1 &>> logFile
		if ! grep -q "Recovering successfully from level ${recoFrom}" logFile; then
			echo "\"Recovering successfully from level ${recoFrom}\" was not found in the log file!"
			echo "LOG:"
			cat logFile
			echo "END OF LOG"
			printFailure 4.2 $config $level
			exit 1
		fi
		printSuccess 4.2 $config $level
	done
done

#case4.2


