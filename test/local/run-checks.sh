#!/usr/bin/env bash
#test
FLAG=1
PROCS=16
diffSize=0
verbose=0
eraseFiles=0
corruptFiles=0
FAILED=0
SUCCEED=0
FAULTY=0
testFailed=0
# Use -gt 1 to consume two arguments per pass in the loop (e.g. each
# argument has a corresponding value to go with it).
# Use -gt 0 to consume one or more arguments per pass in the loop (e.g.
# some arguments don't have a corresponding value to go with it such
# as in the --default example).
# note: if this is set to -gt 0 the /etc/hosts part is not recognized ( may be a bug )
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -d|--diff-size)
    diffSize=1
    echo "[OPTION] Set different checkpoint sizes -> TRUE"
    ;;
    -v|--verbose)
    verbose=1
    echo "[OPTION] Set verbose mode -> TRUE"
    ;;
    -e|--erase-files)
    eraseFiles=1
    echo "[OPTION] Set erase checkpoint files -> TRUE"
    ;;
    -c|--corrupt-files)
    corruptFiles=1
    echo "[OPTION] Set corrupt checkpoint files -> TRUE"
    ;;
    *)
            # unknown option
    ;;
esac
shift # past argument or value
done
#							 #
# ---- HEADER CHECK LOG ---- # 
#							 #
cat <<EOF > check.log
### FTI TEST FOR $PROCS PROCESSES ###
EOF
#							 #
# ---- HEADER FAULTY LOG ---- # 
#							 #
cat <<EOF > failed.log
### FAILED TESTS FOR $PROCS PROCESSES ###
EOF
#							   		    #
# ---- CONFIGURATION FILE TEMPLATE ---- # 
#								        #
cat <<EOF >TMPLT
[basic]
head                           = 0
node_size                      = 4
ckpt_dir                       = Local
glbl_dir                       = Global
meta_dir                       = Meta
ckpt_io                        = 1
ckpt_l1                        = 0
ckpt_l2                        = 0
ckpt_l3                        = 0
ckpt_l4                        = 0
inline_l2                      = 1
inline_l3                      = 1
inline_l4                      = 1
keep_last_ckpt = 0
group_size                     = 4
verbosity                      = 2


[restart]
failure                        = 0
exec_id                        = 2017-05-18_13-35-26


[injection]
rank                           = 0
number                         = 0
position                       = 0
frequency                      = 0


[advanced]
block_size                     = 1024
mpi_tag                        = 2612
local_test                     = 1
EOF
#								 #
# ---- FUNCTION DEFINITIONS ---- # 
#								 #
should_not_fail() {
	if [ $1 = 0 ]; then
	    echo -e "\033[0;32mpassed\033[m"
		let SUCCEED=SUCCEED+1
	elif [ $1 = 255 ]; then
        echo -e "\033[0;31mfailed\033[m (MPI exception)" 
		let FAILED=FAILED+1
		testFailed=1
	elif [ $1 = 30 ]; then
		echo -e "\033[0;31mfailed\033[m (Checkpopint Data Corrupted!)" 
		let FAILED=FAILED+1
		testFailed=1
	elif [ $1 = 20 ]; then
        echo -e "\033[0;31mfailed\033[m (Recovery Failed)" 
		let FAILED=FAILED+1
		testFailed=1
	elif [ $1 = 143 ]; then
        echo -e "\033[0;31mfailed\033[m (Process Killed, Timeout!)" 
		let FAILED=FAILED+1
		testFailed=1
	elif [ $1 = 40 ]; then
        echo -e "\033[0;31mfailed\033[m (Test Data Corrupted!)" 
		let FAULTY=FAULTY+1
		testFailed=1
	else 
		echo "Unknown exit status: "$1
		let FAULTY=FAULTY+1
		testFailed=1
	fi
}

should_fail() {
	if [ $1 = 255 ]; then
	    echo -e "\033[0;31mfailed\033[m (MPI exception)"
		let FAILED=FAILED+1
		testFailed=1
	elif [ $1 = 0 ]; then
        echo -e "\033[0;31mfailed\033[m (Finalized Without Error!)"
		let FAILED=FAILED+1
		testFailed=1
	elif [ $1 = 30 ]; then
		echo -e "\033[0;31mfailed\033[m (Checkpoint Data Corrupted!)" 
		let FAILED=FAILED+1
		testFailed=1
	elif [ $1 = 143 ]; then
        echo -e "\033[0;31mfailed\033[m (Process Killed, Timeout!)" 
		let FAILED=FAILED+1
		testFailed=1
	elif [ $1 = 20 ]; then
        echo -e "\033[0;32mpassed\033[m (Recovery failed)" 
		let SUCCEED=SUCCEED+1
	elif [ $1 = 40 ]; then
        echo -e "\033[0;31mfailed\033[m (Test Data Corrupted!)" 
		let FAULTY=FAULTY+1
		testFailed=1
	else 
		echo "Unknown exit status: "$1
		let FAULTY=FAULTY+1
		testFailed=1
	fi
}

set_inline() {
    case $1 in
        2)
            let l2=$2
            let l3=1
            let l4=1
            ;;
        3)
            let l2=1
            let l3=$2
            let l4=1
            ;;
        4)
            let l2=1
            let l3=1
            let l4=$2
            ;;
    esac
}
#						 #
# ---- TEST SCRIPTS ---- # 
#	#
LEVEL=(1 2 3 4)
LEVEL_FOLDER=(Local Local Local Global)
INLINE_L2=(0 1)
INLINE_L3=(0 1)
INLINE_L4=(0 1)
KEEP=(0 1)
for keep in ${KEEP[*]}; do
    NAME="H0K"$keep"I111"
    #                  #
    # --- HEAD = 0 --- #
    #                  #
    awk -v var="$keep" '$1 == "keep_last_ckpt" {$3 = var}1' TMPLT > tmp; cp tmp $NAME
    for level in ${LEVEL[*]}; do    
        echo -e "[ \033[1mTesting L"$level", head=0, keep="$keep", inline=(1,1,1)...\033[m ]"
        if [ $keep -eq "1" ]; then
            #                  #
            # --- KEEP = 1 --- #
            #                  #
			( set -x; mpirun -n $PROCS ./check.exe $NAME 0 $level $diffSize &>> check.log )
			### SETTING KEEP = 0 TO CLEAN DIRECTORY AFTER TEST
			awk '$1 == "keep_last_ckpt" {$3 = 0}1' $NAME > tmp; cp tmp $NAME; rm tmp
			( cmdpid=$BASHPID; (sleep 10; kill $cmdpid > /dev/null 2>&1 ) & set -x; mpirun -n $PROCS ./check.exe $NAME 0 $level $diffSize &>> check.log )
			should_not_fail $?
			if [ $testFailed = 1 ]; then
				echo -e "L"$level", head=0, keep="$keep", inline=(1,1,1), should not fail" >> failed.log
				testFailed=0
			fi
            awk '$1 == "failure" {$3 = 0}1' $NAME > tmp; cp tmp $NAME; rm tmp
            if [ $eraseFiles = "1" ] && [ $FLAG = "1" ]; then
                awk '$1 == "keep_last_ckpt" {$3 = 1}1' $NAME > tmp; cp tmp $NAME; rm tmp
                ( set -x; mpirun -n $PROCS ./check.exe $NAME 0 $level $diffSize &>> check.log )
                ### DELETE CHECKPOINT FILE
                echo -e "[ \033[1mDeleting Checkpoint File...\033[m ]"
                folder="Global/"$(awk '$1 == "exec_id" {print $3}' < $NAME)"/l4"
                filename=$(ls $folder | grep Rank | head -n 1)
                ( set -x; rm -rf $folder"/"$filename )
                ### SETTING KEEP = 0 TO CLEAN DIRECTORY AFTER TEST
                awk '$1 == "keep_last_ckpt" {$3 = 0}1' $NAME > tmp; cp tmp $NAME; rm tmp
                ( cmdpid=$BASHPID; (sleep 10; kill $cmdpid > /dev/null 2>&1 ) & set -x; mpirun -n $PROCS ./check.exe $NAME 0 $level $diffSize &>> check.log )
                should_fail $?
                if [ $testFailed = 1 ]; then
                    echo -e "L"$level", head=0, keep="$keep", inline=(1,1,1), should not fail" >> failed.log
                    testFailed=0
                fi
                awk '$1 == "failure" {$3 = 0}1' $NAME > tmp; cp tmp $NAME; rm tmp
                let FLAG=2
            fi
        fi
        if [ $keep -eq "0" ]; then
            #                  #
            # --- KEEP = 0 --- #
            #                  #
			( set -x; mpirun -n $PROCS ./check.exe $NAME 1 $level $diffSize &>> check.log )
			( cmdpid=$BASHPID; (sleep 10; kill $cmdpid > /dev/null 2>&1 ) & set -x; mpirun -n $PROCS ./check.exe $NAME 0 $level $diffSize &>> check.log )
			should_not_fail $?
			if [ $testFailed = 1 ]; then
				echo -e "L"$level", head=0, keep="$keep", inline=(1,1,1), should not fail" >> failed.log
				testFailed=0
			fi
            awk '$1 == "failure" {$3 = 0}1' $NAME > tmp; cp tmp $NAME; rm tmp
            if [ $eraseFiles = "1" ]; then
                if [ $level = 1 ]; then
                    ( set -x; mpirun -n $PROCS ./check.exe $NAME 1 $level $diffSize &>> check.log )
                    ### DELETE CHECKPOINT FILE
                    echo -e "[ \033[1mDeleting Checkpoint File...\033[m ]"
                    folder="Local/node0/"$(awk '$1 == "exec_id" {print $3}' < $NAME)"/l1"
                    filename=$(ls $folder | grep Rank | head -n 1)
                    ( set -x; rm -rf $folder"/"$filename )
                    ( cmdpid=$BASHPID; (sleep 10; kill $cmdpid > /dev/null 2>&1 ) & set -x; mpirun -n $PROCS ./check.exe $NAME 0 $level $diffSize &>> check.log )
                    should_fail $?
                    if [ $testFailed = 1 ]; then
                        echo -e "L"$level", head=0, keep="$keep", inline=(1,1,1), should fail" >> failed.log
                        testFailed=0
                    fi
                    awk '$1 == "failure" {$3 = 0}1' $NAME > tmp; cp tmp $NAME; rm tmp
                fi
                if [ $level = 2 ]; then
                    ( set -x; mpirun -n $PROCS ./check.exe $NAME 1 $level $diffSize &>> check.log )
                    ### DELETE Checkpoint FILES
                    echo -e "[ \033[1mDeleting Checkpoint Files...\033[m ]"
                    folder="Local/node0/"$(awk '$1 == "exec_id" {print $3}' < $NAME)"/l2"
                    filename=$(ls $folder | grep Rank | head -n 1)
                    ( set -x; rm -rf $folder"/"$filename )
                    folder="Local/node2/"$(awk '$1 == "exec_id" {print $3}' < $NAME)"/l2"
                    filename=$(ls $folder | grep Rank | head -n 1)
                    ( set -x; rm -rf $folder"/"$filename )
                    ( cmdpid=$BASHPID; (sleep 10; kill $cmdpid > /dev/null 2>&1 ) & set -x; mpirun -n $PROCS ./check.exe $NAME 0 $level $diffSize &>> check.log )
                    should_not_fail $?
                    if [ $testFailed = 1 ]; then
                        echo -e "L"$level", head=0, keep="$keep", inline=(1,1,1), should not fail" >> failed.log
                        testFailed=0
                    fi
                    awk '$1 == "failure" {$3 = 0}1' $NAME > tmp; cp tmp $NAME; rm tmp
                    ( set -x; mpirun -n $PROCS ./check.exe $NAME 1 $level $diffSize &>> check.log )
                    ### DELETE PARTNER FILES
                    echo -e "[ \033[1mDeleting Partner Files...\033[m ]"
                    folder="Local/node0/"$(awk '$1 == "exec_id" {print $3}' < $NAME)"/l2"
                    filename=$(ls $folder | grep Pcof | head -n 1)
                    ( set -x; rm -rf $folder"/"$filename )
                    folder="Local/node2/"$(awk '$1 == "exec_id" {print $3}' < $NAME)"/l2"
                    filename=$(ls $folder | grep Pcof | head -n 1)
                    ( set -x; rm -rf $folder"/"$filename )
                    ( cmdpid=$BASHPID; (sleep 10; kill $cmdpid > /dev/null 2>&1 ) & set -x; mpirun -n $PROCS ./check.exe $NAME 0 $level $diffSize &>> check.log )
                    should_not_fail $?
                    if [ $testFailed = 1 ]; then
                        echo -e "L"$level", head=0, keep="$keep", inline=(1,1,1), should not fail" >> failed.log
                        testFailed=0
                    fi
                    awk '$1 == "failure" {$3 = 0}1' $NAME > tmp; cp tmp $NAME; rm tmp
                    ( set -x; mpirun -n $PROCS ./check.exe $NAME 1 $level $diffSize &>> check.log )
                    ### DELETE NODES
                    echo -e "[ \033[1mDeleting Consecutive Nodes...\033[m ]"
                    ( set -x; rm -rf Local/node0 )
                    ( set -x; rm -rf Local/node1 )
                    ( cmdpid=$BASHPID; (sleep 10; kill $cmdpid > /dev/null 2>&1 ) & set -x; mpirun -n $PROCS ./check.exe $NAME 0 $level $diffSize &>> check.log )
                    should_fail $?
                    if [ $testFailed = 1 ]; then
                        echo -e "L"$level", head=0, keep="$keep", inline=(1,1,1), should not fail" >> failed.log
                        testFailed=0
                    fi
                    awk '$1 == "failure" {$3 = 0}1' $NAME > tmp; cp tmp $NAME; rm tmp
                    ( set -x; mpirun -n $PROCS ./check.exe $NAME 1 $level $diffSize &>> check.log )
                    ### DELETE NODES
                    echo -e "[ \033[1mDeleting Non-Consecutive Nodes...\033[m ]"
                    ( set -x; rm -rf Local/node0 )
                    ( set -x; rm -rf Local/node2 )
                    ( cmdpid=$BASHPID; (sleep 10; kill $cmdpid > /dev/null 2>&1 ) & set -x; mpirun -n $PROCS ./check.exe $NAME 0 $level $diffSize &>> check.log )
                    should_not_fail $?
                    if [ $testFailed = 1 ]; then
                        echo -e "L"$level", head=0, keep="$keep", inline=(1,1,1), should not fail" >> failed.log
                        testFailed=0
                    fi
                    awk '$1 == "failure" {$3 = 0}1' $NAME > tmp; cp tmp $NAME; rm tmp
                fi
                if [ $level = 3 ]; then
                    ( set -x; mpirun -n $PROCS ./check.exe $NAME 1 $level $diffSize &>> check.log )
                    ### DELETE Checkpoint FILES
                    echo -e "[ \033[1mDeleting Checkpoint Files...\033[m ]"
                    folder="Local/node0/"$(awk '$1 == "exec_id" {print $3}' < $NAME)"/l3"
                    filename=$(ls $folder | grep Rank | head -n 1)
                    ( set -x; rm -rf $folder"/"$filename )
                    folder="Local/node2/"$(awk '$1 == "exec_id" {print $3}' < $NAME)"/l3"
                    filename=$(ls $folder | grep Rank | head -n 1)
                    ( set -x; rm -rf $folder"/"$filename )
                    ( cmdpid=$BASHPID; (sleep 10; kill $cmdpid > /dev/null 2>&1 ) & set -x; mpirun -n $PROCS ./check.exe $NAME 0 $level $diffSize &>> check.log )
                    should_not_fail $?
                    if [ $testFailed = 1 ]; then
                        echo -e "L"$level", head=0, keep="$keep", inline=(1,1,1), should not fail" >> failed.log
                        testFailed=0
                    fi
                    awk '$1 == "failure" {$3 = 0}1' $NAME > tmp; cp tmp $NAME; rm tmp
                    ( set -x; mpirun -n $PROCS ./check.exe $NAME 1 $level $diffSize &>> check.log )
                    ### DELETE ENCODED FILES
                    echo -e "[ \033[1mDeleting Encoded Files...\033[m ]"
                    folder="Local/node0/"$(awk '$1 == "exec_id" {print $3}' < $NAME)"/l3"
                    filename=$(ls $folder | grep Rsed | head -n 1)
                    ( set -x; rm -rf $folder"/"$filename )
                    folder="Local/node2/"$(awk '$1 == "exec_id" {print $3}' < $NAME)"/l3"
                    filename=$(ls $folder | grep Rsed | head -n 1)
                    ( set -x; rm -rf $folder"/"$filename )
                    ( cmdpid=$BASHPID; (sleep 10; kill $cmdpid > /dev/null 2>&1 ) & set -x; mpirun -n $PROCS ./check.exe $NAME 0 $level $diffSize &>> check.log )
                    should_not_fail $?
                    if [ $testFailed = 1 ]; then
                        echo -e "L"$level", head=0, keep="$keep", inline=(1,1,1), should not fail" >> failed.log
                        testFailed=0
                    fi
                    awk '$1 == "failure" {$3 = 0}1' $NAME > tmp; cp tmp $NAME; rm tmp
                    ( set -x; mpirun -n $PROCS ./check.exe $NAME 1 $level $diffSize &>> check.log )
                    ### DELETE NODES
                    echo -e "[ \033[1mDeleting Consecutive Nodes...\033[m ]"
                    ( set -x; rm -rf Local/node0 )
                    ( set -x; rm -rf Local/node1 )
                    ( cmdpid=$BASHPID; (sleep 10; kill $cmdpid > /dev/null 2>&1 ) & set -x; mpirun -n $PROCS ./check.exe $NAME 0 $level $diffSize &>> check.log )
                    should_not_fail $?
                    if [ $testFailed = 1 ]; then
                        echo -e "L"$level", head=0, keep="$keep", inline=(1,1,1), should not fail" >> failed.log
                        testFailed=0
                    fi
                    awk '$1 == "failure" {$3 = 0}1' $NAME > tmp; cp tmp $NAME; rm tmp
                    ( set -x; mpirun -n $PROCS ./check.exe $NAME 1 $level $diffSize &>> check.log )
                    ### DELETE NODES
                    echo -e "[ \033[1mDeleting Non-Consecutive Nodes...\033[m ]"
                    ( set -x; rm -rf Local/node0 )
                    ( set -x; rm -rf Local/node2 )
                    ( cmdpid=$BASHPID; (sleep 10; kill $cmdpid > /dev/null 2>&1 ) & set -x; mpirun -n $PROCS ./check.exe $NAME 0 $level $diffSize &>> check.log )
                    should_not_fail $?
                    if [ $testFailed = 1 ]; then
                        echo -e "L"$level", head=0, keep="$keep", inline=(1,1,1), should not fail" >> failed.log
                        testFailed=0
                    fi
                    awk '$1 == "failure" {$3 = 0}1' $NAME > tmp; cp tmp $NAME; rm tmp
                fi
                if [ $level = 4 ]; then
                    ( set -x; mpirun -n $PROCS ./check.exe $NAME 1 $level $diffSize &>> check.log )
                    ### DELETE CHECKPOINT FILE
                    echo -e "[ \033[1mDeleting Checkpoint File...\033[m ]"
                    folder="Global/"$(awk '$1 == "exec_id" {print $3}' < $NAME)"/l4"
                    filename=$(ls $folder | grep Rank | head -n 1)
                    ( set -x; rm -rf $folder"/"$filename )
                    ( cmdpid=$BASHPID; (sleep 10; kill $cmdpid > /dev/null 2>&1 ) & set -x; mpirun -n $PROCS ./check.exe $NAME 0 $level $diffSize &>> check.log )
                    should_fail $?
                    if [ $testFailed = 1 ]; then
                        echo -e "L"$level", head=0, keep="$keep", inline=(1,1,1), should not fail" >> failed.log
                        testFailed=0
                    fi
                    awk '$1 == "failure" {$3 = 0}1' $NAME > tmp; cp tmp $NAME; rm tmp
                fi
            fi
        fi
    done
	rm $NAME
done
for keep in ${KEEP[*]}; do
    for level in ${LEVEL[*]}; do    
        for inline in "0 1"; do
            set_inline $level $inline
            NAME="H1K"$keep"I"$l2""$l3""$l4
            #                  #
            # --- HEAD = 1 --- #
            #                  #
            awk '$1 == "head" {$3 = 1}1' TMPLT > tmp; cp tmp TMPLT
            awk -v var="$keep" '$1 == "keep_last_ckpt" {$3 = var}1' TMPLT > tmp; cp tmp TMPLT
            awk -v var="$l2" '$1 == "inline_l2" {$3 = var}1' TMPLT > tmp; cp tmp TMPLT
            awk -v var="$l3" '$1 == "inline_l3" {$3 = var}1' TMPLT > tmp; cp tmp TMPLT
            awk -v var="$l4" '$1 == "inline_l4" {$3 = var}1' TMPLT > $NAME; rm tmp
            echo -e "[ \033[1mTesting L"$level", head=1, keep="$keep", inline=("$l2","$l3","$l4")...\033[m ]"
            if [ $keep -eq "1" ]; then
            	( set -x; mpirun -n $PROCS ./check.exe $NAME 0 $level $diffSize &>> check.log )
            	### SETTING KEEP = 0 TO CLEAN DIRECTORY AFTER TEST
            	awk '$1 == "keep_last_ckpt" {$3 = 0}1' $NAME > tmp; cp tmp $NAME; rm tmp
            	( cmdpid=$BASHPID; (sleep 10; kill $cmdpid > /dev/null 2>&1 ) & set -x; mpirun -n $PROCS ./check.exe $NAME 0 $level $diffSize &>> check.log )
            	should_not_fail $?
            	if [ $testFailed = 1 ]; then
            		echo -e "L"$level", head=1, keep="$keep", inline=("$l2","$l3","$l4"), should not fail" >> failed.log
                    testFailed=0
            	fi
            	awk '$1 == "failure" {$3 = 0}1' $NAME > tmp; cp tmp $NAME; rm tmp
                if [ $eraseFiles = "1" ] && [ $FLAG = "2" ]; then
                    awk '$1 == "keep_last_ckpt" {$3 = 1}1' $NAME > tmp; cp tmp $NAME; rm tmp
                    ( set -x; mpirun -n $PROCS ./check.exe $NAME 0 $level $diffSize &>> check.log )
                    ### DELETE CHECKPOINT FILE
                    echo -e "[ \033[1mDeleting Checkpoint File...\033[m ]"
                    folder="Global/"$(awk '$1 == "exec_id" {print $3}' < $NAME)"/l4"
                    filename=$(ls $folder | grep Rank | head -n 1)
                    ( set -x; rm -rf $folder"/"$filename )
                    ### SETTING KEEP = 0 TO CLEAN DIRECTORY AFTER TEST
                    awk '$1 == "keep_last_ckpt" {$3 = 0}1' $NAME > tmp; cp tmp $NAME; rm tmp
                    ( cmdpid=$BASHPID; (sleep 10; kill $cmdpid > /dev/null 2>&1 ) & set -x; mpirun -n $PROCS ./check.exe $NAME 0 $level $diffSize &>> check.log )
                    should_fail $?
                    if [ $testFailed = 1 ]; then
                        echo -e "L"$level", head=1, keep="$keep", inline=("$l2","$l3","$l4"), deleted file on PFS, should fail" >> failed.log
                        testFailed=0
                    fi
                    awk '$1 == "failure" {$3 = 0}1' $NAME > tmp; cp tmp $NAME; rm tmp
                    let FLAG=3
                fi
            fi
            if [ $keep -eq "0" ]; then
                ( set -x; mpirun -n $PROCS ./check.exe $NAME 1 $level $diffSize &>> check.log )
                ( cmdpid=$BASHPID; (sleep 10; kill $cmdpid > /dev/null 2>&1 ) & set -x; mpirun -n $PROCS ./check.exe $NAME 0 $level $diffSize &>> check.log )
                should_not_fail $?
                if [ $testFailed = 1 ]; then
                    echo -e "L"$level", head=1, keep="$keep", inline=("$l2","$l3","$l4"), should not fail" >> failed.log
                    testFailed=0
                fi
                awk '$1 == "failure" {$3 = 0}1' $NAME > tmp; cp tmp $NAME; rm tmp
                if [ $eraseFiles = "1" ]; then
                    if [ $level = 1 ]; then
                        ( set -x; mpirun -n $PROCS ./check.exe $NAME 1 $level $diffSize &>> check.log )
                        ### DELETE CHECKPOINT FILE
                        echo -e "[ \033[1mDeleting Checkpoint File...\033[m ]"
                        folder="Local/node0/"$(awk '$1 == "exec_id" {print $3}' < $NAME)"/l1"
                        filename=$(ls $folder | grep Rank | head -n 1)
                        ( set -x; rm -rf $folder"/"$filename )
                        ( cmdpid=$BASHPID; (sleep 10; kill $cmdpid > /dev/null 2>&1 ) & set -x; mpirun -n $PROCS ./check.exe $NAME 0 $level $diffSize &>> check.log )
                        should_fail $?
                        if [ $testFailed = 1 ]; then
                            echo -e "L"$level", head=1, keep="$keep", inline=("$l2","$l3","$l4"), deleted L1 file, should fail" >> failed.log
                            testFailed=0
                        fi
                        awk '$1 == "failure" {$3 = 0}1' $NAME > tmp; cp tmp $NAME; rm tmp
                    fi
                    if [ $level = 2 ]; then
                        ( set -x; mpirun -n $PROCS ./check.exe $NAME 1 $level $diffSize &>> check.log )
                        ### DELETE Checkpoint FILES
                        echo -e "[ \033[1mDeleting Checkpoint Files...\033[m ]"
                        folder="Local/node0/"$(awk '$1 == "exec_id" {print $3}' < $NAME)"/l2"
                        filename=$(ls $folder | grep Rank | head -n 1)
                        ( set -x; rm -rf $folder"/"$filename )
                        folder="Local/node2/"$(awk '$1 == "exec_id" {print $3}' < $NAME)"/l2"
                        filename=$(ls $folder | grep Rank | head -n 1)
                        ( set -x; rm -rf $folder"/"$filename )
                        ( cmdpid=$BASHPID; (sleep 10; kill $cmdpid > /dev/null 2>&1 ) & set -x; mpirun -n $PROCS ./check.exe $NAME 0 $level $diffSize &>> check.log )
                        should_not_fail $?
                        if [ $testFailed = 1 ]; then
                            echo -e "L"$level", head=1, keep="$keep", inline=("$l2","$l3","$l4"), deleted L2 ckpt files, should not fail" >> failed.log
                            testFailed=0
                        fi
                        awk '$1 == "failure" {$3 = 0}1' $NAME > tmp; cp tmp $NAME; rm tmp
                        ( set -x; mpirun -n $PROCS ./check.exe $NAME 1 $level $diffSize &>> check.log )
                        ### DELETE PARTNER FILES
                        echo -e "[ \033[1mDeleting Partner Files...\033[m ]"
                        folder="Local/node0/"$(awk '$1 == "exec_id" {print $3}' < $NAME)"/l2"
                        filename=$(ls $folder | grep Pcof | head -n 1)
                        ( set -x; rm -rf $folder"/"$filename )
                        folder="Local/node2/"$(awk '$1 == "exec_id" {print $3}' < $NAME)"/l2"
                        filename=$(ls $folder | grep Pcof | head -n 1)
                        ( set -x; rm -rf $folder"/"$filename )
                        ( cmdpid=$BASHPID; (sleep 10; kill $cmdpid > /dev/null 2>&1 ) & set -x; mpirun -n $PROCS ./check.exe $NAME 0 $level $diffSize &>> check.log )
                        should_not_fail $?
                        if [ $testFailed = 1 ]; then
                            echo -e "L"$level", head=1, keep="$keep", inline=("$l2","$l3","$l4"), deleted L2 partner files, should not fail" >> failed.log
                            testFailed=0
                        fi
                        awk '$1 == "failure" {$3 = 0}1' $NAME > tmp; cp tmp $NAME; rm tmp
                        ( set -x; mpirun -n $PROCS ./check.exe $NAME 1 $level $diffSize &>> check.log )
                        ### DELETE NODES
                        echo -e "[ \033[1mDeleting Consecutive Nodes...\033[m ]"
                        ( set -x; rm -rf Local/node0 )
                        ( set -x; rm -rf Local/node1 )
                        ( cmdpid=$BASHPID; (sleep 10; kill $cmdpid > /dev/null 2>&1 ) & set -x; mpirun -n $PROCS ./check.exe $NAME 0 $level $diffSize &>> check.log )
                        should_fail $?
                        if [ $testFailed = 1 ]; then
                            echo -e "L"$level", head=1, keep="$keep", inline=("$l2","$l3","$l4"), deleted 2 consecutive nodes, should fail" >> failed.log
                            testFailed=0
                        fi
                        awk '$1 == "failure" {$3 = 0}1' $NAME > tmp; cp tmp $NAME; rm tmp
                        ( set -x; mpirun -n $PROCS ./check.exe $NAME 1 $level $diffSize &>> check.log )
                        ### DELETE NODES
                        echo -e "[ \033[1mDeleting Non-Consecutive Nodes...\033[m ]"
                        ( set -x; rm -rf Local/node0 )
                        ( set -x; rm -rf Local/node2 )
                        ( cmdpid=$BASHPID; (sleep 10; kill $cmdpid > /dev/null 2>&1 ) & set -x; mpirun -n $PROCS ./check.exe $NAME 0 $level $diffSize &>> check.log )
                        should_not_fail $?
                        if [ $testFailed = 1 ]; then
                            echo -e "L"$level", head=1, keep="$keep", inline=("$l2","$l3","$l4"), deleted to non-consecutive nodes, should not fail" >> failed.log
                            testFailed=0
                        fi
                        awk '$1 == "failure" {$3 = 0}1' $NAME > tmp; cp tmp $NAME; rm tmp
                    fi
                    if [ $level = 3 ]; then
                        ( set -x; mpirun -n $PROCS ./check.exe $NAME 1 $level $diffSize &>> check.log )
                        ### DELETE Checkpoint FILES
                        echo -e "[ \033[1mDeleting Checkpoint Files...\033[m ]"
                        folder="Local/node0/"$(awk '$1 == "exec_id" {print $3}' < $NAME)"/l3"
                        filename=$(ls $folder | grep Rank | head -n 1)
                        ( set -x; rm -rf $folder"/"$filename )
                        folder="Local/node2/"$(awk '$1 == "exec_id" {print $3}' < $NAME)"/l3"
                        filename=$(ls $folder | grep Rank | head -n 1)
                        ( set -x; rm -rf $folder"/"$filename )
                        ( cmdpid=$BASHPID; (sleep 10; kill $cmdpid > /dev/null 2>&1 ) & set -x; mpirun -n $PROCS ./check.exe $NAME 0 $level $diffSize &>> check.log )
                        should_not_fail $?
                        if [ $testFailed = 1 ]; then
                            echo -e "L"$level", head=1, keep="$keep", inline=("$l2","$l3","$l4"), deleted L3 ckpt files, should not fail" >> failed.log
                            testFailed=0
                        fi
                        awk '$1 == "failure" {$3 = 0}1' $NAME > tmp; cp tmp $NAME; rm tmp
                        ( set -x; mpirun -n $PROCS ./check.exe $NAME 1 $level $diffSize &>> check.log )
                        ### DELETE ENCODED FILES
                        echo -e "[ \033[1mDeleting Encoded Files...\033[m ]"
                        folder="Local/node0/"$(awk '$1 == "exec_id" {print $3}' < $NAME)"/l3"
                        filename=$(ls $folder | grep Rsed | head -n 1)
                        ( set -x; rm -rf $folder"/"$filename )
                        folder="Local/node2/"$(awk '$1 == "exec_id" {print $3}' < $NAME)"/l3"
                        filename=$(ls $folder | grep Rsed | head -n 1)
                        ( set -x; rm -rf $folder"/"$filename )
                        ( cmdpid=$BASHPID; (sleep 10; kill $cmdpid > /dev/null 2>&1 ) & set -x; mpirun -n $PROCS ./check.exe $NAME 0 $level $diffSize &>> check.log )
                        should_not_fail $?
                        if [ $testFailed = 1 ]; then
                            echo -e "L"$level", head=1, keep="$keep", inline=("$l2","$l3","$l4"), deleted L2 encoded files, should not fail" >> failed.log
                            testFailed=0
                        fi
                        awk '$1 == "failure" {$3 = 0}1' $NAME > tmp; cp tmp $NAME; rm tmp
                        ( set -x; mpirun -n $PROCS ./check.exe $NAME 1 $level $diffSize &>> check.log )
                        ### DELETE NODES
                        echo -e "[ \033[1mDeleting Consecutive Nodes...\033[m ]"
                        ( set -x; rm -rf Local/node0 )
                        ( set -x; rm -rf Local/node1 )
                        ( cmdpid=$BASHPID; (sleep 10; kill $cmdpid > /dev/null 2>&1 ) & set -x; mpirun -n $PROCS ./check.exe $NAME 0 $level $diffSize &>> check.log )
                        should_not_fail $?
                        if [ $testFailed = 1 ]; then
                            echo -e "L"$level", head=1, keep="$keep", inline=("$l2","$l3","$l4"), deleted to consecutive nodes, should not fail" >> failed.log
                            testFailed=0
                        fi
                        awk '$1 == "failure" {$3 = 0}1' $NAME > tmp; cp tmp $NAME; rm tmp
                        ( set -x; mpirun -n $PROCS ./check.exe $NAME 1 $level $diffSize &>> check.log )
                        ### DELETE NODES
                        echo -e "[ \033[1mDeleting Non-Consecutive Nodes...\033[m ]"
                        ( set -x; rm -rf Local/node0 )
                        ( set -x; rm -rf Local/node2 )
                        ( cmdpid=$BASHPID; (sleep 10; kill $cmdpid > /dev/null 2>&1 ) & set -x; mpirun -n $PROCS ./check.exe $NAME 0 $level $diffSize &>> check.log )
                        should_not_fail $?
                        if [ $testFailed = 1 ]; then
                            echo -e "L"$level", head=1, keep="$keep", inline=("$l2","$l3","$l4"), deleted to non-consecutive nodes, should not fail" >> failed.log
                            testFailed=0
                        fi
                        awk '$1 == "failure" {$3 = 0}1' $NAME > tmp; cp tmp $NAME; rm tmp
                    fi
                    if [ $level = 4 ]; then
                        ( set -x; mpirun -n $PROCS ./check.exe $NAME 1 $level $diffSize &>> check.log )
                        ### DELETE CHECKPOINT FILE
                        echo -e "[ \033[1mDeleting Checkpoint File...\033[m ]"
                        folder="Global/"$(awk '$1 == "exec_id" {print $3}' < $NAME)"/l4"
                        filename=$(ls $folder | grep Rank | head -n 1)
                        ( set -x; rm -rf $folder"/"$filename )
                        ( cmdpid=$BASHPID; (sleep 10; kill $cmdpid > /dev/null 2>&1 ) & set -x; mpirun -n $PROCS ./check.exe $NAME 0 $level $diffSize &>> check.log )
                        should_fail $?
                        if [ $testFailed = 1 ]; then
                            echo -e "L"$level", head=1, keep="$keep", inline=("$l2","$l3","$l4"), deleted L4 file, should fail" >> failed.log
                            testFailed=0
                        fi
                        awk '$1 == "failure" {$3 = 0}1' $NAME > tmp; cp tmp $NAME; rm tmp
                    fi
                fi
            fi
        done
    done
    rm $NAME
done
rm -rf TMPLT Global/* Local/* Meta/* chk/*
pkill -f check.exe
echo "---SUMMARY---"
echo "PASSED: "$SUCCEED
echo "FAILED: "$FAILED
echo "FAULTY: "$FAULTY
cat failed.log
