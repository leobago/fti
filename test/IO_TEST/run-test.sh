#!/usr/bin/env bash
PROCS=16
FAILED=0
SUCCEED=0
FAULTY=0
#							 #
# ---- HEADER CHECK LOG ---- # 
#							 #
cat <<EOF > check.log
### FTI TEST FOR $PROCS PROCESSES ###
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
	elif [ $1 = 52 ]; then
		echo -e "\033[0;31mfailed\033[m (wrong result!)" 
		let FAILED=FAILED+1
	elif [ $1 = 1 ]; then
        echo -e "\033[0;31mfailed\033[m (Recovery failed)" 
		let FAILED=FAILED+1
	else 
		echo "Unknown exit status: "$1
		let FAULTY=FAULTY+1
	fi
}

should_fail() {
	if [ $1 = 255 ]; then
        echo -e "\033[0;32mpassed\033[m (MPI exception)"
		let SUCCEED=SUCCEED+1
	elif [ $1 = 0 ]; then
	    echo -e "\033[0;31mfailed\033[m"
		let FAILED=FAILED+1
	elif [ $1 = 52 ]; then
		echo -e "\033[0;31mfailed\033[m (wrong result!)" 
		let FAILED=FAILED+1
	elif [ $1 = 1 ]; then
        echo -e "\033[0;32mpassed\033[m (Recovery failed)" 
		let SUCCEED=SUCCEED+1
	else 
		echo "Unknown exit status: "$1
		let FAULTY=FAULTY+1
	fi
}
#						 #
# ---- TEST SCRIPTS ---- # 
#						 #
LEVEL=(1 3)
IO=(1 2 3)
IO_NAMES=("POSIX" "MPI" "SION")
INLINE_L2=(0 1)
INLINE_L3=(0 1)
INLINE_L4=(0 1)
KEEP=(0 1)
for io in ${IO[*]}; do
    let i=io-1
	for keep in ${KEEP[*]}; do
        NAME=${IO_NAMES[$i]}"H0K"$keep"I111"
        awk -v var="$keep" '$1 == "keep_last_ckpt" {$3 = var}1' TMPLT > tmp; cp tmp TMPLT
        awk -v var="$io" '$1 == "ckpt_io" {$3 = var}1' TMPLT > $NAME; rm tmp
        for level in ${LEVEL[*]}; do    
            echo -e "Testing "${IO_NAMES[$i]}", L"$level", head=0, keep="$keep", inline=(1,1,1)..."
            if [ $keep -eq "1" ]; then
				( set -x; mpirun -n $PROCS ./check.exe $NAME 0 $level &>> check.log )
				### SETTING KEEP = 0 TO CLEAN DIRECTORY AFTER TEST
				awk '$1 == "keep_last_ckpt" {$3 = 0}1' $NAME > tmp; cp tmp $NAME; rm tmp
				( set -x; mpirun -n $PROCS ./check.exe $NAME 0 $level &>> check.log )
				should_not_fail $?
            fi
            if [ $keep -eq "0" ]; then
				( set -x; mpirun -n $PROCS ./check.exe $NAME 1 $level &>> check.log )
				( set -x; mpirun -n $PROCS ./check.exe $NAME 0 $level &>> check.log )
				should_not_fail $?
            fi
        done
		rm $NAME
    done
    for keep in ${KEEP[*]}; do
        for l2 in ${INLINE_L2[*]}; do
            for l3 in ${INLINE_L3[*]}; do
                for l4 in ${INLINE_L4[*]}; do
					NAME=${IO_NAMES[$i]}"H1K"$keep"I"$l2""$l3""$l4
                    awk '$1 == "head" {$3 = 1}1' TMPLT > tmp; cp tmp TMPLT
                    awk -v var="$keep" '$1 == "keep_last_ckpt" {$3 = var}1' TMPLT > tmp; cp tmp TMPLT
        			awk -v var="$io" '$1 == "ckpt_io" {$3 = var}1' TMPLT > tmp; cp tmp TMPLT
                    awk -v var="$l2" '$1 == "inline_l2" {$3 = var}1' TMPLT > tmp; cp tmp TMPLT
                    awk -v var="$l3" '$1 == "inline_l3" {$3 = var}1' TMPLT > tmp; cp tmp TMPLT
                    awk -v var="$l4" '$1 == "inline_l4" {$3 = var}1' TMPLT > $NAME; rm tmp
					for level in ${LEVEL[*]}; do    
						echo -e "Testing "${IO_NAMES[$i]}", L"$level", head=1, keep="$keep", inline=("$l2","$l3","$l4")..."
						if [ $keep -eq "1" ]; then
							( set -x; mpirun -n $PROCS ./check.exe $NAME 0 $level &>> check.log )
							### SETTING KEEP = 0 TO CLEAN DIRECTORY AFTER TEST
							awk '$1 == "keep_last_ckpt" {$3 = 0}1' $NAME > tmp; cp tmp $NAME; rm tmp
							( set -x; mpirun -n $PROCS ./check.exe $NAME 0 $level &>> check.log )
							should_not_fail $?
						fi
						if [ $keep -eq "0" ]; then
							( set -x; mpirun -n $PROCS ./check.exe $NAME 1 $level &>> check.log )
							( set -x; mpirun -n $PROCS ./check.exe $NAME 0 $level &>> check.log )
							should_not_fail $?
						fi
					done
					rm $NAME
                done
            done
        done
    done
done
rm TMPLT
echo "---SUMMARY---"
echo "PASSED: "$SUCCEED
echo "FAILED: "$FAILED
echo "FAULTY: "$FAULTY
