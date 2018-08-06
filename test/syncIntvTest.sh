#!/bin/bash
#
#   @file   syncIntvTest.sh
#   @author Tomasz Paluszkiewicz (tomaszp@man.poznan.pl)
#   @date   October, 2017
#   @brief  tests FTI_Snapshot
#
check () { #$1 - log name $2 - config name
    nextCkptArray=($(grep "Next ckpt. at iter." $1 | awk '{ print $(NF-5) }' | sort -nu))
    ckptMadeArray=($(grep "Checkpoint made i" $1 | awk '{ print $NF }' | sort -nu))
    ckptIntvArray=($(grep "Next ckpt. at iter." $1 | awk '{ print $(NF-11) }' | sort -nu))
    i=0
    while [ -n "${ckptMadeArray[$i]}" ]
    do
        #check if checkpoint was made at correct iteration
        containsElement $((${ckptMadeArray[$i]}+1)) ${nextCkptArray[@]}
        if [ $? -ne 0 ]; then
            #if checkpoint not in nextArray then resync didnt occured between checkpoints
            #so we check if iterations difference is in ckpt interval values
            containsElement $((${ckptMadeArray[$i]}-${ckptMadeArray[$(($i-1))]})) ${ckptIntvArray[@]}
            if [ $? -ne 0 ]; then
                echo "Checkpoint made at i=$((${ckptMadeArray[$i]}+1)) not in next ckpt. values."
                return 1
            fi
        fi
        i=$i+1
    done
    echo "All checkpoints done at correct iterations."
    maxSyncIntv=$(grep "max_sync_intv" $2 | awk '{ print $NF }')
    if [ -z "$maxSyncIntv" ]; then
        maxSyncIntv=512
    fi
    echo "MaxSyncIntv is $maxSyncIntv"
    while read -r line ; do
        iter=$(echo $line | awk '{ print $9 }')
        sync=$(echo $line | awk '{ print $NF }')
        #check if SyncIntv is less than maxSyncIntv
        if [ $sync -gt $maxSyncIntv ]; then
            echo "SyncIntv ($sync) should be less than maxSyncIntv ($maxSyncIntv)."
            return 1
        fi
        #check if resync is done at right iterations
        if [ $(($iter%$sync)) -ne 0 ]; then
            echo "Resync doesnt match."
            return 1
        fi
    done < <(grep "Current iter" $1)
    echo "All resync done at correct iterations."
}

containsElement () { #$1 - element $2 - array
    local e match="$1"
    shift
    for e; do [[ "$e" == "$match" ]] && return 0; done
    return 1
}

cp configs/$1 config.fti
mpirun -n $2 ./syncIntv config.fti > logFile
check logFile config.fti
if [ $? -eq 0 ]; then
    echo "Test passed."
else
    echo "Test failed."
    cat logFile
fi
rm logFile config.fti
rm -r ./Local ./Global ./Meta
