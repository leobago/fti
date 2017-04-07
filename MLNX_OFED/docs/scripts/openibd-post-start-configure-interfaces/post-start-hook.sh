#!/bin/bash

mod=

for ethpath in /sys/class/net/*
do
    if (grep 0x15b3 ${ethpath}/device/vendor > /dev/null 2>&1); then
        # filter by module ?
        if [ ! -z "$mod" ]; then
            if [ "$(basename `readlink -f ${ethpath}/device/driver/module`)" != "$mod" ]; then
                continue
            fi
        fi
        ifname=${ethpath##*/}
        cmd="/bin/mlnx_interface_mgr.sh $ifname"
        echo "Running $cmd"
        eval $cmd
    fi
done
