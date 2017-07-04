#!/bin/bash
#
#   @file   heartbeat.sh
#   @author Karol Sierocinski (ksiero@man.poznan.pl)
#   @date   July, 2017
#   @brief  echo heartbeat for Travis (output every 9 minutes)
#

while true; do
	sleep 9m
	echo "Travis heartbeat"
done
