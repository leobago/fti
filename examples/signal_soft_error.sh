#!/bin/bash


usleep $[ ( $RANDOM % 100 )  + 1 ]

#10 - SIGUSR1 

kill -10 $1

