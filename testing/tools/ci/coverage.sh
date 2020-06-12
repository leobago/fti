#!/bin/bash
#   Copyright (c) 2017 Leonardo A. Bautista-Gomez
#   All rights reserved
#
#   @file   coverage.sh
#   @author Alexandre de Limas Santana (alexandre.delimassantana@bsc.es)
#   @date   June, 2020
#
#   @brief Aggregate coverage metrics in XML format to be used in Jenkins
#   @arguments 
#      output-format: The type of report to generate using gcovr.
#   @details Must be called from the build directory

case $1 in
xml)
    mkdir -p 'coverage'
    gcovr --xml -r ../ -o 'coverage/summary.xml'
    ;;
hmtl)
    mkdir -p 'coverage'
    gcovr --html --html-details -r ../ -o 'coverage/index.html'
    ;;
*)
    echo "Invalid output format"
    ;;
esac
