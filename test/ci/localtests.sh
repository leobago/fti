#!/bin/bash

# Run this scripts from the test folder in build

export MPIRUN_ARGS=--oversubscribe

fixtures=$(find 'local' -name '*.fixture' | sed s/.fixture//)

itf/testrunner ${fixtures[@]}

unset MPIRUN_ARGS