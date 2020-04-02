#!/bin/bash

# Run this scripts from the test folder in build

export MPIRUN_ARGS=--oversubscribe
itf/testrunner -v 2 local/variateProcessorRestart/vpr
#itf/testrunner $(find 'local' -name '*.fixture' | sed s/.fixture//)
unset MPIRUN_ARGS