#!/bin/bash

# Run this scripts from the testing folder in build

export MPIRUN_ARGS=--oversubscribe

#fixtures=$(find '@testing_dir@/local' -name '*.fixture' | sed s/.fixture//)
fixtures=$(find '@testing_dir@/local' -name '*.fixture' | grep 'failtime' | sed s/.fixture//)
@itf_run_cmd@ ${fixtures[@]}
retval=$?

unset MPIRUN_ARGS

exit $retval