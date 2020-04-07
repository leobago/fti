#!/bin/bash

# Run this scripts from the testing folder in build

export MPIRUN_ARGS=--oversubscribe

@itf_run_cmd@ $@
retval=$?

unset MPIRUN_ARGS

exit $retval