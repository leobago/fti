#!/bin/bash

# Run this scripts from the testing folder in build

export MPIRUN_ARGS=--oversubscribe

@itf_run_cmd@ $(find 'local' -name '*.fixture' | grep -v 'vpr' | sed s/.fixture//)

unset MPIRUN_ARGS
