#!/bin/bash

# Run this scripts from the testing folder in build

export MPIRUN_ARGS=--oversubscribe

fixtures=$(find '@testing_dir@/local' -name '*.fixture' | grep 'vpr' | sed s/.fixture//)
@itf_run_cmd@ --dry-run ${fixtures[@]}

unset MPIRUN_ARGS
