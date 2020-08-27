#!/bin/bash
#   Copyright (c) 2017 Leonardo A. Bautista-Gomez
#   All rights reserved
#
#   @file   build.sh
#   @author Alexandre de Limas Santana (alexandre.delimassantana@bsc.es)
#   @date   May, 2020
#
#   @brief script to build FTI for different compilers in the CI environment
#   @arguments
#      $1 - The compiler we want to use in building FTI

root_folder=$(git rev-parse --show-toplevel)
install_script=$root_folder/install.sh

if [ -z $1 ]; then
    exit 1
fi

case $1 in
gcc | GCC)
    export FC=gfortran
    ${install_script} --enable-coverage --enable-hdf5 --enable-sionlib --enable-fortran --sionlib-path=/opt/sionlib
    ;;
intel | Intel)
    export CFLAGS='-D__PURE_INTEL_C99_HEADERS__ -D_Float32=float -D_Float64=double -D_Float32x=_Float64 -D_Float64x=_Float128'
    export PATH="$PATH:/opt/intel/bin"
    /opt/intel/bin/compilervars.sh intel64
    ${install_script} --enable-hdf5 --enable-sionlib --sionlib-path=/opt/sionlib -C $root_folder/CMakeScripts/intel.cmake
    ;;
clang | Clang)
    export OMPI_MPICC=clang
    export OMPI_CXX=clang++
    export CC=clang
    export FC=gfortran
    ${install_script} --enable-hdf5 --enable-sionlib --sionlib-path=/opt/sionlib
    ;;
pgi | PGI)
    export PGICC='/opt/pgi/linux86-64/19.10/bin/'
    export PGIMPICC='/opt/pgi/linux86-64/2019/mpi/openmpi-3.1.3/bin/'
    export LM_LICENSE_FILE="/opt/pgi/license.dat"
    export LD_LIBRARY_PATH='/opt/pgi/linux86-64/19.10/lib'
    export PATH="$PGICC:$PGIMPICC:$PATH"
    export CC=pgcc
    export FC=pgfortran
    ${install_script} --enable-hdf5 --enable-sionlib --sionlib-path=/opt/sionlib
    ;;
esac
