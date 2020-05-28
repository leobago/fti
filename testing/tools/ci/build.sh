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
    ${install_script} --enable-hdf5 --enable-sionlib --sionlib-path=/opt/sionlib
    ;;
intel | Intel)
    export CFLAGS_FIX='-D__PURE_INTEL_C99_HEADERS__ -D_Float32=float -D_Float64=double -D_Float32x=_Float64 -D_Float64x=_Float128'
    export ICCPATH='/opt/intel/compilers_and_libraries_2018.3.222/linux/bin'
    export MPICCPATH='/opt/intel/compilers_and_libraries_2018.3.222/linux/mpi/intel64/bin'
    export LD_LIBRARY_PATH='/opt/HDF5/1.10.4/lib:/opt/intel/compilers_and_libraries_2018.3.222/linux/compiler/lib/intel64:/opt/intel/compilers_and_libraries_2018.3.222/linux/compiler/lib/intel64_lin:/opt/intel/compilers_and_libraries_2018.3.222/linux/mpi/intel64/lib:/opt/intel/compilers_and_libraries_2018.3.222/linux/mpi/mic/lib:/opt/intel/compilers_and_libraries_2018.3.222/linux/ipp/lib/intel64:/opt/intel/compilers_and_libraries_2018.3.222/linux/compiler/lib/intel64_lin:/opt/intel/compilers_and_libraries_2018.3.222/linux/mkl/lib/intel64_lin:/opt/intel/compilers_and_libraries_2018.3.222/linux/tbb/lib/intel64/gcc4.7:/opt/intel/compilers_and_libraries_2018.3.222/linux/tbb/lib/intel64/gcc4.7'
    export PATH="$PATH:$MPICCPATH"

    . $ICCPATH/compilervars.sh intel64
    . $MPICCPATH/mpivars.sh
    export CFLAGS=$CFLAGS_FIX
    ${install_script} --enable-hdf5 -C $root_folder/CMakeScripts/intel.cmake -DHDF5_ROOT=/opt/HDF5/1.10.4
    ;;
clang | Clang)
    export OMPI_MPICC=clang
    export OMPI_CXX=clang++
    export CC=clang
    export FC=gfortran
    ${install_script} --enable-hdf5 --enable-sionlib --sionlib-path=/opt/sionlib
    ;;
pgi | PGI)
    export PGICC='/opt/pgi/linux86-64/18.4/bin/'
    export PGIMPICC='/opt/pgi/linux86-64/2018/mpi/openmpi-2.1.2/bin/'
    export LM_LICENSE_FILE="/opt/pgi/license.dat"
    export LD_LIBRARY_PATH='/opt/pgi/linux86-64/18.4/lib'
    export PATH="$PGICC:$PGIMPICC:$PATH"
    export CC=pgcc
    export FC=pgfortran
    echo $PATH
    ls /opt/pgi/
    ${install_script} --enable-hdf5 -DHDF5_ROOT=/opt/HDF5/1.10.4
    ;;
esac
