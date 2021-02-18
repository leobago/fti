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

set_compiler_env() {
  cat <<EOF > run
#!/bin/bash
cmd="$*"
. /opt/$1/install/activate_all > /dev/null 2>&1
$cmd
EOF
	chmod +x run
  . /opt/$1/install/activate_all  
}

root_folder=$(git rev-parse --show-toplevel)
install_script=$root_folder/install.sh

if [ -z $1 ]; then
    exit 1
fi

case $1 in
gcc | GCC)
    set_compiler_env gnu-openmpi
    ${install_script} --enable-tests --enable-coverage --enable-hdf5 --enable-sionlib --enable-fortran --sionlib-path=$SIONLIB_ROOT
    ;;
mpich | MPICH)
    set_compiler_env gnu-mpich
    ${install_script} --enable-tests --enable-hdf5 --enable-sionlib --enable-fortran --sionlib-path=$SIONLIB_ROOT
    ;;
intel | Intel)
    set_compiler_env intel-impi
    ${install_script} --enable-tests --enable-hdf5 --enable-sionlib --enable-fortran --sionlib-path=$SIONLIB_ROOT --cmake-arg="CMAKE_MODULE_PATH=$CMAKE_MODULE_PATH"
    ;;
llvm | LLVM)
    set_compiler_env llvm-openmpi
    ${install_script} --enable-tests --enable-hdf5
    ;;
pgi | PGI)
    set_compiler_env pgi-openmpi
    ${install_script} --enable-tests --enable-hdf5 --enable-sionlib --enable-fortran --sionlib-path=$SIONLIB_ROOT
    ;;
esac

