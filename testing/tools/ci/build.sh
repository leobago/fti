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
. /opt/$1/install/activate_all
\$*
EOF
	chmod +x run
  . /opt/$1/install/activate_all  
}
#    # $1 -> C         compiler
#    # $2 -> Fortran   compiler
#    # $3 -> C++       compiler
#    
#    # standard compiler
#    export CC=$1
#    export CXX=$3
#    export FC=$2
#    export F95=$2
#    export F77=$2
#    export F90=$2
#    
#    # MPI wrapper
#    export OMPI_CC=$1
#    export OMPI_FC=$2
#    export OMPI_F90=$2
#    export OMPI_F77=$2
#    export OMPI_CXX=$3
#    export OMPI_MPICC=$1
#    export MPICH_CC=$1
#    export MPICH_FC=$2
#    export MPICH_CXX=$3
#    export MPICH_F77=$2
#    export MPICH_F90=$2
#    export I_MPI_CC=$1
#    export I_MPI_CXX=$3
#    export I_MPI_FC=$2
#    export I_MPI_F90=$2
#    export I_MPI_F77=$2
#}
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
    set_compiler_env gcc gfortran g++
    PATH=/opt/mpich/bin:$PATH; export PATH
    PATH=/opt/hdf5-mpich/install/bin:$PATH; export PATH
    PATH=/opt/sionlib-mpich/install/bin:$PATH; export PATH
    LD_LIBRARY_PATH=/opt/mpich/lib:$LD_LIBRARY_PATH; export LD_LIBRARY_PATH
    LD_LIBRARY_PATH=/opt/hdf5-mpich/install/lib:$LD_LIBRARY_PATH; export LD_LIBRARY_PATH
    LD_LIBRARY_PATH=/opt/sionlib-mpich/install/lib:$LD_LIBRARY_PATH; export LD_LIBRARY_PATH
    ${install_script} --enable-tests --enable-hdf5 --enable-sionlib --enable-fortran --sionlib-path=/opt/sionlib-mpich/install
    ;;
intel | Intel)
    set_compiler_env icc ifort icpc
    export CFLAGS='-D__PURE_INTEL_C99_HEADERS__ -D_Float32=float -D_Float64=double -D_Float32x=_Float64 -D_Float64x=_Float128'
    export PATH="$PATH:/opt/intel/bin"
    /opt/intel/bin/compilervars.sh intel64
    ${install_script} --enable-tests --enable-hdf5 --enable-sionlib --sionlib-path=/opt/sionlib -C $root_folder/CMakeScripts/intel.cmake
    ;;
clang | Clang)
    set_compiler_env clang gfortran clang++
    ${install_script} --enable-tests --enable-hdf5 --enable-sionlib --sionlib-path=/opt/sionlib
    ;;
pgi | PGI)
    export PGICC='/opt/nvidia/hpc_sdk/Linux_x86_64/20.11/compilers/bin/'
    export PGIMPICC='/opt/nvidia/hpc_sdk/Linux_x86_64/20.11/comm_libs/openmpi/openmpi-3.1.5/bin/'
    #export LM_LICENSE_FILE="/opt/pgi/license.dat"
    export LD_LIBRARY_PATH='/opt/nvidia/hpc_sdk/Linux_x86_64/20.11/compilers/lib/'
    export PATH="$PGICC:$PGIMPICC:$PATH"
    set_compiler_env pgcc pgfortran pgc++
    ${install_script} --enable-tests --enable-hdf5 --enable-sionlib --sionlib-path=/opt/sionlib
    ;;
esac
