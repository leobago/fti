source ~/.bashrc
CC=pgcc
CXX=pgcpp
FC=pgfortran
export PATH=$PATH:/opt/pgi/linux86-64/2016/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/pgi/linux86-64/2016/lib
export MPI_INCLUDE_PATH=/opt/pgi/linux86-64/2016/mpi/openmpi/include
export MPI_Fortran_INCLUDE_PATH=/opt/pgi/linux86-64/2016/mpi/openmpi/include
export MPI_C_INCLUDE_PATH=/opt/pgi/linux86-64/2016/mpi/openmpi/include
export CPATH=$CPATH:/opt/pgi/linux86-64/2016/include