source ~/.bashrc
CC=pgcc
F90=pgfortran
BPP_COMPILER_ID=PGI
#FC=pgfortran
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/pgi/linux86-64/16.10/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/pgi/linux86-64/16.10/mpi/openmpi/lib
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/lib/x86_64-linux-gnu
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib