source ~/.bashrc
CC=pgcc
F90=pgfortran
#FC=pgfortran
export PGI=/home/travis/pgi
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/travis/pgi/linux86-64/16.10/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/travis/pgi/linux86-64/16.10/mpi/openmpi/lib
export PATH=/home/travis/pgi/linux86-64/16.10/bin:$PATH
export PATH=/home/travis/pgi/linux86-64/16.10/mpi/openmpi/bin:$PATH