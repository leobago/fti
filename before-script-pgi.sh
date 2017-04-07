source ~/.bashrc
CC=pgcc
F90=pgfortran
#FC=pgfortran
export PGI=/opt/pgi
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/pgi/linux86-64/16.10/lib
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/pgi/linux86-64/16.10/mpi/openmpi/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib
export PATH=/usr/bin:$PATH
export PATH=/opt/pgi/linux86-64/16.10/bin:$PATH
export C_INCLUDE_PATH=/usr/lib/openmpi/include
export PATH=/usr/lib/openmpi/include:$PATH
export FORTRAN_INCLUDE_PATH=/usr/lib/openmpi/include
#export PATH=/opt/pgi/linux86-64/16.10/mpi/openmpi/bin:$PATH