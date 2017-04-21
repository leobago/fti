sudo ./travisScripts/install-icc.sh --components icc,mpi,ifort --dest /opt/intel

source ~/.bashrc
source /opt/intel/bin/compilervars.sh intel64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/ism/bin/intel64
export I_MPI_SHM_LMT=shm #https://software.intel.com/en-us/forums/intel-clusters-and-hpc-technology/topic/610561
CC=icc
CXX=icpc
