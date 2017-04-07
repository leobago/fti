wget http://www.mellanox.com/downloads/ofed/MLNX_OFED-4.0-2.0.0.1/MLNX_OFED_LINUX-4.0-2.0.0.1-ubuntu14.04-x86_64.tgz
gunzip < MLNX_OFED_LINUX-4.0-2.0.0.1-ubuntu14.04-x86_64.tgz | tar xvf -
cd MLNX_OFED_LINUX-4.0-2.0.0.1-ubuntu14.04-x86_64
yes | sudo ./mlnxofedinstall
cd ..
./install-pgi.sh --mpi
sudo apt-get install -y libibverbs-dev librdmacm-dev

source ~/.bashrc
CC=pgcc
CXX=pgc++
F90=pgfortran
FC=pgfortran
export PGI=/home/travis/pgi
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/travis/pgi/linux86-64/16.10/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/travis/pgi/linux86-64/16.10/mpi/openmpi/lib
export PATH=/home/travis/pgi/linux86-64/16.10/bin:$PATH
export PATH=/home/travis/pgi/linux86-64/16.10/mpi/openmpi/bin:$PATH