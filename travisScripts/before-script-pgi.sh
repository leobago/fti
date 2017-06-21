wget http://www.mellanox.com/downloads/ofed/MLNX_OFED-4.0-2.0.0.1/MLNX_OFED_LINUX-4.0-2.0.0.1-ubuntu14.04-x86_64.tgz
gunzip < MLNX_OFED_LINUX-4.0-2.0.0.1-ubuntu14.04-x86_64.tgz | tar xvf -
cd MLNX_OFED_LINUX-4.0-2.0.0.1-ubuntu14.04-x86_64
yes | sudo ./mlnxofedinstall
cd ..
./travisScripts/install-pgi.sh --mpi --dest /opt/pgi
sudo apt-get install -y libibverbs-dev librdmacm-dev

source ~/.bashrc
CC=pgcc
CXX=pgc++
F90=pgfortran
FC=pgfortran
export PGI=/opt/pgi
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/pgi/linux86-64/2017/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/pgi/linux86-64/2017/mpi/openmpi/lib
export PATH=/opt/pgi/linux86-64/2017/bin:$PATH
export PATH=/opt/pgi/linux86-64/2017/mpi/openmpi/bin:$PATH
