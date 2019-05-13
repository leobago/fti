PGI_DIR=/opt/pgi
wget http://www.mellanox.com/downloads/ofed/MLNX_OFED-4.0-2.0.0.1/MLNX_OFED_LINUX-4.0-2.0.0.1-ubuntu14.04-x86_64.tgz
gunzip < MLNX_OFED_LINUX-4.0-2.0.0.1-ubuntu14.04-x86_64.tgz | tar xvf -
cd MLNX_OFED_LINUX-4.0-2.0.0.1-ubuntu14.04-x86_64
yes | sudo ./mlnxofedinstall
cd ..
./travisScripts/install-pgi.sh --mpi --dest $PGI_DIR
sudo apt-get install -y libibverbs-dev librdmacm-dev

source ~/.bashrc
CC=pgcc
CXX=pgc++
F90=pgfortran
FC=pgfortran
PGI_VERSION=$(basename "${PGI_DIR}"/linux86-64/*.*/)
export PGI=${PGI_DIR}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${PGI_DIR}/linux86-64/${PGI_VERSION}/lib
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${PGI_DIR}/linux86-64/${PGI_VERSION}/mpi/openmpi/lib
export LD_LIBRARY_PATH=$(${LD_LIBRARY_PATH}:${PGI_DIR}/linux86-64/*/mpi/openmpi*/lib)
export PATH=${PGI_DIR}/linux86-64/${PGI_VERSION}/bin:$PATH
#export PATH=${PGI_DIR}/linux86-64/${PGI_VERSION}/mpi/openmpi/bin:$PATH
export PATH=$(${PGI_DIR}/linux86-64/*/mpi/openmpi*/bin:${PATH})
echo $PATH
