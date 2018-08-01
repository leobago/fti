#!/bin/bash
wget --no-check-certificate https://www.open-mpi.org/software/ompi/v1.10/downloads/openmpi-1.10.7.tar.gz
tar -zxf openmpi-1.10.7.tar.gz
cd openmpi-1.10.7
bash ./configure --prefix=$HOME/openmpi-1.10
make -j
sudo make install
export PATH=$PATH:$HOME/openmpi-1.10/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/openmpi-1.10/lib/
