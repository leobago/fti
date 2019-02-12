#!/bin/bash
MPIVER="1.10.7"
wget --no-check-certificate https://www.open-mpi.org/software/ompi/v1.10/downloads/openmpi-1.10.7.tar.gz
tar -zxf openmpi-1.10.7.tar.gz
cd openmpi-1.10.7
echo "CONFIGURE OPENMPI (VER: $MPIVER)"
bash ./configure --prefix=$HOME/openmpi-$MPIVER CC=gcc CXX=g++ > /dev/null 2>&1
echo "MAKE OPENMPI (VER: $MPIVER)"
make -j 4 > /dev/null 2>&1
echo "INSTALL OPENMPI (VER: $MPIVER)"
sudo make install > /dev/null 2>&1
export PATH=$HOME/openmpi-$MPIVER/bin:$PATH
export LD_LIBRARY_PATH=$HOME/openmpi-$MPIVER/lib/:$LD_LIBRARY_PATH
