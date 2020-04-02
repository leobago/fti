#!/bin/bash

print_usage() {
	echo "usage     : ./install.sh [options]"
	echo "options   : "
	echo "            [--prefix=DIR]                # Installation directory (default: ./install/)"
	echo "            [--enable-hdf5]               # Enable HDF5 extension (default: disabled)"
	echo "            [--enable-sionlib]            # Enable SIONlib extension (default: disabled)"
	echo "            [--enable-ime]                # Enable IME extension (default: disabled)"
	echo "            [--enable-lustre]             # Enable extended Lustre support (default: disabled)"
	echo "            [--enable-fortran]            # Enable Fortran bindings (default: disabled)"
	echo "            [--disable-examples]          # Disable the compilation of examples (default: enabled)"
	echo "            [--enable-testing]            # Enable testing framework (default: disabled)"
	echo "            [--enable-docu]               # Enable creation of FTI documentation (default: disabled)"
	echo "            [--enable-titorial]           # Enable creation of FTI tutorial (default: disabled)"
	echo "            [--enable-fi]                 # Enable FTI fault injection mechanism (default: disabled)"
	echo " "
	echo "            [--sionib-path=DIR]           # Path to SIONlib installation"
	echo "            [--ime-path=DIR]              # Path to DDN IME installation"
	echo " "
	echo "            [--debug]                     # Enable a debug build"
	echo "            [--silent]                    # No output to stdout or stderr during installation"
	echo "            [--uninstall]                 # No output to stdout or stderr during installation"
}

remove_fti() {
    set -x
    rm -rf $FTI_ROOT/build
    rm -rf $FTI_ROOT/install
    rm -f $FTI_ROOT/config.log
    rm -f $FTI_ROOT/install.log
    set +x
}

# Calculate the FTI root directory relative to this file
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
export FTI_ROOT="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"

CMAKE_ARGS=""

#parse arguments
while [ $# -gt 0 ]; do
case $1 in
    -h|--help)
    print_usage
    exit 0
    ;;
    -s=*|--prefix=*)
    FTI_INSTALL_DIR="${1#*=}"
    shift # past argument=value
    ;;
    --debug)
    CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_BUILD_TYPE=Debug"
    shift # past argument=value
    ;;
    --enable-hdf5)
    CMAKE_ARGS="$CMAKE_ARGS -DENABLE_HDF5=1"
    shift # past argument=value
    ;;
    --enable-sionlib)
    CMAKE_ARGS="$CMAKE_ARGS -DENABLE_SIONLIB=1"
    shift # past argument=value
    ;;
    --enable-ime)
    CMAKE_ARGS="$CMAKE_ARGS -DENABLE_IME_NATIVE=1"
    shift # past argument=value
    ;;
    --enable-lustre)
    CMAKE_ARGS="$CMAKE_ARGS -DENABLE_LUSTRE=1"
    shift # past argument=value
    ;;
    --enable-fortran)
    CMAKE_ARGS="$CMAKE_ARGS -DENABLE_FORTRAN=1"
    shift # past argument=value
    ;;
    --disable-examples)
    CMAKE_ARGS="$CMAKE_ARGS -DENABLE_EXAMPLES=0"
    shift # past argument=value
    ;;
    --enable-testing)
    CMAKE_ARGS="$CMAKE_ARGS -DENABLE_TESTS=1"
    shift # past argument=value
    ;;
    --enable-docu)
    CMAKE_ARGS="$CMAKE_ARGS -DENABLE_DOCU=1"
    shift # past argument=value
    ;;
    --enable-tutorial)
    CMAKE_ARGS="$CMAKE_ARGS -DENABLE_TUTORIAL=1"
    shift # past argument=value
    ;;
    --enable-fi)
    CMAKE_ARGS="$CMAKE_ARGS -DENABLE_FI_IO=1"
    shift # past argument=value
    ;;
    --enable-docu)
    CMAKE_ARGS="$CMAKE_ARGS -DENABLE_DOCU=1"
    shift # past argument=value
    ;;
    --sionlib-path=*)
    CMAKE_ARGS="$CMAKE_ARGS -DSIONLIBBASE=${1#*=}"
    shift # past argument=value
    ;;
    --ime-path=*)
    CMAKE_ARGS="$CMAKE_ARGS -DIMEBASE=${1#*=}"
    shift # past argument=value
    ;;
    --silent)
    VERBOSE=false
    shift # past argument=value
    ;;
    --uninstall)
    remove_fti
    exit 0 # past argument=value
    ;;
    --default)
    DEFAULT=YES
    shift # past argument with no value
    ;;
    *)
        echo "unknown option: $1"  
        print_usage
        exit -1
    ;;
esac
done

rm -rf $FTI_ROOT/config.log
rm -rf $FTI_ROOT/install.log

echo "cmake command:" >> $FTI_ROOT/config.log
echo "cmake cmake -DCMAKE_INSTALL_PREFIX:PATH=$FTI_INSTALL_DIR $CMAKE_ARGS .." >> $FTI_ROOT/config.log

# Test if user has an installation directory
if [ -z "$FTI_INSTALL_DIR" ];
then
	FTI_INSTALL_DIR="$FTI_ROOT/install"
fi

# Tutorial steps
cd "$FTI_ROOT"

mkdir -p 'build'
cd 'build'

if [ $VERBOSE ]; then
    cmake cmake -DCMAKE_INSTALL_PREFIX:PATH="$FTI_INSTALL_DIR" $CMAKE_ARGS .. >> $FTI_ROOT/install.log 2>&1 
else
    cmake cmake -DCMAKE_INSTALL_PREFIX:PATH="$FTI_INSTALL_DIR" $CMAKE_ARGS .. 2>&1 | tee -a $FTI_ROOT/install.log 
fi
if [ $? = 1 ]; then
    rm -rf build
    exit -1
fi
if [ $VERBOSE ]; then
    make -j all install >> $FTI_ROOT/install.log 2>&1
else
    make -j all install 2>&1 | tee -a $FTI_ROOT/install.log
fi
if [ $? = 1 ]; then
    exit -1
fi

