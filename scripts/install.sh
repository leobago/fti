#!/bin/bash

# Calculate the FTI root directory relative to this file
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
export FTI_ROOT="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )/.."

# Test if user has an installation directory
if [ -z "$FTI_INSTALL_DIR" ];
then
	FTI_INSTALL_DIR="$FTI_ROOT/install"
fi

# Tutorial steps
cd "$FTI_ROOT"

mkdir -p "$INSTALL_DIR"
mkdir 'build'
cd 'build'

cmake cmake -DCMAKE_INSTALL_PREFIX:PATH="$FTI_INSTALL_DIR" "$@" ..
make -j
make install