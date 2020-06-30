.. Fault Tolerance Library documentation Compilation file


Compilation
===================================================

FTI uses CMake as the build manager to configure and perform the installation.
We also provide an installation script, **install.sh**, in the root directory to faciliate this process.
Both processes are identical, the script merely wraps some common cases and configurations as options.
If your preffered way to build FTI is to use CMake, we recommend to do an out-of-source build.
The following bash code snippets showcase how to build FTI with a given prefix path.

**Default** The default configuration builds the FTI library with Fortran and MPI-IO support for GNU compilers:

.. code-block:: cmake

   mkdir build && cd build
   cmake -DCMAKE_INSTALL_PREFIX:PATH=/install/here/fti ..
   make all install

.. note::
	Notice: THE TWO DOTS AT THE END INVOKE CMAKE IN THE TOP LEVEL DIRECTORY.

**Intel compilers** Fortran and MPI-IO support for Intel compilers:

.. code-block:: cmake

   cmake -C ../intel.cmake -DCMAKE_INSTALL_PREFIX:PATH=/install/here/fti ..
   make all install

**Disable Fortran** Only build FTI C library:

.. code-block:: cmake

   cmake -DCMAKE_INSTALL_PREFIX:PATH=/install/here/fti -DENABLE_FORTRAN=OFF ..
   make all install

**Lustre** For Lustre user who want to use MPI-IO, it is strongly recommended to configure with Lustre support:

.. code-block:: cmake

	cmake -DCMAKE_INSTALL_PREFIX:PATH=/install/here/fti -DENABLE_LUSTRE=ON ..
	make all install


**Cray** For Cray systems, make sure that the modules craype/* and PrgEnv* are loaded (if available). The configuration should be done as:

.. code-block:: cmake

	export CRAY_CPU_TARGET=x86-64
	export CRAYPE_LINK_TYPE=dynamic
	cmake -DCMAKE_INSTALL_PREFIX:PATH=/install/here/fti -DCMAKE_SYSTEM_NAME=CrayLinuxEnvironment ..
	make all install

.. note::
	Notice: MODIFY x86-64 IF YOU ARE USING A DIFFERENT ARCHITECTURE. ALSO, THE OPTION CMAKE_SYSTEM_NAME=CrayLinuxEnvironment IS AVAILABLE ONLY FOR CMAKE VERSIONS 3.5.2 AND ABOVE.


Installing additional IO libraries
===================================================

FTI can work alongside other IO libraries when creating checkpoint files.
Currently, FTI has support for HDF5 and SIONLib.
These libraries can be linked to FTI through CMake options.


HDF5
--------------

FTI can use the `HDF5 library and format <https://www.hdfgroup.org/solutions/hdf5>`_ to generate checkpoint files.
FTI is compatible with the parallel version of the HDF5 library.
Usually package managers have both both HDF5 versions available, so make sure the one installed is the correct one.
Moreover, if you need to compile HDF5 from source, make sure to supply the following option: **--enable-parallel**.
HDF5 support is enabled passing the *ENABLE_HDF5* option to CMake.
Alternatively, the **--enable-hdf5** option can be informed to the install script.

.. code-block:: bash

  mkdir build && cd build
	cmake -DENABLE_HDF5=1 ..
	make all install
  # Or, alternatively
  ./install.sh --enable-hdf5


SIONLib
--------------

FTI also supports `SIONLib <https://www.fz-juelich.de/ias/jsc/EN/Expertise/Support/Software/SIONlib/_node.html>`_ as the IO library.
As it is with HDF5, FTI must be linked with the parallel version of SIONLib.
Inform the *ENABLE_SIONLIB* option to CMake in order to link FTI with SIONLib.
If necessary, use the *SIONLIBBASE* CMake option to assist the linker in finding the library.
The bash script snippet below showcase the commands for a build where SIONLib is installed at */opt/sionlib*.

.. code-block:: bash

  mkdir build && cd build
	cmake -DENABLE_SIONLIB=1 DSIONLIBBASE=/opt/sionlib ..
	make all install
  # Or, alternatively
  ./install.sh --enable-sionlib --sionlib-path=/opt/sionlib
