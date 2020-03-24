.. Fault Tolerance Library documentation Compilation file


Compilation
===================================================

FTI uses Cmake to configure the installation. The recommended way to perform the installation is to create a build directory within the base directory of FTI and perform the cmake command in there. In the following you will find configuration examples. The commands are performed in the build directory within the FTI base directory.

**Default** The default configuration builds the FTI library with Fortran and MPI-IO support for GNU compilers:

.. code-block:: cmake

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