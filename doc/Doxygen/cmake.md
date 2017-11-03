\page cmake CMake configuration

# CMake Options  

Option             |       What it's for                                         | default
-------------------|-------------------------------------------------------------|---------
`ENABLE_FORTRAN`   |  Enables the generation of the Fortran wrapper for FTI      |  ON           
`ENABLE_EXAMPLES`  |  Enables the generation of examples                         |  ON                        
`ENABLE_SIONLIB`   |  Enables the parallel I/O SIONlib for FTI                   |  OFF
`ENABLE_TESTS`     |  Enables the generation of tests                            |  ON
`ENABLE_LUSTRE`    |  Enables Lustre Support                                     |  OFF
`ENABLE_DOCU`      |  Enables the generation of a Doxygen documentation          |  OFF

# Other configurations

To use a certain compiler, set the environment variables `CC` and `FC` to the name of the compiler executables.  
  
For instance if it is desired to use the PGI compiler, try:  

```
    CC=pgcc FC=pgfortran cmake -DCMAKE_INSTALL_PREFIX:PATH=install/here/fti ..
```
  
To use the built-in MD5 rather than OpenSSL, please configure using:  

```
    cmake -DNO_OPENSSL=true -DCMAKE_INSTALL_PREFIX:PATH=/install/here/fti ..
```

On Cray systems, it might be helping to use the cmake flag:   
  
```
    -CMAKE_SYSTEM_NAME=CrayLinuxEnvironment
```

