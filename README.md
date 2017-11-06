What is FTI?
===

FTI stands for Fault Tolerance Interface and is a library that aims to give
computational scientists the means to perform fast and efficient multilevel
checkpointing in large scale supercomputers. FTI leverages local storage plus
data replication and erasure codes to provide several levels of reliability and
performance. FTI is application-level checkpointing and allows users to select
which datasets needs to be protected, in order to improve efficiency and avoid
wasting space, time and energy. In addition, it offers a direct data interface
so that users do not need to deal with files and/or directory names.  All
metadata is managed by FTI in a transparent fashion for the user. If desired,
users can dedicate one process per node to overlap fault tolerance workload and
scientific computation, so that post-checkpoint tasks are executed
asynchronously.

---

Download, compile and install FTI (as easy as 1,2,3)
===

 1) git clone https://github.com/leobago/fti.git
 2) mkdir fti/build && cd fti/build
 3) cmake -DCMAKE_INSTALL_PREFIX:PATH=/install/here/fti .. && make all install

> **REMARK 1** (Intel and GCC)  
> For the case that both, **Intel and GCC**, compilers are installed, please configure using:  
> `cmake -C ../intel.cmake -DCMAKE_INSTALL_PREFIX:PATH=/install/here/fti ..`

> **REMARK 2** (OpenSSL)  
> To use the built-in MD5 rather than OpenSSL, please configure using:  
> `cmake -DNO_OPENSSL=true -DCMAKE_INSTALL_PREFIX:PATH=/install/here/fti ..`

> **REMARK 3** (GNU versions)  
> The usage of different GNU compiler versions for C and Fortran leads currently to an undefined behavior. Please make sure the compiler identification for C and Fortran is the same.

> **REMARK 4** (Cray System)  
> FTI works on Cray system with these modules  
> GNU environment:  
> `module load gcc/5.3.0 CMake/3.6.2 craype/2.5.8 cray-mpich/7.5.0 PrgEnv-gnu/6.0.3 `  
> `export CRAY_CPU_TARGET=x86-64`  
> `export CRAYPE_LINK_TYPE=dynamic`  
> Flag for CMake: `-CMAKE_SYSTEM_NAME=CrayLinuxEnvironment`  
>  
> Intel environment:  
> `module load intel/17.0.1.132 CMake/3.6.2 craype/2.5.8 cray-mpich/7.5.0 PrgEnv-intel/6.0.3`  
> `export CRAY_CPU_TARGET=x86-64`  
> `export CRAYPE_LINK_TYPE=dynamic`  
> Flag for CMake: `-CMAKE_SYSTEM_NAME=CrayLinuxEnvironment`  
>  
> The most important is CMake version: the newer the better.  

---

Configure and run a FTI example
===

The "build/examples" directory contains heat distribution simulations as simple
examples in both, C and Fortran. Usage instructions in file "examples/README".

---

User manual
===

In folder "doc/manual" you will find a user manual, which contains the API description and code snippets for the implementation of FTI as checkpoint I/O. 
  
To generate the documentation wit Doxygen, configure with `-DENABLE_DOCU=ON` and execute in the build directory:  
```
    make doc  
```

---

Acknowledgement (send us a postal card! \\\(^-^\)/\)
===

If you use FTI please consider sending us an email to let us know what you
liked and what could be improved ( :email: leonardo (dot) bautista (at) bsc (dot) es), 
your feedback is important. 

If you use FTI for any research work, please make sure to acknowledge our paper:
Bautista-Gomez, Leonardo, et al. ***"FTI: high performance fault tolerance interface 
for hybrid systems."*** Proceedings of 2011 international conference for high 
performance computing, networking, storage and analysis. ACM, 2011.  

Finally, don't hesitate to send us a postal card to :
Dr. Leonardo Bautista-Gomez (Leo)
Barcelona Supercomputing Center
Carrer de Jordi Girona, 29-31, 08034 Barcelona, SPAIN.
Phone :telephone_receiver: : +34 934 13 77 16
