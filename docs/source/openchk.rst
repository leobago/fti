.. Fault Tolerance Library documentation OpenCHK file


What is OpenCHK?
===================================================
OpenCHK is a pragma based interface that facilitates the implementation of checkpoint-and-restart (C/R) into MPI based applications. The interface acts as an abstraction layer between the application and the C/R library. Besides FTI, OpenCHK supports several other C/R libraries. The supported libraries are recognized by the compiler, so that explicit linking becomes no longer necessary. The C/R library is chosen for each run by setting the corresponding environment variable.

Where to get OpenCHK?
===================================================
OpenCHK is developed and maintained at the BSC. It is open-source and can be downloaded from: https://github.com/bsc-pm/OpenCHK-model