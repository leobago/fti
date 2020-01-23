Post process FTI Posix files
===

A simple library that allows users to read data from POSIX checkpoint files and extract data values from them. To build the utilities pass the -DENABLE_TOOLS=1 flag on the cmake command.  

There is an example on how a use can use it and read variables from all checkpoint files produced by an application execution. The user passes as a first parameter the configuration file path and as a second parameter a directory. For each performed checkpoint the example will export each variable on a separate file with the name "Ckpt_"ID"_"VarId".mpio.

The example mainly serves as a quick demo on how to access all the variables from all the checkpoints of an application execution. 

