# FTI Checkpoints Processor

This is an initial version of FTI's Checkpoint Processor. \
This program allows to read FTI's checkpoints from an external Python script. 

*Input* \
config_file			: FTI's configuration file \
rank_id    			: Id of rank \
ranks (optional)	: Total ranks (if MPI-IO level 4) \
level(optional)     : FTI's level (1, 2, 3, 4) \
					 if not provided, read from last checkpoints. \
output 				: Output format (CSV (default), HDF5, numpy array)



## Pre-requisites

This version processes checkpoints written with POSIX/MPI-IO mode. \
It supports multi-dimensional arrays of data. \
In FTI's configuration file, have the following parameter set as follows: \


## Usage

```python
import read_fti_checkpoints as ckpt

ckpt.read_checkpoints(config_file, rank_id, ranks=None, level=None, output=None)

```

For processing MPI-IO checkpoints, compile the program under /mpiio, for example: 
```
mpiic mpiio_main.c -o mpiio_main
```

## Examples

The /examples sub-folder contains same usage of the checkpoint processor.