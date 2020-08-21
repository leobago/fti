# FTI Checkpoints Processor

This program allows to read FTI's checkpoints from an external Python script. \
Input: FTI's configuration file \
Output: application data in CSV format

This is an initial version of FTI's checkpoint processor
This version works with HeatDistribution application found in tutorial/ directory 
of FTI. Later versions will have support for any application. 

## Pre-requisites

This version processes checkpoints written with POSIX mode.
Pre-requisites: (to be relaxed in the next version)

In FTI In configuration file, have the following parameters set as follows: \
ckpt_io = 1; \
ckpt_dir = ./local; \
meta_dir = ./meta; 

## Usage

```python
import read_fti_checkpoints

read_fti_checkpoints.read_checkpoints(config_file) 

```

