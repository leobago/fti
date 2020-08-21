# FTI Checkpoints Processor

This program allows to read FTI's checkpoints from an external Python script. \
Input: application root folder (application that is checkpointed using FTI) \
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
meta_dir = ./meta; \

## Usage

```python
import read_fti_checkpoints

read_fti_checkpoints.read_checkpoints(app_root) # returns 'words'

```

Note: "app_root" is the directory where the application resides. It is important to have the meta/ and local/ folders directly under app_root for this version to work.
