# FTI Checkpoints Processor

This program allows to read FTI's checkpoints from an external Python script. \
Input: FTI's configuration file and id of rank \
Output: application data in CSV format.

This is an initial version of FTI's Checkpoint Processor.
This version can only read checkpoints if saved to Level 1 of FTI. Next version will be accommodated to all levels.

## Pre-requisites

This version processes checkpoints written with POSIX mode.

In FTI's configuration file, have the following parameter set as follows: 

ckpt_io = 1


## Usage

```python
import read_fti_checkpoints

read_fti_checkpoints.read_checkpoints(config_file, rank_id) 

```

