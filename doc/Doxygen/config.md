\page config Default Configuration File

# Default Configuration File

```
[ Basic ]
head                        = 0
node_size                   = 2
ckpt_dir                    = /scratch/username/
glbl_dir                    = /work/project/
meta_dir                    = /home/username/.fti/
ckpt_L1                     = 3
ckpt_L2                     = 5
ckpt_L3                     = 7
ckpt_L4                     = 11
inline_L2                   = 1
inline_L3                   = 1
inline_L4                   = 1
keep_last_ckpt              = 0
group_size                  = 4
max_sync_intv               = 0
ckpt_io                     = 1
verbosity                   = 2
[ Restart ]
failure                     = 0
exec_id                     = NULL
[ Advanced ]
block_size                  = 1024
transfer_size               = 16
mpi_tag                     = 2612
lustre_striping_unit        = 4194304
lustre_striping_factor      = -1
lustre_striping_offset      = -1
local_test                  = 1
```
