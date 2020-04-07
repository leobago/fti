.. Fault Tolerance Library documentation Configuration examples file

Configuration Examples
=================================

Default Configuration
----------------------------------
.. code-block::

	[basic]
	head                           = 0
	node_size                      = 2
	ckpt_dir                       = ./Local
	glbl_dir                       = ./Global
	meta_dir                       = ./Meta
	ckpt_l1                        = 3
	ckpt_l2                        = 5
	ckpt_l3                        = 7
	ckpt_l4                        = 11
	dcp_l4                         = 0
	inline_l2                      = 1
	inline_l3                      = 1
	inline_l4                      = 1
	keep_last_ckpt                 = 0
	keep_l4_ckpt                   = 0
	group_size                     = 4
	max_sync_intv                  = 0
	ckpt_io                        = 1
	enable_staging                 = 0
	enable_dcp                     = 0
	dcp_mode                       = 0
	dcp_block_size                 = 16384
	verbosity                      = 2

	[restart]
	failure                        = 0
	exec_id                        = 2018-09-17_09-50-30

	[injection]
	rank                           = 0
	number                         = 0
	position                       = 0
	frequency                      = 0

	[advanced]
	block_size                     = 1024
	transfer_size                  = 16
	general_tag                    = 2612
	ckpt_tag                       = 711
	stage_tag                      = 406
	final_tag                      = 3107
	local_test                     = 1
	lustre_striping_unit           = 4194304
	lustre_striping_factor         = -1
	lustre_striping_offset         = -1

**DESCRIPTION**  

..

   This configuration is made of default values (see: 5). FTI processes are not created (\ ``head = 0``\ , notice: if there is no FTI processes, all post-checkpoints must be done by application processes, thus ``inline_L2``\ , ``inline_L3`` and ``inline_L4`` are set to 1), last checkpoint won’t be kept (\ ``keep_last_ckpt = 0``\ ), ``FTI_Snapshot()`` will take L1 checkpoint every 3 min,L2 - every 5 min, L3 - every 7 min and L4 - every 11 min, FTI will print errors and some few important information (\ ``verbosity = 2``\ ) and IO mode is set to POSIX (\ ``ckpt_io = 1``\ ). This is a normal launch of a job, because failure is set to 0 and ``exec_id`` is ``NULL``. ``local_test = 1`` makes this a local test.  

   Using FTI Processes
-------------------


.. code-block::

   [ Basic ]
   head                        = 1
   node_size                   = 2
   ckpt_dir                    = /scratch/username/
   glbl_dir                    = /work/project/
   meta_dir                    = /home/username/.fti/
   ckpt_L1                     = 3
   ckpt_L2                     = 5
   ckpt_L3                     = 7
   ckpt_L4                     = 11
   inline_L2                   = 0
   inline_L3                   = 0
   inline_L4                   = 0
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

**DESCRIPTION**  

..

   FTI processes are created (\ ``head = 1``\ ) and all post-checkpointing is done by them, thus ``inline_L2``\ , ``inline_L3`` and ``inline_L4`` are set to 0. Note that it is possible to select which checkpoint levels should be post-processed by heads and which by application processes (e.g. ``inline_L2 = 1``\ , ``inline_L3 = 0``\ , ``inline_L4 = 0``\ ). L1 post-checkpoint is always done by application processes, because it’s a local checkpoint. Be aware, when ``head = 1``\ , and ``inline_L2``\ , ``inline_L3`` and ``inline_L4`` are set to 1 all post-checkpoint is still made by application processes.


Using only selected ckpt level with FTI_Snapshot
------------------------------------------------


.. code-block::

   [ Basic ]
   head                        = 0
   node_size                   = 2
   ckpt_dir                    = /scratch/username/
   glbl_dir                    = /work/project/
   meta_dir                    = /home/username/.fti/
   ckpt_L1                     = 0
   ckpt_L2                     = 5
   ckpt_L3                     = 0
   ckpt_L4                     = 0
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

**DESCRIPTION**  

..

   ``FTI_Snapshot()`` will take only L2 checkpoint every 5 min Notice that other configurations are also possible (e.g. take L1 ckpt every 5 min and L4 ckpt every 30 min).


Keeping last checkpoint
-----------------------


.. code-block::

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
   keep_last_ckpt              = 1
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

**DESCRIPTION**  

..

   FTI will keep last checkpoint (\ ``Keep_last_ckpt = 1``\ ), thus after finishing the job Failure will be set to 2. 


Using different IO mode
-----------------------


For instance MPI-I/O:  

.. code-block::

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
   ckpt_io                     = 2
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

**DESCRIPTION**  

..

   FTI IO mode is set to MPI IO (\ ``ckpt_io = 2``\ ). Third option is SIONlib IO mode (\ ``ckpt_io = 3``\ ).  


Restart after a failure
-----------------------


.. code-block::

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
   failure                     = 1
   exec_id                     = 2017-07-26_13-22-11
   [ Advanced ]
   block_size                  = 1024
   transfer_size               = 16
   mpi_tag                     = 2612
   lustre_striping_unit        = 4194304
   lustre_striping_factor      = -1
   lustre_striping_offset      = -1
   local_test                  = 1

**DESCRIPTION**  

..

   This config tells FTI that this job is a restart after a failure (\ ``failure`` set to 1 and ``exec_id`` is some date in a format ``YYYY-MM-DD_HH-mm-ss``\ , where ``YYYY`` - year, ``MM`` - month, ``DD`` - day, ``HH`` - hours, ``mm`` - minutes, ``ss`` - seconds). When recovery is not possible, FTI will abort the job (when using ``FTI_Snapshot()``\ ) and/or signal failed recovery by ``FTI_Status()``. 

