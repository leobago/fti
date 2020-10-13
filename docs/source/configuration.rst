.. Fault Tolerance Library documentation Configuration file
.. _configuration:

Configuration
=================

[Basic]
-------

head
^^^^


..

   The checkpointing safety levels L2, L3 and L4 produce additional overhead due to the necessary postprocessing work on the checkpoints. FTI offers the possibility to create an MPI process, called HEAD, in which this postprocessing will be accomplished. This allows it for the application processes to continue the execution immediately after the checkpointing.  


.. list-table::
   :header-rows: 1

   * - Value
     - Meaning
   * - 0
     - The checkpoint postprocessing work is covered by the application processes
   * - 1
     - The HEAD process accomplishes the checkpoint postprocessing work (notice: In this case, the number of application processes will be (n-1)/node)


(\ *default = 0*\ )  

node_size
^^^^^^^^^


..

   Lets FTI know, how many processes will run on each node (ppn). In most cases this will be the amount of processing units within the node (e.g. 2 CPU’s/node and 8 cores/CPU ! 16 processes/node).  


.. list-table::
   :header-rows: 1

   * - Value
     - Meaning
   * - ppn (int > 0)
     - Number of processing units within each node (notice: The total number of processes must be a multiple of ``group_size*node_size``\ )


(\ *default = 2*\ )  

ckpt_dir
^^^^^^^^


..

   This entry defines the path to the local hard drive on the nodes. 


.. list-table::
   :header-rows: 1

   * - Value
     - Meaning
   * - string
     - Path to the local hard drive on the nodes


(\ *default = /scratch/username/*\ )  

glbl_dir
^^^^^^^^


..

   This entry defines the path to the checkpoint folder on the PFS (L4 checkpoints).  


.. list-table::
   :header-rows: 1

   * - Value
     - Meaning
   * - string
     - Path to the checkpoint directory on the PFS


(\ *default = /work/project/*\ )  

meta_dir
^^^^^^^^


..

   This entry defines the path to the meta files directory. The directory has to be accessible from each node. It keeps files with information about the topology of the execution.  


.. list-table::
   :header-rows: 1

   * - Value
     - Meaning
   * - string
     - Path to the meta files directory


(\ *default = /home/user/.fti*\ )  

ckpt_L1
^^^^^^^


..

   Here, the user sets the checkpoint frequency of L1 checkpoints when using ``FTI_Snapshot()``.


.. list-table::
   :header-rows: 1

   * - Value
     - Meaning
   * - L1 intv. (int \>= 0)
     - L1 checkpointing interval in minutes
   * - 0
     - Disable L1 checkpointing


(\ *default = 3*\ )  

ckpt_L2
^^^^^^^


..

   Here, the user sets the checkpoint frequency of L2 checkpoints when using ``FTI_Snapshot()``.


.. list-table::
   :header-rows: 1

   * - Value
     - Meaning
   * - L2 intv. (int \>= 0)
     - L2 checkpointing interval in minutes
   * - 0
     - Disable L2 checkpointing


(\ *default = 5*\ )  

ckpt_L3
^^^^^^^


..

   Here, the user sets the checkpoint frequency of L3 checkpoints when using ``FTI_Snapshot()``.


.. list-table::
   :header-rows: 1

   * - Value
     - Meaning
   * - L3 intv. (int \>= 0)
     - L3 checkpointing interval in minutes
   * - 0
     - Disable L3 checkpointing


(\ *default = 7*\ )  

ckpt_L4
^^^^^^^


..

   Here, the user sets the checkpoint frequency of L4 checkpoints when using ``FTI_Snapshot()``.


.. list-table::
   :header-rows: 1

   * - Value
     - Meaning
   * - L4 intv. (int \>= 0)
     - L4 checkpointing interval in minutes
   * - 0
     - Disable L4 checkpointing


(\ *default = 11*\ )  

dcp_L4
^^^^^^


..

   Here, the user sets the checkpoint frequency of L4 differential checkpoints when using ``FTI_Snapshot()``.


.. list-table::
   :header-rows: 1

   * - Value
     - Meaning
   * - L4 dCP intv. (int \>= 0)
     - L4 dCP checkpointing interval in minutes
   * - 0
     - Disable L4 dCP checkpointing


(\ *default = 0*\ )  

inline_L2
^^^^^^^^^


..

   In this setting, the user chose whether the post-processing work on the L2 checkpoints is done by an FTI process or by the application process.


.. list-table::
   :header-rows: 1

   * - Value
     - Meaning
   * - 0
     - The post-processing work of the L2 checkpoints is done by an FTI process (notice: This setting is only alowed if head = 1)
   * - 1
     - The post-processing work of the L2 checkpoints is done by the application process


(\ *default = 1*\ )  

inline_L3
^^^^^^^^^


..

   In this setting, the user chose whether the post-processing work on the L3 checkpoints is done by an FTI process or by the application process.


.. list-table::
   :header-rows: 1

   * - Value
     - Meaning
   * - 0
     - The post-processing work of the L3 checkpoints is done by an FTI process (notice: This setting is only alowed if head = 1)
   * - 1
     - The post-processing work of the L3 checkpoints is done by the application process


(\ *default = 1*\ )  

inline_L4
^^^^^^^^^


..

   In this setting, the user chose whether the post-processing work on the L4 checkpoints is done by an FTI process or by the application process.


.. list-table::
   :header-rows: 1

   * - Value
     - Meaning
   * - 0
     - The post-processing work of the L4 checkpoints is done by an FTI process (notice: This setting is only alowed if head = 1)
   * - 1
     - The post-processing work of the L4 checkpoints is done by the application process


(\ *default = 1*\ )  

keep_last_ckpt
^^^^^^^^^^^^^^


..

   This setting tells FTI whether the last checkpoint taken during the execution will be kept in the case of a successful run or not.


.. list-table::
   :header-rows: 1

   * - Value
     - Meaning
   * - 0
     - During ``FTI_Finalize()``\ , all checkpoints will be removed (except case 'keep_l4_ckpt=1')
   * - 1
     - After ``FTI_Finalize()``\ , the last checkpoint will be kept and stored on the PFS as a L4 checkpoint (notice: Additionally, the setting failure in the configuration file is set to 2. This will lead to a restart from the last checkpoint if the application is executed again)


(\ *default = 0*\ )  

keep_l4_ckpt
^^^^^^^^^^^^


..

   This setting triggers FTI to keep all level 4 checkpoints taken during the execution. The checkpoint files will be saved in `glbl_dir <Configuration#glbl_dir>`_\ /l4_archive.


.. list-table::
   :header-rows: 1

   * - Value
     - Meaning
   * - 0
     - During ``FTI_Finalize()``\ , all checkpoints will be removed (except case 'keep_last_ckpt=1')
   * - 1
     - All level 4 checkpoints taken during the execution, will be stored under ``glbl_dir/l4_archive``. This folder will not be deleted during the ``FTI_Finalize()`` call.


(\ *default = 0*\ )  

group_size
^^^^^^^^^^


..

   The group size entry sets, how many nodes (members) forming a group.


.. list-table::
   :header-rows: 1

   * - Value
     - Meaning
   * - int i (2 \<= i \<= 32)
     - Number of nodes contained in a group (notice: The total number of processes must be a multiple of ``group_size*node_size``\ )


(\ *default = 4*\ )  

max_sync_intv
^^^^^^^^^^^^^


..

   Sets the maximum number of iterations between synchronisations of the iteration length (used for ``FTI_Snapshot()``\ ). Internally the value will be rounded to the next lower value which is a power of 2.


.. list-table::
   :header-rows: 1

   * - Value
     - Meaning
   * - int i (0 \<= i \<= INT_MAX )
     - maximum number of iterations between measurements of the global mean iteration time (\ ``MPI_Allreduce`` call)
   * - 0
     - Sets the value to 512, the default value for FTI


(\ *default = 0*\ )  

ckpt_io
^^^^^^^


..

   Sets the I/O mode.


.. list-table::
   :header-rows: 1

   * - Value
     - Meaning
   * - 1
     - POSIX I/O mode
   * - 2
     - MPI-IO I/O mode
   * - 3
     - FTI-FF I/O mode
   * - 4
     - SIONLib I/O mode
   * - 5
     - HDF5 I/O mode


(\ *default = 1*\ )  

enable_staging
^^^^^^^^^^^^^^

..

   Enable the staging feature. This feature allows to stage files asynchronously from local (e.g. node local NVMe storage) to the PFS. FTI offers the API functions `FTI_SendFile <API-Reference#fti_sendfile>`_\ , `FTI_GetStageDir <API-Reference#fti_getstagedir>`_ and `FTI_GetStageStatus <API-Reference#FTI_getstagestatus>`_ for that.


.. list-table::
   :header-rows: 1

   * - Value
     - Meaning
   * - 0
     - Staging disabled
   * - 1
     - Stagin enabled (creation of the staging directory in folde 'ckpt_dir')


(\ *default = 0*\ )  

enable_dcp
^^^^^^^^^^


..

   Enable differential checkpointing. In order to use this feature, `ckpt_io <Configuration#ckpt_io>`_ has to be set to 3 (FTI-FF). To trigger differential checkpoints, use either level ``FTI_L4_DCP`` in `FTI_Checkpoint <API-Reference#fti_checkpoint>`_ or set the interval in `dcp_L4 <Configuration#dcp_L4>`_ for usage in `FTI_Snapshot <API-Reference#fti_snapshot>`_.


.. list-table::
   :header-rows: 1

   * - Value
     - Meaning
   * - 0
     - dCP disabled
   * - 1
     - dCP enabled


dcp_mode
^^^^^^^^


..

   Set the hash algorithm used for differential checkpointing.


.. list-table::
   :header-rows: 1

   * - Value
     - Meaning
   * - 0
     - MD5
   * - 1
     - CRC32


(\ *default = 0*\ )  

dcp_block_size
^^^^^^^^^^^^^^


..

   Set the desired partition block size for differential checkpointing in bytes. The block size must be within 512 .. ``USHRT_MAX`` (65535 on most systems). 


.. list-table::
   :header-rows: 1

   * - Value
     - Meaning
   * - b (512 \<= i \<= USHRT_MAX)
     - block size for dataset partition for dCP


(\ *default = 16384*\ )  

verbosity
^^^^^^^^^


..

   Sets the level of verbosity.


.. list-table::
   :header-rows: 1

   * - Value
     - Meaning
   * - 1
     - Debug sensitive. Beside warnings, errors and information, FTI debugging information will be printed
   * - 2
     - Information sensitive. FTI prints warnings, errors and information
   * - 3
     - FTI prints only warnings and errors
   * - 4
     - FTI prints only errors


(\ *default = 2*\ )  

[Restart]
---------

failure
^^^^^^^


..

   This setting should mainly set by FTI itself. The behaviour within FTI is the following:
     


   * Within ``FTI_Init()``\ , it remains on it initial value.
   * After the first checkpoint is taken, it is set to 1.
   * After ``FTI_Finalize()`` and ``keep_last_ckpt`` = 0, it is set to 0.
   * After ``FTI_Finalize()`` and ``keep_last_ckpt`` = 1, it is set to 2.


.. list-table::
   :header-rows: 1

   * - Value
     - Meaning
   * - 0
     - The application starts with its initial conditions (notice: In order to force a clean start, the value may be set to 0 manually. In this case the user has to take care about removing the checkpoint data from the last execution)
   * - 1
     - FTI is searching for checkpoints and starts from the highest checkpoint level (notice: If no readable checkpoints are found, the execution stops)
   * - 2
     - FTI is searching for the last L4 checkpoint and restarts the execution from there (notice: If checkpoint is not L4 or checkpoint is not readable, the execution stops)


(\ *default = 0*\ )  

exec_id
^^^^^^^


..

   This setting should mainly set by FTI itself. During ``FTI_Init()`` the execution ID is set if the application starts for the first time (failure = 0) or the execution ID is used by FTI in order to find the checkpoint files for the case of a restart (\ ``failure`` = 1,2)


.. list-table::
   :header-rows: 1

   * - Value
     - Meaning
   * - yyyy-mm-dd_hh-mm-ss
     - Execution ID (notice: If variate checkpoint data is available, the execution ID may set by the user to assign the desired starting point)


(\ *default = NULL*\ )  

[Advanced]
----------

The settings in this section, should **ONLY** be changed by advanced users.  

block_size
^^^^^^^^^^


..

   FTI temporarily copies small blocks of the L2 and L3 checkpoints to send them through MPI. The size of the data blocks can be set here.


.. list-table::
   :header-rows: 1

   * - Value
     - Meaning
   * - int
     - Size in KB of the data blocks send by FTI through MPI for the checkpoint levels L2 and L3


(\ *default = 1024*\ )  

transfer_size
^^^^^^^^^^^^^


..

   FTI transfers in chunks local checkpoint files to PFS. The size of the chunk can be set here.


.. list-table::
   :header-rows: 1

   * - Value
     - Meaning
   * - int
     - Size in MB of the chunks send by FTI from local to PFS


(\ *default = 16*\ )  

general_tag
^^^^^^^^^^^


..

   FTI uses a certain tags for the MPI messages. The tag for general messages can be set here.


.. list-table::
   :header-rows: 1

   * - Value
     - Meaning
   * - int
     - Tag, used for general MPI messages within FTI


(\ *default = 2612*\ )  

ckpt_tag
^^^^^^^^


..

   FTI uses a certain tags for the MPI messages. The tag for messages related to checkpoint communication can be set here.


.. list-table::
   :header-rows: 1

   * - Value
     - Meaning
   * - int
     - Tag, used for MPI messages related to a checkpoint context within FTI


(\ *default = 711*\ )  

stage_tag
^^^^^^^^^


..

   FTI uses a certain tags for the MPI messages. The tag for messages related to staging communication can be set here.


.. list-table::
   :header-rows: 1

   * - Value
     - Meaning
   * - int
     - Tag, used for MPI messages related to a staging context within FTI


(\ *default = 406*\ )  

final_tag
^^^^^^^^^


..

   FTI uses a certain tags for the MPI messages. The tag for the message to the heads to trigger the end of the execution can be set here.


.. list-table::
   :header-rows: 1

   * - Value
     - Meaning
   * - int
     - Tag, used for the MPI message that marks the end of the execution send from application processes to the heads within FTI


(\ *default = 3107*\ )  

lustre_striping_unit
^^^^^^^^^^^^^^^^^^^^


..

   This option only impacts if ``-DENABLE_LUSTRE`` was added to the Cmake command. It sets the striping unit for the MPI-IO file.


.. list-table::
   :header-rows: 1

   * - Value
     - Meaning
   * - int i (0 \<= i \<= INT_MAX )
     - Striping size in Bytes. The default in Lustre systems is 1MB (1048576 Bytes), FTI uses 4MB (4194304 Bytes) as the dafault value
   * - 0
     - Assigns the Lustre default value


(\ *default = 4194304*\ )  

lustre_striping_factor
^^^^^^^^^^^^^^^^^^^^^^


..

   This option only impacts if ``-DENABLE_LUSTRE`` was added to the Cmake command. It sets the striping factor for the MPI-IO file.


.. list-table::
   :header-rows: 1

   * - Value
     - Meaning
   * - int i (0 \<= i \<= INT_MAX )
     - Striping factor. The striping factor determines the number of OST’s to use for striping.
   * - -1
     - Stripe over all available OST’s. This is the default in FTI.
   * - 0
     - Assigns the Lustre default value


(\ *default = -1*\ )  

lustre_striping_offset
^^^^^^^^^^^^^^^^^^^^^^


..

   This option only impacts if ``-DENABLE_LUSTRE`` was added to the Cmake command. It sets the striping offset for the MPI-IO file.


.. list-table::
   :header-rows: 1

   * - Value
     - Meaning
   * - int i (0 \<= i \<= INT_MAX )
     - Striping offset. The striping offset selects a particular OST to begin striping at.
   * - -1
     - Assigns the Lustre default value


(\ *default = -1*\ )  

local_test
^^^^^^^^^^


..

   FTI is building the topology of the execution, by determining the hostnames of the nodes on which each process runs. Depending on the settings for ``group_size``\ , ``node_size`` and ``head``\ , FTI assigns each particular process to a group and decides which process will be Head or Application dedicated. This is meant to be a local test. In certain situations (e.g. to run FTI on a local machine) it is necessary to disable this function.


.. list-table::
   :header-rows: 1

   * - Value
     - Meaning
   * - 0
     - Local test is disabled. FTI will simulate the situation set in the configuration
   * - 1
     - Local test is enabled (notice: FTI will check if the settings are correct on initialization and if necessary stop the execution)


(\ *default = 1*\ )  


fast_forward
^^^^^^^^^^


..

   This parameter allows the checkpoint interval to be speeded up by the value given to this parameter. In other words, the interval is divided by the fast_forward value. For example, if ckpt_l1 is set to 15, a fast_forward configuration of 5 will result in L1 checkpoints every 3 minutes. A fast_forward rate of 1 keeps the same checkpoint interval frequency.


.. list-table::
   :header-rows: 1

   * - Value
     - Meaning
   * - int
     - Fast forward rate. Must be between 1 and 10.


(\ *default = 1*\ )  



