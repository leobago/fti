.. Fault Tolerance Library documentation Multi-level Ckpt file


Multi-level Checkpointing
===================================================


L1
---------------------------------------------------
L1 denotes the first safety level in the multilevel checkpointing strategy of FTI. The checkpoint of each process is written on the local SSD of the respective node. This is fast but possesses the drawback, that in case of a data loss and corrupted checkpoint data even in only one node, the execution cannot successfully restarted.

L2
---------------------------------------------------
L2 denotes the second safety level of checkpointing. On initialisation, FTI creates a virtual ring for each group of nodes with user defined size (see group_size). The first step of L2 is just a L1 checkpoint. In the second step, the checkpoints are duplicated and the copies stored on the neighbouring node in the group.

That means, in case of a failure and data loss in the nodes, the execution still can be successfully restarted, as long as the data loss does not happen on two neighbouring nodes at the same time.

L3
---------------------------------------------------
L3 denotes the third safety level of checkpointing. In this level, the check- point data trunks from each node getting encoded via the Reed-Solomon (RS) erasure code. The implementation in FTI can tolerate the breakdown and data loss in half of the nodes.

In contrast to the safety level L2, in level L3 it is irrelevant which of nodes encounters the failure. The missing data can get reconstructed from the remaining RS-encoded data files.

L4
---------------------------------------------------
L4 denotes the fourth safety level of checkpointing. All the checkpoint files are flushed to the parallel file system (PFS).