In this page we present a tutorial of FTI.
It also presents a practice section.
The tutorials present a overall description of the library.
The practice section is for familiarization with the library's API and configuration files.
Therefore, information on how to proceed in this section is limited.


#  Installation 

## Preparation

1. Open a terminal, navigate to a folder of your preference and Download FTI
```bash
cd to/desired/path
git clone https://github.com/leobago/fti 
```

2. Change directory to the fti root directory
```bash
cd fti
```

3. Optionally, set the `FTI_INSTALL_DIR` variable with an **absolute path** leading to the desired installation folder.
```base
export FTI_INSTALL_DIR=/my/home/fti
```

## Configure and Install

1. Execute the installation script
```bash
scripts/install.sh '-DENABLE_TUTORIAL=1'
```

This script creates a 'install' and 'build' folder within FTI root directory.
The install folders will contain the compiled FTI library and header files.
If you provided a value for the `FTI_INSTALL_DIR` variable, the 'install' folder will not be created.
Instead, you will find the FTI binary and headers in the location you specified.
The `build` folder will be populated with some installation files and the tutorial binaries.

The `install.sh` script, used in this step, redirects all parameters to the cmake compilation command.
As observed above, the flag -DENABLE\_TUTORIAL=1 will build the tutorial files as well as FTI.


## Executables, tutorial source code, and fti library files

The library is installed at the $FTI\_INSTALL\_DIR or 'install' directory if the variable is not set.
The FTI library source code is under the src/ directory.
The tutorial source code is under tutorial.
The tutorial executables are under build/tutorial/.
For conveniency, on the rest of this tutorial, we refer to these paths by the following variables:

```bash
export FTI_ROOT=$PWD
export TUTORIAL_EXEC=${FTI_ROOT}/build/tutorial/
export TUTORIAL_SRC=${FTI_ROOT}/tutorial/
```

You should always export these variables every time you start/continue the tutorial in another terminal.
Under the ${TUTORIAL\_SRC} directory you can find various directories.
Each of which corresponds to a step presented in the tutorial.


# Demonstration of FTI

We demonstrate an example of the API function 'FTI\_Snapshot()' to showcase the various safety levels of FTI.
To observe it, run the example in each case for at least one minute.
Then, interrupt the execution by pressing `ctrl+c` in the terminal.
In some systems `ctrl+c` does not kill all executing MPI processes.
If this is the case in your machine, use the following command to kill all processes.

```bash
killall 'executable'
```


## L1 - Local checkpoint on the nodes

Change directory to ${TUTORIAL\_EXEC}/L1 and run the execution with `make hdl1`.
This will launch an mpi application protected by FTI with mpirun.
FTI will store the protected variables in the 'local' folder in this directory.
You may follow the checkpointing events by observing the contents in this folder while the program is running.
For that, before executing, open another terminal and use one of the following commands.

```bash
watch -n 1 "find local"
```
or
```bash
cd local; watch -n 1 "ls -lR"
```
or
```bash
watch -n 1 "du -kh local"
```

Interrupt the application after some execution time.
Then run the application again with `make hdl1`.
FTI will attemp to resume from the last checkpoint.
If successfull, it will output the following message to the terminal.

```
[ FTI  Information ] : This is a restart. The execution ID is: YYYY-MM-DD_hh-mm-ss
[ FTI  Information ] : Recovering successfully from level 1 with Ckpt. 1. 
```

Take some time to look into the 'meta/' folder and the 'config.L1.fti' configuration file.
Note that FTI detects a failure and the last previous execution ID annotated in these files.
After the successful restart, interrupt the execution and delete one of the checkpoint files.
The files are stored in: ${TUTORIAL\_EXEC}/L1//local/<NODE>/<EXEC-ID>/l1/.
They are names as ckpt<ID>-Rank<RANK>.fti.
You can also simply delete the whole node directory.
You will notice that, after deletion, the program won’t be able to resume the execution.


## L2 – local checkpoint on the nodes + copy to the neighbor node:

Change directory to the L2 example in folder ${TUTORIAL\_EXEC}/L2.
Then, run the application with `make hdl2`.
As before, you can **watch** the contents of the 'local' folder while the program is running.

Interrupt the execution after a checkpoint is made.
Restart the application again with `make hdl2`.
Notice that the application resumes from where the checkpoint was taken as in the L1 example.
Now, interrupt the execution once again.

This time, before executing the application we will go into the local folder and remove **one** of the node folders.
You will notice that the program resumes execution despite the missing node files.
Now, try to delete more than one file/directories in different combinations and answer the following questions.


### Questions: In order to keep the execution able to resume:
1. How many files you can delete?

<details>
<summary>Try to figure it out yourself first</summary>

> You can delete up to half of the node folders in the best case scenario.

</details>

2. Which files can you delete?

<details>
<summary>Try to figure it out yourself first</summary>

> The L2 recovery works as long as all data can be reconstructed from the neighbouring nodes.
> You can observe this by deleting the **node0** and **node2** folders.
> Notice that, in this scenario, **node1** will be able to restore the **node0** checkpoint.
> Also, **node3** will be able to restore the **node2** checkpoint.
> 
> A failure to restore the checkpoint can happen deleting less files however.
> You can do this by deleting the **node0** folder and one of the **CkptX-PcofY.fti** files in **node1**.
> This scenario represents a node failure and a corruption in the neighbor's copy (the Pcof file).
> Notice that, if you delete a **CkptX-RankY.fti** file instead in **node1**, FTI is able to restore.
> This happens because **node0** files can be reconstructed in node 1 and **node2** can restore the missing **node1** file.
> This is the worst case scenario where the failure of only two nodes will cause the checkpoints to be lost.

</details>


## L3 – local checkpoint on the nodes + RS encoding:

Change directory to the L3 example in folder ${TUTORIAL\_EXEC}/L3.
Then, run the application with `make hdl3`.
As before, you can **watch** the contents of the 'local' folder while the program is running.

Interrupt the execution after a checkpoint is made.
Restart the application again with `make hdl3`.
Notice that the application resumes from where the checkpoint was taken as in the L1 and L2 examples.
Now, interrupt the execution once again.

This time, before executing the application we will go into the local folder and remove **two** of the node folders.
You will notice that the program resumes execution despite the missing node files.
Now, try to delete more than one file/directories in different combinations and answer the following questions.

### Questions: In order to keep the execution able to resume:
1. How many files you can delete?

<details>
<summary>Try to figure it out yourself first</summary>

> You can delete up to half of the node folders in any scenario.

</details>

2. Which files can you delete?

<details>
<summary>Try to figure it out yourself first</summary>

> The L3 recovery works as long all the data can be recovered from the Reed Solomon encodings.
> You can observe that deleting **any two nodes** will not prevent FTI from recovering the application state.
> 
> FTI groups the MPI ranks and guarantees that each node will contain exactly 1 process of each group.
> It uses Reed Solomon to encode the checkpoints for each group so that they whitstand a loss of at most half of the data.
> As such, since there is one process in each group, information can be reconstructed if, at most, half of the nodes fail.
> You can force a failure by deleting **any two node folders** and then **any other file** in another remaining node.

</details>


## L4 – flush of the checkpoints to the parallel file system:

Change directory to the L4 example in folder ${TUTORIAL\_EXEC}/L4.
Then, run the application with `make hdl4`.
As before, you can **watch** the contents of the 'local' folder while the program is running.

Interrupt the execution after a checkpoint is made.
Restart the application again with `make hdl4`.
Notice that the application resumes from where the checkpoint was taken as in the prior examples.


## L4 – Differential Checkpoint:

Change directory to the differential checkpoint in folder ${TUTORIAL\_EXEC}/DCP.
Then, run the application with `make hdDCP`.
You may follow the “blue” (dCP) messages in the terminal while the progam is running.
After a couple of checkpoints, interrupt the application and restart it.
What is actually happening?

Delete all files under the 'local/', 'global/' and 'meta/' folders.
Now open file 'config.DCP.fti' with your favorite text editor.
Change the following parameters:
1. ckpt\_io = 3 to ckpt\_io = 1
2. failure = “x” to failure = 0

The first option changes the file format and the second indicates that we will do a fresh run (not a recovery).
Run the execution again with `make hdDCP`.
Do you observe any difference in the timings of the checkpoints?


# Practice 

1. In the ‘${TUTORIAL\_SRC}/practice’ folder you will find the source code of the program we used to demonstrate the FTI features.
This time, the application is not integrated with FTI.
Try to include FTI in the application.
You can use either the ‘FTI\_Snapshot’ or ‘FTI\_Checkpoint’ function to cause FTI to take a checkpoint.
To build the code changes you implemented you can:

```bash
cd $FTI_ROOT/build
make
```

To execute the application, change directory to ${TUTORIAL\_EXEC}/practice and execute the following script.

```bash
cd $TUTORIAL_EXEC/practice
make
mpirun -n 4 ./hdp.exe GRID_SIZE
```

GRID\_SIZE is an integer number defining the size of the grid to be solved in Mb.
(Remark: <GRIDSIZE> denotes the dynamic memory of each mpi process in MB)

Besides altering the source code, you also need to create an appropriate configuration file.
Information about the options in the configuration file can be found [here](https://github.com/leobago/fti/wiki/Configuration) and example configuration files can be found [here](https://github.com/leobago/fti/wiki/Configuration-Examples).

2. Change into the folder ‘${TUTORIAL\_EXEC}/tutorial/experiment’ and play with the settings of the configuration file.
To execute the program, type: `mpirun -n 8 --oversubscribe hdp.exe  <GRIDSIZE> config.fti`.
Perform executions with ‘Head=0’ and ‘Head=1’ and try to see if there are any differences in the application execution time?
If there are no differeneces, you may take frequent L3 checkpointing and a gridsize of 256 or higher.
Then, you will most likely see a difference.
For more information, check on the research paper that originated from this tool.

Bautista-Gomez, Leonardo, et al. ***"FTI: high performance fault tolerance interface
for hybrid systems."*** Proceedings of 2011 international conference for high
performance computing, networking, storage and analysis. ACM, 2011.