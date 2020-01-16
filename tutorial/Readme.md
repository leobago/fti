I this page we present a tutorial of FTI. The purpose of the practice section is for you to get familiar with the FTI API as well as with the configuration file. Therefore there is limited information on how you should proceed.  


#  Installation 

## Preparation
1. Create FTI directory
```bash
mkdir FTI
cd FTI
```
2. Create Installation Directory
```bash
mkdir install-fti
```
3. Set enviromental variable to installation path
```bash
export FTI_INSTALL_DIR=$PWD/install-fti
```
4. Download FTI.
```bash
git clone https://github.com/leobago/fti 
```
5. Change into base directory
```bash
cd fti
```
6. Set Enviroment Variable to FTI root
```bash
export FTI_ROOT=$PWD
```

## Configure and Install
1. Create build directory and change into it
```bash
mkdir build
cd build
```
2. Build FTI
```bash
cmake -DCMAKE_INSTALL_PREFIX:PATH=$FTI_INSTALL_DIR -DENABLE_TUTORIAL=1 ..
make
make install
```

The flag -DENABLE\_TUTORIAL=1 besides building FTI, will also build the tutorial files

## Executables, tutorial source code, and fti library files
The library is installed at the $FTI\_INSTALL\_DIR the source code of the FTI library is in ${FTI\_ROOT}/src and the source code of the tutorial is under ${FTI\_ROOT}/tutorial, the executables of the tutorial are under ${FTI\_ROOT}/build/tutorial/. For conveniency on the rest of the tutorial set also the following variables: 

```bash
export TUTORIAL_EXEC=${FTI_ROOT}/build/tutorial/
export TUTORIAL_SRC=${FTI_ROOT}/tutorial/
```

You should always export this variables every time you try to start/continue the tutorial. Under the ${TUTORIAL\_SRC} directory you can find various directories, each directory corresponds to a step presented in the tutorial. 

# Demonstration of FTI

To demonstrate the various safety levels of FTI, we will execute an example which uses the API function ‘FTI\_Snapshot()’. Run the example in each case for at least one minute and interrupt the execution after that time by pressing ‘ctrl+c’. In some systems 'ctrl+c' does not kill all executing MPI processes, to kill all processes just killall 'executable'.

## L1 - Local checkpoint on the nodes

Change into folder ${TUTORIAL\_EXEC}/L1 and run the execution with ‘make hdl1’. While the program is running, you may follow the events by observing the contents in the ‘local’ folder. In order to do that you can use the commands: 

```bash
watch -n 1 $(find local)
watch -n 1 $(du -kh local)
```
or
```bash
cd local; watch -n 1 $(ls -lR)
```

(It may be illuminating to open the files in the ‘${TUTORIAL\_EXEC}/L1/meta’ folder, using a text editor. What kind of information do you think is kept in these files?)

After interrupting the execution, run again ‘make hdl1’. The execution will (hopefully) resume from where the checkpoint was taken.

After the successful restart, interrupt the execution and delete one of the checkpoint files. The files are stored as (you can also simply delete the whole node directory): ${TUTORIAL\_EXEC}/L1//local/<NODE>/<EXEC-ID>/l1/ckpt<ID>-Rank<RANK>.fti. You will notice, that in that case the program won’t be able to resume the execution.

## L2 – local checkpoint on the nodes + copy to the neighbor node:

Change into folder ${TUTORIAL\_EXEC}/L2 and run the execution with ‘make hdl2’. While the program is running, you may follow the events by observing the contents in the ‘local’ folder.

After interrupting the execution, run again ‘make hdl2’. The execution will also in this case (hopefully) resume from where the checkpoint was taken.

After the successful restart, interrupt the execution and delete one of the checkpoint files. You will notice that now the program (hopefully) will be able to resume the execution. Try to delete more then one file.

### Questions: In order to keep the execution able to resume:
1. How many files you can delete?
2. Which files can you delete?

L3 – local checkpoint on the nodes + copy to the neighbor node + RS encoding:

Change into folder ${TUTORIAL\_EXEC}/L3 and run the execution with ‘make hdl3’. While the program is running, you may follow the events by observing the contents in the ‘local’ folder.

After interrupting the execution, run again ‘make hd3’. The execution will (surprisingly) also in this case resume from where the checkpoint was taken.

After the successful restart, interrupt the execution and delete one of the checkpoint files, the
program will be able to resume.

### Questions: In order to keep the execution able to resume:
1. How many files you can delete?
2. Which files can you delete?

## L4 – flush of the checkpoints to the parallel file system:
Change into folder ${TUTORIAL\_EXEC}/L4 and run the execution with ‘make hdl4’. While the program is running, you may follow the events by observing the contents in the ‘global’ folder. After interrupting the execution, run again ‘make hdl4’. The execution will resume from where the checkpoint was taken.


## L4 – Differential Checkpoint:

Change into folder ${TUTORIAL\_EXEC}/DCP/ and run the execution with ‘make hdDCP’. While the progam is running you may follow the “blue” messages in the terminal. What is actually happening? After a couple of checkpoints, you can kill the application and restart it. 

Delete all files under ./local, ./global/ ./meta/ and open file config.DCP.fti with your favorite text editor. Change the following parameters :
1. ckpt\_io = 3 to ckpt\_io = 1
2. failure = “x” to failure = 0

The first option changes the file format and the second option indicates that we will do a fresh run (not a recovery). Run the execution with ‘make hdDCP’, do you observe any difference in the timings of the checkpoints?

# Practice 

1. In the ‘${TUTORIAL\_SRC}/practice’ folder you will find the source code of the program we used to demonstrate the FTI features. In this case without FTI being implemented. Try to implement FTI. You can use either the ‘FTI\_Snapshot’ or ‘FTI\_Checkpoint’ function to cause FTI taking a checkpoint. To build the code changes you implemented you can :

```bash
cd $FTI_ROOT/build
make
```

To execute your implementation change directory to ${TUTORIAL\_EXEC}/practice and execute the binary hdp.exe. 

Besides implementing the source code you need also to create an appropriate configuration file. Information about the options in the configuration file can be found [here](https://github.com/leobago/fti/wiki/Configuration) and example configuration files can be found [here](https://github.com/leobago/fti/wiki/Configuration-Examples).  


```bash
cd $TUTORIAL_EXEC/practice
make
mpirun -n 4 ./hdp.exe GRID_SIZE
```

GRID\_SIZE is an integer number defining the size of the grid to be solved in Mb. 


2. Change into the folder ‘${TUTORIAL\_EXEC}/tutorial/experiment’ and play with the settings of the configuration file. To run the program, type: ‘mpirun -n 8 
hdex.exe  <GRIDSIZE> config.fti’. Perform executions with ‘Head=0’ and ‘Head=1’, do you notice any difference in the execution duration? (Note: You may take frequent L3 checkpointing and a gridsize of 256 or higher. In that case you will most likely see a difference). (Remark: <GRIDSIZE> denotes the dynamic memory of each mpi process in MB)

