1) Make sure that FTI library is accessible in your LD_LIBRARY_PATH.

2) Go to "fti/build/examples".

3) With your favorite text editor open and edit the FTI configuration file
   configBkp.fti to set the three directory variables.

    - vim configBkp.fti (Edit Ckpt_dir, Glbl_dir, Meta_dir)

4) Run examples by going to fti/build/examples and:
	- execute:	"make init"	<- copies configBkp.fti to config.fti
	- execute:	"make hd"	<- runs first example (heatdis.c)
	- after FTI creates checkpoint press "ctrl + c" to abort a process
	- execute:	"make hd"
	- see that work will continue from last checkpoint
	- wait for program to finish

   If you want run different example instead of "make hd" execute:
	- "make hd2" 	<- runs second example (heatd2.c)
	- "make hdf"	<- runs third example (fheatdis.f90)
	- "make runall" 	<- runs all examples

   After running hd2 execute plot.sh and go to "results" folder to see PNG files with graphs.
