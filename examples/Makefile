## * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
##   Makefile to test FTI library with the heat distribution program.

##   PLEASE SET THIS VARIABLE BEFORE COMPILING
FTIPATH		= /path/to/fti/install/directory

##   COMPILERS
MPICC 		?= mpicc
MPIFC		?= mpif90
MPIRUN 		?= mpirun

##   FLAGS
# Compiling using shared library
FTIFLAG 	= -I$(FTIPATH)/include -L$(FTIPATH)/lib -lfti -lm
FFTIFLAG 	= -I$(FTIPATH)/include -L$(FTIPATH)/lib -lfti_f90 -lfti -lm
# Compiling using static library
#FTIFLAG 	= -I$(FTIPATH)/include $(FTIPATH)/lib/libfti.a
#FFTIFLAG 	= -I$(FTIPATH)/include $(FTIPATH)/lib/libfti_f90.a $(FTIPATH)/lib/libfti.a

##   TARGETS
all: 		hd hd2 hdf

hd:		heatdis.c
		$(MPICC) -o hd.exe heatdis.c $(FTIFLAG)

hd2:		heatd2.c
		$(MPICC) -o hd2.exe heatd2.c $(FTIFLAG)

hdf:		fheatdis.f90
		$(MPIFC) -o hdf.exe fheatdis.f90 $(FFTIFLAG)

hdt:
		$(MPIRUN) -np 8 ./hd.exe 32 config.fti

hdv:
		./plot.sh

clean:
		rm -f hd.exe hd2.exe hdf.exe results/*

.PHONY:		hd hd2 hdf hdt hdv clean

