## * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
##   Makefile to test FTI library with the heat distribution program.

##   PLEASE SET THIS VARIABLE BEFORE COMPILING
FTIPATH		= ../../build

##   COMPILERS
MPICC 		?= mpicc
MPIFC		?= mpif90
MPIRUN 		?= mpirun

##   FLAGS
# Compiling using shared library
#FTIFLAG 	= -I$(FTIPATH)/include -L$(FTIPATH)/lib -lfti -lm
#FFTIFLAG 	= -I$(FTIPATH)/include -L$(FTIPATH)/lib -lfti_f90 -lfti -lm
# Compiling using static library
FTIFLAG 	= -I$(FTIPATH)/include $(FTIPATH)/lib/libfti.a -lm
FFTIFLAG 	= -I$(FTIPATH)/include $(FTIPATH)/lib/libfti_f90.a $(FTIPATH)/lib/libfti.a -lm

##   TARGETS
all:		tokenRing.c
#		C_INCLUDE_PATH=../../include
#		export C_INCLUDE_PATH
		$(MPICC) -o tokenRing tokenRing.c $(FTIFLAG)
