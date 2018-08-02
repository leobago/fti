# SET TO FTI SOURCE DIRECTORY
FTI_HOME :=
# SET TO FTI RELEASE DIRECTORY
FTI_RELEASE :=
# SET TO FTI BUILD DIRECTORY
FTI_BUILD :=
FTI_INC_DIR := $(FTI_RELEASE)/include
FTI_LIB_DIR := $(FTI_RELEASE)/lib
FTI_SRC := $(FTI_HOME)/src/*.h $(FTI_HOME)/src/*.c $(FTI_HOME)/include/fti.h
WORK_DIR := $(FTI_HOME)/test/local/staging

.PHONY: clean all run-test fti

all: run-test

export LD_LIBRARY_PATH := $(LD_LIBRARY_PATH):$(FTI_RELEASE)/lib

fti: $(FTI_SRC) clean
	cd $(FTI_BUILD) && $(MAKE) all install
	cd $(WORK_DIR)

massive: massive.c fti 
	mpicc -o massive -g $(CDEF) $< -I$(FTI_INC_DIR) -L$(FTI_LIB_DIR) -lfti

run-test: massive Makefile
	cp cfg/config.fti .
	mpirun -n 12 ./$<

clean:
	rm -rf *.o massive rdir Global Local Meta config.fti

