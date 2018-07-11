FTI_HOME := # SET TO FTI SOURCE DIRECTORY
FTI_RELEASE := # SET TO FTI RELEASE DIRECTORY
FTI_BUILD := # SET TO FTI BUILD DIRECTORY
FTI_INC_DIR := $(FTI_RELEASE)/include
FTI_LIB_DIR := $(FTI_RELEASE)/lib
FTI_SRC := $(FTI_HOME)/src/*.h $(FTI_HOME)/src/*.c $(FTI_HOME)/include/fti.h
WORK_DIR := $(FTI_HOME)/test/local/diffckpt

.PHONY: clean all run fti share env

all: diff_test

export LD_LIBRARY_PATH := $(LD_LIBRARY_PATH):$(FTI_RELEASE)/lib
#export FTI_DCP_BLOCK_SIZE=16384
#export FTI_DCP_HASH_MODE=2

fti: $(FTI_SRC) clean
	cd $(FTI_BUILD) && $(MAKE) all install
	cd $(WORK_DIR)

diff_test_func.o: diff_test_func.c diff_test.h Makefile fti
	mpicc -o $@ -c $< -g $(CDEF) -I$(FTI_INC_DIR) -L$(FTI_LIB_DIR) -lfti

diff_test: diff_test_func.o diff_test.c diff_test.h Makefile fti
	mpicc -o diff_test diff_test.c -g $(CDEF) $< -I$(FTI_INC_DIR) -L$(FTI_LIB_DIR) -lfti -lcrypto

small: test.c Makefile fti
	mpicc -o $@ $< -g -I$(FTI_INC_DIR) -L$(FTI_LIB_DIR) -lfti
	cp cfg/config.fti .
	mpirun -n 8 ./$@

small-valgrind: test.c Makefile fti
	mpicc -o $@ $< -g -I$(FTI_INC_DIR) -L$(FTI_LIB_DIR) -lfti
	cp cfg/config.fti .
	mpirun -n 8 valgrind ./$@ 2>&1 | tee smallDbgOut

run: diff_test Makefile fti
	cp cfg/config.fti .
	mpirun -n 8 ./$<

speed: test_speed.c Makefile fti
	mpicc -o $@ $< -g -I$(FTI_INC_DIR) -L$(FTI_LIB_DIR) -lfti
	cp cfg/config.fti .
	mpirun -n 8 ./$@

share: test_share.c diff_test.h Makefile fti
	mpicc -o $@ $< -g -I$(FTI_INC_DIR) -L$(FTI_LIB_DIR) -lfti
	cp cfg/config.fti .
	mpirun -n 8 ./$@
	mpirun -n 8 ./$@

realloc: realloc-test.c Makefile fti
	mpicc -o $@ $< -g -I$(FTI_INC_DIR) -L$(FTI_LIB_DIR) -lfti
	cp cfg/config.fti .
	mpirun -n 8 ./$@
	mpirun -n 8 ./$@

run-test: diff_test Makefile
	cp cfg/config.fti .
	mpirun -n 8 ./$<
	mpirun -n 8 ./$<

run-valgrind: diff_test Makefile
	cp cfg/config.fti .
	#mpirun -n 1 valgrind ./$< : -n 7 ./$<
	mpirun -n 8 valgrind ./$<

clean:
	rm -rf *.o diff_test Global Local Meta config.fti

