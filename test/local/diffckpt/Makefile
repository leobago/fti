FTI_HOME := /home/kellekai/WORK/FTI/FTI-REPO-KAI/build/RELEASE
FTI_INC_DIR := $(FTI_HOME)/include
FTI_LIB_DIR := $(FTI_HOME)/lib

.PHONY: clean all run

all: diff_test

diff_test_func.o: diff_test_func.c diff_test.h Makefile
	mpicc -o $@ -c $< -g $(CDEF) -I$(FTI_INC_DIR) -L$(FTI_LIB_DIR) -lfti

diff_test: diff_test_func.o diff_test.c diff_test.h Makefile
	mpicc -o diff_test diff_test.c -g $(CDEF) $< -I$(FTI_INC_DIR) -L$(FTI_LIB_DIR) -lfti -lcrypto

run: diff_test Makefile
	cp cfg/config.fti .
	mpirun -n 8 ./$<

run-test: diff_test Makefile
	cp cfg/config.fti .
	mpirun -n 8 ./$<
	mpirun -n 8 ./$<
#	mpirun -n 1 valgrind ./$< : -n 7 ./$<

clean:
	rm -rf *.o diff_test Global Local Meta config.fti

