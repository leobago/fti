FTI_HOME := /home/kellekai/WORK/FTI/FTI-REPO-KAI/build/RELEASE
FTI_BUILD := /home/kellekai/WORK/FTI/FTI-REPO-KAI/build
FTI_INC_DIR := $(FTI_HOME)/include
FTI_LIB_DIR := $(FTI_HOME)/lib
FTI_SRC := /home/kellekai/WORK/FTI/FTI-REPO-KAI/src/*.h /home/kellekai/WORK/FTI/FTI-REPO-KAI/src/*.c /home/kellekai/WORK/FTI/FTI-REPO-KAI/include/fti.h
WORK_DIR := /home/kellekai/WORK/FTI/FTI-REPO-KAI/test/local/diffckpt

.PHONY: clean all run fti

all: diff_test

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

run: diff_test Makefile
	cp cfg/config.fti .
	mpirun -n 8 ./$<

run-test: diff_test Makefile
	cp cfg/config.fti .
	mpirun -n 8 ./$<
	mpirun -n 8 ./$<

run-valgrind: diff_test Makefile
	cp cfg/config.fti .
	mpirun -n 1 valgrind ./$< : -n 7 ./$<

clean:
	rm -rf *.o diff_test Global Local Meta config.fti

