CC=mpicc
CLFLAGS=-Wl,-rpath /data/WORK/FTI/FTI_L2FIX/RELEASE/lib -L/data/WORK/FTI/FTI_L2FIX/RELEASE/lib -lfti
CFLAGS=-I/data/WORK/FTI/FTI_L2FIX/RELEASE/include -g -Werror

all: check.exe

check.exe: check.c
	$(CC) -o $@ $+ $(CLFLAGS) $(CFLAGS)

clean:
	rm -rf check.exe Local/* Global/* Meta/* chk/* config.fti
