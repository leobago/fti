CC=mpicc
CLFLAGS=-Wl,-rpath /data/WORK/FTI/FTI_GENERIC/RELEASE/FTI/lib -L/data/WORK/FTI/FTI_GENERIC/RELEASE/FTI/lib -lfti
CFLAGS=-I/data/WORK/FTI/FTI_GENERIC/RELEASE/FTI/include -g -Werror

all: check.exe

check.exe: check.c
	$(CC) -o $@ $+ $(CLFLAGS) $(CFLAGS)

clean:
	rm -rf check.exe Local/* Global/* Meta/* chk/* config.fti
