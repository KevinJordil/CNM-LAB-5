CC=nvcc
LD=nvcc

OPT_LEVEL=-O3 -lcublas

CC_OPTS=$(OPT_LEVEL) -Xcompiler -fPIC,-Wall,-Wno-unused-variable,-Werror,-Wfatal-errors,-fopenmp

HEADERS:=$(wildcard *.h) Makefile

.SUFFIXES:

TARGETS=mlp
all : $(TARGETS)
	rm -rf *.o

%.o: %.cu $(HEADERS)
	$(CC) -c $< -o $@ $(CC_OPTS)

$(TARGETS): mlp.o rand.o
	$(LD) $^ -o $@ $(CC_OPTS) -lm

clean:
	rm -rf $(TARGETS) *.o