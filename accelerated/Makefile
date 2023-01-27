NVCC=nvcc

OPT_LEVEL=-O3 -lcublas

NVCC_OPTS=$(OPT_LEVEL)

HEADERS:=$(wildcard *.h) Makefile

TARGET=mlp
all: $(TARGET)

%.o: %.cu $(HEADERS)
	$(NVCC) -c $< -o $@ $(NVCC_OPTS)

$(TARGET): mlp.o
	$(NVCC) $^ -o $@ $(NVCC_OPTS)

clean:
	rm -rf $(TARGET) *.o