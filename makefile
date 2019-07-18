# Compiler paths
HOST_COMPILER=g++
CUDA_PATH=/usr/local/cuda-9.0
NVCC=$(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

# Set some debug flags
CXX_FLAGS= -g -G -Iinclude
NVCC_FLAGS=$(CXX_FLAGS)

test_allocator: test.o
	$(NVCC) $(NVCC_FLAGS) -o $@ test.o

test.o: test/main.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ -c test/main.cu

clean:
	rm -f test test.o

