NVCC := nvcc
# Change to sm_80 for A100, sm_89 for RTX 4090, sm_90a for H100
ARCH := sm_89
NVCC_FLAGS := -O3 -std=c++17 -arch=$(ARCH) -w

GENERATED := $(wildcard generated/*.cu)

.PHONY: all clean

all: scan_bench

scan_bench: main.cu scan_harness.cuh $(GENERATED)
	$(NVCC) $(NVCC_FLAGS) main.cu -o $@

bench: bench.cu $(GENERATED)
	$(NVCC) $(NVCC_FLAGS) bench.cu -o $@

clean:
	rm -f scan_bench bench
