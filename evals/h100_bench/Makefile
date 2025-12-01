NVCC_PATH := $(shell which nvcc 2>/dev/null)
ifeq ($(NVCC_PATH),)
    $(error nvcc not found on PATH. Install the CUDA toolkit or set NVCC_PATH manually.)
endif

NVCC_FLAGS = -std=c++17 -O3 -DNDEBUG -w
NVCC_LDFLAGS = -lcublas -lcuda
OUT_DIR = out

CUDA_OUTPUT_FILE = -o $(OUT_DIR)/$@

NVCC_FLAGS += --use_fast_math
NVCC_FLAGS += -gencode arch=compute_90a,code=sm_90a

NVCC_BASE = $(NVCC_PATH) $(NVCC_FLAGS) $(NVCC_LDFLAGS) -lineinfo

test: main.cu utils.cuh cutlass_my_h100_match_no_tail.cu
	mkdir -p $(OUT_DIR)
	$(NVCC_BASE) main.cu $(CUDA_OUTPUT_FILE)

clean:
	rm -rf $(OUT_DIR)

.PHONY: test clean
