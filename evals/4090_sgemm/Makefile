# CUDA Compiler
NVCC := nvcc

# Target executable
TARGET := sgemm_sweep

# Source files
SRCS := main.cu runner.cu

# Compute capability (adjust based on your GPU)
# RTX 4070 Ti SUPER = 8.9, A100 = 8.0, etc.
CUDA_ARCH := 89

# Compiler flags
NVCC_FLAGS := -arch=sm_$(CUDA_ARCH) \
              -O3 \
              --use_fast_math \
              -w \
              -I. \
              -lcublas

# Default target
all: $(TARGET)

# Build the executable
$(TARGET): $(SRCS)
	$(NVCC) $(NVCC_FLAGS) $(SRCS) -o $(TARGET)

# Clean build artifacts
clean:
	rm -f $(TARGET)

# Run the benchmark
run: $(TARGET)
	./$(TARGET)

# Phony targets
.PHONY: all clean run
