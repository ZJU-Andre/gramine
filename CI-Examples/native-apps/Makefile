# CI-Examples/native-apps/Makefile
# Makefile to build native GPU benchmark applications.

# Compiler
CC := gcc
NVCC := nvcc

# Common Flags
CFLAGS := -O2 -Wall
LDFLAGS :=
INCFLAGS := -I./src 

# CUDA Configuration (User may need to adjust these)
CUDA_HOME ?= /usr/local/cuda
CUDA_INCFLAGS := -I$(CUDA_HOME)/include
CUDA_LDFLAGS := -L$(CUDA_HOME)/lib64 -Wl,-rpath=$(CUDA_HOME)/lib64 -lcudart

# ONNX Runtime Configuration (User may need to adjust these)
ONNXRUNTIME_HOME ?= /usr/local/onnxruntime # Base directory of ONNX Runtime installation
ONNXRUNTIME_INCFLAGS := -I$(ONNXRUNTIME_HOME)/include
ONNXRUNTIME_LDFLAGS := -L$(ONNXRUNTIME_HOME)/lib -Wl,-rpath=$(ONNXRUNTIME_HOME)/lib -lonnxruntime

# cuBLAS Configuration (Often part of CUDA_HOME, but can be separate)
CUBLAS_HOME ?= $(CUDA_HOME) # Assume cuBLAS is in CUDA_HOME
CUBLAS_LDFLAGS := -L$(CUBLAS_HOME)/lib64 -Wl,-rpath=$(CUBLAS_HOME)/lib64 -lcublas

# Source files
VECTOR_ADD_NATIVE_SRC := src/vector_add_native_benchmark.c
# For vector_add_native, we also need the .cu file from the shared_enclave example.
# This Makefile assumes it's copied or linked here, or path is adjusted.
# For simplicity, let's assume it's linked/copied to ./src/
SHARED_VECTOR_ADD_CU_SRC := ../shared-enclave/src/vector_add_kernel.cu 
SHARED_VECTOR_ADD_CU_OBJ := src/vector_add_kernel.o

ONNX_NATIVE_SRC := src/onnx_native_benchmark.c
GEMM_NATIVE_SRC := src/gemm_native_benchmark.c

# Executables
TARGET_DIR := bin
VECTOR_ADD_NATIVE_EXE := $(TARGET_DIR)/vector_add_native_benchmark
ONNX_NATIVE_EXE := $(TARGET_DIR)/onnx_native_benchmark
GEMM_NATIVE_EXE := $(TARGET_DIR)/gemm_native_benchmark

.PHONY: all clean vector_add_native onnx_native gemm_native

all: vector_add_native onnx_native gemm_native

$(TARGET_DIR):
	mkdir -p $(TARGET_DIR)

# Rule to compile the .cu file from shared-enclave example into an object file
$(SHARED_VECTOR_ADD_CU_OBJ): $(SHARED_VECTOR_ADD_CU_SRC) | $(TARGET_DIR)
	@echo "Compiling CUDA kernel object $(SHARED_VECTOR_ADD_CU_OBJ) from $(SHARED_VECTOR_ADD_CU_SRC)..."
	$(NVCC) $(CUDA_INCFLAGS) -O2 -Xcompiler=-fPIC --device-c -c $< -o $@

vector_add_native: $(VECTOR_ADD_NATIVE_SRC) $(SHARED_VECTOR_ADD_CU_OBJ) | $(TARGET_DIR)
	@echo "Building Native Vector Add Benchmark..."
	$(CC) $(CFLAGS) $(INCFLAGS) $(CUDA_INCFLAGS) $^ -o $(VECTOR_ADD_NATIVE_EXE) $(CUDA_LDFLAGS)
	@echo "$(VECTOR_ADD_NATIVE_EXE) built."

onnx_native: $(ONNX_NATIVE_SRC) | $(TARGET_DIR)
	@echo "Building Native ONNX Benchmark..."
	$(CC) $(CFLAGS) $(INCFLAGS) $(ONNXRUNTIME_INCFLAGS) $(CUDA_INCFLAGS) $< -o $(ONNX_NATIVE_EXE) $(ONNXRUNTIME_LDFLAGS) $(CUDA_LDFLAGS)
	@echo "$(ONNX_NATIVE_EXE) built."

gemm_native: $(GEMM_NATIVE_SRC) | $(TARGET_DIR)
	@echo "Building Native GEMM Benchmark..."
	$(CC) $(CFLAGS) $(INCFLAGS) $(CUDA_INCFLAGS) $< -o $(GEMM_NATIVE_EXE) $(CUBLAS_LDFLAGS) $(CUDA_LDFLAGS)
	@echo "$(GEMM_NATIVE_EXE) built."

clean:
	@echo "Cleaning native apps build directory..."
	rm -rf $(TARGET_DIR) src/*.o
	@echo "Native apps clean complete."

# Notes:
# 1. This Makefile assumes that for vector_add_native_benchmark, the `vector_add_kernel.cu`
#    and `vector_add.h` are either copied into ./src or the path to 
#    `../shared-enclave/src/vector_add_kernel.cu` is correct and accessible.
#    The header `vector_add.h` is expected to be in `./src` or found via `INCFLAGS`.
#    For simplicity, one might copy `vector_add.h` from `../shared-enclave/src/` to `./src/`.
# 2. Users must ensure CUDA_HOME and ONNXRUNTIME_HOME point to correct installations,
#    or that the libraries are in standard system paths.
# 3. The rpath settings help the dynamic linker find the .so files at runtime.
# 4. For ONNX and GEMM, this links against libcudart as well, as some CUDA setup/teardown
#    (like cudaEvent for timing) might be used even if main computation is via another lib.
#    If not, cudart linking can be removed for those.
# 5. The compilation of vector_add_kernel.cu uses --device-c for separate compilation.
#    This object file is then linked by the C compiler.
#    If vector_add_native_benchmark.c was a .cu file itself, nvcc could compile and link all in one go.
#    But since it's a .c file calling an extern "C" CUDA wrapper, this two-step is common.
#    Alternatively, link the .cu object with nvcc as the linker.
#    This setup assumes gcc can link .o files produced by nvcc for device code wrappers.
#    A more robust approach for linking CUDA device code with host C code might involve
#    nvcc for the final linking step if there are issues.
#    Example: $(NVCC) $(CFLAGS_NVCC) $(INCFLAGS) $(CUDA_INCFLAGS) $(VECTOR_ADD_NATIVE_SRC) $(SHARED_VECTOR_ADD_CU_OBJ) -o $(VECTOR_ADD_NATIVE_EXE) $(CUDA_LDFLAGS)
#    For now, using GCC for final link, which is fine if vector_add_kernel.o only contains device code and its C-stubs.
#    The `-Xcompiler=-fPIC` is for the host part of the .cu file if any. `--device-c` ensures relocatable device code.
#    If `vector_add_kernel.cu` itself had host code that needed linking, this might need adjustment.
#    Given `launch_vector_add_cuda` is `extern "C"`, this should work.
