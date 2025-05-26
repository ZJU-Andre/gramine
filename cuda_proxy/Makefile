CC = gcc
# Adjust CUDA_PATH if your CUDA installation is elsewhere
CUDA_PATH ?= /usr/local/cuda
# Adjust MBEDTLS_PATH if mbedTLS is not in a standard system path
# If Gramine builds mbedTLS as a subproject, these paths might need to point there.
# For now, assume system paths or that CPPFLAGS/LDFLAGS will be set externally if needed.
MBEDTLS_PATH ?= /usr
MBEDTLS_INCLUDE_PATH ?= $(MBEDTLS_PATH)/include
MBEDTLS_LIB_PATH ?= $(MBEDTLS_PATH)/lib

# Adjust LIBOS_INCLUDE_PATH to the location of libos_gpu_ipc.h
# This assumes the cuda_proxy directory is at the root of the gramine repository.
LIBOS_INCLUDE_PATH ?= ../libos/include

CFLAGS = -g -Wall -Wextra -I$(CUDA_PATH)/include -I$(MBEDTLS_INCLUDE_PATH) -I$(LIBOS_INCLUDE_PATH)
LDFLAGS = -L$(CUDA_PATH)/lib64 -lcuda -L$(MBEDTLS_LIB_PATH) -lmbedtls -lmbedx509 -lmbedcrypto -lrt # Added -lrt for shm_open

TARGET = gramine-cuda-proxy

all: $(TARGET)

$(TARGET): proxy.c
	$(CC) $(CFLAGS) -o $(TARGET) proxy.c $(LDFLAGS)

clean:
	rm -f $(TARGET) *.o $(PIPE_ENCLAVE_TO_PROXY) $(PIPE_PROXY_TO_ENCLAVE) # Also clean up pipes if created by make run or similar

# Define fixed pipe names (must match what proxy.c uses)
PIPE_ENCLAVE_TO_PROXY = /tmp/gramine_gpu_ipc_enclave_to_proxy
PIPE_PROXY_TO_ENCLAVE = /tmp/gramine_gpu_ipc_proxy_to_enclave

# Example phony target to help test pipe creation (optional)
create_pipes:
	@echo "Creating FIFOs for testing..."
	@mkfifo $(PIPE_ENCLAVE_TO_PROXY) || true
	@mkfifo $(PIPE_PROXY_TO_ENCLAVE) || true
	@echo "FIFOs created (or already exist): $(PIPE_ENCLAVE_TO_PROXY), $(PIPE_PROXY_TO_ENCLAVE)"

# Example phony target to run the proxy (optional, for testing)
# Note: This assumes the proxy creates/manages its own pipes as per current proxy.c logic.
# If pipes are created externally (e.g. via create_pipes), proxy logic might need adjustment.
run: $(TARGET)
	./$(TARGET)

.PHONY: all clean create_pipes run
