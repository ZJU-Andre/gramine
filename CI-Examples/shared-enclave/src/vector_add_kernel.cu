#include <stdio.h>
#include <cuda_runtime.h>

// CUDA Kernel for vector addition
__global__ void vectorAddKernel(float *a, const float *b, const float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        a[i] = b[i] + c[i];
    }
}

#include <stdbool.h> // For bool type
#include <stdint.h>  // For uint64_t

// C-callable wrapper function to launch the CUDA kernel
extern "C" int launch_vector_add_cuda(
    bool use_dma_for_output_a,       // New: Flag to indicate if DMA output to client host is used
    uint64_t dest_client_host_ptr_a, // New: Client-provided host pointer for output A (if DMA)
    float* h_A_out_fallback,         // Existing h_A_out, now a fallback if DMA output is not used
    const float* h_B_in, 
    const float* h_C_in, 
    int n,
    int* cuda_error_code,
    const char** cuda_error_str,
    bool use_dma_for_b,
    uint64_t d_ptr_b_client,
    bool use_dma_for_c,
    uint64_t d_ptr_c_client
) {
    float *d_A = NULL; // Device pointer for result A (always allocated by enclave)
    float *d_B_enclave = NULL; // Device pointer for B if allocated by enclave
    float *d_C_enclave = NULL; // Device pointer for C if allocated by enclave

    // Effective device pointers to be used by the kernel
    float *d_B_effective = NULL;
    float *d_C_effective = NULL;

    cudaError_t err = cudaSuccess;

    // Initialize output error pointers
    if (cuda_error_code) *cuda_error_code = cudaSuccess;
    if (cuda_error_str) *cuda_error_str = cudaGetErrorString(cudaSuccess);

    // Allocate memory for result array d_A on the GPU (always done by enclave)
    err = cudaMalloc((void**)&d_A, n * sizeof(float));
    if (err != cudaSuccess) {
        if (cuda_error_code) *cuda_error_code = err;
        if (cuda_error_str) *cuda_error_str = cudaGetErrorString(err);
        fprintf(stderr, "CUDA_WRAPPER: Failed to allocate d_A: %s\n", cudaGetErrorString(err));
        goto Error;
    }

    // Handle input array B
    if (use_dma_for_b) {
        printf("CUDA_WRAPPER: Using client-provided device pointer for B (0x%lx)\n", d_ptr_b_client);
        d_B_effective = (float*)d_ptr_b_client; // Cast client's device pointer
    } else {
        printf("CUDA_WRAPPER: Allocating and copying data for B from host pointer h_B_in.\n");
        err = cudaMalloc((void**)&d_B_enclave, n * sizeof(float));
        if (err != cudaSuccess) {
            if (cuda_error_code) *cuda_error_code = err;
            if (cuda_error_str) *cuda_error_str = cudaGetErrorString(err);
            fprintf(stderr, "CUDA_WRAPPER: Failed to allocate d_B_enclave: %s\n", cudaGetErrorString(err));
            goto Error;
        }
        err = cudaMemcpy(d_B_enclave, h_B_in, n * sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            if (cuda_error_code) *cuda_error_code = err;
            if (cuda_error_str) *cuda_error_str = cudaGetErrorString(err);
            fprintf(stderr, "CUDA_WRAPPER: Failed to copy h_B_in to d_B_enclave: %s\n", cudaGetErrorString(err));
            goto Error;
        }
        d_B_effective = d_B_enclave;
    }

    // Handle input array C
    if (use_dma_for_c) {
        printf("CUDA_WRAPPER: Using client-provided device pointer for C (0x%lx)\n", d_ptr_c_client);
        d_C_effective = (float*)d_ptr_c_client; // Cast client's device pointer
    } else {
        printf("CUDA_WRAPPER: Allocating and copying data for C from host pointer h_C_in.\n");
        err = cudaMalloc((void**)&d_C_enclave, n * sizeof(float));
        if (err != cudaSuccess) {
            if (cuda_error_code) *cuda_error_code = err;
            if (cuda_error_str) *cuda_error_str = cudaGetErrorString(err);
            fprintf(stderr, "CUDA_WRAPPER: Failed to allocate d_C_enclave: %s\n", cudaGetErrorString(err));
            goto Error;
        }
        err = cudaMemcpy(d_C_enclave, h_C_in, n * sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            if (cuda_error_code) *cuda_error_code = err;
            if (cuda_error_str) *cuda_error_str = cudaGetErrorString(err);
            fprintf(stderr, "CUDA_WRAPPER: Failed to copy h_C_in to d_C_enclave: %s\n", cudaGetErrorString(err));
            goto Error;
        }
        d_C_effective = d_C_enclave;
    }

    // Define block and grid sizes
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    // Launch the kernel with effective pointers
    vectorAddKernel<<<gridSize, blockSize>>>(d_A, d_B_effective, d_C_effective, n);
    err = cudaGetLastError(); // Check for errors from kernel launch
    if (err != cudaSuccess) {
        if (cuda_error_code) *cuda_error_code = err;
        if (cuda_error_str) *cuda_error_str = cudaGetErrorString(err);
        fprintf(stderr, "CUDA_WRAPPER: Kernel launch failed: %s\n", cudaGetErrorString(err));
        goto Error;
    }

    // Synchronize device to ensure kernel completion before copying back results
    // This is good practice, though cudaMemcpy DeviceToHost is blocking by default for default stream.
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        if (cuda_error_code) *cuda_error_code = err;
        if (cuda_error_str) *cuda_error_str = cudaGetErrorString(err);
        fprintf(stderr, "CUDA_WRAPPER: cudaDeviceSynchronize failed after kernel launch: %s\n", cudaGetErrorString(err));
        goto Error;
    }

    // Copy the result array from device (d_A) to the appropriate host destination
    if (use_dma_for_output_a) {
        printf("CUDA_WRAPPER: Copying result d_A to client's host pointer 0x%lx via DMA.\n", dest_client_host_ptr_a);
        err = cudaMemcpy((void*)dest_client_host_ptr_a, d_A, n * sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            if (cuda_error_code) *cuda_error_code = err;
            if (cuda_error_str) *cuda_error_str = cudaGetErrorString(err);
            fprintf(stderr, "CUDA_WRAPPER: Failed to copy d_A to client's host pointer (dest_client_host_ptr_a): %s\n", cudaGetErrorString(err));
            goto Error;
        }
    } else {
        printf("CUDA_WRAPPER: Copying result d_A to fallback host buffer h_A_out_fallback.\n");
        err = cudaMemcpy(h_A_out_fallback, d_A, n * sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            if (cuda_error_code) *cuda_error_code = err;
            if (cuda_error_str) *cuda_error_str = cudaGetErrorString(err);
            fprintf(stderr, "CUDA_WRAPPER: Failed to copy d_A to h_A_out_fallback: %s\n", cudaGetErrorString(err));
            goto Error;
        }
    }

    // Free GPU memory
Error: // Common cleanup point
    if (d_A) cudaFree(d_A); // d_A is always allocated by enclave
    
    // Free d_B_enclave and d_C_enclave only if they were allocated by this function
    if (d_B_enclave) {
        printf("CUDA_WRAPPER: Freeing enclave-allocated d_B_enclave.\n");
        cudaFree(d_B_enclave);
    }
    if (d_C_enclave) {
        printf("CUDA_WRAPPER: Freeing enclave-allocated d_C_enclave.\n");
        cudaFree(d_C_enclave);
    }

    if (err != cudaSuccess) {
        // If an error occurred before this point, cuda_error_code and cuda_error_str are already set.
        // If a cudaFree fails, this will update them.
        cudaError_t free_err = cudaGetLastError(); // Check if any cudaFree failed
        if (free_err != cudaSuccess && err == cudaSuccess) { // Only update if no prior error and free failed
             if (cuda_error_code) *cuda_error_code = free_err;
             if (cuda_error_str) *cuda_error_str = cudaGetErrorString(free_err);
             fprintf(stderr, "CUDA_WRAPPER: Error during cudaFree: %s\n", cudaGetErrorString(free_err));
             return -1; // Indicate error from cleanup
        }
        return -1; // Indicate prior error
    }

    return 0; // Success
}
