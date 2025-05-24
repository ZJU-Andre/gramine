#include <stdio.h>
#include <cuda_runtime.h>

// CUDA Kernel for vector addition
__global__ void vectorAddKernel(float *a, const float *b, const float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        a[i] = b[i] + c[i];
    }
}

// C-callable wrapper function to launch the CUDA kernel
extern "C" int launch_vector_add_cuda(
    float* h_A_out, 
    const float* h_B_in, 
    const float* h_C_in, 
    int n,
    int* cuda_error_code,       // Output CUDA error code
    const char** cuda_error_str   // Output CUDA error string pointer
) {
    float *d_A = NULL, *d_B = NULL, *d_C = NULL;
    cudaError_t err = cudaSuccess;

    // Initialize output error pointers
    if (cuda_error_code) *cuda_error_code = cudaSuccess;
    if (cuda_error_str) *cuda_error_str = cudaGetErrorString(cudaSuccess);

    // Allocate memory on the GPU
    err = cudaMalloc((void**)&d_A, n * sizeof(float));
    if (err != cudaSuccess) {
        if (cuda_error_code) *cuda_error_code = err;
        if (cuda_error_str) *cuda_error_str = cudaGetErrorString(err);
        fprintf(stderr, "CUDA_WRAPPER: Failed to allocate d_A: %s\n", cudaGetErrorString(err));
        goto Error;
    }

    err = cudaMalloc((void**)&d_B, n * sizeof(float));
    if (err != cudaSuccess) {
        if (cuda_error_code) *cuda_error_code = err;
        if (cuda_error_str) *cuda_error_str = cudaGetErrorString(err);
        fprintf(stderr, "CUDA_WRAPPER: Failed to allocate d_B: %s\n", cudaGetErrorString(err));
        goto Error;
    }

    err = cudaMalloc((void**)&d_C, n * sizeof(float));
    if (err != cudaSuccess) {
        if (cuda_error_code) *cuda_error_code = err;
        if (cuda_error_str) *cuda_error_str = cudaGetErrorString(err);
        fprintf(stderr, "CUDA_WRAPPER: Failed to allocate d_C: %s\n", cudaGetErrorString(err));
        goto Error;
    }

    // Copy input arrays from host to device
    err = cudaMemcpy(d_B, h_B_in, n * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        if (cuda_error_code) *cuda_error_code = err;
        if (cuda_error_str) *cuda_error_str = cudaGetErrorString(err);
        fprintf(stderr, "CUDA_WRAPPER: Failed to copy h_B_in to d_B: %s\n", cudaGetErrorString(err));
        goto Error;
    }

    err = cudaMemcpy(d_C, h_C_in, n * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        if (cuda_error_code) *cuda_error_code = err;
        if (cuda_error_str) *cuda_error_str = cudaGetErrorString(err);
        fprintf(stderr, "CUDA_WRAPPER: Failed to copy h_C_in to d_C: %s\n", cudaGetErrorString(err));
        goto Error;
    }

    // Define block and grid sizes
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    // Launch the kernel
    vectorAddKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, n);
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

    // Copy the result array from device to host
    err = cudaMemcpy(h_A_out, d_A, n * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        if (cuda_error_code) *cuda_error_code = err;
        if (cuda_error_str) *cuda_error_str = cudaGetErrorString(err);
        fprintf(stderr, "CUDA_WRAPPER: Failed to copy d_A to h_A_out: %s\n", cudaGetErrorString(err));
        goto Error;
    }

    // Free GPU memory
Error: // Common cleanup point
    if (d_A) cudaFree(d_A);
    if (d_B) cudaFree(d_B);
    if (d_C) cudaFree(d_C);

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
