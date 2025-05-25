#include "poc_batch_kernels.h"
#include <stdio.h> // For printf in kernels if needed for debug (use with caution)

// Kernel to copy data from one device buffer to another
__global__ void poc_copy_kernel_dto_d_impl(unsigned char* output, const unsigned char* input, uint32_t size_bytes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Using byte-wise copy, adjust threads/blocks for performance if this were real.
    // For POC, simple stride. Max 1024 threads per block.
    int stride = gridDim.x * blockDim.x; 
    for (uint32_t i = idx; i < size_bytes; i += stride) {
        output[i] = input[i];
    }
}

// Kernel to fill a device buffer with a pattern
__global__ void poc_generate_data_kernel_impl(unsigned char* buffer, uint32_t size_bytes, unsigned char pattern_value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (uint32_t i = idx; i < size_bytes; i += stride) {
        buffer[i] = pattern_value;
    }
}

// Kernel to transform a float array (multiply by scale factor)
__global__ void poc_transform_kernel_float_impl(float* output, const float* input, uint32_t num_elements, float scale_factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        output[idx] = input[idx] * scale_factor;
    }
}

// Wrapper for poc_copy_kernel_dto_d_impl
extern "C" cudaError_t launch_poc_copy_kernel_dto_d(void* output_device_ptr, 
                                                  const void* input_device_ptr, 
                                                  uint32_t size_bytes, 
                                                  cudaStream_t stream) {
    if (!output_device_ptr || !input_device_ptr || size_bytes == 0) {
        return cudaErrorInvalidValue;
    }
    // For byte-wise operations, block/grid can be generic.
    // Example: 256 threads per block. Grid size to cover all bytes.
    int threads_per_block = 256;
    int blocks_per_grid = (size_bytes + threads_per_block - 1) / threads_per_block;
    // Ensure gridDim is not excessively large if size_bytes is huge. MaxGridDim an issue.
    // For POC, assume size_bytes is manageable for this simple grid.
    
    poc_copy_kernel_dto_d_impl<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        (unsigned char*)output_device_ptr, (const unsigned char*)input_device_ptr, size_bytes);
    return cudaGetLastError(); // Check for launch errors
}

// Wrapper for poc_generate_data_kernel_impl
extern "C" cudaError_t launch_poc_generate_data_kernel(void* device_ptr, 
                                                     uint32_t size_bytes, 
                                                     unsigned char pattern_value, 
                                                     cudaStream_t stream) {
    if (!device_ptr || size_bytes == 0) {
        return cudaErrorInvalidValue;
    }
    int threads_per_block = 256;
    int blocks_per_grid = (size_bytes + threads_per_block - 1) / threads_per_block;

    poc_generate_data_kernel_impl<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        (unsigned char*)device_ptr, size_bytes, pattern_value);
    return cudaGetLastError();
}

// Wrapper for poc_transform_kernel_float_impl
extern "C" cudaError_t launch_poc_transform_kernel_float(float* output_device_ptr, 
                                                       const float* input_device_ptr, 
                                                       uint32_t num_elements, 
                                                       float scale_factor, 
                                                       cudaStream_t stream) {
    if (!output_device_ptr || !input_device_ptr || num_elements == 0) {
        return cudaErrorInvalidValue;
    }
    int threads_per_block = 256;
    int blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;

    poc_transform_kernel_float_impl<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        output_device_ptr, input_device_ptr, num_elements, scale_factor);
    return cudaGetLastError();
}
